"""
Part 9: Leakage-safe preprocessing.

Generic, dataset-agnostic pipeline:
 1. Load raw file(s) for a DatasetSpec.
 2. Derive binary (and, if available, multiclass) labels via label_maps.
 3. Split BEFORE any fitting: session/entity-disjoint if session_id_cols are
    available (grouped split so the same 5-tuple/session never appears in more
    than one of train/val/test), else stratified random split as a documented
    fallback (recorded in the manifest).
 4. Remove exact cross-split duplicates (on the full feature+label row) after
    the split is fixed, dropping the duplicate from val/test (keep train copy)
    so no information about a test row leaks via an identical training row.
 5. Fit imputer/scaler/encoder on TRAIN ONLY; apply to val/test.
 6. Save a split manifest (row counts, class balance per split, dedup counts,
    split strategy) to manifests/<dataset>_split_manifest.json for auditability.

No oversampling happens before the split (SMOTE etc. is applied only inside
model-training folds, on the resampled TRAIN partition, never before splitting
and never on val/test).
"""
import os, json, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.dataset_specs import SPECS, DatasetSpec
from common.label_maps import BINARY_FN, MULTICLASS_FN, heuristic_attack_stage

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)


def _clean_colnames(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw(spec: DatasetSpec, nrows_per_file=None, encoding="utf-8-sig"):
    frames = []
    for fp in spec.files:
        try:
            df = pd.read_csv(fp, nrows=nrows_per_file, encoding=encoding,
                              na_values=spec.na_values, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, nrows=nrows_per_file, encoding="latin1",
                              na_values=spec.na_values, on_bad_lines="skip", low_memory=False)
        df = _clean_colnames(df)
        df["__source_file__"] = os.path.basename(fp)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def build_labels(df: pd.DataFrame, spec: DatasetSpec):
    y_bin = BINARY_FN[spec.binary_positive_fn](df[spec.label_col]) if spec.binary_positive_fn else None
    y_multi = None
    if spec.multiclass_map_fn:
        multi_col = spec.label_col
        if spec.multiclass_map_fn == "tonoiot_type":
            multi_col = "type"
        elif spec.multiclass_map_fn == "unsw_attack_cat":
            multi_col = "attack_cat"
        elif spec.multiclass_map_fn == "edge_iiot_attack_type":
            multi_col = "Attack_type"
        if multi_col in df.columns:
            y_multi = MULTICLASS_FN[spec.multiclass_map_fn](df[multi_col])
    return y_bin, y_multi


def make_group_key(df: pd.DataFrame, spec: DatasetSpec):
    cols = [c for c in spec.session_id_cols if c in df.columns]
    if not cols:
        return None
    key = df[cols].astype(str).agg("_".join, axis=1)
    return key


def leakage_safe_split_native(dataset_name: str, nrows_per_file=None, random_state=42,
                               max_rows_per_split=None):
    """For datasets whose curators already provide a train/val/test split by file
    (currently: CICIoT2023). Loads each part separately, builds labels, and skips
    the group/stratified re-split logic -- only cross-split exact-duplicate removal
    and train-only fitting are applied downstream."""
    spec = SPECS[dataset_name]
    assert spec.native_split is not None
    parts = {}
    for split_name, files in spec.native_split.items():
        sub_spec = DatasetSpec(**{**spec.__dict__, "files": files})
        part_df = load_raw(sub_spec, nrows_per_file=nrows_per_file)
        if max_rows_per_split is not None and len(part_df) > max_rows_per_split:
            part_df = part_df.sample(n=max_rows_per_split, random_state=random_state).reset_index(drop=True)
        parts[split_name] = part_df

    df = pd.concat([parts["train"], parts["val"], parts["test"]], ignore_index=True)
    y_bin, y_multi = build_labels(df, spec)
    df = df.assign(__y_bin__=y_bin.values if y_bin is not None else np.nan)
    if y_multi is not None:
        df = df.assign(__y_multi__=y_multi.values)

    n_train, n_val, n_test = len(parts["train"]), len(parts["val"]), len(parts["test"])
    splits = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n_train + n_val + n_test),
    }

    feature_cols = [c for c in df.columns if c not in ("__y_bin__", "__y_multi__", "__source_file__")]
    row_hash = pd.util.hash_pandas_object(df[feature_cols].astype(str), index=False)
    train_hashes = set(row_hash.values[splits["train"]])
    dedup_report = {}
    for split_name in ["val", "test"]:
        sidx = splits[split_name]
        keep_mask = ~np.isin(row_hash.values[sidx], list(train_hashes))
        dedup_report[split_name] = {"before": int(len(sidx)), "dropped_exact_dupe_of_train": int((~keep_mask).sum())}
        splits[split_name] = sidx[keep_mask]

    manifest = {
        "dataset": dataset_name,
        "split_strategy": "native_curator_split_by_file",
        "n_rows_total": int(len(df)),
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "dedup_report": dedup_report,
        "class_balance": {
            k: (df["__y_bin__"].iloc[v].value_counts().to_dict() if y_bin is not None else None)
            for k, v in splits.items()
        },
        "random_state": random_state,
    }
    with open(os.path.join(MANIFEST_DIR, f"{dataset_name}_split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return df, splits, spec, manifest


def leakage_safe_split(dataset_name: str, val_frac=0.15, test_frac=0.15,
                        nrows_per_file=None, random_state=42, max_total_rows=None):
    spec = SPECS[dataset_name]
    if spec.native_split is not None:
        return leakage_safe_split_native(dataset_name, nrows_per_file=nrows_per_file,
                                          random_state=random_state,
                                          max_rows_per_split=max_total_rows)
    df = load_raw(spec, nrows_per_file=nrows_per_file)
    if max_total_rows is not None and len(df) > max_total_rows:
        df = df.sample(n=max_total_rows, random_state=random_state).reset_index(drop=True)

    y_bin, y_multi = build_labels(df, spec)
    df = df.assign(__y_bin__=y_bin.values if y_bin is not None else np.nan)
    if y_multi is not None:
        df = df.assign(__y_multi__=y_multi.values)

    group_key = make_group_key(df, spec)
    strategy = "group_disjoint" if group_key is not None else "stratified_random"

    idx = np.arange(len(df))
    if group_key is not None:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
        trainval_idx, test_idx = next(gss1.split(idx, groups=group_key.values))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=random_state)
        train_idx, val_idx = next(gss2.split(trainval_idx, groups=group_key.values[trainval_idx]))
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]
    else:
        strat = df["__y_bin__"] if y_bin is not None else None
        trainval_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=random_state,
                                                    stratify=strat.values if strat is not None else None)
        strat2 = df["__y_bin__"].values[trainval_idx] if y_bin is not None else None
        train_idx, val_idx = train_test_split(trainval_idx, test_size=val_frac / (1 - test_frac),
                                                random_state=random_state, stratify=strat2)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # --- exact cross-split duplicate removal (drop from val/test, keep train copy) ---
    feature_cols = [c for c in df.columns if c not in ("__y_bin__", "__y_multi__", "__source_file__")]
    row_hash = pd.util.hash_pandas_object(df[feature_cols].astype(str), index=False)
    train_hashes = set(row_hash.values[train_idx])
    dedup_report = {}
    for split_name in ["val", "test"]:
        sidx = splits[split_name]
        keep_mask = ~np.isin(row_hash.values[sidx], list(train_hashes))
        dedup_report[split_name] = {"before": int(len(sidx)), "dropped_exact_dupe_of_train": int((~keep_mask).sum())}
        splits[split_name] = sidx[keep_mask]

    manifest = {
        "dataset": dataset_name,
        "split_strategy": strategy,
        "n_rows_total": int(len(df)),
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "dedup_report": dedup_report,
        "class_balance": {
            k: (df["__y_bin__"].iloc[v].value_counts().to_dict() if y_bin is not None else None)
            for k, v in splits.items()
        },
        "random_state": random_state,
    }
    with open(os.path.join(MANIFEST_DIR, f"{dataset_name}_split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return df, splits, spec, manifest


def fit_transform_train_only(df, splits, spec: DatasetSpec, exclude_cols=None):
    """Fit imputer + ordinal-encode categoricals + robust-scale numerics on TRAIN ONLY."""
    exclude_cols = set(exclude_cols or []) | {"__y_bin__", "__y_multi__", "__source_file__", spec.label_col}
    exclude_cols |= set(spec.drop_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    train_df = df.iloc[splits["train"]]
    # Any column that isn't numeric dtype is treated as categorical regardless of whether
    # it was hand-listed in spec.categorical_cols -- IDS CSVs routinely carry extra
    # string/boolean flag columns (e.g. 'T'/'F' TLS flags) that are easy to miss by hand.
    auto_cat = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = sorted(set(spec.categorical_cols) | set(auto_cat))
    cat_cols = [c for c in cat_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_imputer = SimpleImputer(strategy="median").fit(train_df[num_cols]) if num_cols else None
    scaler = RobustScaler().fit(num_imputer.transform(train_df[num_cols])) if num_cols else None
    cat_imputer = SimpleImputer(strategy="most_frequent").fit(train_df[cat_cols]) if cat_cols else None
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
        cat_imputer.transform(train_df[cat_cols])) if cat_cols else None

    def transform(part_df):
        arrs = []
        if num_cols:
            arrs.append(scaler.transform(num_imputer.transform(part_df[num_cols])))
        if cat_cols:
            arrs.append(encoder.transform(cat_imputer.transform(part_df[cat_cols])))
        return np.concatenate(arrs, axis=1) if arrs else np.zeros((len(part_df), 0))

    X = {name: transform(df.iloc[idx]) for name, idx in splits.items()}
    feature_names = num_cols + cat_cols
    return X, feature_names, dict(num_cols=num_cols, cat_cols=cat_cols)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max_total_rows", type=int, default=None)
    args = ap.parse_args()
    df, splits, spec, manifest = leakage_safe_split(args.dataset, max_total_rows=args.max_total_rows)
    print(json.dumps(manifest, indent=2, default=str))
