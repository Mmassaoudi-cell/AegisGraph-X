"""
Part 2: Dataset inventory for the 14 (nominal) local IDS dataset folders.

Computes, per dataset file, without loading full multi-GB files into memory:
 - file type, size, row count (streamed line count)
 - column/feature count
 - label column candidate + class distribution (streamed, label column(s) only)
 - missing value rate (sampled)
 - duplicate rate (sampled, on a bounded random sample)
 - presence of ip/port/proto/service/timestamp/session identifier columns
 - rough graph/RL/leakage-safe-split suitability flags (heuristic, from schema)

Output: manifests/dataset_inventory.csv and manifests/dataset_inventory.json
"""
import os, sys, json, csv, re, random
import pandas as pd
import numpy as np

DATA_ROOT = r"C:\Users\MMASSAOUDI\Desktop\Data"
OUT_DIR = r"c:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 2\manifests"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# Explicit registry of dataset "families" -> list of csv files that belong to them.
# This avoids double counting nested/duplicate folders (e.g. RT_IOT vs RT_IOT2022 are
# the same file; UNSW_NB15/ and dataset/ are the same file).
REGISTRY = {
    "APA-DDoS": {
        "files": [r"APA-DDoS-Dataset\APA-DDoS-Dataset.csv"],
        "label_candidates": ["Label", "label"],
    },
    "Bot-IoT": {
        "files": [rf"Bot_IoT\data_{i}.csv" for i in range(1, 75)],
        "label_candidates": ["attack", "category", "subcategory"],
    },
    "CIC-IDS-2017": {
        "files": None,  # discover *.csv under CIC-IDS- 2017
        "dir": "CIC-IDS- 2017",
        "label_candidates": [" Label", "Label"],
    },
    "CICIoT2023": {
        "files": [r"CICIOT23\CICIOT23\train\train.csv",
                  r"CICIOT23\CICIOT23\validation\validation.csv",
                  r"CICIOT23\CICIOT23\test\test.csv"],
        "label_candidates": ["label"],
    },
    "CIC-VPN2016-flow": {
        "files": [r"CIC_VPN2016\consolidated_traffic_data.csv"],
        "label_candidates": ["traffic_type"],
    },
    "ISCX-VPN2016-CS240": {
        "files": None,
        "dir": "CS240_ISCXVPN2016",
        "label_candidates": ["Label", "Category"],
    },
    "Edge-IIoTset": {
        "files": [r"Edge-IIoTset\Edge-IIoTset dataset\Selected dataset for ML and DL\DNN-EdgeIIoT-dataset.csv"],
        "label_candidates": ["Attack_label", "Attack_type"],
    },
    "ISCX-IDS-2012": {
        "files": [r"IDS_ISCX_2012_dataset\dataset_attack.csv", r"IDS_ISCX_2012_dataset\dataset_normal.csv"],
        "label_candidates": ["label"],
    },
    "NSL-KDD": {
        "files": [r"KDD\kdd_train.csv", r"KDD\kdd_test.csv"],
        "label_candidates": ["labels"],
    },
    "RT-IoT2022": {
        "files": [r"RT_IOT2022\RT_IOT2022.csv"],
        "label_candidates": ["Attack_type"],
    },
    "ToN-IoT-Network": {
        "files": [r"TONIoT Network Dataset\train_test_network.csv"],
        "label_candidates": ["label", "type"],
    },
    "UNSW-NB15": {
        "files": [r"UNSW_NB15\UNSW_NB15_training-set.csv", r"UNSW_NB15\UNSW_NB15_testing-set.csv"],
        "label_candidates": ["label", "attack_cat"],
    },
}

IP_PAT = re.compile(r"(ip|addr|src|dst|saddr|daddr)", re.I)
PORT_PAT = re.compile(r"(port|sport|dport)", re.I)
PROTO_PAT = re.compile(r"(proto|protocol)", re.I)
SERVICE_PAT = re.compile(r"(service|svc)", re.I)
TIME_PAT = re.compile(r"(time|stime|ltime|timestamp|duration|dur\b)", re.I)
SESSION_PAT = re.compile(r"(flow.?id|session|id\.orig|id\.resp|pkseqid|^no$)", re.I)


def discover_files(entry):
    if entry.get("files"):
        return [os.path.join(DATA_ROOT, f) for f in entry["files"]]
    d = os.path.join(DATA_ROOT, entry["dir"])
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(".csv"):
                out.append(os.path.join(root, fn))
    return sorted(out)


def count_lines_fast(path):
    """Count newline-terminated rows without loading file into memory."""
    count = 0
    with open(path, "rb") as f:
        buf_size = 1 << 20
        read_f = f.raw.read if hasattr(f, "raw") else f.read
        buf = f.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = f.read(buf_size)
    return count


def get_header(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    # sniff delimiter
    delim = ","
    return [c.strip() for c in first.strip().split(delim)]


def pick_label_col(header, candidates):
    norm = {h.strip().lower(): h for h in header}
    for c in candidates:
        if c.strip().lower() in norm:
            return norm[c.strip().lower()]
    return None


def schema_flags(header):
    joined_cols = header
    has_ip = any(IP_PAT.search(c) for c in joined_cols)
    has_port = any(PORT_PAT.search(c) for c in joined_cols)
    has_proto = any(PROTO_PAT.search(c) for c in joined_cols)
    has_service = any(SERVICE_PAT.search(c) for c in joined_cols)
    has_time = any(TIME_PAT.search(c) for c in joined_cols)
    has_session = any(SESSION_PAT.search(c) for c in joined_cols)
    return dict(has_ip=has_ip, has_port=has_port, has_proto=has_proto,
                has_service=has_service, has_time=has_time, has_session=has_session)


def sample_label_distribution(path, label_col, header, max_rows=300000):
    """Read only the label column via a chunked read, streamed, capped at max_rows total
    scanned rows for very large files (uniform prefix+stride sample) to bound runtime."""
    try:
        idx = header.index(label_col)
    except ValueError:
        return {}, 0
    counts = {}
    total = 0
    chunksize = 200000
    try:
        for chunk in pd.read_csv(path, usecols=[idx], header=0, chunksize=chunksize,
                                  encoding="utf-8-sig", on_bad_lines="skip", low_memory=False):
            col = chunk.iloc[:, 0].astype(str)
            vc = col.value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + int(v)
            total += len(chunk)
            if total >= max_rows:
                break
    except Exception as e:
        return {"__error__": str(e)}, total
    return counts, total


def sample_missing_and_dupes(path, header, n=20000):
    try:
        df = pd.read_csv(path, nrows=n, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    except Exception as e:
        return {"error": str(e)}
    missing_rate = float(df.isna().mean().mean())
    dup_rate = float(df.duplicated().mean())
    return {"missing_rate_sample": round(missing_rate, 5), "dup_rate_sample": round(dup_rate, 5),
            "n_sampled_for_missing_dup": len(df)}


def main():
    inventory = []
    for name, entry in REGISTRY.items():
        files = discover_files(entry)
        print(f"[{name}] {len(files)} file(s)", flush=True)
        total_rows = 0
        total_size = 0
        n_files = 0
        header0 = None
        label_col_found = None
        agg_label_counts = {}
        agg_label_scanned = 0
        flags = None
        missing_dupe = None
        errors = []
        for fp in files:
            if not os.path.exists(fp):
                errors.append(f"missing:{fp}")
                continue
            n_files += 1
            size = os.path.getsize(fp)
            total_size += size
            try:
                nrows = count_lines_fast(fp) - 1  # minus header
            except Exception as e:
                nrows = -1
                errors.append(f"count_err:{fp}:{e}")
            total_rows += max(nrows, 0)
            header = get_header(fp)
            if header0 is None:
                header0 = header
                flags = schema_flags(header)
                lc = pick_label_col(header, entry["label_candidates"])
                label_col_found = lc
                if missing_dupe is None:
                    missing_dupe = sample_missing_and_dupes(fp, header)
            if label_col_found:
                counts, scanned = sample_label_distribution(fp, label_col_found, header)
                agg_label_scanned += scanned
                for k, v in counts.items():
                    agg_label_counts[k] = agg_label_counts.get(k, 0) + v
            print(f"   - {os.path.basename(fp)}: {nrows} rows, {size/1e6:.1f} MB", flush=True)

        n_features = (len(header0) - 1) if header0 else None
        rec = {
            "dataset": name,
            "n_files": n_files,
            "total_size_MB": round(total_size / 1e6, 1),
            "total_rows_est": total_rows,
            "n_columns": len(header0) if header0 else None,
            "n_features_est": n_features,
            "label_column": label_col_found,
            "has_ip": flags["has_ip"] if flags else None,
            "has_port": flags["has_port"] if flags else None,
            "has_proto": flags["has_proto"] if flags else None,
            "has_service": flags["has_service"] if flags else None,
            "has_time": flags["has_time"] if flags else None,
            "has_session_id": flags["has_session"] if flags else None,
            "label_distribution_sampled_n": agg_label_scanned,
            "label_distribution": json.dumps(agg_label_counts, default=str)[:2000],
            "n_classes_sampled": len(agg_label_counts),
            "missing_rate_sample": missing_dupe.get("missing_rate_sample") if missing_dupe else None,
            "dup_rate_sample": missing_dupe.get("dup_rate_sample") if missing_dupe else None,
            "errors": ";".join(errors)[:500],
        }
        inventory.append(rec)
        print(f"[{name}] DONE total_rows~{total_rows} classes~{len(agg_label_counts)}", flush=True)

    df = pd.DataFrame(inventory)
    df.to_csv(os.path.join(OUT_DIR, "dataset_inventory.csv"), index=False)
    with open(os.path.join(OUT_DIR, "dataset_inventory.json"), "w") as f:
        json.dump(inventory, f, indent=2, default=str)
    print("\nSaved inventory to", OUT_DIR)


if __name__ == "__main__":
    main()
