"""Targeted follow-up checks on the shortlisted candidate datasets before final selection:
 - full (not prefix-capped) label distribution for Edge-IIoTset (prefix sample was degenerate)
 - CICIoT2023 cross-split exact-duplicate rate (train vs val vs test) -> leakage check
 - ToN-IoT-Network label vs type consistency
 - CIC-IDS-2017 label encoding fix confirmation
 - RT-IoT2022 quick class balance
"""
import pandas as pd, os, hashlib

DATA_ROOT = r"C:\Users\MMASSAOUDI\Desktop\Data"

def full_label_scan(path, col_idx, chunksize=500000):
    counts = {}
    total = 0
    for chunk in pd.read_csv(path, usecols=[col_idx], header=0, chunksize=chunksize,
                             encoding="utf-8-sig", on_bad_lines="skip", low_memory=False):
        vc = chunk.iloc[:, 0].astype(str).value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
        total += len(chunk)
    return counts, total

print("=== Edge-IIoTset full label scan ===")
p = os.path.join(DATA_ROOT, "Edge-IIoTset", "Edge-IIoTset dataset", "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv")
header = pd.read_csv(p, nrows=0).columns.tolist()
idx = header.index("Attack_label")
counts, total = full_label_scan(p, idx)
print("total rows scanned:", total, "counts:", counts)

print("\n=== CICIoT2023 cross-split hash overlap (sampled) ===")
base = os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23")
def row_hashes(path, n=200000):
    df = pd.read_csv(path, nrows=n)
    h = pd.util.hash_pandas_object(df, index=False)
    return set(h.values)
h_train = row_hashes(os.path.join(base, "train", "train.csv"))
h_val = row_hashes(os.path.join(base, "validation", "validation.csv"))
h_test = row_hashes(os.path.join(base, "test", "test.csv"))
print("train-val overlap (of sampled 200k each):", len(h_train & h_val))
print("train-test overlap (of sampled 200k each):", len(h_train & h_test))
print("val-test overlap (of sampled 200k each):", len(h_val & h_test))

print("\n=== ToN-IoT-Network label vs type ===")
p2 = os.path.join(DATA_ROOT, "TONIoT Network Dataset", "train_test_network.csv")
df2 = pd.read_csv(p2, usecols=["label", "type"], encoding="utf-8-sig")
print(pd.crosstab(df2["label"], df2["type"]))

print("\n=== CIC-IDS-2017 label encoding check (Wednesday file, web-attack rows) ===")
p3 = os.path.join(DATA_ROOT, "CIC-IDS- 2017", "Wednesday-workingHours.pcap_ISCX.csv")
for enc in ["utf-8", "cp1252", "latin1"]:
    try:
        df3 = pd.read_csv(p3, usecols=[" Label"], encoding=enc, nrows=5000)
        print(enc, "->", df3[" Label"].unique()[:8])
    except Exception as e:
        print(enc, "FAILED", e)

print("\n=== RT-IoT2022 class balance ===")
p4 = os.path.join(DATA_ROOT, "RT_IOT2022", "RT_IOT2022.csv")
df4 = pd.read_csv(p4, usecols=["Attack_type"])
print(df4["Attack_type"].value_counts())
