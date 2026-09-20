"""
Dataset-specific raw-label -> {binary, multiclass, attack_stage} mappings.
Kept separate from dataset_specs.py so mapping logic is auditable and testable.
attack_stage is a coarse MITRE-ATT&CK-ish stage index used only by the
neuro-symbolic candidate (Candidate 8) and the POMDP escalation logic; it is a
heuristic label, not ground truth, and is documented as such in the paper.
"""
import numpy as np
import pandas as pd

BENIGN_TOKENS = {"benign", "normal", "0", "0.0"}


def identity_binary(y_raw: pd.Series) -> pd.Series:
    """For datasets whose label column is already {0,1}, an exact benign token, or a
    class-name string that starts with a benign-like word (e.g. CICIoT2023's
    'BenignTraffic'). Substring/prefix matching is required here because several
    datasets embed the benign token inside a longer class-name string rather than
    using it verbatim."""
    s = y_raw.astype(str).str.strip().str.lower()
    if set(s.unique()) <= {"0", "1"}:
        return s.astype(int)
    is_benign = s.isin(BENIGN_TOKENS) | s.str.startswith("benign") | s.str.startswith("normal")
    return (~is_benign).astype(int)


def rtiot_binary(y_raw: pd.Series) -> pd.Series:
    s = y_raw.astype(str).str.strip().str.lower()
    benign_classes = {"mqtt_publish", "thing_speak", "wipro_bulb", "arp_poisioning",
                       "idle", "normal", "benign"}
    return (~s.isin(benign_classes)).astype(int)


def identity_multiclass(y_raw: pd.Series) -> pd.Series:
    s = y_raw.astype(str).str.strip()
    return s


def tonoiot_type(y_raw: pd.Series) -> pd.Series:
    return y_raw.astype(str).str.strip().str.lower()


def unsw_attack_cat(y_raw: pd.Series) -> pd.Series:
    s = y_raw.astype(str).str.strip()
    s = s.replace({"": "Normal", "nan": "Normal"})
    return s


def edge_iiot_attack_type(y_raw: pd.Series) -> pd.Series:
    return y_raw.astype(str).str.strip()


BINARY_FN = {
    "identity_binary": identity_binary,
    "rtiot_binary": rtiot_binary,
}

MULTICLASS_FN = {
    "identity_multiclass": identity_multiclass,
    "tonoiot_type": tonoiot_type,
    "unsw_attack_cat": unsw_attack_cat,
    "edge_iiot_attack_type": edge_iiot_attack_type,
}

# Coarse attack-stage heuristic (Reconnaissance=0, InitialAccess/Exploit=1,
# C2/Lateral=2, ActionOnObjectives/Impact=3) used only for candidate 8 (neuro-symbolic)
# action masking and reward shaping -- NOT presented as ground-truth ATT&CK labels.
STAGE_KEYWORDS = {
    0: ["scan", "recon", "probe", "sniff"],
    1: ["exploit", "backdoor", "injection", "password", "brute", "shellcode", "sql"],
    2: ["c2", "command", "lateral", "mitm", "spoof", "poison"],
    3: ["ddos", "dos", "ransomware", "exfilt", "impact"],
}


def heuristic_attack_stage(multiclass_label: str) -> int:
    s = str(multiclass_label).lower()
    for stage, kws in STAGE_KEYWORDS.items():
        if any(k in s for k in kws):
            return stage
    return -1  # benign / unknown
