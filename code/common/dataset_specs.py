"""
Canonical column-role specification for each candidate final dataset.

Each DatasetSpec maps a dataset's raw schema onto a common set of *roles* used by
every downstream component (leakage-safe splitter, graph builder, POMDP env,
baselines). Only datasets that pass screening (see manifests/dataset_inventory.csv
and manifests/dataset_selection_report.md) are actually instantiated in experiments,
but specs are kept for all inspected candidates so the screening step is reproducible.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os

DATA_ROOT = r"C:\Users\MMASSAOUDI\Desktop\Data"


@dataclass
class DatasetSpec:
    name: str
    files: List[str]
    label_col: str
    binary_positive_fn: Optional[str]   # name of a function key in LABEL_TO_BINARY
    multiclass_map_fn: Optional[str]    # name of a function key in LABEL_TO_MULTICLASS
    src_ip_col: Optional[str] = None
    dst_ip_col: Optional[str] = None
    src_port_col: Optional[str] = None
    dst_port_col: Optional[str] = None
    proto_col: Optional[str] = None
    service_col: Optional[str] = None
    time_col: Optional[str] = None
    session_id_cols: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)  # leakage-prone / identifier cols to exclude from X
    categorical_cols: List[str] = field(default_factory=list)
    na_values: List[str] = field(default_factory=lambda: ["-", "?", "", "NaN", "nan"])
    native_split: Optional[Dict[str, List[str]]] = None  # if set, {"train":[files],"val":[files],"test":[files]}


SPECS: Dict[str, DatasetSpec] = {

    "CICIoT2023": DatasetSpec(
        name="CICIoT2023",
        files=[os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "train", "train.csv"),
               os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "validation", "validation.csv"),
               os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "test", "test.csv")],
        label_col="label",
        binary_positive_fn="identity_binary",
        multiclass_map_fn="identity_multiclass",
        src_ip_col=None, dst_ip_col=None,
        src_port_col=None, dst_port_col=None,
        proto_col="Protocol Type", service_col=None,
        time_col=None,
        session_id_cols=[],
        drop_cols=[],
        categorical_cols=[],
        # The original curators already provide a train/validation/test split by file;
        # we respect it rather than re-shuffling, and only apply post-hoc exact
        # cross-split duplicate removal (see leakage_safe_split.load_native_split).
        native_split={
            "train": [os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "train", "train.csv")],
            "val": [os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "validation", "validation.csv")],
            "test": [os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "test", "test.csv")],
        },
    ),

    "ToN-IoT-Network": DatasetSpec(
        name="ToN-IoT-Network",
        files=[os.path.join(DATA_ROOT, "TONIoT Network Dataset", "train_test_network.csv")],
        label_col="label",
        binary_positive_fn="identity_binary",
        multiclass_map_fn="tonoiot_type",
        src_ip_col="src_ip", dst_ip_col="dst_ip",
        src_port_col="src_port", dst_port_col="dst_port",
        proto_col="proto", service_col="service",
        time_col=None,
        session_id_cols=["src_ip", "src_port", "dst_ip", "dst_port", "proto"],
        # 'type' is the multiclass label; src_ip/dst_ip are raw identifiers used only for
        # graph construction and host-isolation bookkeeping, never as a model feature.
        drop_cols=["type", "src_ip", "dst_ip"],
        categorical_cols=["proto", "service", "conn_state",
                           "dns_query", "ssl_version", "ssl_cipher", "http_method"],
    ),

    "UNSW-NB15": DatasetSpec(
        name="UNSW-NB15",
        files=[os.path.join(DATA_ROOT, "UNSW_NB15", "UNSW_NB15_training-set.csv"),
               os.path.join(DATA_ROOT, "UNSW_NB15", "UNSW_NB15_testing-set.csv")],
        label_col="label",
        binary_positive_fn="identity_binary",
        multiclass_map_fn="unsw_attack_cat",
        src_ip_col=None, dst_ip_col=None,   # NB15 train/test-set CSVs do not carry raw IPs
        src_port_col=None, dst_port_col=None,
        proto_col="proto", service_col="service",
        time_col=None,
        # NB15's release CSVs carry only a row index in `id`, not a real flow 5-tuple or
        # session id, so a group-disjoint split on it would be cosmetic; we fall back to
        # stratified-random splitting honestly (documented in the split manifest) instead.
        session_id_cols=[],
        drop_cols=["attack_cat", "id"],
        categorical_cols=["proto", "service", "state"],
    ),

    "Edge-IIoTset": DatasetSpec(
        name="Edge-IIoTset",
        files=[os.path.join(DATA_ROOT, "Edge-IIoTset", "Edge-IIoTset dataset",
                             "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv")],
        label_col="Attack_label",
        binary_positive_fn="identity_binary",
        multiclass_map_fn="edge_iiot_attack_type",
        src_ip_col="ip.src_host", dst_ip_col="ip.dst_host",
        src_port_col="tcp.srcport", dst_port_col="tcp.dstport",
        proto_col=None, service_col=None,
        time_col="frame.time",
        session_id_cols=["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport"],
        drop_cols=["Attack_type", "ip.src_host", "ip.dst_host", "frame.time"],
        categorical_cols=["http.request.method", "dns.qry.name", "mqtt.protoname"],
    ),

    "RT-IoT2022": DatasetSpec(
        name="RT-IoT2022",
        files=[os.path.join(DATA_ROOT, "RT_IOT2022", "RT_IOT2022.csv")],
        label_col="Attack_type",
        binary_positive_fn="rtiot_binary",
        multiclass_map_fn="identity_multiclass",
        src_ip_col=None, dst_ip_col=None,
        src_port_col="id.orig_p", dst_port_col="id.resp_p",
        proto_col="proto", service_col="service",
        time_col=None,
        session_id_cols=["id.orig_p", "id.resp_p", "proto", "service"],
        drop_cols=["no"],
        categorical_cols=["proto", "service"],
    ),

    "CIC-IDS-2017": DatasetSpec(
        name="CIC-IDS-2017",
        files=[],  # populated by discover_cic_files() at load time (8 daily CSVs)
        label_col="Label",
        binary_positive_fn="identity_binary",
        multiclass_map_fn="identity_multiclass",
        src_ip_col=None, dst_ip_col=None,   # ISCX version aggregated: no raw IPs, only Destination Port
        src_port_col=None, dst_port_col="Destination Port",
        proto_col=None, service_col=None,
        time_col=None,
        session_id_cols=[],
        drop_cols=[],
        categorical_cols=[],
    ),

    "NSL-KDD": DatasetSpec(
        name="NSL-KDD",
        files=[os.path.join(DATA_ROOT, "KDD", "kdd_train.csv"),
               os.path.join(DATA_ROOT, "KDD", "kdd_test.csv")],
        label_col="labels",
        binary_positive_fn="identity_binary",
        multiclass_map_fn="identity_multiclass",
        proto_col="protocol_type", service_col="service",
        session_id_cols=[],
        drop_cols=[],
        categorical_cols=["protocol_type", "service", "flag"],
    ),

    "APA-DDoS": DatasetSpec(
        name="APA-DDoS",
        files=[os.path.join(DATA_ROOT, "APA-DDoS-Dataset", "APA-DDoS-Dataset.csv")],
        label_col="Label",
        binary_positive_fn="identity_binary",
        multiclass_map_fn=None,
        src_ip_col="ip.src", dst_ip_col="ip.dst",
        src_port_col="tcp.srcport", dst_port_col="tcp.dstport",
        proto_col="ip.proto", service_col=None,
        time_col="frame.time",
        session_id_cols=["ip.src", "ip.dst", "tcp.srcport", "tcp.dstport"],
        drop_cols=["ip.src", "ip.dst", "frame.time"],
        categorical_cols=[],
    ),
}


def discover_cic_files():
    d = os.path.join(DATA_ROOT, "CIC-IDS- 2017")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".csv"))


SPECS["CIC-IDS-2017"].files = discover_cic_files()
