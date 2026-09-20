# Dataset Screening and Selection Report

Source: `manifests/dataset_inventory.csv` (full recursive scan, 2026-07-28) and
`results/logs/verify_candidates2.log` (targeted follow-up checks). All 14 nominal
dataset folders under `C:\Users\MMASSAOUDI\Desktop\Data` resolve to **12 unique
dataset families** (two folder pairs are duplicate copies of the same file:
`RT_IOT/` == `RT_IOT2022/RT_IOT2022.csv`, and `UNSW_NB15/` == `dataset/`).

## Inventory summary (Table I source data)

| Dataset | Files | Size | Rows (est.) | Features | Label col | Classes (sampled) | IP? | Port? | Proto? | Time? | Dup rate (sample) |
|---|---:|---:|---:|---:|---|---:|:-:|:-:|:-:|:-:|---:|
| APA-DDoS | 1 | 21.7 MB | 151,200 | 22 | Label | 3 | Y | Y | Y | Y | 0.0% |
| Bot-IoT | 74 | 15.0 GB | 73,370,443 | 34 | attack | 2 (sampled) | Y | Y | Y | Y | 17.1%* |
| CIC-IDS-2017 | 8 | 884.6 MB | 2,830,743 | 78 | Label | 14 | N | Y | N | N | 1.4% |
| CICIoT2023 | 3 | 2.32 GB | 7,845,673 | 46 | label | 34 | N | N | Y | N | 0.02% (+0.2% cross-split) |
| CIC-VPN2016-flow | 1 | 13.1 MB | 59,706 | 23 | traffic_type | 14 | N | N | N | Y | 16.7% |
| ISCX-VPN2016-CS240 | 4 | 128.8 MB | 271,028 | 88 | Label | 10 | Y | Y | Y | Y | 0.0% |
| Edge-IIoTset | 1 | 1.22 GB | 2,219,201 | 62 | Attack_label | 2 (full scan) | Y | Y | N | Y | 0.0% |
| ISCX-IDS-2012 | 2 | 783.6 MB | 5,658,998 | 29 | label | 2 | Y | Y | Y | Y | **60.3%** |
| NSL-KDD | 2 | 17.5 MB | 148,517 | 41 | labels | 37 | N | Y | Y | Y | 0.0% |
| RT-IoT2022 | 1 | 54.8 MB | 123,117 | 84 | Attack_type | 12 | N | Y | Y | Y | 0.0% |
| ToN-IoT-Network | 1 | 29.9 MB | 211,043 | 43 | label | 2 (bin.) / 9 (multi.) | Y | Y | Y | N | 6.4% |
| UNSW-NB15 | 2 | 47.7 MB | 257,673 | 44 | label | 2 (bin.) / ~10 (multi.) | N | Y | Y | N | 0.0% |

\* Bot-IoT duplicate rate and class balance are unreliable from prefix sampling because the
74 shard files are attack-concentrated; a representative estimate would require a full
stratified scan of all 15 GB, which was judged computationally disproportionate for this
study (see rejection reason below).

Follow-up checks resolved two sampling artifacts: Edge-IIoTset's prefix sample looked
100%-benign because attack rows are concentrated later in the concatenated file; a full
column scan shows a real 72.8%/27.2% benign/attack split. CICIoT2023's train/validation/test
files (provided pre-split by the original curators) show a small (~0.19%, 375-392 of 200k
sampled rows) exact-duplicate rate across splits, which our pipeline removes post-hoc from
val/test before any model sees them.

## Screening against the seven predeclared selection criteria

| Criterion | APA-DDoS | Bot-IoT | CIC-IDS-2017 | CICIoT2023 | VPN2016 (both) | Edge-IIoTset | ISCX-2012 | NSL-KDD | RT-IoT2022 | ToN-IoT-Net | UNSW-NB15 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1. Leakage-safe split feasible | partial (no session id) | Y (session id) | partial | Y (native split + session-less dedup) | Y | partial (time only) | partial | partial | Y (5-tuple) | Y (5-tuple) | partial (has `id`) |
| 2. Usable labels | Y | Y (label only in sampled shards) | Y (encoding quirk, fixable) | Y | **N — traffic-type, not attack/benign** | Y | Y | Y | Y | Y | Y |
| 3. Graph/temporal/sequential construction | Y (IP) | Y (IP) | protocol-role only | protocol-role only | protocol-role only | Y (IP) | Y (IP) | protocol-role only | protocol-role only | Y (IP) | protocol-role only |
| 4. Not saturated by all strong baselines | risk (3-class DDoS, small) | unknown (untested at scale) | Y | Y | n/a (rejected on #2) | Y | **risk (60% dup rate)** | **known-saturated (literature)** | Y (imbalanced, 12-class) | Y | Y |
| 5. Meaningful binary and/or multiclass | binary only | binary (sampled) | Y both | Y both | n/a | binary (+ Attack_type avail.) | binary only | Y both | Y both | Y both | Y both |
| 6. Enough samples for repeated-seed testing | marginal (151K, 3 classes) | Y (huge) | Y | Y | n/a | Y | Y | marginal | marginal (123K) | Y | Y |
| 7. Supports meaningful RL decision process | weak (single DDoS pattern) | Y | Y | Y | n/a | Y | weak (dup-heavy) | weak (literature-saturated) | Y | Y | Y |

## Final selection (4 datasets)

1. **CICIoT2023** — modern large-scale IoT/IIoT slot. 34 attack/benign classes, native
   train/validation/test split by the original curators (respected, with post-hoc exact-duplicate
   removal), large enough for repeated-seed testing and low-label-fraction experiments,
   protocol-role graph construction (no raw host IPs released).
2. **ToN-IoT-Network** — CPS/industrial-pattern slot. Originates from the UNSW IoT/IIoT
   telemetry-fusion testbed; the network CSV retains real source/destination IPs and ports,
   enabling genuine host-flow graphs and host-isolation actions; 8 balanced attack subtypes
   (20,000 rows each) plus benign.
3. **UNSW-NB15** — general network-IDS anchor. The most widely benchmarked dataset of the
   twelve, moderate size (257K rows) keeps repeated-seed / multi-baseline experiments
   tractable, supports both binary and ~10-class `attack_cat` multiclass evaluation.
4. **RT-IoT2022** (supplementary) — small (123K rows) real-time IoT protocol dataset (MQTT,
   ARP poisoning, NMAP scan variants, cloud telemetry) used only for the cross-dataset-transfer
   check (Experiment 11) and as an additional robustness data point; **not used alone to support
   the paper's primary superiority claims**, consistent with its small size and heavy class
   skew (77% single attack type).

Per the task's explicit instruction, **NSL-KDD and APA-DDoS are held out as supplementary
sanity-control datasets only** (NSL-KDD is a widely-documented saturated/duplicated derivative
of KDD Cup 99; APA-DDoS covers a single DDoS pattern with only 3 classes and 151K rows) and are
never used as primary evidence for the method-comparison claims.

## Rejected candidates and reasons

- **Bot-IoT**: real dataset, real IDS value, but at 15 GB / 74 shard files it is
  disproportionate to the compute/time budget of this study; the prefix-sampled label
  distribution is also unreliable (shards are attack-concentrated), meaning a trustworthy
  minority-class estimate would require a full re-scan we did not have budget for. This is a
  **scope limitation, not a claim that Bot-IoT is unsuitable** — flagged explicitly in the
  paper's Limitations section.
- **CIC-IDS-2017**: legitimate general-IDS dataset, but redundant with UNSW-NB15 in the
  "general network IDS" slot, larger (79 features, 2.8M rows) which would cost tractability
  without adding a new pattern, and has a known label-encoding artifact (`Web Attack – XSS`
  written with a non-UTF8 dash) that must be repaired before use. Kept as a reserve dataset.
- **Edge-IIoTset**: legitimate and well-balanced (72.8/27.2%) once the sampling artifact is
  corrected, but redundant with CICIoT2023 in the "modern IoT/IIoT" slot within a 3-4 dataset
  budget. Kept as a reserve dataset.
- **CIC-VPN2016-flow / ISCX-VPN2016-CS240**: fail criterion 2 — labels encode traffic/application
  type (chat, VoIP, streaming, VPN-vs-plain) rather than benign/attack, so they are not IDS
  datasets in the required sense. Not used.
- **ISCX-IDS-2012**: fails criterion 4 — 60.3% duplicate rate in a 20K-row sample indicates a
  serious data-quality/leakage risk for this legacy corpus. Not used.
