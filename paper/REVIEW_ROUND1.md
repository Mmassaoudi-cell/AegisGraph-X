# Peer Review Package — AegisGraph-X (Round 1)

**Manuscript:** AegisGraph-X: Tree-Guided Graph Mixture-of-Experts Reinforcement Learning for Cost-Aware Intrusion Detection
**Venue:** IEEE Transactions (tier-1)
**Review date:** 2026-07-29
**Panel:** EIC + 3 peer reviewers + Devil's Advocate
**Decision:** MAJOR REVISION (borderline reject — reframing required)

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Attribute | Determination |
|---|---|
| Primary discipline | Network security / intrusion detection (systems-security ML) |
| Secondary disciplines | Applied ML (MoE, distillation, calibration), reinforcement learning, applied statistics |
| Research paradigm | Quantitative, empirical benchmark study with ablation |
| Methodology type | Comparative benchmarking + component ablation + simulated sequential decision evaluation |
| Target journal tier | IEEE TIFS / TDSC / TNSM class |
| Paper maturity | Late-stage draft; complete experiments, unresolved framing |
| Distinguishing feature | Radical negative-result disclosure — the paper argues against four of its own five layers |

### Reviewer Personas

| Role | Configured identity |
|---|---|
| **EIC** | Editor, IEEE TIFS. Systems-security background. Cares about: does a *claim* survive? Is the venue right? Allergic to contribution-by-architecture-diagram. |
| **R1 — Methodology** | Applied statistician in ML benchmarking. Expertise: multiple-comparison procedures, equivalence testing, seed variance, reproducibility. |
| **R2 — Domain** | Network IDS researcher, 15 yrs. Expertise: CIC/UNSW/ToN corpora pathologies, class-imbalance evaluation, SOC operations. |
| **R3 — Perspective** | Decision theory / operations research. Expertise: optimal stopping, value of information, cost-sensitive classification, active sensing. |
| **DA — Devil's Advocate** | Adversarial reader. Mandate: find the strongest argument that this paper should not be published as written. |

---

## Phase 1 — Independent Reviews

### Reviewer 0 — Editor-in-Chief

**Recommendation:** Major Revision (reframing) | **Overall: 42/100**

| Dimension | Score |
|---|---|
| Originality | 45 |
| Significance | 35 |
| Fit to readership | 60 |
| Presentation | 40 |

**Strengths.** The disclosure discipline is the best I have seen in this subfield in several years. Section III's seven predeclared screening criteria (p.3), the leakage-safe split protocol (III-C), the retention of the failed legacy design as baseline #39 rather than quietly dropping it (IV-A), and the decision to report the reward-scale incomparability as a limitation rather than papering over it — these are genuinely exemplary and I want to see them survive revision intact.

**Central concern — no claim survives.** By the paper's own Section VII, in bold: the MoE router is *"not statistically distinguishable from a much simpler unweighted expert average"*; the RL investigation layer *"did not improve classification accuracy over the backbone alone"*; hard-negative mining *"did not improve the compact student."* Table VIII then reports the full system is *nominally below* the strongest single tree baseline on all three datasets (0.9378 vs 0.9403; 0.9967 vs 0.9984; 0.9533 vs 0.9599). After the honesty is accounted for, the residual claim is: *a nine-expert ensemble ties with XGBoost, and four added layers do nothing measurable.* That is a valuable finding, but it is not the finding the title, abstract, and Section IV architecture are organized around.

**Framing mismatch.** The manuscript is written as a methods paper that keeps apologizing. It should be written as what it is: a rigorous negative-results and structural-diagnosis paper. The current framing forces the reader through 5 pages of architecture (IV-C through IV-G) for components the paper then disowns. Reorganize around the diagnosis, not the artifact.

**Presentation defects.**
- Abstract is ~370 words against a 250-word limit, and consists of six sentences averaging 55 words. It is unreadable at speed.
- Section II contains a **duplicated "Graph IDS." paragraph** (p.2, col.1 and col.2) with near-identical wording. Copy-editing failure.
- Tables IX and X use two incompatible definitions of "tied" (IX: 29 tied on UNSW; X: 6 tied + 23 mean-superior-only). Reconcile into one table.
- No Data/Code Availability statement, no ethics statement, no compute disclosure beyond hardware.

**Venue judgment.** If the reframing in R1/DA is executed, this is publishable — but I would steer it toward a venue that explicitly welcomes reproducibility and negative-results work. As a standard IEEE Transactions methods paper it will not survive a second round.

---

### Reviewer 1 — Methodology & Statistics

**Recommendation:** Major Revision | **Rigor: 48/100 · Reproducibility: 50/100 · Evidence quality: 40/100**

**Strengths.** Leakage-safe stacking (IV-C) with contiguous `expert_fit`/`router_fit` splits and the explicit justification for contiguous-over-shuffled given windowed graph construction is correct and well-reasoned. Calibration fit on VAL only, thresholds never tuned on TEST (IV-G), and split manifests under `manifests/` are the right practices. Holm correction is the appropriate family-wise procedure.

**MAJOR-1 — "Statistically ties" is an inverted null.** The headline claim (Abstract; VI-D; VIII) that AegisGraph-X *"statistically ties the strongest gradient-boosted-tree baseline on all three datasets"* is derived from *failure to reject* H₀ at p_Holm ≥ 0.60 with **n = 3 seeds**. Non-significance is not equivalence. With three paired observations the test has essentially no power to detect the 0.2–0.7 point differences at issue. The correct instrument is a **two-one-sided-tests (TOST) equivalence procedure with a pre-specified margin** (e.g. δ = 0.005 macro-F1, justified operationally). Until that is run, the paper cannot claim a tie — only that it did not detect a difference.

**MAJOR-2 — Asymmetric interpretation of the same underlying power.** The same n = 3 design yields "28 statistically superior comparisons" (VI-E) against weaker baselines. So low power is treated as *evidence of parity* when the comparator is strong, and adequate power is assumed when the comparator is weak. Both readings cannot be right. Note also that the direction is negative on **3 of 3** datasets in Table VIII — a consistency the tie framing suppresses.

**MAJOR-3 — Holm at n = 3 is variance-fragile.** To clear Holm's most stringent threshold (p < 0.05/41 ≈ 0.0012) with df = 2 requires t ≳ 25, which is achievable only when the seed-level standard deviation is very small (Table VII reports std as low as 0.0002). With three samples the std estimate is itself extremely unstable, so significance here is driven by a possibly-underestimated denominator. Report the paired differences and their CIs, not just labels.

**MAJOR-4 — Seed determinism bug undermines the pairing assumption.** Section VI-G discloses *"a network-weight-initialization determinism bug — fixed after these runs completed — means the nominal seed value did not fully control initial conditions."* The paper argues this strengthens the instability finding. Perhaps — but a paired t-test *requires* matched pairs. If seed *k* does not identify a controlled initial condition, the pairing across models is not established, and every paired test in Tables VIII–X inherits the problem. State explicitly whether the 3 seeds used for the **main** results (Table VII/VIII) were affected. If they were, the tests must be re-run post-fix.

**MAJOR-5 — Supporting experiments run at n = 1 and n = 2.** Table XIV (low-label) is seed 0 only. Table XVI (calibration) is 1 seed. Table XV (robustness) is 2 seeds. Yet Table XVI's conclusions turn on differences in the fourth decimal (ECE 0.0038 → 0.0023). At n = 1 these are noise. Either add seeds or remove the comparative claims.

**MAJOR-6 — Three seeds is below the paper's own stated preference.** Section VII concedes 3 seeds is "below the task's preferred 5–10." Given that the entire contribution now rests on *null* results, statistical power is not a nicety — it is the whole argument. 10 seeds minimum.

**MINOR.**
- CICIoT2023 is row-capped at 100K of 7.85M rows (Table II) — 1.3% of the corpus. The claim that tree baselines sit "near ceiling" is established only at this subsample.
- Table XVIII reports identical inference latency (4.67e-05) for `Full9ExpertEnsemble+MoERouter` and `FullAegisGraphX_backbone+RL`. Identical to three significant figures suggests the RL forward pass was not actually included in the measurement; the text's "adds no additional inference latency" needs a measured basis.
- Report effect sizes (Cohen's d or paired mean difference with CI) alongside every significance label, as Section V promises but the tables do not deliver.

---

### Reviewer 2 — Domain (Network IDS)

**Recommendation:** Major Revision | **Domain contribution: 35/100 · Literature: 60/100 · Evaluation validity: 45/100**

**Strengths.** The fourteen-dataset screening (Table I) with explicit rejection rationale is the single most useful artifact in this paper — rejecting ISCX-IDS-2012 for a 60.3% duplicate rate and CIC-VPN2016 for label-type mismatch is exactly the discipline this literature lacks. Retaining ToN-IoT-Network specifically because it preserves real source/destination IPs (III-D) and therefore permits genuine host-flow graph construction is a sophisticated choice most graph-IDS papers get wrong.

**MAJOR-7 — Binary framing on near-saturated corpora, while the multiclass reality is disclosed and ignored.** Section II correctly notes these corpora are *"near-saturated for binary classification under strong tree ensembles."* The paper then evaluates binary. Table XVII shows the multiclass picture: HistGradientBoosting reaches macro-F1 **0.311** on CICIoT2023 (34 classes) and **0.502** on UNSW-NB15. The operationally meaningful problem — *which* attack, so the analyst knows what to do — is nowhere near solved, and the near-ceiling binary numbers that motivate "we tie the strongest baseline" are an artifact of the easy framing. A cost-aware investigation paper in particular cannot justify binary-only: investigation cost depends on attack type.

**MAJOR-8 — The cost-aware evaluation never exercises the cost-aware machinery.** Tables XII and XIII report **telemetry cost = 0.0000 on every dataset in both reward modes**, and escalation rate ≈ 0 (0.0023 / 0 / 0 in Mode 1; 0 / 0 / 0 in Mode 2). Layer 4's entire premise is budgeted information acquisition — REQUESTTELEMETRY, INSPECTNODE, EXPANDSUBGRAPH. The learned policy uses none of them, ever. Meanwhile isolation rate is 0.62 (UNSW) and 0.67 (CICIoT2023). The paper flags the CICIoT2023 always-ISOLATE collapse but presents UNSW as non-collapsed — yet a policy that isolates 62% of hosts and acquires zero telemetry is also degenerate. This should be reported as a general failure of the acquisition layer, not a dataset-specific one.

**MAJOR-9 — Operational realism of the alert volume.** Table XVI reports ~7,948 alerts per 10,000 flows, and the conformal FPR-control variant at α = 0.05 reduces this to 7,934.9 — a **0.17% reduction**. Layer 5 is presented as "calibration and low-FPR control." On this evidence it controls nothing. Either evaluate at a realistic base rate (these corpora are 64–98% attack, which is not a SOC) or drop the low-FPR framing.

**MAJOR-10 — Robustness collapse is reported but not reckoned with.** Table XV: ToN-IoT-Network falls from 0.996 clean to **0.38–0.44** under feature noise; UNSW to 0.67–0.76; CICIoT2023 to 0.56–0.65. Missing-feature injection at 50% costs 0.09–0.36. The paper deserves credit for leading with the *most* fragile result (VI-H) rather than the most flattering. But the Conclusion still describes a "credible... honestly-evaluated classifier" without integrating this. A detector that loses 55 macro-F1 points under moderate feature noise is not deployable in the adversarial setting IDS inhabits — where feature perturbation is the *threat model*, not a robustness curiosity. Add an evasion-oriented discussion.

**MINOR.**
- Section II's duplicated "Graph IDS." paragraph (p.2).
- No comparison against recent flow-transformer IDS or self-supervised IDS pretraining; the deep-tabular family stops at FT-Transformer/TCN. SAINT and TabTransformer are disclosed as unimplemented (Section V) — but they are precisely the baselines that would test the "tabular-attention expert" premise.
- UNSW-NB15's stratified-random split (Table II) is honestly disclosed but this corpus has documented train/test distribution artifacts; a note on their effect would help.
- RT-IoT2022 is introduced as a transfer-check dataset and then never used for AegisGraph-X (Section VII). Either run it or remove it from Tables I–II.

---

### Reviewer 3 — Cross-Disciplinary Perspective (Decision Theory / OR)

**Recommendation:** Major Revision | **Cross-disciplinary grounding: 30/100 · Practical impact: 25/100**

**Strengths.** The structural insight — that RL's comparative advantage is sequential, budget-constrained decision-making, not boundary learning, and that asking it to classify "wastes RL's actual comparative advantage" (Section I) — is correct and worth stating plainly in a security venue. The separation of *what to emit* from *when and how thoroughly to decide it* is the right decomposition.

**MAJOR-11 — The obvious decision-theoretic baseline is missing, and it likely dominates.** Once Layer 5 produces a *calibrated* posterior p(attack | x) and Table III/IV specifies a cost structure, the cost-minimizing action is **analytically computable** by expected-cost minimization — no learning required. For the information-acquisition actions, the classical instrument is **value of information**: acquire telemetry iff expected reduction in Bayes risk exceeds acquisition cost. This is textbook (Howard 1966; Raiffa & Schlaifer). The paper compares *RL vs. no-RL* but never *RL vs. the analytic Bayes-optimal policy on the same calibrated probabilities and the same cost table*. This is the single most important missing baseline. My strong prior is that the analytic policy beats the PPO policy, costs nothing to train, cannot collapse to always-ISOLATE, and would explain *why* Layer 4 failed: PPO is being asked to rediscover a closed-form solution from a few hundred episodes.

**MAJOR-12 — "Structural, not a tuning accident" is over-claimed from two configurations.** Section IV-A and the Conclusion assert the RL-as-classifier failure has a *structural* root cause. But the evidence is: one composite reward with eight uncorrected terms, later fixed, plus (in this revision) two reward-weight modes. Two configurations of a ~12-term reward (Table III) is not a search. The structural argument is *plausible* — and I find the a-priori reasoning persuasive — but it is presently an argument, not a result. Either weaken to "consistent with a structural explanation" or supply the missing evidence: a learning-curve / sample-complexity analysis showing the gap does not close with budget, or a comparison against the analytic policy per MAJOR-11.

**MAJOR-13 — The structural guarantee is a tautology being marketed as a contribution.** "Classification quality can never fall below the calibrated ensemble's own quality regardless of how the RL policy trains" (Abstract; IV-F) is true *by construction* — COMMIT reads the label. This is not a theorem about the method; it is a restatement of the design choice. Worse, it is load-bearing in exactly the wrong direction: it guarantees the RL layer cannot hurt the metric the paper reports, while Tables XII–XIII show it *does* hurt the metrics it can affect (isolation rate 0.62–0.67, zero telemetry acquisition). The guarantee protects the paper from its own null result rather than delivering value to a user.

**MINOR.**
- MoE routing is cited from large-scale LM/vision work but the relevant nearby literature is **stacking / super-learner** (Wolpert; van der Laan) and **mixture-of-experts for tabular data**. The finding that a learned router does not beat an unweighted average is a well-known stacking result at small sample sizes; connecting to that literature would let the authors *explain* the null rather than merely report it.
- The active-sensing framing should cite the optimal-stopping and sequential-analysis literature (Wald's SPRT is the canonical single-hypothesis case of exactly this problem).
- **No ethics or deployment-harm discussion.** A policy that auto-isolates 62–67% of hosts, evaluated on simulated response, has real potential for operational harm. The reward's isolation penalty is not a substitute for a paragraph on false-isolation consequences and human-in-the-loop requirements.

---

### Devil's Advocate

**Verdict: two CRITICAL issues. Editorial Decision cannot be Accept.**

#### Strongest counter-argument

The paper's ablation table refutes the paper. Table XI reports the full AegisGraph-X backbone at macro-F1 **0.9627**. Removing the expert ensemble entirely and using the single best expert gives **0.9648**. Removing the MoE router and using an equal-weight average gives **0.9639**. Both ablations are *better* than the full system. Table VIII then shows the full system is nominally below the best off-the-shelf tree baseline on **all three** datasets. Table XI's distillation block shows the same pattern one level down: "no hard-class mining" (0.9001) and "plain KD" (0.8829) both beat the full distilled student (0.8721).

So: every novel component, tested individually, is neutral or harmful, and the assembled system is worse than a single XGBoost you could have downloaded. The honest reading is not "we contribute a five-layer architecture with some unvalidated layers." It is **"we built a five-layer architecture and our own experiments show it should not have been built."** That is a legitimate and publishable finding — arguably a more useful one than the intended contribution. But the manuscript will not say it. Title, abstract structure, Section IV's five-layer exposition, and the Conclusion's "credible, statistically-tied-with-the-strongest-baseline, honestly-evaluated classifier" all continue to present the artifact as the contribution. The paper's rhetoric and its evidence point in opposite directions, and the reader is left to resolve the contradiction.

#### Issue list

| ID | Sev | Dimension | Location | Issue |
|---|---|---|---|---|
| **DA-1** | **CRITICAL** | Logic / evidence | Table XI vs. Abstract, VIII | Full system underperforms its own ablations (0.9627 vs 0.9648 / 0.9639) and the strongest single baseline on 3/3 datasets, yet is presented as a contribution. No positive claim survives the paper's own evidence. |
| **DA-2** | **CRITICAL** | Statistical inference | Abstract; VI-D; VI-E; VIII | The two headline claims — "statistically ties the strongest baseline" and "never significantly beaten by any real baseline" — are both *manufactured by low power*. With n = 3 and Holm over 41 comparisons, not-being-beaten is close to guaranteed a priori. The design cannot produce the negative claims it advertises. |
| **DA-3** | MAJOR | Circular reasoning | IV-F; Abstract | "Classification quality can never fall below the ensemble's" is true by construction, not a result. It is used to convert a null result into a safety feature. |
| **DA-4** | MAJOR | Overgeneralization | IV-A; VIII | "Root cause is structural, not a tuning accident" generalizes from two reward configurations to an impossibility claim. |
| **DA-5** | MAJOR | Selective emphasis | VI-B; XII–XIII | Always-ISOLATE collapse is framed as a CICIoT2023-specific event. Telemetry cost is 0.0000 and escalation ≈ 0 on *all three* datasets in *both* modes — the acquisition layer is universally unused. The general failure is reported as a special case. |
| **DA-6** | MAJOR | Missing alternative | IV-F; V | No analytic expected-cost / value-of-information policy baseline. The most likely explanation for Layer 4's failure — that it is relearning a closed-form solution — is never tested. |
| **DA-7** | MAJOR | Unsupported deployment claim | VI-I; VIII | Distilled student marketed at 740× speedup with "8.1K parameters vs. the full pool," but Table XVIII's only comparable param count is the RL policy's 9,918 — an 18% reduction, not a compression story. Tree ensembles have no param entry, so the headline comparison is apples-to-oranges. |
| **DA-8** | MINOR | Reproducibility | VI-G | Seed determinism bug is disclosed and then argued to *strengthen* the finding. It may — but it also breaks the pairing assumption underlying every paired test in the paper. The disclosure is used rhetorically rather than propagated to the analysis. |

#### Ignored alternative explanations

1. **Small-sample stacking.** The router is trained on 30% of a 70–100K-row split's out-of-sample scores. Learned combiners are well documented to lose to simple averaging in this regime. The null is expected, not surprising — and this reframes it from "MoE doesn't work here" to "we confirmed a known stacking result."
2. **Ceiling effect, not parity.** On corpora at 0.94–0.998 binary macro-F1, everything ties everything. The "28 superior comparisons" are against models far from ceiling; the "tie" is with models at it. Both are properties of the benchmark, not the method.
3. **The RL layer may be solving the wrong POMDP.** With 64-flow windows and terminal actions per flow, the horizon may be too short for information-acquisition to pay off at all — an environment-design explanation the paper never separates from its structural one.

#### Missing stakeholder perspectives

- **The SOC analyst**, who receives 7,948 alerts per 10,000 flows and 62% host isolation.
- **The isolated benign host's owner** — no discussion of false-isolation harm beyond a reward penalty.
- **The reproducer**, given the seed determinism bug and an anonymized repository with no archival DOI.

#### "So what?" test

*Passes* as: "a disciplined post-mortem showing that MoE routing, budgeted RL investigation, conformal FPR control, and hard-negative distillation each fail to improve on a plain gradient-boosted-tree IDS, with a structural diagnosis of why RL-as-classifier cannot work."
*Fails* as: "we propose AegisGraph-X."

#### Observations (non-defects)

The negative-result reporting in VI-B, VI-F, VI-G, and VI-H is of a standard the field should adopt. Leading with the *most* fragile robustness result rather than the most flattering (VI-H) is a deliberate choice against self-interest and should be preserved verbatim through revision. My criticism is entirely about framing and inference, not about integrity.

---

## Phase 2 — Editorial Decision

### Decision: **MAJOR REVISION** (borderline Reject; reframing is not optional)

Per Checkpoint Rule #4, the Devil's Advocate raised CRITICAL findings (DA-1, DA-2); Accept is unavailable.

### Consensus (all five reviewers)

1. **The paper's evidence does not support the paper's framing.** EIC, R1, R2, R3, and DA reached this independently from different directions — contribution structure (EIC), inferential validity (R1), operational validity (R2), decision-theoretic redundancy (R3), and internal contradiction (DA).
2. **The integrity of the reporting is exceptional and must be preserved.** Unanimous. No reviewer asked for a negative result to be softened or removed.
3. **n = 3 is disqualifying for a paper whose contribution is null results.**

### Disagreement requiring arbitration

**Is the paper salvageable as-is with reframing, or does it need new experiments?**
- EIC and DA: reframing alone could carry it.
- R1 and R3: new work is required — TOST equivalence testing (R1) and the analytic Bayes-optimal baseline (R3).

**Arbitration.** R1 and R3 prevail. Reframing without MAJOR-1 (equivalence testing) merely relabels an unsupported claim; the "tie" is the paper's only positive statement and it currently rests on absence of evidence. And R3's MAJOR-11 is decisive: without the analytic policy comparison, the paper cannot distinguish "RL is the wrong tool for investigation" from "RL is the wrong tool for *this* investigation problem as we posed it" — which is the difference between a general contribution and a local one. Both are required.

### Revision Roadmap (prioritized)

**Tier 1 — blocking**

| # | Action | Addresses |
|---|---|---|
| 1 | **Reframe the paper as a structural diagnosis + negative-results study.** New title (drop the artifact-forward form), abstract rebuilt around the diagnosis, Section IV compressed from five-layer exposition to a design description subordinate to the finding. | DA-1, EIC |
| 2 | **Replace every "statistically ties" claim with a TOST equivalence test** at a pre-registered margin (suggest δ = 0.005 macro-F1, operationally justified). If equivalence is not established, say so. | MAJOR-1, DA-2 |
| 3 | **Raise seeds to ≥ 10** for Tables VII–XI; ≥ 5 for XIV–XVI. Re-run all paired tests **after** the determinism fix and state explicitly which results are pre- vs. post-fix. | MAJOR-4, MAJOR-5, MAJOR-6, DA-8 |
| 4 | **Add the analytic expected-cost / value-of-information policy** as a Layer-4 baseline on the same calibrated probabilities and cost table. | MAJOR-11, DA-6 |
| 5 | **Report the acquisition-layer failure as general, not dataset-specific.** Telemetry cost 0.0000 and escalation ≈ 0 across all datasets and both modes belongs in the abstract, not in a table footnote. | MAJOR-8, DA-5 |

**Tier 2 — required before acceptance**

| # | Action | Addresses |
|---|---|---|
| 6 | Soften "structural, not a tuning accident" to a supported claim, or supply sample-complexity evidence. | MAJOR-12, DA-4 |
| 7 | Reposition the structural guarantee as a design property, not a contribution. | MAJOR-13, DA-3 |
| 8 | Integrate the robustness collapse (Table XV) into the Conclusion; add an adversarial-perturbation framing. | MAJOR-10 |
| 9 | Drop or re-evaluate the "low-FPR control" claim given the 0.17% alert reduction. | MAJOR-9 |
| 10 | Fix the distillation compression claim; give a like-for-like size/latency comparison. | DA-7 |
| 11 | Add multiclass discussion; either evaluate the router/RL layers on multiclass or explicitly bound the contribution to binary. | MAJOR-7 |
| 12 | Add ethics / deployment-harm paragraph covering false isolation and human-in-the-loop. | R3 minor |

**Tier 3 — presentation**

| # | Action |
|---|---|
| 13 | Cut abstract to ≤ 250 words; break sentences averaging 55 words. |
| 14 | Delete the duplicated "Graph IDS." paragraph (p.2). |
| 15 | Merge Tables IX and X into one table with a single definition of "tied." |
| 16 | Add Data/Code Availability, ethics, and compute statements. |
| 17 | Cite stacking / super-learner literature to explain the router null; cite optimal stopping / SPRT for the acquisition framing. |
| 18 | Remove RT-IoT2022 from the dataset tables or run it. |

### Score summary

| Dimension | Score | Note |
|---|---|---|
| Originality | 45 | Composition of known components; the diagnosis is the original part |
| Significance | 35 | High as a negative result, low as framed |
| Methodological rigor | 48 | Excellent protocol design, weak inferential statistics |
| Evidence quality | 40 | n=3 for main results, n=1–2 for supporting |
| Reproducibility | 50 | Exemplary manifests, undermined by determinism bug |
| Literature coverage | 60 | Good IDS coverage, missing decision theory and stacking |
| Clarity | 40 | Dense abstract, duplicated text, framing/evidence mismatch |
| **Overall** | **45** | Major Revision |

---

*Reviewers did not modify the manuscript. This document is the sole review output.*
