# Research brief

Prior work on WM-811K, candidate additions, scope, and draft resume material. Written before any modelling.

Verification legend: **[V]** means the bibliographic record or paper page was checked. **[U]** means unverified (from memory or a secondary citation). Nothing marked [U] goes into the README until it is checked.

## 1. Prior work on WM-811K

### Key papers
| Paper | What it did | Status |
|---|---|---|
| Wu, Jang & Chen, "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets", *IEEE TSM* 28(1):1–12, 2015, doi:10.1109/TSM.2014.2364237 | Released the dataset (811,457 maps). Proposed rotation/scale-invariant features (density, Radon, geometry) + SVM, plus similarity ranking. Method details are from memory and still need checking against the paper | [V] Crossref record (title, authors, venue, pages, Feb 2015 print), [U] method details |
| Nakazawa & Kulkarni (Intel), "Wafer Map Defect Pattern Classification and Image Retrieval Using CNN", *IEEE TSM* 31(2):309–314, 2018, doi:10.1109/TSM.2018.2795466 | CNN trained on **synthetic** maps and tested on real ones. Image retrieval from an 18k library, with a reported 3.7% retrieval error on 1,191 real wafers. These details come from a *Nature Electronics* research highlight about the paper (M. Lee, "Deep learning spots the defect", Nat. Electron. 1:153, 2018, doi:10.1038/s41928-018-0050-z), not from the paper itself | [V] Crossref record of the paper, [U] method details and numbers (the earlier link was to the highlight, not the paper) |
| Saqlain, Jargalsaikhan & Lee, "A Voting Ensemble Classifier for Wafer Map Defect Patterns Identification…", *IEEE TSM* 32(2):171–182, 2019, doi:10.1109/TSM.2019.2904306 | 66 handcrafted features + LR/RF/GBM/ANN soft voting. Reported 86.7% accuracy (per an author page, not checked in the paper) | [V] Crossref record, [U] numbers |
| Kahng & Kim, "Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern Classification", *IEEE TSM* 34(1):74–86, 2021. Code: github.com/hgkahng/WaPIRL | Contrastive pretraining on unlabeled maps; helps most when labels are scarce | [V] record |
| Li, Jiang, Chen & Wang, "Wafer Defect Pattern Labeling and Recognition Using Semi-Supervised Learning", *IEEE TSM* 35(2):291–299, 2022, doi:10.1109/TSM.2022.3159246 | Pseudo-labels the unlabeled maps and clusters the leftovers to find new defect types | [V] record, authors from university portal records (see docs/root_causes.md) |
| Kwak, Lee & Kim, SWaCo, 2023; Baek, Hwang & Kim, CowSSL, 2024 | Self-supervised / open-world semi-supervised with out-of-distribution handling | [V] titles |
| **Cao & Tao (Fudan), "Zero-Shot Multi-Class Wafer Map Classification with Rule-Ordered Prompts for Large Multimodal Models", arXiv 2610.35080, 2026** | GPT-4o (gpt-4o-2024-08-06), zero-shot, 6 merged classes, **no "none" class**, random 9,000-map WM-811K sample. Overall accuracy 43.7% with raw prompts and 81.3% with rule prompts. Reports macro *accuracy*, not F1. Latency 0.5–1 s per map; no dollar costs | [V] read the paper |
| Kaushik, "SemiFA: Agentic Multi-Modal Framework for … Failure Analysis Report Generation", arXiv 2604.13236, 2026 | 4-agent LLM pipeline (DINOv2 + LLaVA) that writes failure-analysis reports. Graded by a GPT-4o judge; **no hallucination measurement** | [V] read the abstract page |
| Ezzat, Liu, Hochbaum & Ding, arXiv 2006.13824 | Gives pattern→cause statements with references (see §3, root-cause sources) | [V] read the paper |

**Checked since (record level, full details in docs/root_causes.md):** Kyeong & Kim 2018, *IEEE TSM* 31(3):395–402 [V] record; Shim, Kang & Cho 2020, *IEEE TSM* 33(2):258–266 [V] record; Hwang & Kim 2020, *IEEE TSM* 33(3):466–475 [V] record. The "Kong & Ni 2020" paper I had in mind did **not** turn up in search, so I'm dropping it.

### Typical macro-F1
I **can't give you a reliable typical macro-F1 under a lot split**. Most papers report accuracy on random wafer-level splits, often after dropping "none" or subsampling. One 2025 Springer paper (EURASIP JIVP, doi:10.1186/s13640-025-00666-3) reports about 93 macro-F1, but on an unclear split. Supervised CNNs routinely claim 96–98% accuracy. That gap is itself a finding: **a lot-split, 9-class, macro-F1 benchmark is rare**, so your numbers will look lower and be more honest. Say exactly that in the interview.

### Common pitfalls (each one becomes a DECISIONS.md entry)
1. Random wafer split leaks lot signatures. Fix: group the split by `lotName`. **Audit:** all 811,457 maps have a well-formed lot name (46,293 lots, at most 25 wafers each), so no special handling is needed. A secondary source's 0.3% missing-lot figure does not hold for this file.
2. The built-in `trianTestLabel` split (typo is in the data) may not be lot-disjoint. **Audit:** it is lot-disjoint (0 shared lots), but it is lopsided: 54,355 training vs 118,595 test maps. 92 exact-copy groups of defect maps also straddle it (see 7).
3. Headline accuracy is dominated by "none". Also, "none" is often silently dropped, which inflates scores.
4. Near-full has only 149 maps (confirmed by the audit), so per-class recall on it has huge variance. Report confidence intervals or the raw counts.
5. Map sizes vary widely (die size). Resizing changes the meaning of "Edge" and "Loc", so the resize method is a decision.
6. Labels are noisy: Loc vs Edge-Loc and Center vs Donut are ambiguous (Cao & Tao note this too).
7. Duplicate or near-duplicate maps may cross splits. Hash them to check (not established in the literature, my own check). **Audit:** about 115k extra copies exist, but almost all are clean maps with zero or one fail that match by chance. The real issue is 187 defect-labeled maps in 93 exact-copy groups, every one spanning different lots, so a pure lot split would not keep them apart. 14 copy groups carry conflicting labels, which is direct evidence of label noise (6).
8. Augmenting before the split leaks augmented copies into test.

### Overdone (skip)
- Chasing CNN accuracy: ResNet/VGG transfer learning, 98%+ claims.
- GAN/VAE/autoencoder augmentation to fix imbalance (arXiv 2411.11029, the 2025 Springer paper and many more).
- Voting ensembles of handcrafted features (Saqlain 2019 already did this).
- Plain self-supervised or contrastive pretraining as the main contribution (Kahng & Kim, SWaCo, CowSSL).
- **Plain "GPT-4o zero-shot on WM-811K"**: Cao & Tao published this in Oct 2026. You need a different angle (§2, item E).

---

## 2. Candidate additions (scored for a fab/yield-engineer audience)

| # | Addition | Why a fab engineer cares | Risk | What you must be able to defend |
|---|---|---|---|---|
| A | **Yield loss per pattern + Pareto**: die yield per map; mean, median and lot-level distribution per class; "dies lost per 1,000 wafers" by pattern | This is how yield engineers prioritize: fix the biggest Pareto bar first | Low | Yield definition, why the median beats the mean for skewed loss, normalizing by die count, why loss ≠ cause (Near-full is high loss but rare) |
| B | **Lot-level / excursion view**: pattern co-occurrence within lots, run-length across `waferIndex` | "Is it one wafer or the whole lot?" splits tool excursions from random events; this is the first thing a yield engineer asks | Low | Lot vs wafer variation, why clustering within a lot points to a tool or chamber |
| C | **Sourced root-cause table** (pattern → candidate process modules → first fab step to check → signal to confirm) | Turns a classifier into a triage tool | Medium: few peer-reviewed pattern→cause sources, so frame entries as *hypotheses* | Physics of each mechanism (radial deposition/etch non-uniformity, CMP edge roll-off, litho focus, handling scratches). Your own spin-coat edge-bead and ALD uniformity work backs the radial rows |
| D | **RF baseline vs small CNN under a lot split**, macro-F1 + per-class recall + confusion matrix | Core credibility | Low | Grouped split, class weights, why Radon features work, CNN receptive field |
| E | **VLM zero-shot benchmark, differentiated from Cao & Tao**: same lot-split test set as your CNN, **all 9 classes including "none"**, macro-F1, an open-weight VLM run locally plus Luna 6 via ChatGPT credits, measured latency and token counts priced at published API rates per 1k wafers, raw vs rule-prompted | Answers "can we skip labeling?" and makes the cost/accuracy tradeoff explicit | Medium: prompt sensitivity, local VLM quality | Stratified subsampling, why "none" matters for the false-alarm rate, cost per wafer vs a fab's wafer volume |
| F | **Grounded triage assistant + hallucination eval**: an LLM gets the predicted class, confidence, yield and lot stats, and may cite **only** rows of the table from C. It drafts an 8D D2–D4 first response | Shows controlled AI use: closed-world grounding plus measurement | Medium | Grounding; automatic hallucination check (any cause not in the table = hallucination); agreement with the table (correctness); why D2–D4 only |
| F′ | **Your "model on synthetic data" idea → process-fault simulator**: generate maps from *parameterized* faults (radial thickness profile → Center/Donut/Edge-Ring; edge exclusion → Edge-Loc; arc paths → Scratch; Poisson → Random), train on synthetic data only, test on real | Purely process-physics driven, and it inverts the problem: a map tells you *which parameter drifted*. Nakazawa & Kulkarni (Intel) showed synthetic→real works for classification | Med-High: sim-to-real gap | Spatial models of each fault, why they're plausible, what the sim gets wrong |
| G | **Reject option / abstain**: confidence threshold → "send to engineer"; precision vs coverage curve | Fabs prefer "I don't know" over a wrong auto-disposition | Low | Calibration, coverage tradeoff |
| H | **Label-noise audit**: review about 100 high-confidence disagreements by hand | Shows judgment; fab data is messy | Low | How you decided an existing label was wrong |
| I | **Pseudo-labeling on about 640k unlabeled maps** | Real fabs have far more unlabeled maps than labeled ones | Medium: Colab RAM, already done (Kahng & Kim, Li 2022) | Confirmation bias, confidence thresholds, keeping the test set untouched |
| J | Grad-CAM sanity check | Confirms the CNN looks at the defect, not at map borders | Low | What the saliency map shows and what it doesn't |
| K | Small Streamlit demo (upload map → class, yield, root-cause row, assistant draft) | Recruiters actually click it | Low | — |
| L | New-pattern / out-of-distribution detection on unlabeled data | Interesting but overdone (CowSSL) | High | — |

## 3. Ranking and recommended scope
No time budget. Scope is set by value and by what each step depends on, not by a calendar.

**Rank** (value to a process recruiter): A > C > B > D > F′ > E > F > G > H > K > J > I > L.

**Build order (by dependency):**
1. **Foundation:** data audit (counts, lot IDs, sizes, duplicates, `trianTestLabel` check) → lot-grouped stratified split.
2. **Process analysis:** **A** yield Pareto → **B** lot/excursion view → **C** root-cause table with checked sources.
3. **Classification:** **D** RF baseline → small CNN on the desktop GPU → **G** abstain curve → **J** Grad-CAM → **H** label audit.
4. **AI-native:** **E** VLM benchmark on the same test set → **F** grounded assistant + hallucination eval.
5. **Process-physics extension:** **F′** fault simulator, synthetic→real test.
6. **Optional:** I pseudo-labeling. K (demo) was decided against (d07). L is skipped as overdone.

Why E + F rather than I: E and F are measurable and novel enough (lot split, 9 classes, cost, hallucination rate). Pseudo-labeling is the most-trodden path.

**Root-cause sources to check for C:** Hwang & Kuo 2007 [V] record; Kim, Lee & Kim 2018 [V] record (it is in *IISE Transactions*, not IEEE TSM); Cunningham & MacKinnon 1998 [V] record; Neo et al. 2017, *Microelectronics Reliability* 76–77:255–260, doi:10.1016/j.microrel.2017.07.036 [V] Crossref, after the reference was read off Ezzat et al.'s reference list (all via Ezzat et al.). Details and status are in docs/root_causes.md. I'll also add standard textbooks (e.g. Quirk & Serda, *Semiconductor Manufacturing Technology*) [U, need a page reference]. Each table row cites ≥1 checked source, Mehnaj's own lab work, or is marked "engineering hypothesis".

**Mehnaj's hands-on work as evidence (see d09):** spin-coating edge beads and ALD thickness-uniformity work. The edge bead is a thicker rim of film left at the wafer edge when surface tension holds liquid there during spin-off. That is the same radial, edge-located thickness non-uniformity proposed as one cause of Edge-Ring. ALD uniformity work covers centre-to-edge thickness gradients from precursor dosing, purge or temperature non-uniformity, which map to Center, Donut and Edge-Ring hypotheses. Each row this applies to should cite the specific run or slide. No measured values are quoted until the slides are in the repo.

## 4. Resume bullets + pitch (filled from results/, 2026-11-03)
- Built a yield-analysis pipeline on 811K production wafer maps (WM-811K). Ranked defect signatures by excess die loss against a per-product clean baseline: Edge-Ring, Edge-Loc and Random account for 69% of pattern-driven die loss. Showed every pattern clusters inside lots far beyond chance (up to 127×), with 7.1% of lots being single-pattern excursions.
- Classified 9 defect patterns under a leakage-safe lot-level split (copy-linked lots merged). A 0.58M-weight CNN reached 0.877 macro-F1 (95% CI 0.855–0.893) vs 0.867 for a random forest on engineered features, with per-class recall ≥0.85 except Scratch (0.73). It cut defects passed as clean from 27% to 11% for Scratch, and at a 5% miss rate sent half as many clean wafers to review (10 vs 19 per 1,000).
- Benchmarked two vision-language models zero-shot on the same lot-split test wafers. GPT-6 Luna went from 0.45 to 0.61 macro-F1 with a corrected domain prompt, and Claude Opus 5.5 from 0.53 to 0.72, vs 0.92 for the CNN, at $0.47 and $13.95 per 1,000 wafers. Built an LLM triage assistant grounded in a sourced root-cause table: 0 of 60 replies made an unsupported claim under code-based hallucination checks, vs 58 of 60 without the table.
- Built a physics-based process-fault simulator. A model trained only on simulated wafers reached 0.45 macro-F1 on real data, catching 0.97/0.88/0.73 of Random/Edge-Ring/Scratch wafers, and showed real "clean" wafers carry structure the simulator lacks.

**30-second pitch:** "Fabs see thousands of wafer maps a day, and the useful question isn't *what does this look like?* but *which tool do I check first, and how many dies is it costing me?* I took 811,000 real wafer maps and ranked defect patterns by the yield they destroy: edge patterns and random defects cause about two thirds of it, and they pile up by lot, which points at tools. Then I built a classifier evaluated the way a fab would deploy it: split by lot, scored on rare patterns, not accuracy. It reaches 0.88 macro-F1, and at a 5% miss rate it sends half as many good wafers to an engineer as the baseline. I tested whether vision-language models can read wafer maps without labels. The best one, Claude Opus 5.5, gets 0.72 at about 1.4 cents a wafer, still well behind the CNN, so it can't replace labelling. And I built an assistant that drafts a first-response root-cause hypothesis only from a sourced process table. Code checks every claim it makes, and with the table it made nothing up in 60 of 60 cases."

## Sources checked
- Wu et al. 2015 (Crossref): https://doi.org/10.1109/TSM.2014.2364237
- Nakazawa & Kulkarni 2018 (Crossref): https://doi.org/10.1109/TSM.2018.2795466. The *Nature Electronics* highlight about it, which is not the paper: https://doi.org/10.1038/s41928-018-0050-z
- Saqlain et al. 2019 (Crossref): https://doi.org/10.1109/TSM.2019.2904306
- Kahng & Kim 2021: https://pure.korea.ac.kr/en/publications/self-supervised-representation-learning-for-wafer-bin-map-defect-/
- Cao & Tao 2026: https://arxiv.org/html/2610.35080
- SemiFA 2026: https://arxiv.org/abs/2604.13236
- Ezzat et al.: https://ar5iv.labs.arxiv.org/html/2006.13824
- EURASIP JIVP 2025: https://link.springer.com/article/10.1186/s13640-025-00666-3
- Autoencoder augmentation: https://arxiv.org/abs/2411.11029
