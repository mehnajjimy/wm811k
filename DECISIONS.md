# Decisions

Every choice in the project, with the options considered, the recommendation, why, and what Mehnaj decided. Code points here with tags such as `# d03`.

---

### d01 Budget
- **Options:** paid API calls for the VLM benchmark; free only (local models, existing credits).
- **Recommended:** free only.
- **Why:** the project should be reproducible by anyone at no cost, and token counts priced at published rates still answer "what would this cost a fab".
- **Decided:** free only. No paid API calls.

### d02 Compute
- **Options:** laptop only; free Colab; Mehnaj's desktop GPU (RTX 5070 Ti 16 GB, 32 GB RAM, Ryzen 7 7800X3D).
- **Recommended:** Mac for data work and analysis, desktop GPU for CNN training and the local VLM, Colab as fallback.
- **Why:** 16 GB of VRAM is enough for a small CNN and a 7B-class VLM, without Colab session limits. The card needs a PyTorch build with CUDA 12.8 or newer.
- **Decided:** as recommended. GPU jobs run over SSH from the Mac.

### d03 Train/test split
- **Options:** random wafer split; the dataset's built-in `trianTestLabel`; split grouped by `lotName`.
- **Recommended:** grouped by lot, stratified by class as far as lot grouping allows.
- **Why:** wafers in one lot share tools, recipes and time, so their maps are correlated. A random split puts near-copies in train and test and inflates scores.
- **Decided:** split by lot (non-negotiable from the project brief). The built-in split gets audited for lot overlap, and maps with no valid lot ID get handled as separate groups.

### d04 Metrics
- **Options:** accuracy; macro-F1; per-class recall; balanced accuracy.
- **Recommended:** macro-F1 plus per-class recall and a confusion matrix, with counts or intervals for small classes.
- **Why:** about 85% of labelled maps are "none", so accuracy mostly measures the majority class. Near-full has only about 149 maps, so its recall is noisy.
- **Decided:** macro-F1 and per-class recall. Headline accuracy is never reported.

### d05 Class set
- **Options:** 9 classes including "none"; 8 defect classes only; both.
- **Recommended:** 9-class headline, plus an 8-class side table.
- **Why:** in a fab most wafers are clean, so false alarms on clean wafers matter. The 8-class table allows comparison with papers that drop "none".
- **Decided:** as recommended.

### d06 Assistant form
- **Options:** LLM + root-cause table assistant (F); process-fault simulator trained on synthetic maps (F′); both.
- **Recommended:** both, with F′ as a core piece.
- **Why:** the simulator is the most process-driven part of the project and the least done before. The assistant shows controlled AI use, with a measured hallucination rate.
- **Decided:** both, F′ core.

### d07 Demo
- **Options:** Streamlit demo; README with figures only.
- **Recommended:** README first, demo later if wanted.
- **Why:** strong figures do most of the work, and most reviewers never run a demo.
- **Decided:** no demo, README only.

### d08 VLM benchmark setup
- **Options:** paid frontier APIs; an open-weight VLM run locally; Luna 6 via ChatGPT credits; Claude via sub-agent.
- **Recommended:** a local open-weight VLM plus Luna 6.
- **Why:** both are free. The test differs from Cao & Tao (arXiv 2610.35080) by using the same lot-split test set as the CNN, all 9 classes, macro-F1, and measured latency and token cost.
- **Decided:** local open-weight VLM + Luna 6.

### d09 Root-cause table sourcing
- **Options:** cite only peer-reviewed sources; mix sources with labelled engineering hypotheses.
- **Recommended:** every row cites a verified source or is labelled "engineering hypothesis".
- **Why:** peer-reviewed links from pattern to cause are thin, so the table needs other evidence. Mehnaj has hands-on uniformity work that supplies it.
- **Decided:** as recommended. **Corrected 2026-11-03:** an earlier version said the table rested on coursework only, which was wrong. Rows also draw on Mehnaj's own lab work:
  - **Spin-coat edge beads.** A thicker film rim at the wafer edge, which is the same radial edge non-uniformity behind the Edge-Ring hypothesis.
  - **ALD thickness uniformity.** Centre-to-edge gradients, which bear on the Center, Donut and Edge-Ring hypotheses.
- Rows cite the specific run or slide. No measured values are quoted until the slides are in the repo. This is the angle an ML-only portfolio can't offer: the radial patterns are tied to film physics Mehnaj has measured.

### d10 Repository and license
- **Options:** private; public under MIT.
- **Recommended:** public, MIT, with the dataset kept out of git.
- **Why:** this is a portfolio project. The dataset is 2 GB and not ours to redistribute, so the README links to the Kaggle page.
- **Decided:** public, MIT.

### d11 Agents
- **Options:** Codex Sol/Astra; Claude sub-agents; a mix.
- **Recommended:** Opus orchestrates, Fable/Sonnet/Haiku sub-agents do the work, Luna 6 handles cheap cross-checks.
- **Why:** on 2026-11-03, Sol, Astra and the 6.1 Terra/Luna models all returned "not supported" under the ChatGPT login. Only `gpt-6-luna` and `gpt-5.6-terra` answered, and Terra 5.6 is weaker than Haiku in Mehnaj's experience.
- **Decided:** as recommended. See ORCHESTRATOR.md.

### d12 Code style
- **Options:** minimal comments; decision-tagged comments; teaching-heavy notes.
- **Recommended / decided:** teaching-heavy notes in Mehnaj's voice (lowercase, no punctuation), in a package with thin notebooks and light tooling. See STYLE.md.

### d13 Exact-copy defect maps across lots
- **Finding:** 187 defect-labeled maps form 93 groups of pixel-identical copies. Every group spans more than one lot, and 92 groups straddle the dataset's built-in train/test split (results/audit_facts.csv). Clean maps also repeat a lot, but those are maps with zero or one fail that match by chance, not copies.
- **Options:** keep everything and accept the leak; drop all but one copy; keep all copies but force lots linked by a shared defect map into the same split group.
- **Recommended:** merge linked lots into one split group.
- **Why:** this closes the leak without throwing data away, and it's easy to explain: "if two lots share an identical defect map, they are treated as one unit."
- **Decided:** approved as recommended (2026-11-03).

### d14 Use the built-in split or our own
- **Finding:** the built-in `trianTestLabel` split is lot-disjoint, but it puts 31% of labeled maps in training and 69% in test, and it is broken by the d13 copies.
- **Options:** use it as is; build our own lot-grouped, class-stratified split (for example 70/15/15 train/validation/test).
- **Recommended:** our own split, with d13 applied and a fixed seed.
- **Why:** rare classes need more training maps than a 31% share gives, a validation set is needed for thresholds (abstain, item G), and the copy leak has to be closed either way.
- **Decided:** approved as recommended (2026-11-03).

### d15 Split result (implements d13 and d14)
- **What was built:** `src/wm811k/split.py` uses 20 stratified group folds (3 test, 3 validation, 14 train) with seed 811 from `config.yaml`. Groups are lots, merged where they share an identical defect map, which produced 61 merged groups.
- **Checks:** every class keeps the same share in train, validation and test to within 0.0001 (results/split_class_counts.csv). Zero copy groups cross the split. No lot appears in two splits (tests/test_split.py).
- **Unlabeled maps:** unlabeled maps follow their lot's split. Lots with no labelled map at all (566,611 maps) form an `unlabeled_only` pool, so pseudo-labelling can never pull in test lots.
- **Stored as:** `results/split_lots.csv`, one row per lot.

### d16 Yield loss measure
- **Options:** raw yield per class; mean yield gap; extra dies lost compared with a clean-wafer baseline, per wafer and per 1,000 wafers.
- **Recommended:** extra dies lost against the median yield of "none" wafers, reported per 1,000 labelled wafers as the Pareto number.
- **Why:** clean wafers still lose dies to random defects, so only the loss above that baseline belongs to the pattern. "Per 1,000 wafers" combines how often a pattern appears with how bad it is, which is how a yield engineer ranks what to fix first. The median is used because a few very bad clean wafers would drag a mean.
- **Decided:** implemented in `src/wm811k/yield_stats.py`. Results are in results/yield_by_class.csv and figures/yield_pareto.png.

### d17 Baseline per product
- **Options:** one global clean baseline; a baseline per product.
- **Recommended:** per product, using map grid size as the product marker. Sizes with fewer than 30 clean wafers fall back to the global clean median of 90.2%.
- **Why:** different products run at different normal yields. Comparing every wafer to one global median would blame a product's normal loss on the pattern. Switching moved Center's share of extra loss from 18.5% to 13.5% and lifted Edge-Loc above Random.
- **Decided:** implemented. The grid size is an assumption, since the data has no product ID.

### d18 Finding: the labelled subset is not a random sample
- **Finding:** labelled wafers have a median yield of 89.4%, against 94.8% for unlabelled wafers (results/yield_labeled_vs_unlabeled.csv).
- **Implication:** the labelled set leans toward worse wafers, so "per 1,000 labelled wafers" is not "per 1,000 wafers the fab ran". The README says this plainly rather than scaling numbers up to fab volume.
- **Decided:** report as a caveat. If pseudo-labelling (item I) happens, it can test how far the Pareto moves on the full population.

### d19 Finding: Scratch costs fewer dies than a clean wafer
- **Finding:** Scratch wafers have a median yield of 90.8%, slightly above clean wafers, so their extra loss is negative (results/yield_by_class.csv).
- **Implication:** scratches rank last by die loss, but they matter for other reasons: they point to handling or CMP damage and can be a reliability risk that wafer-sort yield does not see. The root-cause table should say so, rather than reading the low loss as "unimportant".
- **Decided:** keep the negative value and explain it.

### d20 Lot-level analysis pool
- **Options:** all labelled wafers; only lots with enough labelled wafers.
- **Recommended / decided:** lots with at least 10 labelled wafers: 7,025 lots and 165,591 wafers (results/lots_facts.csv).
- **Why:** a lot with two labelled wafers cannot show whether a pattern hits the whole lot.

### d21 Does a pattern pile up inside lots
- **Method:** for each class, the average number of other same-pattern wafers in the same lot. This is compared with 200 shuffles of labels across all wafers, which keep lot sizes and class frequencies but break any link between lot and pattern.
- **Result:** every defect class clusters inside lots far above chance, with p ≤ 0.005 for all classes (the smallest p-value 200 shuffles can give). Ratios are Donut 127×, Random 111×, Center 30×, Edge-Ring 17×, Near-full 11×, Loc 5.5×, Edge-Loc 5.0× and Scratch 4.8× (results/lots_clustering.csv, figures/lots_clustering.png).
- **Process reading:** patterns that come in lots point to something every wafer in the lot shared (a chamber, a recipe, a time window), not to one-off wafer events. That decides whether an engineer pulls tool history or a single wafer's handling log.
- **Caveat:** labels may have been assigned lot by lot, which could inflate clustering. That can't be ruled out from the data alone.

### d22 Do patterns run in neighbouring slots
- **Method:** the chance that the next slot holds the same pattern. This is compared with 200 shuffles inside each lot, which keep each lot's pattern mix and scramble only the order.
- **Result:** most patterns show little ordering beyond their lot mix: Center 1.06×, Random 1.08× and Edge-Ring 1.00×. Loc is the exception at 1.28× (p = 0.005). Near-full shows 3.2× (p = 0.015) but rests on very few pairs. Donut, Edge-Loc and Scratch are not different from chance (results/lots_slot_order.csv).
- **Process reading:** the lot is the unit, not the slot. Edge-Ring in particular tends to fill whole lots. That fits a lot-level process condition better than a tool drifting wafer by wafer.

### d23 Excursion lots
- **Definition:** a lot where one defect pattern covers at least half of its labelled wafers.
- **Result:** 502 of the 7,025 pooled lots (7.1%) are excursion lots, and 361 of them are Edge-Ring. Center accounts for 76 (results/lots_excursions_by_class.csv).
