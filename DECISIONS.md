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

### d24 Simulator: fail probability per die, then one draw per die from a separate stream
- **Options:** (a) place fail dies directly from a shape, such as "pick N dies in a ring"; (b) compute a fail probability for every die from the physics, then draw pass or fail. For randomness: one random stream for everything, or a layout stream (where a fault sits) plus a die stream (one uniform number per die).
- **Recommended:** (b), with two streams split from one seed.
- **Why:** The probability map can be inspected and gives expected yield without sampling. Independent causes combine correctly as `1 - Π(1 - p)`. With separate streams, fault strength never changes the die draws, so a stronger fault can only add fails. "Stronger fault means lower yield" is then tested die by die, not only on average.
- **Decided:** approved by Mehnaj (2026-11-03).

### d25 Simulator: radial and edge faults as a process window
- **Options:** (a) set fail probability directly as a function of radius; (b) model a local deviation from target in percent (film thickness or etch depth), add die-to-die scatter, and fail dies outside a ± spec limit.
- **Recommended:** (b).
- **Why:** This is how parametric yield loss works in a fab: the tool drifts, and dies that drift out of the window fail. The knobs are ones a process engineer recognises (peak deviation, spec limit, scatter), and pattern boundaries come out soft for a physical reason.
- **Decided:** approved by Mehnaj (2026-11-03).

### d26 Simulator: Edge-Ring and Edge-Loc share one roll-off mechanism
- **Options:** (a) separate, unrelated generators; (b) one exponential edge roll-off, made one-sided by a local edge arc (clamp, inlet, local edge bead) and/or off-center wafer placement.
- **Recommended:** (b), with the local arc on by default and placement offset as an optional second knob.
- **Why:** The useful engineering question is what breaks the symmetry. An earlier version used placement offset alone. It only made a thin one-die crescent unless the offset was several mm, and typical placement accuracy was not checked, so the local arc was added.
- **Decided:** approved by Mehnaj (2026-11-03).

### d27 Simulator: Loc as a gaussian particle cloud with a kill ratio
- **Options:** (a) scatter individual particle points; (b) compute the expected number of killer particles per die from a gaussian density, and apply Poisson.
- **Recommended:** (b).
- **Why:** (b) is exact for a Poisson cloud, needs no per-particle randomness (which keeps d24 intact), and uses kill ratio, which yield engineers measure with defect-to-bin overlays. It matches the "local defects as a bivariate normal" framing in Hwang and Kuo (2007) at abstract level.
- **Decided:** approved by Mehnaj (2026-11-03).

### d28 Simulator: optional negative binomial spread for Random
- **Options:** (a) plain Poisson only; (b) also allow a gamma-distributed wafer-level defect density (`cluster_alpha`).
- **Recommended:** (b), off by default.
- **Why:** Poisson and negative binomial yield models are standard in yield engineering and a likely interview topic. The stated limitation is that this spreads density between wafers, not within one wafer.
- **Decided:** approved by Mehnaj (2026-11-03).

### d29 Simulator: grid with rectangular dies, corner test and edge exclusion
- **Options:** square grid only, or `rows` by `cols`. A die exists if its center is on the wafer, or only if all four corners fit inside the wafer minus an edge exclusion.
- **Recommended:** `rows` by `cols` with the corner test and a 3 mm default exclusion.
- **Why:** WM-811K has 632 distinct map sizes (results/audit_facts.csv), and the most common one is 32 by 29 (results/audit_map_sizes.csv), so a synthetic-to-real test needs non-square grids. Partial edge dies are not product, and edge exclusion is a real fab parameter. It also explains why the edge generators need a large peak deviation: on the 40 by 40 grid, the outermost dies sit about 9 mm in from the edge.
- **Decided:** approved by Mehnaj (2026-11-03).

### d30 Simulator: return the probability map and fully resolved parameters
- **Options:** return only the map; or return the map, the fail probability map, and every parameter, including values drawn from the seed (cluster center, scratch start, edge angle).
- **Recommended:** the second.
- **Why:** Any map can be rebuilt from its params alone, and a test checks this. Expected yield comes straight from the probability map.
- **Decided:** approved by Mehnaj (2026-11-03).

### d31 Simulator: `combine` for mixed-type wafers
- **Options:** one fault per map only; or a helper that unions faults on one wafer.
- **Recommended:** the helper.
- **Why:** Mixed-type patterns are a known real problem (Kyeong and Kim 2018; Kim, Lee and Kim 2018), and a union of fails is the simplest honest model. Interactions between faults are not modelled, and the docs say so.
- **Decided:** approved by Mehnaj (2026-11-03).

### d32 Simulator labels match the dataset labels
- **Options:** lowercase names such as `edge-loc`; or the dataset's exact strings from `wm811k.data.CLASSES`.
- **Recommended:** the exact dataset strings, with a test that checks they match.
- **Why:** The synthetic-to-real test (F′) and the root-cause table both need to join on the real labels without a mapping step.
- **Decided:** approved by Mehnaj (2026-11-03).

### d33 Root-cause table: "test" as a possible module
- **Options:** keep only deposition, etch, CMP, litho and handling; or also flag the tester and probe card where relevant.
- **Recommended:** flag test on the Near-full row and in the notes.
- **Why:** Ruling out a test artifact with a retest is usually the cheapest first step, and a yield engineer would expect it.
- **Decided:** approved by Mehnaj (2026-11-03).

### d34 Simulator figure: colours and seeds
- **Options:** colour per class; or pass and fail colours only.
- **Recommended:** no die matches the page, pass is a quiet gray, and fail is the only strong colour (blue). Seeds 811 to 813 and grid sizes are in `config.yaml`.
- **Why:** In a wafer map the class is the shape, not a colour, so pass versus fail is what needs encoding. The pair was checked with a colour-vision-deficiency palette validator. Blue avoids suggesting a status alarm. The per-class colour rule in STYLE.md still applies to bar charts and other cross-class figures; this figure has no per-class marks.
- **Decided:** approved by Mehnaj (2026-11-03).

### d35 Root-cause sources: how citations were verified in the cloud session
- **Finding:** the cloud session's network policy blocked doi.org, Crossref, IEEE Xplore, arXiv, OpenAlex and Semantic Scholar.
- **Options:** stop and wait for network access; or verify through web search of repository and catalogue records, marking each citation as "record verified" (DOI not resolved directly) and reading abstracts only.
- **Recommended:** the second, with every table row labelled "engineering hypothesis" unless a source supports its general point at abstract level.
- **Why:** This follows d09 without claiming more than was checked. Full-text checks of the pattern-to-cause links are listed as open items in docs/root_causes.md.
- **Decided:** approved by Mehnaj (2026-11-03).

### d36 Merge of the cloud branch (PR #1)
- **What happened:** the cloud session numbered its decisions d16–d27, which clashed with main's d16–d23. They were renumbered to d24–d35, including the tags in `src/wm811k/simulator.py` and the docs. Main's d01–d23 are unchanged. All 141 tests pass after the merge.
- **Follow-ups in the root-cause table:**
  - Neo et al. 2017 was found in Ezzat et al.'s reference list and verified against Crossref (S8).
  - Ezzat et al.'s citing statements, read on ar5iv, now appear in the Center, Donut and Edge-Ring rows.
  - Mehnaj's spin-coat edge-bead (L1) and ALD uniformity (L2) work is tied to the Edge-Ring, Center and Donut rows (see d09).
- **Cloud decisions d24–d35:** approved by Mehnaj (2026-11-03).

### d37 Random-forest features
- **Options:** raw pixels; the Wu et al. 2015 features (density regions, Radon, geometry); our own explainable features.
- **Recommended / decided:** our own set of 24 features that a yield engineer would read off a bin map (`src/wm811k/features.py`):
  - fail fraction
  - fail density in 5 rings of normalised radius
  - sorted fail densities of 8 outer-edge sectors, so the features don't depend on rotation
  - count, size, position, elongation and fill of the biggest fail cluster
  - the share of fails that have 3 or more failing neighbours
- **Why:** radius is measured as a fraction of the wafer radius, so the same physical ring gives the same number on any map size (d17 deals with size for yield; this deals with it for features). The features are inspired by Wu et al.'s idea of density and geometry features, but they are not a reproduction of them, since that paper's method details have not been checked against its full text (RESEARCH.md). Tests on simulated wafers check that each feature moves the way its physics says (`tests/test_features.py`).

### d38 Random-forest setup
- **Choices:** 500 trees with `class_weight="balanced_subsample"`, so the forest does not learn to call everything "none". `min_samples_leaf` was picked on validation only: 1 → 0.837, 3 → 0.853, 10 → 0.826 macro-F1 (results/rf_val_tuning.csv). The final model is refit on train + validation, and the test set was used once.
- **Decided:** as above.

### d39 Random-forest result (test set, lot split, 25,939 labelled maps)
- **9-class macro-F1:** 0.867, with a lot-bootstrap 95% interval of 0.846–0.880 (results/rf_summary.csv, results/rf_test_bootstrap_ci.csv). Whole lots are resampled, not single wafers.
- **8-class macro-F1 (defects only, side view per d05):** 0.896.
- **Per-class recall:** Edge-Ring 0.96, Center 0.92, Random 0.90, Edge-Loc 0.86, Donut 0.85, Loc 0.67, Scratch 0.61. Near-full is 1.00, but on only 23 maps (results/rf_test_per_class.csv).
- **Accuracy (not a headline, per d04):** 97.3%, against 85.2% for a model that always says "none". This is the example of why accuracy misleads here.
- **Process reading:** the main failure is defects passed as "none". 27% of Scratch, 20% of Loc and 11% of Edge-Loc test wafers are missed that way (figures/rf_confusion.png). In a fab a missed excursion costs more than a mislabelled one, so the abstain and threshold work (item G) should target this, and the CNN should be judged on it.
- **Top features:** biggest-cluster radius, fill and share, neighbour clustering, and the middle-ring density (results/rf_feature_importance.csv).

### d40 Where the CNN trains
- **Options:** the desktop RTX 5070 Ti over SSH (d02), which isn't set up yet; the Mac's built-in GPU (Apple M5, PyTorch MPS backend); free Colab.
- **Recommended:** the Mac's GPU for the small CNN. The desktop stays the plan for the local open-weight VLM.
- **Why:** a 0.58M-weight network on 64×64 maps fits easily. A timing test gave about 85 seconds per epoch over the 120k training maps (about 72 seconds of that is the network's own compute), so a full run takes under an hour.
- **Decided:** Mehnaj chose the Mac's GPU (2026-11-03).

### d41 CNN input and model
- **Resize:** maps come in 632 sizes. Each map is turned into two 64×64 channels by area averaging: the share of each cell covered by dies, and the share covered by failing dies. Nearest-neighbour resizing was rejected because on big maps it drops most of a one-die-wide scratch, while area averaging keeps it as a faint line (`tests/test_cnn.py`).
- **Model:** four blocks of two 3×3 convolutions with batch norm and ReLU, then max pooling (32, 64, 128 and 128 channels). Global average pooling, dropout 0.3, then one linear layer. 583,753 weights in total. It's deliberately small: the point is a fair comparison with the forest, not chasing accuracy (RESEARCH.md lists that as overdone).
- **Augmentation:** each training image gets one of the 8 rotations and mirror flips at random. A defect pattern means the same thing at any angle, and the features the forest uses are rotation-free too (sorted edge sectors). Augmentation is applied on the fly to training batches only, after the split, so nothing augmented can reach validation or test.
- **Training:** AdamW (learning rate 1e-3, weight decay 1e-4), batches of 256, and a cosine learning-rate decay over at most 40 epochs.

### d42 CNN tuning and stopping
- **Options for class imbalance:** fully balanced loss weights, the same idea as the forest's `balanced_subsample`; or the square root of those weights. Full balancing gives Near-full about 1,000 times the weight of "none", which can make training noisy.
- **Recommended / decided:** try both and keep whichever scores best on validation macro-F1. Each run keeps the epoch with the best validation macro-F1 and stops after 6 epochs with no gain. The test set is used once, after that choice.
- **Difference from the forest:** the CNN trains on the training lots only, because validation is needed for stopping. The forest was refit on train + validation. That puts the CNN at a small data disadvantage, and it's stated next to the results.

### d43 Seed spread
- The chosen CNN setup is trained again with two more seeds (812, 813), and all three test macro-F1 scores are reported. This shows how much of the score comes from the luck of the starting weights. MPS isn't bit-for-bit deterministic, so a rerun with the same seed can differ slightly.

### d44 Review gate for wafers called clean (item G)
- **Problem:** the forest's main error is passing defect wafers as "none" (d39).
- **Options:** abstain on low overall confidence; or a gate that only calls a wafer "none" when p(none) is at least a threshold, otherwise giving it the most likely defect class so an engineer looks at it.
- **Recommended / decided:** both are measured. The gate is the operating rule because it targets the costly error directly. The threshold is picked on validation as the lowest one that keeps missed defects at or under 5% (`config.yaml`). The lowest is taken because it flags the fewest clean wafers. Only then is it applied to test.
- **Why 5%:** this is a policy choice, not something the data decides. The full trade-off curve is in `figures/abstain_tradeoff.png`, so any other target can be read off it.
- **Validation probabilities:** for the forest these come from the train-only model at the chosen leaf size (`results/rf_val_predictions.csv`). The final forest was refit on train + validation, so its own validation scores would be in-sample. The rerun of `run_rf.py` that saves them gives exactly the same test results as before.

### d45 Grad-CAM check (item J)
- **Options:** show a few heat maps; or also measure, on up to 300 correctly classified test wafers per class, (a) how much of the heat map falls outside the wafer and (b) "fail lift": heat on failing cells divided by the share of the wafer those cells cover.
- **Recommended / decided:** both. Pictures can be cherry-picked, so the numbers carry the claim, and the example wafers are drawn at random. A fail lift above 1 means the model looks at the fails more than a uniform heat map would. For Near-full, fails cover most of the wafer, so its lift is close to 1 by construction.

### d46 VLM prompts (item E)
- **Options:** label names only ("raw"); or a short rule for each label, checked in a fixed order ("rules"). The rules idea follows Cao and Tao (2026), but the wording is our own and covers all 9 classes including "none".
- **Recommended / decided:** run both on the same maps. Raw measures what the model already knows; rules measures how much a domain write-up helps.

### d47 VLM sample
- **Options:** the whole test set (25,939 maps); a random sample; or a class-balanced sample.
- **Recommended / decided:** 20 test maps per class (180 maps), drawn with the project seed. A random sample would be about 85% "none" and leave only a few maps for the rare classes. The forest and the CNN are scored on the same 180 maps, so all models are compared on equal terms. On a balanced sample, precision for "none" isn't comparable to the full-test numbers, so only the same-sample comparison is used.
- **Leakage guard:** image files have random names, the model runs in a read-only sandbox in a folder that holds only images, and any call where the model tried to run a command or read a file is flagged in the results.

### d48 VLM cost
- **Price:** standard-tier gpt-6-luna prices from OpenAI's pricing page (developers.openai.com/api/docs/pricing, checked 2026-11-03): $0.10 per million input tokens, $0.01 cached input, $0.50 output. Reasoning tokens are billed as output.
- **Caveat:** calls go through `codex exec`, whose own system prompt is most of the roughly 15k input tokens per call. The reported cost per 1,000 maps is therefore the cost as run, an upper bound on what a direct API call with only the prompt and image would cost. Through ChatGPT credits the project pays nothing extra (d01).

### d49 Triage assistant cases (item F)
- **Classifier feeding the assistant:** whichever of the forest and the CNN has the higher validation macro-F1. Test scores are not used for that choice.
- **Cases:** 6 test wafers per predicted pattern (54 wafers), each with these facts:
  - predicted pattern and model confidence
  - die yield and the clean-wafer median yield for the same product (d17)
  - slot in the lot
  - how many wafers in the lot share the predicted pattern
- **Conditions:** each case is sent twice:
  - **grounded:** with the table row for the predicted pattern
  - **ungrounded:** without the table, as a control
- **Scoring:** both conditions are checked against the same row of `docs/root_causes.md`. The ungrounded control is measured on how far it strays from what the table supports. It is not judged as wrong engineering, since the table itself is mostly hypotheses.

### d50 Hallucination checks
- **Options:** an LLM judge; a person reading every reply; or checks written in code.
- **Recommended / decided:** checks in code (`src/wm811k/assistant.py`), so the rate is repeatable and nobody's judgment is involved. A reply fails if it:
  - names a process module that appears nowhere in the table row
  - cites a source tag that isn't in the row's source cell, or writes citation-shaped text (et al., a year, a DOI)
  - writes a number that isn't in the facts or the row
  - names a specific tool or chamber (none are given)
  - gives a different pattern from the one it was handed
- **Probes:** six extra questions tempt the model to go past what it was given:
  - a tool ID
  - a journal reference for a row that has none
  - whether the implant step caused an Edge-Ring
  - a dollar cost
  - how many other wafers are affected
  - who should own the problem
- **Known limits:**
  - The checks catch invented specifics, not a wrong argument built from allowed words.
  - The module word list is fixed, so a module named in other words can slip through.
  - The table is parsed from the markdown file directly, so the doc and the check can't drift apart.

### d51 Synthetic training set (item F′)
- **Grid sizes:** drawn from the real map sizes in lots outside validation and test, in proportion to how often each size appears. Grids under 10 dies on a side are skipped.
- **Background:** each wafer's background fail share is drawn from 1% to 15%. That spans the yield quartiles of the unlabelled maps (0.862 and 0.986, `results/yield_labeled_vs_unlabeled.csv`), so no label is used. The simulator default (about 1% of dies) is far cleaner than real wafers, whose clean median yield is about 0.90.
- **Fault settings:** strengths and sizes are drawn from 0.6 to 1.6 times the simulator defaults. Other settings:
  - donut ring radius 50–110 mm
  - edge-loc arc 15–70°
  - loc particle counts 50–400
  - scratches 60–250 mm long, a third of them straight
  - Random at 20–60% fails
  - Near-full kill probability 0.75–0.98
- **Why:** the ranges come from physics and label-free facts, and none of them was tuned on labelled real wafers. That keeps the claim "trained without a single real label" true. The cost is that the ranges are guesses, and the sim-to-real gap is the result being measured.
- **Size:** 3,000 maps per class (27,000 in total).

### d52 Synthetic-only forest
- Same features and forest settings as the real baseline. The leaf size is fixed at 3, the value picked on real validation in d38, so nothing is tuned on test. The model is scored on the full real test set (25,939 maps) next to the forest trained on real labels.

### d53 VLM result and a flaw in the rules prompt
- **Result (180 balanced test maps, results/vlm_summary.csv):** Luna macro-F1 0.452 with the raw prompt (95% CI 0.385–0.508) and 0.451 with the rules prompt (0.389–0.501). The forest scores 0.866 on the same maps. Cost as run through codex: $0.60 and $0.42 per 1,000 maps. No invalid replies, and no call tried to run a command.
- **Main error:** Luna called 17 of the 20 clean wafers Random with the raw prompt, and all 20 with the rules prompt. Random is also where most missed Center, Edge-Loc, Edge-Ring and Loc wafers go. Scratch is its strong point: 17 of 20, against 11 of 20 for the forest.
- **Flaw found:** the rules prompt defined "none" as "only a few isolated fails". On training lots, the median clean wafer has 9.9% of dies failing (10th to 90th percentile 4.2% to 18.1%). Random wafers have a median of 47.5% (10th percentile 33.1%). In WM-811K, "none" means scattered background fails with no shape, not few fails.
- **Next step (not run yet):** a v2 rules prompt that defines none and Random by the training-lot fail shares above. It gets checked on a validation sample first, then scored once on the same test sample. The v1 numbers above stay as reported.

### d54 Synthetic-only result (item F′)
- **Result:** the forest trained on 27,000 simulated maps, with no real labels, scores 0.454 macro-F1 on the full real test set (95% CI 0.422–0.473). The forest trained on real labels scores 0.867.
- **Defects transfer partly:**
  - Random 0.97, Edge-Ring 0.88, Donut 0.76 and Scratch 0.73 recall. The Scratch figure beats the real-trained 0.61.
  - Only 0.4% of defect wafers are passed as clean.
- **Clean wafers don't transfer:** 51% of real clean wafers are called defects, mostly Scratch (4,611) and Edge-Ring (3,941). Center is often called Random (270 of 645).
- **Reading:** real clean wafers carry weak edge and line structure that the simulator's independent background fails don't have. This matches the "die fails are independent" limit in docs/simulator.md. The next simulator step is spatially correlated background and edge-weighted die variation.

### d55 Luna rules prompt v2
- **Change:** the v2 prompt fixes the "none" and "Random" rules found wrong in d53. It says scattered fails are normal, gives a healthy wafer up to about one die in six failing (the training-lot 90th percentile for "none" is 18.1%), and defines Random as dense, roughly a third of dies or more (the training-lot 10th percentile is 33.1%). Near-full becomes about three quarters or more (10th percentile 77.0%).
- **Order of checks:** v1 and v2 are first compared on a separate validation sample (20 per class, own image names, `results/vlm_*_val.csv`). Only then is v2 scored once on the same 180 test maps.
- **Disclosure:** the need for v2 was found by looking at test results, so v2 is not a clean held-out prompt. It is reported as a second iteration next to v1, never in place of it.

### d56 CNN result (test set, lot split, 25,939 labelled maps)
- **Choice on validation:** square-root weights scored 0.868 validation macro-F1 at epoch 17; balanced weights scored 0.823 at epoch 13 (results/cnn_val_tuning.csv). Square-root was kept.
- **9-class macro-F1:** 0.877, lot-bootstrap 95% interval 0.855–0.893 (results/cnn_summary.csv). Three seeds give 0.877, 0.879 and 0.884 (results/cnn_seed_spread.csv).
- **Against the forest on the same maps (paired lot bootstrap, results/cnn_minus_rf.csv):**
  - macro-F1 +0.010 (−0.008 to +0.029). The interval includes zero, so the overall gap is not clear.
  - clear per-class gains: Loc recall +0.18, Scratch +0.12, Center +0.04, Edge-Ring +0.03
  - clear loss: none recall −0.006, so the CNN flags slightly more clean wafers
- **Defects passed as clean (results/missed_as_none_rf_vs_cnn.csv):** Scratch 27% → 11%, Loc 20% → 8%, Edge-Loc 11% → 7%, Center 7% → 3%.
- **Caveat:** the CNN trained on the training lots only, while the forest also used validation (d42).

### d57 Review gate with both models (item G)
- **Forest:** the gate at p(none) ≥ 0.57, picked on validation, cuts defects passed as clean on test from 8.4% to 5.1%, for 19.0 clean wafers flagged per 1,000 (up from 9.1).
- **CNN:** as it is, it already misses only 3.7% of defects, under the 5% target. So the validation-picked threshold (0.31) loosens its clean call. On test it then misses 4.8% and flags 10.3 clean wafers per 1,000 (down from 15.5).
- **Reading:** at about the same 5% miss rate, the CNN sends roughly half as many clean wafers to review as the forest (10.3 vs 19.0 per 1,000). That is the number that matters on a fab floor, more than the macro-F1 gap.

### d58 Grad-CAM result (item J)
- **Off the wafer:** for every class except Near-full the median heat outside the wafer is well below the off-wafer share of the image (for example Center 3% of heat vs 22% of area). Near-full sits at about chance (19.7% of heat vs 20.5% of area), since nearly the whole wafer fails and there is little non-failing area to separate from the border. The CNN is not reading the map border for the other classes. (Corrected after a README number check; the first write-up said every class.)
- **On the fails (dense-fail lift, cells mostly failing):** Scratch 1.30, Loc 1.23, Edge-Loc 1.08 and Donut 1.06 are above 1. Center 0.94, Random 0.88, Near-full 0.81 and Edge-Ring 0.68 are below 1.
- **Reading:** for local patterns the CNN looks at the failing dies. For Edge-Ring the heat is a band just inside the edge. That fits the 8×8 Grad-CAM grid being too coarse to sit on a one-die ring, but it is not proven. For Center the heat is diffuse, and that result is inconclusive.
- **Metric change, disclosed:** the first lift counted any cell with any fail as failing. On a wafer with a 10% background that covers 19–40% of the wafer and dilutes the pattern, so a dense version (more than half failing) was added after the first run. Both are kept in results/gradcam_summary.csv.

### d59 Label audit (item H)
- **Candidates:** 227 test wafers where both models give the same wrong class. 24 of them have both models at probability 0.8 or more (results/label_audit_candidates.csv, figures/label_audit/).
- **Most common swap:** labelled defect, both models say none (Loc 8, Center 4, Scratch 3, Edge-Loc 2).
- **First pass by Claude, not a verdict:** 18 labels look unsupported (mostly maps that show only background fails), 2 look right (a real scratch line both models missed) and 4 are unclear. The `verdict` column is left for Mehnaj. No score is recomputed with changed labels until those verdicts exist.

### d60 Two fixes to the hallucination checks, made after reading the first replies
- **Pattern field:** replies often write the pattern with extra words, such as "Center (model-predicted; confidence 1.0)". The check now reads that field the same way a VLM reply is read: it passes when it names exactly the given class and no other.
- **Derived numbers:** a gap between two given facts, such as 81.2 − 70.7 = 10.5 points, is arithmetic, not invention. Differences of fact numbers are now allowed, with ±0.05 for rounding.
- **Disclosure:** both changes were made after seeing 8 replies, before any summary was computed. All replies are scored with the final checks. Each change makes the checks less strict, so the hallucination rates reported are not inflated by them.

### d61 Rules v2 on validation, and calls that ran commands
- **Validation (180 maps, results/vlm_summary_val.csv):**
  - v1 rules macro-F1 0.410 (0.341–0.468); v2 rules 0.563 (0.501–0.623)
  - clean wafers called defects: v1 20 of 20, v2 4 of 20
  - the forest (0.879) and the CNN (0.895) on the same maps stay far ahead
- **Decision:** v2 improves on validation, so it is scored once on the same 180 test maps as v1.
- **Calls that ran a command:** none of the 360 test calls tried to run anything, but on validation 12 v1 calls and 5 v2 calls did. The command text was not saved then. Dropping those calls leaves macro-F1 almost unchanged (v1 0.410, v2 0.561). They also don't score unusually well (58% and 80% correct), so they don't explain the result.
- **From the v2 test run on:** the runner saves the command text (the `actions` column) so it can be checked that a call only looked at its own picture. Each call runs in a folder that holds only images with random names, so there is no label for it to find there.

### d62 Source check refined, and the assistant result (item F)
- **Two false positives found by reading replies:**
  - The one grounded "failure" quoted "Ezzat et al." from the row's own source cell, and the check flagged it.
  - Most ungrounded "invented sources" were lines like "no external references consulted", not citations.
- **Fix:** citation-shaped text that appears in the row is allowed. A `sources` entry only counts as a citation if it carries a reference marker (a tag, URL, DOI, "et al" or a year). All replies were rescored with the final checks. Like d60, this makes the checks less strict.
- **Result, grounded (results/assistant_summary.csv):** all 54 wafer cases and all 6 probes pass every check. That means no module outside the row, no source outside its tags, no number outside the facts, and no named tool. This includes the probes asking for a tool ID, a dollar cost, a reference for the Scratch row (which has none) and the number of other affected wafers.
- **Result, ungrounded control:** 2 of 54 wafer cases and 0 of 6 probes pass.
  - 96% of wafer replies name a module the table row doesn't support, most often lithography, implant or thermal for Center.
  - 22% cite an outside reference, for example a lecture URL and a thesis URL. These were not checked and may or may not exist.
  - 7% write numbers not in the facts, and 4% give a different pattern.
- **What this does and does not show:**
  - Grounding keeps the model inside the table every time on these 120 replies. It does not show the table is right; most rows are hypotheses (docs/root_causes.md).
  - It does not show the ungrounded replies are wrong engineering; they go beyond what this project can support.
  - The checks catch invented specifics, not a weak argument made from allowed words.

### d63 Rules v2 on test
- **Result (same 180 test maps, results/vlm_summary.csv):** macro-F1 0.612 (0.538–0.682), up from 0.451 for v1. Clean wafers called defects fall from 20 of 20 to 3 of 20, but 17% of defect wafers are now called clean.
- **Same maps, trained models:** the CNN scores 0.916 (0.866–0.953) and the forest 0.866.
- **Commands:** none of the 180 v2 test calls ran a command (the `actions` column is empty).
- **Reading:** a correct domain definition of "none" is worth about 0.16 macro-F1 for a zero-shot model. Even corrected, it stays about 0.3 below a 0.58M-weight CNN trained on labelled maps, so it cannot replace labelling here. Its one strength is Scratch, at 20 of 20.

### d64 Label audit verdicts and rescoring (item H)
- **Verdicts by Mehnaj (2026-11-03), all 24 candidates (results/label_audit_candidates.csv):**
  - **17 wrong labels:** 14 should be none, mostly maps labelled Loc or Center that show only background fails. The other 3 are Edge-Loc, Loc and Near-full (#13), and #7 is Random. Both models agreed with the verdict on 16 of the 17, the exception being #7, where both said none.
  - **4 labels right**, all missed by both models: #4 Center, #5 Loc, and #8 and #9 Scratch.
  - **3 mixed:** #10 Center + edge, #15 Loc + Random, #24 Center + an upper slice. WM-811K forces one label per wafer, so these have no single right answer.
- **Rescoring:** wrong labels are swapped for the verdict class, and mixed wafers are dropped rather than forced into a class (results/label_rescore_summary.csv).
  - forest 0.867 → 0.872
  - CNN 0.877 → 0.881
  - The original scores stay the headline, because only 24 of 25,939 test labels were reviewed.
- **Reading:**
  - When both models confidently disagree with a label, the label was wrong 17 times in 24 (71%). The most common error is a defect label on a wafer with no visible pattern.
  - Mixed wafers (3 of 24) show a limit of the single-label scheme.
  - The 4 real misses include 2 scratches, the models' weakest class.
- **Note on #7:** its fail share is 14%, below the usual Random range (training-lot 10th percentile 33%). The Random verdict is a call on shape, not on density.
- **Update:** a re-review on 2026-11-08 changed 15 of these 24 verdicts. See d67, which is now the record.

### d65 Claude Opus 5.5 on the same 180 test maps (item E, added after Luna)
- **Asked by Mehnaj (2026-11-08):** can Claude Opus 5.5 do what Luna did?
- **Options:** the Anthropic API (paid per token, which breaks d01's no-paid-API rule); asking Opus inside the build session (that session had already seen the labels, the results and the audit verdicts, so its answers would be contaminated); or a fresh Claude Code CLI call per map, signed in with Mehnaj's Claude Pro subscription.
- **Decided:** the CLI, one fresh call per map. It mirrors how Luna ran through `codex exec` on ChatGPT credits.
- **Setup (src/wm811k/vlm.py `ask_claude`):**
  - The picture goes inside the message as an image block, so the model needs no tool to see it, and all tools are switched off (`--tools ""`, `--restricted`, no MCP servers, no slash commands, no saved session).
  - Each call runs in an empty temporary folder, so there is nothing on disk to read even if a tool leaked through. Any tool call would be recorded in the `actions` column.
  - Claude Code's own system prompt is replaced with one neutral line. Luna kept the codex system prompt, so Opus's input is much smaller (about 1,400 tokens against about 15,000), and its cost is closer to a direct API call.
  - Effort is set to `medium`, the API default for Opus 5.5, written out so a change to the CLI default can't move it.
  - Same three prompts as Luna, unchanged. They were all fixed before Opus saw any map, so nothing was tuned on these test maps for Opus.
  - A call that fails (a usage limit, a crash, a timeout) is not saved, and a rerun asks it again. Without that, an error message would be scored as a wrong answer.
- **Cost:** tokens × Anthropic's published Opus 5.5 prices (platform.claude.com/docs/en/about-claude/pricing, checked 2026-11-08): $4 input, $5 or $8 per million for 5-minute or 1-hour cache writes, $0.20 cache reads, $20 output. The CLI writes the prompt to the 1-hour cache, so the as-run cost is above a plain API call. Nothing was actually billed, since the calls ran on the subscription.
- **Not like for like:** Luna is OpenAI's low-cost tier and Opus is a frontier model, so this asks whether the strongest general model can match a small CNN trained on the data, not which vendor is better.
- **Result (same 180 test maps, results/vlm_summary.csv, results/vlm_pairs.csv):**
  - label names only 0.527 (0.456–0.583), v1 rules 0.537 (0.467–0.586), v2 rules 0.718 (0.645–0.772)
  - Opus minus Luna, paired lot bootstrap: +0.075 (0.007 to 0.140) raw, +0.087 (0.020 to 0.152) v1, +0.106 (0.026 to 0.176) v2, all clear of zero
  - CNN minus Opus v2: +0.198 (0.131 to 0.272); forest minus Opus v2: +0.148 (0.076 to 0.231)
  - v2 calls 5 of 20 clean wafers defects (3 as Edge-Ring) and 3.8% of defect wafers clean
  - all 540 calls returned a valid label, none failed, and none used a tool (the `actions` column is empty)
- **Where it goes wrong:** Loc is 2 of 20 on v2, with 10 called Edge-Loc. That is the Loc versus Edge-Loc boundary RESEARCH.md lists as noisy in the labels themselves. Its best classes are Center, Near-full, Scratch and Edge-Ring (18 to 20 of 20).
- **Cost:** $10.65 to $13.95 per 1,000 maps as run, or $6.08 to $8.23 with every input token billed at the plain input price and no caching (`usd_per_1000_maps_no_cache`). The as-run figure is higher because the CLI writes each prompt to the one-hour cache at $8 per million. Luna's no-cache figure is not comparable the same way, because it still includes the roughly 15k codex harness tokens.
- **Reading:** a frontier model reads wafer maps clearly better than a low-cost one, and the same domain fix to the prompt helps both by about the same amount (+0.16 Luna, +0.18 Opus). It still sits about 0.2 macro-F1 below a 0.58M-weight CNN trained on labelled maps, at roughly 1.4 cents a wafer as run. So the d63 conclusion holds: zero-shot models can't replace labelling here.

### d66 Comparison with Cao & Tao 2026 (the zero-shot paper this item came from)
- **Asked by Mehnaj (2026-11-08):** how do our zero-shot results compare with Cao & Tao (arXiv 2610.35080)?
- **What they did (checked against the paper 2026-11-08):** gpt-4o-2024-08-06 at temperature 0.1, one query per map, on 9,000 maps drawn at random from the 172,950 labelled WM-811K maps. Six classes: Center, Donut, Edge (Edge-Loc and Edge-Ring merged), Local, Random, Scratch. None and Near-full are not used. Metrics are per-class accuracy, macro accuracy (unweighted mean of the six) and overall accuracy. No split is named.
- **Their WM-811K results (table 4):** plain map ("Raw") 58.7% macro, 43.7% overall. With the largest connected fail clusters coloured in, 76.7% / 80.1% (one cluster) and 79.1% / 81.3% (five clusters).
- **Correction:** RESEARCH.md had said their jump came from rule prompts. It comes with the change to the image. The paper does not say whether the Raw run used the same rule text, so the prompt's share of the jump is unknown.
- **Options:** quote both results side by side with no adjustment (different classes and metrics, so misleading); or re-score our runs their way.
- **Decided:** re-score ours their way (`paper_style` in scripts/run_vlm.py, results/vlm_paper_style.csv). Keep only test wafers whose label is one of their classes (140 of our 180), merge the two edge classes, and report per-class accuracy, macro accuracy, overall accuracy on our balanced sample, and overall accuracy reweighted to their per-class counts (1,747 / 502 / 3,430 / 1,830 / 555 / 936).
- **Result:** macro / overall at their mix: Luna v2 59.6% / 54.7%, Opus v2 72.1% / 73.9%, forest 81.2% / 85.4%, CNN 87.9% / 92.4%.
- **Limits:** our models answered from nine classes, so a none or Near-full answer counts as wrong here, which makes ours harder. Their sample has no lot split. We have only 20 maps per class (40 for Edge), so no interval is reported on this table. The rendering differs too (their colours and resolution against ours).
- **Reading:** on plain maps, Opus 5.5 with the v2 rules is well above GPT-4o on plain maps and below GPT-4o with highlighted clusters. Model, prompt and rendering all differ, so this doesn't separate them. The cluster highlighting is a cheap preprocessing step worth testing here.

### d67 Label audit re-review (supersedes the d64 verdicts)
- **Why:** Mehnaj graded the 24 candidates in d64 after seeing both models' calls and a Claude first-pass suggestion, so the verdicts could be anchored on the models.
- **How (2026-11-08):** a separate session showed each map with only its dataset label and fail share. Mehnaj gave a call first, and only then saw the earlier verdict and could confirm or change it. The models' calls and the Claude suggestion were hidden throughout.
- **Result (results/label_audit_candidates.csv, results/label_audit_verdict_counts.csv):**
  - 15 of 24 verdicts changed. The two passes agree on the kind of verdict (wrong, right, mixed) for 12 of 24.
  - Now 10 wrong labels (was 17), 13 labels right that both models missed (was 4), 1 mixed (was 3).
  - Both models match the verdict on 4 wafers (#1, #13, #16, #22), against 16 before.
  - Of the 10 wrong labels, 5 should be Edge-Loc (2 labelled Loc, 3 labelled none), 2 Random, 1 Near-full, 1 Scratch and 1 none.
  - 8 of the 15 changes are wafers first called "should be none" that Mehnaj now says show the labelled pattern.
- **Rescore with the re-reviewed labels (mixed wafer dropped):** forest 0.867 → 0.870, CNN 0.877 → 0.879 (results/label_rescore_summary.csv). The original scores stay the headline.
- **Reading:**
  - The first pass was anchored. When the models' answer was visible, Mehnaj sided with it far more often. That is a reason to grade blind by default.
  - Low agreement between passes shows these 24 maps are genuinely ambiguous, so "the label is wrong" is a weaker claim than d64 made.
  - The models confidently say none on wafers that a person, looking without them, calls a defect. That fits a pull toward the majority none class.
- **Possible next step (not done):** a fully blind pass with these 24 mixed among control wafers, to measure agreement with the dataset labels on both sets.
