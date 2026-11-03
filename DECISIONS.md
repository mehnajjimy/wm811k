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
- **Why:** peer-reviewed links from pattern to cause are thin. Mehnaj's background is coursework, not hands-on fab work, so rows rest on literature and textbook physics.
- **Decided:** as recommended.

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

### d24 Simulator: fail probability per die, then one draw per die from a separate stream
- **Options:** (a) place fail dies directly from a shape, such as "pick N dies in a ring"; (b) compute a fail probability for every die from the physics, then draw pass or fail. For randomness: one random stream for everything, or a layout stream (where a fault sits) plus a die stream (one uniform number per die).
- **Recommended:** (b), with two streams split from one seed.
- **Why:** The probability map can be inspected and gives expected yield without sampling. Independent causes combine correctly as `1 - Π(1 - p)`. With separate streams, fault strength never changes the die draws, so a stronger fault can only add fails. "Stronger fault means lower yield" is then tested die by die, not only on average.
- **Decided:** pending, for Mehnaj to approve.

### d25 Simulator: radial and edge faults as a process window
- **Options:** (a) set fail probability directly as a function of radius; (b) model a local deviation from target in percent (film thickness or etch depth), add die-to-die scatter, and fail dies outside a ± spec limit.
- **Recommended:** (b).
- **Why:** This is how parametric yield loss works in a fab: the tool drifts, and dies that drift out of the window fail. The knobs are ones a process engineer recognises (peak deviation, spec limit, scatter), and pattern boundaries come out soft for a physical reason.
- **Decided:** pending, for Mehnaj to approve.

### d26 Simulator: Edge-Ring and Edge-Loc share one roll-off mechanism
- **Options:** (a) separate, unrelated generators; (b) one exponential edge roll-off, made one-sided by a local edge arc (clamp, inlet, local edge bead) and/or off-center wafer placement.
- **Recommended:** (b), with the local arc on by default and placement offset as an optional second knob.
- **Why:** The useful engineering question is what breaks the symmetry. An earlier version used placement offset alone. It only made a thin one-die crescent unless the offset was several mm, and typical placement accuracy was not checked, so the local arc was added.
- **Decided:** pending, for Mehnaj to approve.

### d27 Simulator: Loc as a gaussian particle cloud with a kill ratio
- **Options:** (a) scatter individual particle points; (b) compute the expected number of killer particles per die from a gaussian density, and apply Poisson.
- **Recommended:** (b).
- **Why:** (b) is exact for a Poisson cloud, needs no per-particle randomness (which keeps d24 intact), and uses kill ratio, which yield engineers measure with defect-to-bin overlays. It matches the "local defects as a bivariate normal" framing in Hwang and Kuo (2007) at abstract level.
- **Decided:** pending, for Mehnaj to approve.

### d28 Simulator: optional negative binomial spread for Random
- **Options:** (a) plain Poisson only; (b) also allow a gamma-distributed wafer-level defect density (`cluster_alpha`).
- **Recommended:** (b), off by default.
- **Why:** Poisson and negative binomial yield models are standard in yield engineering and a likely interview topic. The stated limitation is that this spreads density between wafers, not within one wafer.
- **Decided:** pending, for Mehnaj to approve.

### d29 Simulator: grid with rectangular dies, corner test and edge exclusion
- **Options:** square grid only, or `rows` by `cols`. A die exists if its center is on the wafer, or only if all four corners fit inside the wafer minus an edge exclusion.
- **Recommended:** `rows` by `cols` with the corner test and a 3 mm default exclusion.
- **Why:** WM-811K has 632 distinct map sizes (results/audit_facts.csv), and the most common one is 32 by 29 (results/audit_map_sizes.csv), so a synthetic-to-real test needs non-square grids. Partial edge dies are not product, and edge exclusion is a real fab parameter. It also explains why the edge generators need a large peak deviation: on the 40 by 40 grid, the outermost dies sit about 9 mm in from the edge.
- **Decided:** pending, for Mehnaj to approve.

### d30 Simulator: return the probability map and fully resolved parameters
- **Options:** return only the map; or return the map, the fail probability map, and every parameter, including values drawn from the seed (cluster center, scratch start, edge angle).
- **Recommended:** the second.
- **Why:** Any map can be rebuilt from its params alone, and a test checks this. Expected yield comes straight from the probability map.
- **Decided:** pending, for Mehnaj to approve.

### d31 Simulator: `combine` for mixed-type wafers
- **Options:** one fault per map only; or a helper that unions faults on one wafer.
- **Recommended:** the helper.
- **Why:** Mixed-type patterns are a known real problem (Kyeong and Kim 2018; Kim, Lee and Kim 2018), and a union of fails is the simplest honest model. Interactions between faults are not modelled, and the docs say so.
- **Decided:** pending, for Mehnaj to approve.

### d32 Simulator labels match the dataset labels
- **Options:** lowercase names such as `edge-loc`; or the dataset's exact strings from `wm811k.data.CLASSES`.
- **Recommended:** the exact dataset strings, with a test that checks they match.
- **Why:** The synthetic-to-real test (F′) and the root-cause table both need to join on the real labels without a mapping step.
- **Decided:** pending, for Mehnaj to approve.

### d33 Root-cause table: "test" as a possible module
- **Options:** keep only deposition, etch, CMP, litho and handling; or also flag the tester and probe card where relevant.
- **Recommended:** flag test on the Near-full row and in the notes.
- **Why:** Ruling out a test artifact with a retest is usually the cheapest first step, and a yield engineer would expect it.
- **Decided:** pending, for Mehnaj to approve.

### d34 Simulator figure: colours and seeds
- **Options:** colour per class; or pass and fail colours only.
- **Recommended:** no die matches the page, pass is a quiet gray, and fail is the only strong colour (blue). Seeds 811 to 813 and grid sizes are in `config.yaml`.
- **Why:** In a wafer map the class is the shape, not a colour, so pass versus fail is what needs encoding. The pair was checked with a colour-vision-deficiency palette validator. Blue avoids suggesting a status alarm. The per-class colour rule in STYLE.md still applies to bar charts and other cross-class figures; this figure has no per-class marks.
- **Decided:** pending, for Mehnaj to approve.

### d35 Root-cause sources: how citations were verified in the cloud session
- **Finding:** the cloud session's network policy blocked doi.org, Crossref, IEEE Xplore, arXiv, OpenAlex and Semantic Scholar.
- **Options:** stop and wait for network access; or verify through web search of repository and catalogue records, marking each citation as "record verified" (DOI not resolved directly) and reading abstracts only.
- **Recommended:** the second, with every table row labelled "engineering hypothesis" unless a source supports its general point at abstract level.
- **Why:** This follows d09 without claiming more than was checked. Full-text checks of the pattern-to-cause links are listed as open items in docs/root_causes.md.
- **Decided:** pending, for Mehnaj to approve.
