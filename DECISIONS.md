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
