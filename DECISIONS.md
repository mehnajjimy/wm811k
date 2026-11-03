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
