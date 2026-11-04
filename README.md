# wm811k

Yield-pattern analysis of the WM-811K wafer maps (811,457 production wafer maps; Wu, Jang & Chen, IEEE TSM 2015), written for process and yield engineers.

The question a fab asks of a bin map isn't "what does this look like?" It's "how many dies is it costing me, and which tool do I check first?" This project answers that:

- It ranks defect patterns by the yield they destroy.
- It classifies them the way a fab would deploy a classifier: split by lot, scored on rare patterns, with a review gate for the costly errors.
- It ties each pattern to a sourced root-cause table.
- It tests whether a vision-language model and a physics simulator can stand in for labelled data.
- It adds an LLM triage assistant whose claims are checked by code.

Every number below comes from a file in `results/`. Every choice behind it, with the options that were turned down, is in [DECISIONS.md](DECISIONS.md).

## How it was evaluated

- **Split by lot, not by wafer.** Wafers in one lot share tools and timing, so a random wafer split leaks. 187 defect maps also appear as exact copies across lots (93 copy groups). Those lots are merged before splitting, so no copy crosses train and test (d13–d15). The result is about 70/15/15 train/validation/test, with 25,939 labelled test wafers.
- **Macro-F1 and per-class recall, never headline accuracy.** 85.2% of test wafers are "none", so a model that never finds a defect is 85.2% accurate.
- **Intervals resample whole lots** (500 lot-bootstrap rounds), because wafers within a lot are not independent.
- **The test set was used once per model.** Every setting was picked on validation.

## Results

### Which patterns cost the most yield

Excess die loss is measured against the median yield of clean wafers of the same product, using grid size as the product marker (d16–d17). Edge-Ring (25.0% of excess loss), Edge-Loc (22.2%) and Random (21.4%) account for 69% of the dies lost to patterns. Near-full is the worst per wafer (median yield 12%) but rare. Scratch wafers lose no dies on net against their product baseline (`results/yield_by_class.csv`, `figures/yield_pareto.png`).

### Is it one wafer or the whole lot

Every defect pattern piles up inside lots far more than chance allows. Donut is 127× and Random 111× more likely than expected to have a lot-mate with the same pattern, and Edge-Ring 17× (permutation p < 0.005 for every class). Slot order within a lot adds only a little on top of that. Loc is the clearest case, 1.28× more likely than chance to repeat in the next slot. 502 of 7,025 lots (7.1%) are excursion lots, where one defect pattern covers half or more of the labelled wafers, and 361 of those are Edge-Ring. That points to tools and chambers, not random events (`results/lots_*.csv`, `figures/lots_clustering.png`).

### Classifiers

| Test set, 9 classes | Macro-F1 (95% CI) | Defect wafers passed as clean |
|---|---|---|
| Random forest on 24 engineered features | 0.867 (0.846–0.880) | 8.4% |
| Small CNN (0.58M weights), 3 seeds | 0.877 (0.855–0.893); seeds 0.877–0.884 | 3.7% |

The overall macro-F1 gap between the CNN and the forest is +0.010, with a paired interval of −0.008 to +0.029. That interval includes zero, so neither model is clearly better overall. Where they differ is the expensive error, a defect called clean:

- Scratch missed as clean falls from 27% (forest) to 11% (CNN).
- Loc falls from 20% to 8%.
- The CNN's recall gains on Loc (+0.18) and Scratch (+0.12) are clear of zero.

Sources: `results/cnn_minus_rf.csv`, `results/missed_as_none_rf_vs_cnn.csv`, `figures/rf_confusion.png`, `figures/cnn_confusion.png`.

### When to send a wafer to an engineer

A review gate only calls a wafer clean when the model is confident, and its threshold is picked on validation. With missed defects held near 5%, the forest flags 19.0 clean wafers per 1,000 for review and the CNN 10.3. At the same miss rate, that's about half the review load. With the CNN deciding the 90% of wafers it is surest about, macro-F1 on those wafers is 0.986 (`results/abstain_operating_points.csv`, `figures/abstain_tradeoff.png`).

### Is the CNN looking at the defect

Grad-CAM heat maps were measured on up to 300 correct test wafers per class. For every class, far less heat falls outside the wafer than the off-wafer area would get by chance; for Center it is 3% of heat against 22% of the image. For Scratch, Loc, Edge-Loc and Donut, the heat is concentrated on the failing dies (lift 1.06–1.30). For Edge-Ring it sits as a band just inside the edge, which may be the coarse 8×8 heat-map grid; for Center it is diffuse (`results/gradcam_summary.csv`, `figures/gradcam_examples.png`).

### Suspect labels

227 test wafers get the same wrong class from both models, and 24 of them get it with high confidence from both. A first visual pass suggests 18 of those 24 labels aren't supported by the map, mostly wafers labelled Loc or Center that show only background fails. Those verdicts still need a human check, so no score has been recomputed (`results/label_audit_candidates.csv`, `figures/label_audit/`).

### Can a vision-language model skip the labelling

GPT-6 Luna, zero-shot, on 180 test wafers (20 per class) that the forest and CNN were also scored on:

| Same 180 wafers | Macro-F1 (95% CI) | Clean wafers called defects | Defect wafers called clean | Cost per 1,000 wafers |
|---|---|---|---|---|
| Luna, label names only | 0.452 (0.385–0.508) | 20 of 20 | 0% | $0.60 |
| Luna, one rule per label (v1) | 0.451 (0.389–0.501) | 20 of 20 | 0.6% | $0.42 |
| Luna, corrected rules (v2) | 0.612 (0.538–0.682) | 3 of 20 | 17% | $0.47 |
| Random forest | 0.866 (0.810–0.917) | 0 of 20 | 7% | — |
| Small CNN | 0.916 (0.866–0.953) | 0 of 20 | 2.5% | — |

The v1 rule prompt defined "none" as "only a few fails". On training lots, a clean wafer has a median of 9.9% failing dies, and Luna read that normal background as a Random pattern. The v2 prompt says scattered fails are normal and defines Random as dense, from the training-lot fail shares.

On a separate validation sample, v2 raised macro-F1 from 0.410 to 0.563, and only then was it scored once on test. v2 was prompted by looking at test results, so it is reported next to v1, not in place of it (d53, d55, d61).

Even corrected, Luna stays far behind a small trained model. Its strong point is Scratch: 20 of 20 with v2, against 11 of 20 for the forest and 15 of 20 for the CNN.

Costs are at OpenAI's published gpt-6-luna prices. They are an upper bound, because the codex harness used for the calls adds about 15k input tokens to every call (`results/vlm_summary.csv`, d46–d48).

### A triage assistant that can't make things up

Luna drafts an 8D D2–D4 first response for a flagged wafer from the CNN's prediction, the die yield, the product's clean-wafer yield and the lot context. It is told to use only the matching row of the root-cause table. Code checks every reply, with no LLM judge, for:

- process modules the row doesn't support
- sources outside the row's tags
- numbers not in the facts
- named tools
- the wrong pattern

| 54 test wafers + 6 trick questions | All checks pass | Module beyond the table | Outside reference cited | Number not in the facts |
|---|---|---|---|---|
| With the table row | 60 of 60 | 0% | 0% | 0% |
| Without the table (control) | 2 of 60 | 96% | 22% | 7% |

The trick questions ask for a tool ID, a dollar cost, a journal reference for a row that has none, and how many other wafers are affected. With the table, five of the six listed the requested item as unknown. The sixth, asked whether implant caused an Edge-Ring, never named implant and listed the causal step as unknown. Without the table, the replies regularly add lithography, implant or thermal causes for a Center wafer, and some cite lecture or thesis URLs that weren't checked.

This shows that grounding keeps the model inside what the project can support. It doesn't show the table is right; most rows are hypotheses. The checks catch invented specifics, not a weak argument made from allowed words (`results/assistant_summary.csv`, d49–d50, d60, d62).

### Training on simulated wafers only

A process-fault simulator ([docs/simulator.md](docs/simulator.md)) builds wafer maps from physical fault settings: radial thickness profiles, edge roll-off, particle clouds, scratch paths and Poisson defects. A forest trained on 27,000 simulated maps, with no real label, scores 0.454 macro-F1 (0.422–0.473) on the real test set:

- **What transfers:** it catches most defect types (Random 0.97, Edge-Ring 0.88, Donut 0.76, Scratch 0.73 recall) and passes only 0.4% of defects as clean.
- **What doesn't:** it calls 51% of real clean wafers defects.

Real clean wafers carry weak edge and line structure that independent background fails don't produce. That's the next thing the simulator has to learn (`results/synthetic_*.csv`, d51–d54).

## Root causes

[docs/root_causes.md](docs/root_causes.md) maps each pattern to:

- likely process modules and the physical mechanism
- the first fab step to check
- the signal that would confirm the cause

Each row cites a verified source or is marked as an engineering hypothesis. The Edge-Ring, Center and Donut rows draw on hands-on spin-coat edge-bead and ALD uniformity work.

## Layout

- `src/wm811k/` holds all the logic: loading, audit, split, yield, lots, features, simulator, CNN, gate, VLM and assistant.
- `scripts/run_*.py` runs one analysis each and writes `results/` and `figures/`.
- `tests/` holds pytest checks: the yield formula, a lot-disjoint split, each feature's physics on simulated wafers, the hallucination checks, and more.
- `config.yaml` holds every seed and threshold.

## Data

Download `LSWMD.pkl` from the [WM-811K Kaggle page](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and put it in the repository root. It isn't included because of its size.

## License

MIT
