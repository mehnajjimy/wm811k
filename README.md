# wm811k

Which wafer defects cost a chip fab the most chips, and can a small model catch them? I worked through that the way a process or yield engineer would, using 811,457 real production wafer maps from the WM-811K dataset (Wu, Jang & Chen, IEEE TSM 2015). Each map shows which chips (dies) on a silicon wafer failed testing.

## Key results

- Two edge patterns, Edge-Ring and Edge-Loc, cause 47% of the extra dies lost to defects.
- Defects come in lots (batches of wafers processed together). 502 of 7,025 lots are excursions, which points at tools and chambers.
- A small convolutional neural network (CNN) passes 3.7% of defective wafers as clean, against 8.4% for a random forest.
- On the same 180 wafers, the CNN scores 0.916 macro-F1 against 0.718 for Claude Opus 5.5 with no training. Macro-F1 is a score that weighs every defect type equally.
- An AI assistant drafting incident reports passed every made-up-detail check on 60 of 60 drafts when limited to my root-cause table, against 2 of 60 without it.

![Extra dies lost per 1,000 labelled wafers for each defect pattern](figures/yield_pareto.png)

Each bar compares a pattern with clean wafers of the same product.

Contents: [Key results](#key-results) · [How I evaluated it](#how-i-evaluated-it) · [Results](#results) · [Root causes](#root-causes) · [Limits](#limits) · [Run it](#run-it) · [Layout](#layout)

Every number below comes from a file in `results/`. Every choice behind it, including the options I turned down, is in [DECISIONS.md](DECISIONS.md).

## How I evaluated it

- I split by lot, not by wafer. Wafers in one lot share tools and timing, so a random wafer split leaks.
- 187 defect maps also show up as exact copies across lots (93 copy groups), so I merged those lots before splitting and no copy crosses train and test. That gives about 70/15/15 train, validation and test, with 25,939 labelled test wafers.
- I report macro-F1 and per-class recall (the share of a defect type the model finds), never headline accuracy. 85.2% of test wafers are "none", so a model that never finds a single defect is already 85.2% accurate.
- 95% confidence intervals (CI) resample whole lots (500 lot-bootstrap rounds), because wafers in the same lot aren't independent.
- Each model touched the test set once. Every setting was picked on validation.

Sources: d13 to d15

## Results

### Edge defects cost the most dies

Edge-Ring and Edge-Loc cost the most dies. I measured excess die loss, the dies lost beyond what clean wafers of the same product lose, using map grid size as the product marker.

- Edge-Ring: 25.0% of excess loss
- Edge-Loc: 22.2%
- Random: 21.4%

Together these three make up 69% of the dies lost to patterns. Near-full is the worst per wafer, with a median yield of 12%, but it's rare. Scratch wafers lose no dies on net against their product baseline.

Sources: `results/yield_by_class.csv`, `figures/yield_pareto.png`, d16, d17

### Defects come in lots

Every defect pattern piles up inside lots much more than chance would allow. That points at tools and chambers more than random events.

How much more likely than expected a wafer is to have a lot-mate with the same pattern:

- Donut: 127 times
- Random: 111 times
- Edge-Ring: 17 times

Every class passes a permutation test at p < 0.005. A permutation test shuffles the labels to see what chance alone would give.

502 of 7,025 lots (7.1%) are excursion lots, where one defect pattern covers half or more of the labelled wafers. 361 of those are Edge-Ring.

Slot order inside a lot only adds a little on top of that. Loc is the clearest case, at 1.28 times more likely than chance to repeat in the next slot.

Sources: `results/lots_*.csv`, `figures/lots_clustering.png`

### The CNN misses fewer defects

Overall the two models are close, but the CNN misses far fewer defects. The random forest works from 24 features I engineered from each map, and the CNN reads the map image directly.

- Random forest: 0.867 macro-F1 (95% CI 0.846 to 0.880), 8.4% of defect wafers passed as clean
- Small CNN with 0.58M weights, trained 3 times from different random seeds: 0.877 (0.855 to 0.893), seeds 0.877 to 0.884, 3.7% passed as clean

The CNN beats the forest by 0.010 macro-F1, with a paired interval of −0.008 to +0.029. That interval includes zero, so overall neither one is clearly better.

The real difference is in the expensive mistake, a defect wafer called clean. The CNN drops Scratch missed as clean from 27% to 11% and Loc from 20% to 8%. Its recall gains on Loc (+0.18) and Scratch (+0.12) are clear of zero.

Sources: `results/cnn_minus_rf.csv`, `results/missed_as_none_rf_vs_cnn.csv`, `figures/rf_confusion.png`, `figures/cnn_confusion.png`

### The CNN halves the review work

A review gate only calls a wafer clean when the model is confident enough, and sends the rest to a person. The threshold is picked on validation.

With missed defects held near 5%, the forest sends 19.0 clean wafers per 1,000 to review and the CNN sends 10.3. So at the same miss rate the CNN needs about half the review work.

If the CNN only decides the 90% of wafers it's surest about, its macro-F1 on those wafers is 0.986.

Sources: `results/abstain_operating_points.csv`, `figures/abstain_tradeoff.png`

### Where the CNN looks

For most patterns the CNN looks at the failing dies, not at unrelated parts of the image. I checked this with Grad-CAM (a heat map of where the model looked) on up to 300 correct test wafers per class. For every class except Near-full, much less heat lands outside the wafer than chance would give.

[Full analysis](docs/gradcam.md)

Sources: `results/gradcam_summary.csv`, `figures/gradcam_examples.png`

### Some labels are wrong

I hand-checked the 24 test wafers that both models confidently gave a different class from their label. On a blind re-review, made before seeing the models' answers, 10 labels were wrong. My first pass, done after seeing their answers, had found 17, so it was anchored on the models.

Fixing them barely moves the scores. Macro-F1 goes from 0.867 to 0.870 for the forest and from 0.877 to 0.879 for the CNN.

[Full analysis](docs/label-audit.md)

Sources: `results/label_audit_candidates.csv`, `results/label_rescore_summary.csv`, d64, d67

### General AI models fall short

Vision-language models (AI models that read images and text) can't replace the trained models here when given no training. On the same 180 test wafers, with my best prompt for the two AI models:

- Small CNN: 0.916 macro-F1
- Random forest: 0.866
- Claude Opus 5.5: 0.718
- GPT-6 Luna: 0.612

Opus beats Luna on every prompt, and both are good at Scratch. The full analysis also compares against GPT-4o results from the Cao and Tao paper.

[Full analysis](docs/vlm.md)

Sources: `results/vlm_summary.csv`, `results/vlm_pairs.csv`, `results/vlm_paper_style.csv`, d46 to d48, d53, d55, d61, d65, d66

### An AI that drafts incident reports without making things up

When the CNN flags a wafer, GPT-6 Luna drafts the first steps of an 8D report (a standard manufacturing problem-solving format), using only the matching row of my root-cause table. Code, not another AI, checks every draft for invented tools, numbers, causes and references. With the table row, 60 of 60 drafts passed every check, and without it 2 of 60 did.

That shows the limit works. It doesn't show the table is right, since most rows are still hypotheses.

[Full analysis](docs/triage-assistant.md)

Sources: `results/assistant_summary.csv`, d49, d50, d60, d62

### Simulated wafers catch defects but not clean ones

I built a process-fault simulator ([docs/simulator.md](docs/simulator.md)) that makes wafer maps from physical fault settings. A random forest trained only on 27,000 simulated maps catches most real defect types and passes only 0.4% of defects as clean. But it calls 51% of real clean wafers defects, because real clean wafers carry weak edge and line structure the simulator doesn't make yet.

[Full analysis](docs/synthetic-training.md)

Sources: `results/synthetic_*.csv`, d51 to d54

## Root causes

[docs/root_causes.md](docs/root_causes.md) maps each pattern to its likely process modules and physical mechanism, the first fab step to check, and the signal that would confirm the cause. Each row cites a verified source or is marked as an engineering hypothesis. The Edge-Ring, Center and Donut rows draw on my own hands-on spin-coat edge-bead and ALD uniformity work.

## Limits

- The data has no product ID, so I use map grid size as a stand-in for product.
- The labelled wafers lean toward worse ones (median yield 89.4% against 94.8% for unlabelled), so "per 1,000 labelled wafers" isn't a rate for the whole fab.
- Labels may have been assigned lot by lot, which could inflate the lot clustering. The data can't rule that out.
- There are no tool, chamber or recipe fields, so the root-cause table stays a set of hypotheses until someone checks it against real process history.

## Run it

Download `LSWMD.pkl` from the [WM-811K Kaggle page](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and put it in the repository root. It isn't included because of its size.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then run the scripts from the repo root in this order. Each one writes to `results/` and `figures/`.

```bash
python scripts/run_audit.py
python scripts/run_split.py
python scripts/run_yield.py
python scripts/run_lots.py
python scripts/run_rf.py
python scripts/run_cnn.py
python scripts/run_abstain.py
python scripts/run_gradcam.py
python scripts/run_label_audit.py
python scripts/run_label_rescore.py
python scripts/run_synthetic.py
python scripts/run_vlm.py
python scripts/run_vlm.py --backend opus --prompts raw rules rules_v2
python scripts/run_assistant.py
```

`run_cnn.py` trains on an Apple GPU if there is one and falls back to the CPU. On an Apple M5 each of its four training runs took 30 to 50 minutes.

`run_vlm.py` and `run_assistant.py` call GPT-6 Luna through the codex CLI bundled with the ChatGPT desktop app, so they need that installed and signed in. The Opus run calls the Claude Code CLI that comes with the Claude desktop app, which needs to be signed in with `claude auth login`. I've only run the whole pipeline on my Mac.

## Layout

- `src/wm811k/` holds all the logic: loading, audit, split, yield, lots, features, simulator, CNN, review gate, VLM and assistant.
- `scripts/run_*.py` each run one analysis.
- `tests/` has 177 pytest checks, including the yield formula, a lot-disjoint split, each feature's physics on simulated wafers and the hallucination checks.
- `config.yaml` holds every seed and threshold.

## License

MIT
