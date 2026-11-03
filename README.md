# wm811k

Yield-pattern analysis of the WM-811K wafer map dataset (Wu, Jang & Chen, IEEE TSM 2015), written for process and yield engineers. Work in progress.

The project:
- classifies wafer defect patterns
- measures how much yield each pattern costs
- maps each pattern to the process steps worth checking first

## Status
Research and planning are done. See [RESEARCH.md](RESEARCH.md) for prior work and scope, and [DECISIONS.md](DECISIONS.md) for the choices made so far. No results yet.

Built so far:
- data loading, audit checks and a lot-grouped train/validation/test split (`src/wm811k/`, `results/`)
- a root-cause table that maps each pattern to process modules, first checks and confirming signals, with sources and their verification status ([docs/root_causes.md](docs/root_causes.md))
- a process fault simulator that builds synthetic wafer maps from physical fault parameters ([docs/simulator.md](docs/simulator.md), [figure](figures/simulator_examples.png))

## Data
Download `LSWMD.pkl` from the [WM-811K Kaggle page](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and put it in the repository root. It is not included here because of its size.

## License
MIT
