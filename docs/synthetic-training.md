# Training on simulated wafers only

A model trained only on simulated wafer maps catches most real defects, but it flags half of the real clean wafers as defective. This is the full write-up behind the simulator section of the [README](../README.md#training-on-simulated-wafers).

## What I built

I built a process-fault simulator that makes wafer maps from physical fault settings. How it works is in [simulator.md](simulator.md). It covers:

- radial thickness profiles (film thicker or thinner from center to edge)
- edge roll-off (the film thinning at the wafer edge)
- particle clouds
- scratch paths
- Poisson defects (random background fails scattered across the wafer)

## What I tested

I trained a random forest on 27,000 simulated maps, with no real labels at all. Then I scored it on the real test set.

## What came out

It scores 0.454 macro-F1 (0.422 to 0.473) on real wafers. Macro-F1 averages the F1 score of each class equally, so rare defect types count as much as common ones.

It catches most defect types. Recall, the share of a class the model finds, is:

- Random: 0.97
- Edge-Ring: 0.88
- Donut: 0.76
- Scratch: 0.73

It passes only 0.4% of defects as clean.

## What it doesn't show

But it calls 51% of real clean wafers defects. Real clean wafers carry weak edge and line structure that independent background fails don't produce. That's the next thing the simulator needs.

Sources: `results/synthetic_*.csv`, d51 to d54
