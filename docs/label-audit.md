# Suspect labels

Of the 24 test wafers both models confidently disagreed with, 10 had wrong labels when I checked them blind. My first check had found 17, because I'd looked at the models' answers first. This is the full write-up behind the label section of the [README](../README.md#some-labels-are-wrong).

## What I checked

A label is the defect pattern the dataset says a wafer has. If two different models confidently agree that a label is wrong, it's worth checking whether the label itself is the problem.

227 test wafers get the same wrong class from both the random forest and the CNN. 24 of those get it with at least 0.8 probability from both models, and I went through all 24 by hand, twice.

## Two passes

The first time, I looked at each map after seeing both models' calls. Then I re-reviewed them, making my own call before seeing the models' answers or my first verdict. 15 of the 24 verdicts changed.

| 24 suspect wafers | First pass (saw the models' calls) | Re-review (my call first) |
|---|---|---|
| Label wrong | 17 | 10 |
| Label right, both models missed it | 4 | 13 |
| Two patterns on one wafer | 3 | 1 |
| Both models right | 16 | 4 |

## What changed and why

Most of the change is wafers I'd first called clean that do show the labelled pattern when I look without the models' answer next to it. My two passes agree on the kind of verdict for only 12 of the 24.

So these maps really are ambiguous, and my first pass was anchored on the models. I'm treating the re-review as the record.

In the re-review, 5 of the 10 wrong labels should be Edge-Loc (2 were labelled Loc and 3 none). That's the same blurry line Claude Opus 5.5 struggles with in the [vision-language model test](vlm.md).

## Effect on the scores

Macro-F1 averages the F1 score of each class equally, so rare defect types count as much as common ones. F1 combines how many of a class the model catches with how many of its calls for that class are right.

With the re-reviewed labels (the one mixed wafer dropped), macro-F1 goes from 0.867 to 0.870 for the forest and from 0.877 to 0.879 for the CNN. The original scores stay the headline, since only 24 of 25,939 labels were reviewed.

Sources: `results/label_audit_candidates.csv`, `results/label_rescore_summary.csv`, d64, d67
