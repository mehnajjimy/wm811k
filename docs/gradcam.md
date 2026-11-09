# Grad-CAM check

For most defect patterns, the CNN looks at the failing dies when it makes its call, not at some unrelated part of the image. This is the full write-up behind the Grad-CAM section of the [README](../README.md#grad-cam).

## What I measured

Grad-CAM (gradient-weighted class activation mapping) makes a heat map of which parts of an image pushed the model toward its answer. If a model is right for the wrong reason, the heat usually lands somewhere it shouldn't, like outside the wafer.

I ran it on up to 300 correctly classified test wafers per class and measured two things:

- How much of the heat falls outside the wafer, compared with how much of the image is outside the wafer.
- Lift: how much more heat lands on the failing dies than their share of the image would give. A lift of 1.0 means the model isn't focusing on them at all.

## What came out

For every class except Near-full, much less heat lands outside the wafer than chance would give. For Center it's 3% of the heat against 22% of the image.

Near-full sits at about chance (20% of the heat against 21% of the image). That makes sense, since almost the whole wafer fails on a Near-full map.

For Scratch, Loc, Edge-Loc and Donut, the heat is concentrated on the failing dies, with a lift of 1.06 to 1.30.

## What it doesn't show

For Edge-Ring the heat sits as a band just inside the edge, not on the ring itself. That might just be the coarse 8×8 heat-map grid, which can't resolve a thin ring at the very edge.

For Center the heat is spread out, so I can't say much about where the model looks.

Sources: `results/gradcam_summary.csv`, `figures/gradcam_examples.png`
