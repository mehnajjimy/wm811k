"""item j  does the cnn look at the defect or at something else on the map
run from the repo root after run_cnn.py  python scripts/run_gradcam.py
pictures alone can be cherry picked so two numbers are measured on many maps
off wafer mass  share of the heat map that sits outside the wafer
fail lift  heat on failing cells over the share of the wafer those cells cover"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import ListedColormap

from wm811k import cnn, viz
from wm811k.data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"
CACHE = ROOT / "cache"
DEFECTS = [c for c in CLASSES if c != "none"]
PER_CLASS = 300

# same wafer colours as the simulator figure  d34
PASS_COLOR = "#d3d1ca"
FAIL_COLOR = viz.ACCENT


def measures(cam: np.ndarray, img: np.ndarray) -> dict:
    """the two checks for one map  cam and img are at the same 64x64 size"""
    die, fail = img[0], img[1]
    on_wafer = die >= 0.5
    failing = fail > 0
    # a uniform heat map would put on the failing cells exactly the share of
    # the wafer they cover so a lift above 1 means the model looks at the fails
    area = failing[on_wafer].mean()
    # d45 the first version counts any cell with any fail  on a wafer with a
    # 10 percent background that is a big part of the wafer so it dilutes the
    # pattern  the dense version only counts cells that are mostly failing
    dense = fail > 0.5
    dense_area = dense[on_wafer].mean()
    return {
        "off_wafer_mass": float(cam[~on_wafer].sum()),
        "off_wafer_area": float((~on_wafer).mean()),
        "failing_area": float(area),
        "fail_lift": float(cam[failing].sum() / area) if area > 0 else np.nan,
        "dense_fail_lift": float(cam[dense].sum() / dense_area) if dense_area > 0 else np.nan,
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed = cfg["seed"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    viz.setup()

    cached = np.load(CACHE / f"cnn_images_{cfg['cnn']['size']}.npz")
    pos = pd.Series(np.arange(len(cached["index"])), index=cached["index"])
    preds = pd.read_csv(OUT / "cnn_test_predictions.csv", index_col=0)
    model = cnn.SmallCNN(len(CLASSES)).to(device)
    model.load_state_dict(torch.load(CACHE / "cnn_main.pt", map_location=device))

    # only maps the model got right are checked because the question is
    # whether a right answer came from the right place
    rng = np.random.default_rng(seed)
    rows, examples = [], {}
    for k, c in enumerate(CLASSES):
        if c == "none":
            continue
        right = preds.index[(preds["label"] == c) & (preds["pred"] == c)].to_numpy()
        pick = rng.choice(right, size=min(PER_CLASS, len(right)), replace=False)
        imgs = cached["images"][pos[pick].to_numpy()].astype(np.float32) / 255
        x = torch.from_numpy(imgs).to(device)
        cams = cnn.grad_cam(model, x, torch.full((len(pick),), k, device=device)).cpu().numpy()
        for i, idx in enumerate(pick):
            rows.append({"map": idx, "label": c, **measures(cams[i], imgs[i])})
        examples[c] = (imgs[0], cams[0])

    per_map = pd.DataFrame(rows)
    per_map.to_csv(OUT / "gradcam_per_map.csv", index=False)
    summary = per_map.groupby("label").agg(
        maps=("map", "size"),
        off_wafer_mass_median=("off_wafer_mass", "median"),
        off_wafer_area_median=("off_wafer_area", "median"),
        fail_lift_median=("fail_lift", "median"),
        share_with_lift_above_1=("fail_lift", lambda s: float((s > 1).mean())),
        failing_area_median=("failing_area", "median"),
        dense_fail_lift_median=("dense_fail_lift", "median"),
        share_with_dense_lift_above_1=("dense_fail_lift", lambda s: float((s > 1).mean())),
    )
    summary.to_csv(OUT / "gradcam_summary.csv")
    print(summary.round(3).to_string())

    # one random right answer per class  the wafer on top and the heat map under it
    wafer_cmap = ListedColormap([viz.SURFACE, PASS_COLOR, FAIL_COLOR])
    fig, axes = plt.subplots(2, len(DEFECTS), figsize=(14, 4.2))
    for j, c in enumerate(DEFECTS):
        img, cam = examples[c]
        code = np.where(img[0] < 0.5, 0, np.where(img[1] > 0.25, 2, 1))
        axes[0, j].imshow(code, cmap=wafer_cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[0, j].set_title(c, fontsize=10)
        # one hue light to dark because heat is a magnitude
        axes[1, j].imshow(cam, cmap="Oranges", interpolation="bilinear")
        axes[1, j].contour(img[0] >= 0.5, levels=[0.5], colors=[viz.TEXT_SECONDARY], linewidths=0.6)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("wafer map\nfails in blue", fontsize=9)
    axes[1, 0].set_ylabel("where the cnn looked\ndarker is more", fontsize=9)
    fig.suptitle("where the cnn looked  one random correct test wafer per pattern")
    fig.tight_layout()
    fig.savefig(FIG / "gradcam_examples.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
