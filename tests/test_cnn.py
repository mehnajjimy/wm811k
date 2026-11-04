import numpy as np
import torch

from wm811k.cnn import SIZE, SmallCNN, class_weights, dihedral, to_image
from wm811k.simulator import WaferGrid, generate


def test_image_keeps_the_fail_share_of_the_map():
    # area averaging should keep the overall fail share about the same as the raw map
    m = generate("Center", WaferGrid(rows=49, cols=39), seed=3).wafer_map
    img = to_image(m)
    assert img.shape == (2, SIZE, SIZE)
    raw_share = (m == 2).sum() / (m > 0).sum()
    img_share = img[1].sum() / img[0].sum()
    assert abs(raw_share - img_share) < 0.01


def test_thin_scratch_survives_downsizing():
    # a one die wide line on a map bigger than 64 must still show up after resizing
    m = np.ones((130, 130), dtype=int)
    m[65, 10:120] = 2
    img = to_image(m)
    assert img[1].max() > 0.3
    assert (img[1] > 0).sum() >= 50


def test_fail_channel_never_exceeds_die_channel():
    m = generate("Edge-Ring", seed=1).wafer_map
    img = to_image(m)
    assert (img[1] <= img[0] + 1e-6).all()


def test_model_gives_one_score_per_class():
    model = SmallCNN(9).eval()
    out = model(torch.zeros(4, 2, SIZE, SIZE))
    assert out.shape == (4, 9)


def test_dihedral_only_moves_pixels():
    # rotations and flips must not change how many fails a map has
    x = torch.rand(16, 2, SIZE, SIZE)
    y = dihedral(x, torch.Generator().manual_seed(0))
    assert torch.allclose(x.sum(dim=(1, 2, 3)), y.sum(dim=(1, 2, 3)))
    assert not torch.equal(x, y)


def test_rare_classes_get_more_weight():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    full = class_weights(y, 3, "balanced")
    soft = class_weights(y, 3, "sqrt")
    assert full[2] > full[1] > full[0]
    # the square root keeps the order but shrinks the gap
    assert soft[2] / soft[0] < full[2] / full[0]
    assert soft[2] > soft[0]


def test_grad_cam_gives_one_normalised_map_per_image():
    from wm811k.cnn import grad_cam

    torch.manual_seed(0)
    model = SmallCNN(9)
    x = torch.rand(3, 2, SIZE, SIZE)
    cam = grad_cam(model, x, torch.tensor([0, 4, 8]))
    assert cam.shape == (3, SIZE, SIZE)
    assert (cam >= 0).all()
    sums = cam.sum(dim=(1, 2))
    # each map sums to 1 unless the model found no positive evidence at all
    assert torch.all(((sums - 1).abs() < 1e-4) | (sums == 0))
