"""small convolutional network for wafer maps
maps are turned into fixed size two channel images then a short stack of
conv layers learns the pattern straight from the pixels"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from wm811k import evaluate

SIZE = 64


def to_image(wafer_map: np.ndarray, size: int = SIZE) -> np.ndarray:
    """one wafer map as a 2 x size x size float array
    channel 0 is the share of each cell covered by dies and channel 1 is the
    share of each cell covered by failing dies"""
    # d40 maps come in 632 sizes so they have to be brought to one size
    # area averaging keeps a thin scratch visible as a faint line where
    # nearest neighbour resizing would drop most of its pixels on big maps
    m = torch.as_tensor(np.asarray(wafer_map), dtype=torch.float32)
    channels = torch.stack([(m > 0).float(), (m == 2).float()])
    return F.adaptive_avg_pool2d(channels, size).numpy()


def to_images(maps: pd.Series, size: int = SIZE) -> np.ndarray:
    """every map as uint8 images in one array
    values are stored as 0 to 255 so 170k maps fit in about 1.4 gb"""
    out = np.empty((len(maps), 2, size, size), dtype=np.uint8)
    for i, m in enumerate(maps):
        out[i] = np.round(to_image(m, size) * 255)
    return out


def _block(c_in: int, c_out: int) -> nn.Sequential:
    # two 3x3 convs see a 5x5 patch then pooling halves the map so each block
    # looks at a wider part of the wafer than the one before it
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """four conv blocks then global average pooling and one linear layer
    about 0.6 million weights which is small next to resnet style models"""

    def __init__(self, n_classes: int = 9, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(2, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 4 * w)
        )
        # global average pooling throws away where on the 4x4 grid a pattern
        # sat so the last layer reads what was found not where
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def dihedral(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """each image in the batch gets one of the 8 rotations and mirror flips
    d41 a pattern means the same thing at any angle so this is free extra data"""
    # the generator lives on the cpu so the same seed gives the same flips
    # on any device
    k = torch.randint(0, 8, (x.shape[0],), generator=gen).to(x.device)
    out = x.clone()
    for t in range(8):
        pick = k == t
        if not pick.any():
            continue
        y = torch.rot90(x[pick], t % 4, dims=(2, 3))
        if t >= 4:
            y = torch.flip(y, dims=(3,))
        out[pick] = y
    return out


def class_weights(y: np.ndarray, n_classes: int, scheme: str) -> torch.Tensor:
    """loss weight per class from how often each class shows up in training
    balanced is n over k times count like the forest and sqrt is its square root"""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * counts)
    # sqrt is a softer push toward rare classes  it is tried because full
    # balancing gives near full about 1000 times the weight of none
    if scheme == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_proba(
    model: nn.Module, images: np.ndarray, device: str, batch: int = 1024
) -> np.ndarray:
    """softmax class probabilities for every image"""
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i : i + batch]).to(device).float() / 255
        out.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    classes: list[str],
    seed: int,
    device: str,
    weights: str = "sqrt",
    max_epochs: int = 40,
    patience: int = 6,
    batch: int = 256,
    lr: float = 1e-3,
    log=print,
) -> tuple[nn.Module, pd.DataFrame]:
    """trains on train and keeps the epoch with the best validation macro f1
    gives back the best model and one row of history per epoch
    labels come in as integer codes into classes"""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n = len(classes)
    model = SmallCNN(n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # d41 the learning rate falls toward zero over the run on a cosine curve so
    # late epochs make small careful steps instead of bouncing around the minimum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_y, n, weights).to(device))
    val_true = pd.Series(np.asarray(classes)[val_y])

    best, best_state, since_best, rows = -1.0, None, 0, []
    # the training images sit on the gpu once as uint8 so each batch is a
    # lookup there instead of a copy from main memory
    x_all = torch.from_numpy(train_x).to(device)
    y_all = torch.from_numpy(train_y).to(device)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            x = x_all[idx].float() / 255
            x = dihedral(x, gen)
            y = y_all[idx]
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        # d42 validation picks the epoch so the test set is never looked at
        # while training
        pred = predict_proba(model, val_x, device).argmax(1)
        f1 = evaluate.macro_f1(val_true, pd.Series(np.asarray(classes)[pred]), classes)
        rows.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_macro_f1": f1})
        log(f"epoch {epoch + 1}  loss {total / len(order):.4f}  val macro f1 {f1:.4f}")
        if f1 > best:
            best, since_best = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            # stop once validation has not improved for a while because more
            # epochs would only fit the training maps more tightly
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def grad_cam(model: SmallCNN, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """heat map of where on the wafer the model looked for each image
    gives back one map per image at the input size scaled to sum to 1
    d45 grad cam weights the last conv layer by how much each channel pushes
    the score of the target class up"""
    model.eval()
    # the last relu of the last block sits before pooling so it still has an
    # 8x8 grid  after pooling it would be a 4x4 grid which is too coarse to read
    layer = model.features[3][5]
    saved = {}

    def keep(_module, _inp, out):
        out.retain_grad()
        saved["act"] = out

    handle = layer.register_forward_hook(keep)
    try:
        scores = model(x)
        model.zero_grad(set_to_none=True)
        scores.gather(1, target[:, None]).sum().backward()
    finally:
        handle.remove()
    act = saved["act"]
    weights = act.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    total = cam.sum(dim=(1, 2), keepdim=True)
    # a map with no positive evidence at all gets a flat zero map instead of nan
    return (cam / total.clamp_min(1e-12)).detach()
