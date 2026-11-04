"""zero shot wafer map reading with a vision language model through codex exec
item e  the model sees a picture of the map and a prompt and never any labels"""

import json
import re
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from wm811k.data import CLASSES

CODEX = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"

# white page for no die  grey for pass  blue for fail  same idea as d34 but on
# pure white so the wafer edge is easy to see in a small picture
_CMAP = ListedColormap(["#ffffff", "#d3d1ca", "#2a78d6"])

_FORMAT = (
    "Do not run any commands or read any files. "
    'Reply with only a JSON object like {"pattern": "<one label>"}.'
)

_INTRO = (
    "This image is a semiconductor wafer bin map seen from above. "
    "White is outside the wafer, grey is a die that passed test and blue is a die that failed. "
)

# d46 raw gives only the label names so it measures what the model already knows
RAW = (
    _INTRO
    + "Classify the spatial pattern of the failing dies as exactly one of: "
    + ", ".join(CLASSES)
    + ". "
    + _FORMAT
)

# d46 rules gives a short definition of each label checked in a fixed order
# the order matters because a few labels overlap and the first match wins
RULES = (
    _INTRO
    + "Classify the spatial pattern of the failing dies using these rules, checked in order. "
    "Pick the first rule that matches.\n"
    "1. Near-full: most of the wafer, well over half the dies, has failed.\n"
    "2. Scratch: the fails form a thin line or arc, one or two dies wide, often long.\n"
    "3. Donut: the fails form a ring around the middle of the wafer with mostly passing dies "
    "at the very center.\n"
    "4. Center: the fails form one solid patch at the middle of the wafer.\n"
    "5. Edge-Ring: the fails run around most of the outer edge of the wafer.\n"
    "6. Edge-Loc: the fails form a patch on one part of the outer edge only.\n"
    "7. Loc: the fails form one solid patch away from the center and away from the edge.\n"
    "8. Random: the fails are scattered over the wafer with no shape, well above a few dies.\n"
    "9. none: only a few isolated fails or none at all.\n" + _FORMAT
)

PROMPTS = {"raw": RAW, "rules": RULES}


def render(wafer_map: np.ndarray, path: Path, pixels: int = 320) -> None:
    """saves the map as a square png with every die drawn as a flat block"""
    inch = pixels / 100
    fig = plt.figure(figsize=(inch, inch), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    # nearest keeps each die a sharp block so no colours get blended at the edges
    ax.imshow(wafer_map, cmap=_CMAP, vmin=0, vmax=2, interpolation="nearest")
    fig.savefig(path)
    plt.close(fig)


def parse(text: str) -> str:
    """pulls the label out of the reply or gives invalid
    a reply that does not name exactly one known label counts as wrong later"""
    lookup = {c.lower(): c for c in CLASSES}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            got = json.loads(m.group(0)).get("pattern", "")
            return lookup.get(str(got).strip().lower(), "invalid")
        except (json.JSONDecodeError, AttributeError):
            pass
    # longer labels are matched and cut out first so edge loc is not also read as loc
    found, rest = [], text
    for c in sorted(CLASSES, key=len, reverse=True):
        pattern = rf"\b{re.escape(c)}\b"
        if re.search(pattern, rest, flags=re.IGNORECASE):
            found.append(c)
            rest = re.sub(pattern, " ", rest, flags=re.IGNORECASE)
    return found[0] if len(found) == 1 else "invalid"


def cost_usd(usage: dict, price: dict) -> float:
    """dollars for one call at the published per million token prices
    cached input is billed at its own lower price and reasoning is billed as output"""
    cached = usage.get("cached_input_tokens", 0)
    fresh = usage.get("input_tokens", 0) - cached
    out = usage.get("output_tokens", 0)
    return (fresh * price["input"] + cached * price["cached_input"] + out * price["output"]) / 1e6


def _codex(args: list[str], prompt: str, model: str, cwd: Path, timeout: int) -> dict:
    """runs codex exec once and reads the reply  token use and seconds out of its event stream"""
    t0 = time.time()
    proc = subprocess.run(
        [CODEX, "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check", "--json",
         "-C", str(cwd), *args, "-"],
        input=prompt, capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip
    seconds = time.time() - t0
    text, usage, ran_something = "", {}, False
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = ev.get("item", {})
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text", "")
        elif item.get("type") not in (None, "agent_message", "reasoning"):
            # anything else such as a shell command means the model went looking around
            ran_something = True
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage", {})
    return {
        "reply": text,
        "seconds": seconds,
        "ran_something": ran_something,
        "returncode": proc.returncode,
        **{k: usage.get(k, 0) for k in
           ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")},
    }  # fmt: skip


def ask(image: Path, prompt: str, model: str, timeout: int = 300) -> dict:
    """one call to the model with one image
    gives back the raw reply  the parsed label  token use  seconds  and whether
    the model tried to run anything which would make the answer suspect"""
    r = _codex(["-i", str(image)], prompt, model, image.parent, timeout)
    return {**r, "pred": parse(r["reply"]) if r["reply"] else "invalid"}


def ask_text(prompt: str, model: str, cwd: Path, timeout: int = 300) -> dict:
    """one text only call  used by the triage assistant in item f"""
    return _codex([], prompt, model, cwd, timeout)
