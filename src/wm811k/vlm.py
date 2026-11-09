"""zero shot wafer map reading with a vision language model through codex exec
or through the claude code cli for claude opus
item e  the model sees a picture of the map and a prompt and never any labels"""

import base64
import json
import re
import shutil
import subprocess
import tempfile
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

# d55 rules v2 fixes the none and random rules in v1 which said none means
# only a few fails  on training lots a clean wafer has a median of about 10
# percent failing dies and random wafers start around a third so the rules now
# say that scattered fails alone are normal
RULES_V2 = (
    _INTRO
    + "Classify the spatial pattern of the failing dies using these rules, checked in order. "
    "Pick the first rule that matches. Scattered single failing dies are normal on every "
    "wafer, and a healthy wafer often has up to about one die in six failing, so look for a "
    "shape, not just for fails.\n"
    "1. Near-full: most of the wafer, about three quarters of the dies or more, has failed.\n"
    "2. Scratch: some fails line up into a thin line or arc, one or two dies wide.\n"
    "3. Donut: the fails form a ring around the middle of the wafer with mostly passing dies "
    "at the very center.\n"
    "4. Center: the fails form one solid patch at the middle of the wafer.\n"
    "5. Edge-Ring: the fails run around most of the outer edge of the wafer.\n"
    "6. Edge-Loc: the fails form a patch on one part of the outer edge only.\n"
    "7. Loc: the fails form one solid patch away from the center and away from the edge.\n"
    "8. Random: fails are spread over the whole wafer with no shape and are dense, roughly a "
    "third of the dies or more.\n"
    "9. none: no shape at all, only scattered fails at the normal background level.\n" + _FORMAT
)

PROMPTS = {"raw": RAW, "rules": RULES, "rules_v2": RULES_V2}


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
    cached input is billed at its own lower price and reasoning is billed as output
    anthropic also bills writing to the cache so those tokens get their own price
    input_tokens always counts every input token  fresh  cached and written"""
    cached = usage.get("cached_input_tokens", 0)
    w5 = usage.get("cache_write_5m_tokens", 0)
    w1h = usage.get("cache_write_1h_tokens", 0)
    fresh = usage.get("input_tokens", 0) - cached - w5 - w1h
    out = usage.get("output_tokens", 0)
    usd = fresh * price["input"] + cached * price["cached_input"] + out * price["output"]
    usd += w5 * price.get("cache_write_5m", 0) + w1h * price.get("cache_write_1h", 0)
    return usd / 1e6


def _codex(args: list[str], prompt: str, model: str, cwd: Path, timeout: int) -> dict:
    """runs codex exec once and reads the reply  token use and seconds out of its event stream"""
    t0 = time.time()
    proc = subprocess.run(
        [CODEX, "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check", "--json",
         "-C", str(cwd), *args, "-"],
        input=prompt, capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip
    seconds = time.time() - t0
    text, usage, ran_something, actions = "", {}, False, []
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
            # what it ran is kept so a person can check it only looked at its own picture
            ran_something = True
            if ev.get("type") == "item.completed":
                actions.append(str(item.get("command", item.get("type")))[:300])
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage", {})
    return {
        "reply": text,
        "seconds": seconds,
        "ran_something": ran_something,
        "actions": " || ".join(actions),
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


# d65 the same test through claude opus  run with the claude code cli that comes
# with the claude desktop app so it uses the subscription and no paid api key
_CLAUDE_GLOB = "Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude"

# replaces the long claude code system prompt so the call is the picture and the
# prompt and nothing about coding  and so the input tokens stay small
CLAUDE_SYSTEM = "You read semiconductor wafer bin map images and answer in the format asked."


def claude_cli() -> str:
    """path to the claude code cli  the one on the path first  else the newest copy
    the desktop app keeps in application support"""
    found = shutil.which("claude")
    if found:
        return found
    copies = sorted(Path.home().glob(_CLAUDE_GLOB), key=lambda p: p.stat().st_mtime)
    if not copies:
        raise FileNotFoundError("no claude code cli found")
    return str(copies[-1])


def _claude_message(image: Path, prompt: str) -> str:
    """one user turn with the picture first and the prompt after it  as stream json
    the picture goes inside the message so the model never needs a tool to see it"""
    data = base64.standard_b64encode(image.read_bytes()).decode()
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        {"type": "text", "text": prompt},
    ]
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def read_claude_stream(stdout: str) -> dict:
    """the reply  token use and any tool calls out of the cli stream json output
    the input count is rebuilt to hold every input token like the codex numbers do"""
    # a stream with no result event at all is a failed call  for example a crash
    text, usage, actions, failed = "", {}, [], True
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                # tools are switched off so any tool use here means something leaked
                if block.get("type") == "tool_use":
                    actions.append(f"{block.get('name')} {json.dumps(block.get('input'))}"[:300])
        if ev.get("type") == "result":
            # a usage limit or a sign in problem comes back as an error result
            # its text is not an answer so it is never parsed as one
            failed = bool(ev.get("is_error"))
            text = "" if failed else ev.get("result") or ""
            usage = ev.get("usage") or {}
    writes = usage.get("cache_creation") or {}
    w5 = writes.get("ephemeral_5m_input_tokens", 0)
    w1h = writes.get("ephemeral_1h_input_tokens", 0)
    # older usage blocks only give the total written so count it all as the 5 minute kind
    if not writes:
        w5 = usage.get("cache_creation_input_tokens", 0)
    cached = usage.get("cache_read_input_tokens", 0)
    return {
        "reply": text,
        "failed": failed,
        "ran_something": bool(actions),
        "actions": " || ".join(actions),
        "input_tokens": usage.get("input_tokens", 0) + cached + w5 + w1h,
        "cached_input_tokens": cached,
        "cache_write_5m_tokens": w5,
        "cache_write_1h_tokens": w1h,
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_output_tokens": (usage.get("output_tokens_details") or {}).get(
            "thinking_tokens", 0
        ),
    }


def ask_claude(image: Path, prompt: str, model: str, effort: str, timeout: int = 300) -> dict:
    """one call to claude with one image  same fields as ask so both score the same way
    it runs in an empty temporary folder with every tool and setting switched off so it
    cannot read the repo  the labels or anything else on disk"""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [claude_cli(), "-p", "--model", model, "--effort", effort, "--tools", "",
                 "--restricted", "--strict-mcp-config", "--no-session-persistence",
                 "--disable-slash-commands", "--system-prompt", CLAUDE_SYSTEM,
                 "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"],
                input=_claude_message(image, prompt), capture_output=True, text=True,
                timeout=timeout, check=False, cwd=tmp,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            # a hung call is a failed call and is asked again on the next run
            return {"reply": "", "failed": True, "pred": "invalid", "seconds": time.time() - t0}
    r = read_claude_stream(proc.stdout)
    r["failed"] = r["failed"] or proc.returncode != 0
    pred = parse(r["reply"]) if r["reply"] else "invalid"
    return {**r, "seconds": time.time() - t0, "returncode": proc.returncode, "pred": pred}
