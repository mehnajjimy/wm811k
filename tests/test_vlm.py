import numpy as np
from PIL import Image

from wm811k.vlm import PROMPTS, cost_usd, parse, render


def test_parse_reads_the_json_reply():
    assert parse('{"pattern": "Edge-Loc"}') == "Edge-Loc"
    assert parse('sure\n{"pattern": "edge-ring"}') == "Edge-Ring"


def test_parse_marks_unknown_or_unclear_replies_invalid():
    assert parse('{"pattern": "Ring"}') == "invalid"
    # two labels named in plain text is not an answer
    assert parse("could be Center or Donut") == "invalid"
    assert parse("Scratch") == "Scratch"
    # edge loc in plain text must not also count as loc
    assert parse("this looks like Edge-Loc") == "Edge-Loc"


def test_cost_bills_cached_input_at_the_lower_price():
    price = {"input": 0.10, "cached_input": 0.01, "output": 0.50}
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 500_000, "output_tokens": 100_000}
    assert abs(cost_usd(usage, price) - (0.05 + 0.005 + 0.05)) < 1e-9


def test_prompts_name_every_class_and_never_hint_a_label():
    for p in PROMPTS.values():
        for c in ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random",
                  "Scratch", "none"]:  # fmt: skip
            assert c in p


def test_render_draws_fails_in_blue(tmp_path):
    m = np.ones((20, 20), dtype=int)
    m[:, :2] = 0
    m[8:12, 8:12] = 2
    path = tmp_path / "w.png"
    render(m, path, pixels=200)
    img = np.asarray(Image.open(path).convert("RGB"))
    assert img.shape[:2] == (200, 200)
    blue = (img[..., 2] > 180) & (img[..., 0] < 80)
    # the 4x4 fail block is 16 of 400 dies so about 4 percent of the picture
    assert 0.02 < blue.mean() < 0.06
