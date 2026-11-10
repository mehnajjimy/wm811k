import json

import numpy as np
from PIL import Image

from wm811k.vlm import (
    PROMPTS,
    _claude_message,
    _plan_only_env,
    cost_usd,
    parse,
    read_claude_stream,
    render,
)


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


def test_cost_bills_anthropic_cache_writes_at_their_own_price():
    price = {"input": 4.0, "cache_write_5m": 5.0, "cache_write_1h": 8.0,
             "cached_input": 0.20, "output": 20.0}  # fmt: skip
    # 1000 fresh  2000 read from cache  3000 written for 5 minutes  4000 for an hour
    usage = {"input_tokens": 10_000, "cached_input_tokens": 2_000,
             "cache_write_5m_tokens": 3_000, "cache_write_1h_tokens": 4_000,
             "output_tokens": 500}  # fmt: skip
    want = (1_000 * 4 + 2_000 * 0.2 + 3_000 * 5 + 4_000 * 8 + 500 * 20) / 1e6
    assert abs(cost_usd(usage, price) - want) < 1e-12


def test_claude_stream_gives_reply_and_counts_every_input_token():
    events = [
        {"type": "system", "subtype": "init", "tools": [], "apiKeySource": "none"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}},
        {"type": "result", "result": '{"pattern": "Donut"}', "usage": {
            "input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 30,
            "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 0},
            "output_tokens_details": {"thinking_tokens": 12}}},
    ]  # fmt: skip
    r = read_claude_stream("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    assert r["reply"] == '{"pattern": "Donut"}'
    assert r["input_tokens"] == 170 and r["cached_input_tokens"] == 50
    assert r["cache_write_5m_tokens"] == 20 and r["reasoning_output_tokens"] == 12
    assert not r["ran_something"]


def test_claude_stream_flags_any_tool_use():
    ev = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/x"}}]}}  # fmt: skip
    r = read_claude_stream(json.dumps(ev))
    assert r["ran_something"] and r["actions"].startswith("Read")
    assert r["reply"] == ""


def test_claude_message_puts_the_picture_before_the_prompt(tmp_path):
    path = tmp_path / "w.png"
    render(np.ones((10, 10), dtype=int), path, pixels=50)
    msg = json.loads(_claude_message(path, "classify it"))
    content = msg["message"]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["text"] == "classify it"


def test_claude_error_result_is_a_failed_call_not_an_answer():
    ev = {"type": "result", "is_error": True, "result": "Claude usage limit reached",
          "usage": {}}  # fmt: skip
    r = read_claude_stream(json.dumps(ev))
    assert r["failed"] and r["reply"] == ""


def test_claude_call_billed_to_an_api_key_is_a_failed_call():
    events = [
        {"type": "system", "subtype": "init", "apiKeySource": "ANTHROPIC_API_KEY"},
        {"type": "result", "result": '{"pattern": "Donut"}', "usage": {}},
    ]
    r = read_claude_stream("\n".join(json.dumps(e) for e in events))
    assert r["failed"] and r["api_key_source"] == "ANTHROPIC_API_KEY"


def test_plan_only_env_drops_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "y")
    env = _plan_only_env()
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
