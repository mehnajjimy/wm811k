from pathlib import Path

from wm811k.assistant import check, load_table, prompt
from wm811k.data import CLASSES

TABLE = load_table(Path(__file__).resolve().parents[1] / "docs" / "root_causes.md")
FACTS = {
    "predicted_pattern": "Center",
    "model_confidence": 0.94,
    "die_yield_percent": 71.3,
    "wafers_in_lot_with_same_pattern": "7 of 24",
}


def _reply(**over):
    base = {
        "pattern": "Center",
        "modules": ["CMP", "deposition"],
        "first_check": "post-CMP film thickness map for the affected lots",
        "confirm_with": "center thickness outside the spec window",
        "sources": ["[S5]"],
        "d2_problem": "center fails with die yield 71.3 percent on 7 of 24 wafers in the lot",
        "d3_containment": "hold the lot",
        "d4_hypothesis": "radial CMP removal profile off target at the center",
        "unknowns": "which tool ran the lot",
    }
    return base | over


def test_table_has_one_row_per_class_with_its_tags():
    assert set(TABLE) == set(CLASSES)
    assert "S5" in TABLE["Center"]["source_tags"]
    # rows marked as engineering hypothesis only carry no tags
    assert TABLE["Scratch"]["source_tags"] == []


def test_a_grounded_reply_passes_every_check():
    out = check(_reply(), FACTS, TABLE["Center"])
    assert all(out[k] for k in out if k not in ("detail",))


def test_made_up_source_and_citation_are_caught():
    assert not check(_reply(sources=["[S9]"]), FACTS, TABLE["Center"])["no_invented_source"]
    bad = _reply(d4_hypothesis="as shown by Smith et al 2019")
    assert not check(bad, FACTS, TABLE["Center"])["no_invented_source"]


def test_made_up_number_and_tool_are_caught():
    bad = _reply(d2_problem="yield dropped 12.5 percent on ETCH03")
    out = check(bad, FACTS, TABLE["Center"])
    assert not out["no_invented_number"]
    assert not out["no_invented_tool"]
    # step names like d2 and chemistry like sio2 are not flagged
    ok = _reply(d3_containment="D3 hold the lot and check the SiO2 film")
    assert check(ok, FACTS, TABLE["Center"])["no_invented_number"]
    assert check(ok, FACTS, TABLE["Center"])["no_invented_tool"]


def test_module_outside_the_row_is_caught():
    bad = _reply(modules=["implant"])
    assert not check(bad, FACTS, TABLE["Center"])["no_unsupported_module"]
    bad_text = _reply(d4_hypothesis="an implant dose error")
    assert not check(bad_text, FACTS, TABLE["Center"])["no_unsupported_module"]


def test_ungrounded_prompt_never_shows_the_table():
    assert "TABLE ROW" not in prompt(FACTS, None)
    assert TABLE["Center"]["first_check"] in prompt(FACTS, TABLE["Center"])


def test_pattern_with_extra_words_and_simple_gaps_still_pass():
    facts = FACTS | {"clean_wafer_median_yield_percent_same_product": 81.2}
    ok = _reply(
        pattern="Center (model-predicted)", d2_problem="yield 71.3 is 9.9 points below 81.2"
    )
    out = check(ok, facts, TABLE["Center"])
    assert out["right_pattern"] and out["no_invented_number"]
    # naming two classes is not an answer
    assert not check(_reply(pattern="Center or Donut"), facts, TABLE["Center"])["right_pattern"]


def test_no_reference_statements_are_not_citations_but_urls_are():
    ok = _reply(sources=["no external references used", "[S5]"])
    assert check(ok, FACTS, TABLE["Center"])["no_invented_source"]
    bad = _reply(sources=["https://example.org/wafer-paper"])
    assert not check(bad, FACTS, TABLE["Center"])["no_invented_source"]
    # the row itself says ezzat et al so quoting that is fine
    quoted = _reply(d4_hypothesis="Ezzat et al. [E] link center patterns to stains")
    assert check(quoted, FACTS, TABLE["Center"])["no_invented_source"]
