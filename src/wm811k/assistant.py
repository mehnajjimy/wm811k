"""item f  a triage assistant that may only use the root cause table
the model gets the facts for one wafer and the table and drafts the first part
of an 8d response  every claim in its reply is then checked by code against
what it was given so a made up module  source  number or tool counts as a
hallucination without anyone having to judge it by eye"""

import json
import re
from pathlib import Path

TABLE_HEADER = "| Pattern | Likely process modules |"

# process words checked in the free text  a word that is not anywhere in the
# table row the model was pointed at is a claim the table does not support
MODULE_WORDS = [
    "CMP", "deposition", "CVD", "PVD", "ALD", "etch", "litho", "lithography", "photo",
    "implant", "thermal", "RTP", "anneal", "handling", "diffusion", "oxidation",
    "plating", "epi", "wet clean", "probe card", "tester",
]  # fmt: skip

SOURCE_TAG = re.compile(r"\[(S\d+|E|L\d+)\]")

# named tools or chambers such as etch03  chamber 2  tool b7  none exist in the
# input so any of these in a reply was invented
TOOL_ID = re.compile(
    r"\b(?:tool|chamber|etcher|polisher|furnace|reactor|module)\s*[#-]?\s*[A-Z]?\d+\b",
    flags=re.IGNORECASE,
)
# tool style names in capitals like ETCH03 or CMP-2  kept case sensitive so
# chemistry like SiO2 is not caught
TOOL_NAME = re.compile(r"\b[A-Z]{3,}[-_]?\d{1,3}\b")

# citation shaped text such as smith et al  2019 or a doi
CITATION = re.compile(r"\bet al\b|\b(?:19|20)\d{2}\b|\bdoi\b|10\.\d{4,}/", flags=re.IGNORECASE)

NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


def load_table(path: Path) -> dict[str, dict]:
    """the root cause table from docs root_causes md as one dict per pattern
    the markdown file stays the single source so the doc and the check never drift"""
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(TABLE_HEADER))
    keys = ["pattern", "modules", "mechanism", "first_check", "confirm_with", "source"]
    rows = {}
    for ln in lines[start + 2 :]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        row = dict(zip(keys, cells, strict=True))
        row["source_tags"] = sorted(set(SOURCE_TAG.findall(row["source"])))
        rows[row["pattern"]] = row
    return rows


GROUNDED = """You are drafting a first response for a wafer yield excursion, in the style of
8D steps D2 (problem description), D3 (containment) and D4 (root cause hypothesis).

Use ONLY the FACTS and the TABLE ROW below. Rules:
- Name only process modules, mechanisms, checks and signals that appear in the TABLE ROW.
- Cite only source tags that appear in the TABLE ROW's source cell, written like [S5]. If it
  has no tags, cite nothing and say the row is an engineering hypothesis.
- Do not write any number that is not in FACTS. Do not name specific tools, chambers or people.
- If something an engineer would want is not given, put it under "unknowns".

FACTS
{facts}

TABLE ROW
{row}

Reply with only a JSON object with these keys:
"pattern", "modules" (list), "first_check", "confirm_with", "sources" (list of tags),
"d2_problem", "d3_containment", "d4_hypothesis", "unknowns".
Do not run any commands or read any files."""

UNGROUNDED = """You are drafting a first response for a wafer yield excursion, in the style of
8D steps D2 (problem description), D3 (containment) and D4 (root cause hypothesis).
Use your own process engineering knowledge.

FACTS
{facts}

Reply with only a JSON object with these keys:
"pattern", "modules" (list), "first_check", "confirm_with", "sources" (list of references),
"d2_problem", "d3_containment", "d4_hypothesis", "unknowns".
Do not run any commands or read any files."""


def prompt(facts: dict, row: dict | None) -> str:
    """the grounded prompt when a row is given and the ungrounded control when not"""
    f = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    if row is None:
        return UNGROUNDED.format(facts=f)
    r = "\n".join(f"- {k}: {row[k]}" for k in
                  ("pattern", "modules", "mechanism", "first_check", "confirm_with", "source"))  # fmt: skip
    return GROUNDED.format(facts=f, row=r)


def parse_reply(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def _free_text(reply: dict) -> str:
    keys = ("first_check", "confirm_with", "d2_problem", "d3_containment", "d4_hypothesis")
    return " ".join(str(reply.get(k, "")) for k in keys)


def _allowed_numbers(facts: dict, row: dict) -> set[float]:
    text = " ".join(str(v) for v in [*facts.values(), *row.values()])
    nums = {float(n) for n in NUMBER.findall(text)}
    # a share given as 0.94 may be written as 94 percent and the other way round
    nums |= {round(n * 100, 1) for n in nums if n < 1} | {round(n / 100, 3) for n in nums}
    return nums


def check(reply: dict | None, facts: dict, row: dict) -> dict:
    """every check is true when the reply passes it
    row is always the table row for the facts so the ungrounded control is
    checked against the same table it never saw"""
    if reply is None:
        return {"parsed": False}
    text = _free_text(reply)
    row_text = " ".join(str(v) for v in row.values()).lower()

    # modules listed or named in the text that the row never mentions
    listed = [str(m) for m in reply.get("modules", []) or []]
    bad_listed = [m for m in listed if m.lower() not in row_text]
    bad_words = [
        w for w in MODULE_WORDS
        if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE)
        and w.lower() not in row_text
    ]  # fmt: skip

    # sources must be tags from the row  anything else is a made up citation
    cited = [str(s) for s in reply.get("sources", []) or []]
    tags = {t.strip("[]") for t in cited}
    bad_sources = sorted(tags - set(row["source_tags"]))
    bad_citation_text = CITATION.findall(text)

    allowed = _allowed_numbers(facts, row)
    # d2 d3 d4 and 8d are step names not quantities
    clean = re.sub(r"\b(?:8D|D[1-8])\b", " ", text)
    bad_numbers = [n for n in NUMBER.findall(clean) if float(n) not in allowed]
    tools = [m.group(0) for r in (TOOL_ID, TOOL_NAME) for m in r.finditer(text)]

    return {
        "parsed": True,
        "right_pattern": str(reply.get("pattern", "")).strip() == facts["predicted_pattern"],
        "no_unsupported_module": not bad_listed and not bad_words,
        "no_invented_source": not bad_sources and not bad_citation_text,
        "no_invented_number": not bad_numbers,
        "no_invented_tool": not tools,
        "detail": json.dumps(
            {
                "modules": bad_listed + bad_words,
                "sources": bad_sources + bad_citation_text,
                "numbers": bad_numbers,
                "tools": tools,
            }
        ),
    }


CHECKS = [
    "right_pattern",
    "no_unsupported_module",
    "no_invented_source",
    "no_invented_number",
    "no_invented_tool",
]
