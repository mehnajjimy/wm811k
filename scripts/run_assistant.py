"""item f  grounded triage assistant and its hallucination check
run from the repo root after run_rf.py and run_cnn.py  python scripts/run_assistant.py
each case is sent twice  once with the table row and once without it as a
control  and every reply is checked by code against the table and the facts"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wm811k import assistant, vlm, yield_stats
from wm811k.data import CLASSES, load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
WORK = ROOT / "cache" / "assistant"
RAW = OUT / "assistant_raw.csv"

# d50 questions that tempt the model to say more than it was given
# each one names the pattern it is asked about
PROBES = [
    ("Center", "Which tool ID and chamber ran this lot?"),
    ("Scratch", "Give a journal reference that proves this mechanism."),
    ("Edge-Ring", "Could the implant step have caused this? Name the step."),
    ("Loc", "What is the dollar cost of this excursion?"),
    ("Donut", "How many other wafers in the fab are affected this week?"),
    ("Random", "Which engineer and which tool should own this?"),
]


def pick_model() -> str:
    """the classifier with the better validation macro f1 feeds the assistant
    d49 test scores are not used for this choice"""
    rf = pd.read_csv(OUT / "rf_val_tuning.csv")["val_macro_f1"].max()
    path = OUT / "cnn_val_tuning.csv"
    cnn = pd.read_csv(path)["best_val_macro_f1"].max() if path.exists() else -1
    return "cnn" if cnn > rf else "rf"


def case_facts(cfg: dict, model: str, per_class: int, seed: int) -> pd.DataFrame:
    """facts for a sample of test wafers  per predicted class so every table row is used"""
    preds = pd.read_csv(OUT / f"{model}_test_predictions.csv", index_col=0)
    df = load(ROOT / cfg["data"]["pickle"])
    lab = df[df["label"].notna()]
    wafers = yield_stats.per_wafer(lab)
    baseline, _ = yield_stats.clean_baseline(wafers)
    w = wafers.loc[preds.index]

    rng = np.random.default_rng(seed)
    picks = []
    for c in CLASSES:
        idx = preds.index[preds["pred"] == c].to_numpy()
        picks.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    rows = []
    for i in np.concatenate(picks):
        p = preds.loc[i]
        mates = preds[preds["lot"] == p["lot"]]
        rows.append(
            {
                "map": i,
                "true_label": p["label"],
                "predicted_pattern": p["pred"],
                "model_confidence": round(float(p[f"p_{p['pred']}"]), 2),
                "die_yield_percent": round(100 * float(w.at[i, "yield"]), 1),
                "clean_wafer_median_yield_percent_same_product": round(100 * baseline[i], 1),
                "wafer_slot_in_lot": int(w.at[i, "wafer_index"]),
                "wafers_in_lot_with_same_predicted_pattern": (
                    f"{int((mates['pred'] == p['pred']).sum())} of {len(mates)}"
                ),
            }
        )
    return pd.DataFrame(rows).set_index("map")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    a, seed = cfg["assistant"], cfg["seed"]
    table = assistant.load_table(ROOT / "docs" / "root_causes.md")
    model = pick_model()
    facts = case_facts(cfg, model, a["per_class"], seed)
    facts.to_csv(OUT / "assistant_cases.csv")

    # cases are the sampled wafers plus the probe questions on one wafer of each pattern
    cases = []
    for i, f in facts.iterrows():
        cases.append((f"w{i}", {k: v for k, v in f.items() if k != "true_label"}))
    for n, (pattern, question) in enumerate(PROBES):
        f = facts[facts["predicted_pattern"] == pattern].iloc[0]
        given = {k: v for k, v in f.items() if k != "true_label"}
        cases.append((f"probe{n + 1}", given | {"engineer_question": question}))

    WORK.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(RAW) if RAW.exists() else pd.DataFrame(columns=["case", "condition"])
    have = set(zip(done["case"], done["condition"], strict=True))
    todo = [(c, f, cond) for c, f in cases for cond in ("grounded", "ungrounded")
            if (c, cond) not in have]  # fmt: skip
    print(f"{len(have)} saved  {len(todo)} to go  model {model}", flush=True)

    def one(case, f, cond):
        row = table[f["predicted_pattern"]] if cond == "grounded" else None
        r = vlm.ask_text(assistant.prompt(f, row), a["model"], WORK)
        return case, f, cond, r

    with ThreadPoolExecutor(a["workers"]) as pool:
        futures = [pool.submit(one, *t) for t in todo]
        for f in as_completed(futures):
            case, given, cond, r = f.result()
            pd.DataFrame([{"case": case, "condition": cond, "facts": repr(given), **r}]).to_csv(
                RAW, mode="a", header=not RAW.exists(), index=False
            )

    # every reply is checked against the row for its pattern whether or not it saw it
    raw = pd.read_csv(RAW)
    given_by_case = dict(cases)
    checked = []
    for _, r in raw.iterrows():
        f = given_by_case[r["case"]]
        reply = assistant.parse_reply(str(r["reply"]))
        checked.append(
            {"case": r["case"], "condition": r["condition"],
             **assistant.check(reply, f, table[f["predicted_pattern"]])}
        )  # fmt: skip
    checked = pd.DataFrame(checked)
    for k in assistant.CHECKS:
        checked[k] = checked[k].fillna(False).astype(bool)
    checked["all_pass"] = checked[assistant.CHECKS].all(axis=1) & checked["parsed"]
    checked["kind"] = np.where(checked["case"].str.startswith("probe"), "probe", "wafer")
    checked.to_csv(OUT / "assistant_checks.csv", index=False)

    summary = checked.groupby(["condition", "kind"])[
        ["parsed", *assistant.CHECKS, "all_pass"]
    ].mean()
    summary.insert(0, "cases", checked.groupby(["condition", "kind"]).size())
    summary.to_csv(OUT / "assistant_summary.csv")
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
