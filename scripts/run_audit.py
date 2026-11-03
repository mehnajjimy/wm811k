"""runs every audit check and writes the answers to results
run from the repo root  python scripts/run_audit.py"""

from pathlib import Path

import pandas as pd

from wm811k import audit
from wm811k.data import load
from wm811k.yield_stats import die_yield

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load(ROOT / "LSWMD.pkl")

    audit.class_counts(df).to_csv(OUT / "audit_class_counts.csv", index=False)
    audit.size_summary(df).to_csv(OUT / "audit_map_sizes.csv", index=False)

    dups = audit.duplicates(df)
    dups.to_csv(OUT / "audit_duplicates.csv")

    # one flat table of single number facts so the readme can quote them
    facts: dict[str, float] = {"maps_total": len(df)}
    facts["maps_labeled"] = int(df["label"].notna().sum())
    facts.update(audit.lot_summary(df))
    facts.update(audit.lot_overlap_in_builtin_split(df))
    facts["distinct_map_sizes"] = int(df.groupby(["rows", "cols"]).ngroups)
    facts["duplicate_groups"] = len(dups)
    # extra copies beyond the first one in each group are what could leak
    facts["duplicate_extra_copies"] = int((dups["copies"] - 1).sum())
    facts["duplicate_groups_spanning_lots"] = int((dups["lots"] > 1).sum())
    facts["duplicate_groups_with_conflicting_labels"] = int((dups["labels"] > 1).sum())
    facts.update(audit.defect_duplicates(df))
    facts["pixel_values"] = " ".join(str(v) for v in sorted(audit.pixel_values(df)))

    # maps with no tested die would break yield so count them up front
    yields = df["wafer_map"].map(die_yield)
    facts["maps_with_no_tested_dies"] = int(yields.isna().sum())

    pd.Series(facts, name="value").rename_axis("fact").to_csv(OUT / "audit_facts.csv")
    print(pd.Series(facts).to_string())


if __name__ == "__main__":
    main()
