"""loading and cleaning the wm811k pickle"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


class _OldPandasUnpickler(pickle.Unpickler):
    """unpickler that points old pandas module paths at where they live now"""

    # the pickle was written in 2019 with a pandas that kept index classes
    # under pandas.indexes  newer pandas moved them to pandas.core.indexes
    # so any lookup under the old path gets redirected before it fails
    def find_class(self, module: str, name: str):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        return super().find_class(module, name)


def _unwrap(cell) -> str | None:
    """turns the nested label cells into a plain string or none"""
    # labels are stored as tiny 2d arrays like [['Center']] and unlabeled
    # maps hold an empty array so dig out the string if there is one
    arr = np.asarray(cell)
    if arr.size == 0:
        return None
    return str(arr.ravel()[0])


def load_raw(path: Path) -> pd.DataFrame:
    """reads the original pickle exactly as it was saved"""
    # the file was pickled under python 2 so numpy byte buffers have to be
    # read as latin1 or the decoder chokes on the first non ascii byte
    with open(path, "rb") as f:
        return _OldPandasUnpickler(f, encoding="latin1").load()


def load(path: Path) -> pd.DataFrame:
    """reads the pickle and gives back one tidy row per wafer
    columns  wafer_map  lot  wafer_index  die_size  label  split  rows  cols"""
    raw = load_raw(path)

    df = pd.DataFrame(
        {
            "wafer_map": raw["waferMap"],
            "lot": raw["lotName"].astype(str),
            # wafer index comes in as a float but it is a slot number in the lot
            "wafer_index": raw["waferIndex"].astype(int),
            "die_size": raw["dieSize"].astype(float),
            "label": raw["failureType"].map(_unwrap),
            # the original column name is misspelled in the source data
            "split": raw["trianTestLabel"].map(_unwrap),
        }
    )

    # map height and width are kept so size effects can be checked later
    shapes = df["wafer_map"].map(np.shape)
    df["rows"] = shapes.map(lambda s: s[0])
    df["cols"] = shapes.map(lambda s: s[1])
    return df
