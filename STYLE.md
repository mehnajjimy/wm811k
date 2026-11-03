# Code style

## Layout
- `src/wm811k/` holds all the logic as a normal Python package.
- `notebooks/` holds thin, numbered notebooks (`01_audit.ipynb`, `02_yield.ipynb` …). They import from the package and draw figures, nothing else.
- `tests/` holds pytest checks.
- `results/` holds every number the README quotes, as CSV files written by the code.
- `config.yaml` holds paths, seeds and thresholds.

## Comment voice
Comments are teaching notes in Mehnaj's voice. They should read like a person's own lab notes explaining what the code does and why. They should not read like generated text.

- All lowercase, no punctuation.
- Plain and professional. Simple words.
- A short block before each chunk of code explains the idea. Shorter notes go on the lines a reader would stop at.
- Explain *why*, not what the line literally says.
- Decision tags use the same voice and point to DECISIONS.md, for example `# d03 split by lot so wafers from one lot never sit in both train and test`.
- Docstrings use the same voice and stay short: what it takes, what it gives back, why it exists.
- Banned: "let's", "we'll", "note that", "simply", "robust", "leverage", "seamlessly", emoji, cheerful filler, restating the code.

Example of the target density:

```python
def die_yield(wafer_map: np.ndarray) -> float:
    """fraction of tested dies that passed
    0 is no die 1 is pass 2 is fail so only 1s and 2s count"""
    # dies off the wafer edge are 0 and are not tested so they are left out
    passed = (wafer_map == 1).sum()
    failed = (wafer_map == 2).sum()
    # a map with no tested dies has no yield so return nan instead of crashing
    if passed + failed == 0:
        return float("nan")
    return passed / (passed + failed)
```

## Tooling
- Python 3.12 or newer, in a `.venv` virtual environment.
- Type hints on public functions.
- `ruff format` and `ruff check`. Rules about comment punctuation and capitalisation are turned off so the voice above passes lint.
- `pytest` covers the things that must never break: the yield formula, the lot-disjoint split, and no leakage into the test set.
- Fixed seeds everywhere, set in `config.yaml`.

## Prose (README, DECISIONS.md, RESEARCH.md)
- Normal sentences with normal punctuation.
- Run the `humanize-docs` skill over the README before it goes public.

## Figures
- Labelled axes with units.
- The same colour for each defect class across every figure.
- Titles in plain process language, without ML jargon.
