# GBERT

A bert based model for political analysis.

## Layout

- `src/model/`: Python package (preprocess, dataset, model, utils)
- `scripts/`: runnable entrypoints
- `notebooks/`: the original notebook snapshot

## Quickstart

1) Create an environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Put your raw data CSV at the repo root:

- `manifesto_sentences.csv`

The preprocessing expects at least these columns:
- `sentence`, `category`, `country`, `year`
- macro columns: `battle_value`, `gdp_value`, `trade_value`, `debt_value`, `inflation_value`, `unemployment_value`

3) Preprocess to produce `train.csv`, `val.csv`, `preprocess_meta.joblib`

```bash
python scripts/preprocess.py
```

4) Train

```bash
python scripts/train.py --epochs 5
```

For a tiny debug run:

```bash
python scripts/train.py --debug_n 256 --epochs 1 --batch_size 16 --val_batch_size 16
```

## Notes

- This repo keeps your original model decomposition (text + macro additive NAM + country/year main effects)
  and includes the interaction + FiLM blocks already present in your notebook model.
