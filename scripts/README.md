<h1 align="center">Beta–Binomial Scripts Documentation</h1>

# 1. Overview

This directory contains simple scripts to generate grouped count data, classify with a Beta–Binomial model, and plot a mastery map. They are designed for educational clarity and small datasets.

# 2. Scripts

## 2.1 generate_beta_binomial_data.py
- Generates synthetic grouped binary outcomes.
- Output CSV columns: `group_id, attempts, successes`.
- Example:
```bash
python scripts/generate_beta_binomial_data.py --num_groups 200 --alpha 3 --beta_param 4 --out data/beta_binomial_synthetic.csv
```

## 2.2 beta_binomial_classifier.py
- Computes posterior mean and credible interval per row and assigns categories.
- Categories (default): Attempted, Familiar, Proficient via lower-credible-bound thresholds.
- Example:
```bash
python scripts/beta_binomial_classifier.py --input data/beta_binomial_examples.csv --output data/beta_binomial_classified.csv --alpha_prior 2 --beta_prior 2 --familiar 0.3 --proficient 0.5 --confidence 0.8
```

## 2.3 plot_mastery_map.py
- Plots attempts×successes mastery regions given prior and thresholds.
- Saves a PNG into `figures/`.
- Example:
```bash
python scripts/plot_mastery_map.py --max_attempts 20 --alpha_prior 2 --beta_prior 2 --familiar 0.3 --proficient 0.5 --confidence 0.8 --out figures/mastery_map.png
```

# 3. Notes

- Keep datasets small for quick iteration.
- For probabilistic sampling (e.g., full Bayesian posterior draws), consider PyMC; not required here.
