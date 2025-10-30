<h1 align="center">Beta–Binomial Dataset Information</h1>

# 1. Dataset Overview

This folder contains small educational datasets for Beta–Binomial classification. Each row represents grouped binary outcomes: number of attempts and number of successes for a group (e.g., learner, item, cohort).

# 2. Files

## 2.1 beta_binomial_examples.csv
- Columns: `group_id, attempts, successes`
- Purpose: Tiny hand-crafted examples used in docs and notebooks.

## 2.2 beta_binomial_synthetic.csv
- Columns: `group_id, attempts, successes`
- Purpose: A slightly larger synthetic sample for experimentation.

# 3. Schema

- group_id: string identifier of the group/entity
- attempts: total trials (non-negative integer)
- successes: number of successful trials (integer in [0, attempts])

# 4. Usage

You can load these datasets with pandas:
```python
import pandas as pd

examples = pd.read_csv('data/beta_binomial_examples.csv')
synthetic = pd.read_csv('data/beta_binomial_synthetic.csv')
```

# 5. Notes

- Data are illustrative and small by design for clarity.
- For larger experiments, see the script `scripts/generate_beta_binomial_data.py` (to be added) which can create customized synthetic datasets.
