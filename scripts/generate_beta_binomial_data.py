#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy.stats import beta, binom


def generate_beta_binomial_dataset(num_groups: int, attempts_low: int, attempts_high: int,
                                   alpha: float, beta_param: float, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    group_ids = [f"G{i+1}" for i in range(num_groups)]
    attempts = rng.integers(low=attempts_low, high=attempts_high + 1, size=num_groups)
    p = beta.rvs(alpha, beta_param, random_state=rng, size=num_groups)
    successes = np.array([binom.rvs(n=int(n_i), p=float(p_i), random_state=rng) for n_i, p_i in zip(attempts, p)])
    return pd.DataFrame({"group_id": group_ids, "attempts": attempts, "successes": successes})


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Beta–Binomial grouped data")
    parser.add_argument("--num_groups", type=int, default=100)
    parser.add_argument("--attempts_low", type=int, default=3)
    parser.add_argument("--attempts_high", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--beta_param", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/beta_binomial_synthetic.csv")
    args = parser.parse_args()

    df = generate_beta_binomial_dataset(args.num_groups, args.attempts_low, args.attempts_high,
                                        args.alpha, args.beta_param, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
