#!/usr/bin/env python3
import argparse
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
from scipy.stats import beta


@dataclass
class Thresholds:
    familiar: float
    proficient: float
    confidence: float


def posterior(alpha_prior: float, beta_prior: float, attempts: int, successes: int) -> Tuple[float, float, float]:
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + (attempts - successes)
    mean = alpha_post / (alpha_post + beta_post)
    return alpha_post, beta_post, mean


def credible_interval(alpha_post: float, beta_post: float, conf: float) -> Tuple[float, float]:
    lower = (1 - conf) / 2
    upper = 1 - lower
    return beta.ppf(lower, alpha_post, beta_post), beta.ppf(upper, alpha_post, beta_post)


def categorize(mean: float, ci_low: float, thresholds: Thresholds) -> str:
    if ci_low >= thresholds.proficient:
        return "Proficient"
    if ci_low >= thresholds.familiar:
        return "Familiar"
    return "Attempted"


def classify(df: pd.DataFrame, alpha_prior: float, beta_prior: float, thresholds: Thresholds) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        a_post, b_post, mean = posterior(alpha_prior, beta_prior, int(row.attempts), int(row.successes))
        ci_lo, ci_hi = credible_interval(a_post, b_post, thresholds.confidence)
        label = categorize(mean, ci_lo, thresholds)
        rows.append({
            "group_id": row.group_id,
            "attempts": int(row.attempts),
            "successes": int(row.successes),
            "post_alpha": a_post,
            "post_beta": b_post,
            "post_mean": mean,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "label": label,
        })
    return pd.DataFrame(rows)


def get_project_root() -> Path:
    """Find the project root by locating the scripts directory's parent."""
    script_path = Path(__file__).resolve()
    # If script is in scripts/, project root is parent
    if script_path.parent.name == "scripts":
        return script_path.parent.parent
    # Otherwise, assume we're already at project root
    return script_path.parent


def main():
    parser = argparse.ArgumentParser(description="Classify grouped outcomes with Beta–Binomial")
    parser.add_argument("--input", type=str, default="data/beta_binomial_examples.csv")
    parser.add_argument("--output", type=str, default="data/beta_binomial_classified.csv")
    parser.add_argument("--alpha_prior", type=float, default=2.0)
    parser.add_argument("--beta_prior", type=float, default=2.0)
    parser.add_argument("--familiar", type=float, default=0.3)
    parser.add_argument("--proficient", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.8)
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = get_project_root()
    input_path = project_root / args.input
    output_path = project_root / args.output

    thresholds = Thresholds(args.familiar, args.proficient, args.confidence)
    df = pd.read_csv(input_path)
    out = classify(df, args.alpha_prior, args.beta_prior, thresholds)
    out.to_csv(output_path, index=False)
    print(f"Wrote classifications to {output_path}")


if __name__ == "__main__":
    main()
