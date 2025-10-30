#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


def posterior_params(alpha_prior: float, beta_prior: float, n: int, k: int):
    return alpha_prior + k, beta_prior + (n - k)


def ci_low(alpha_post: float, beta_post: float, conf: float) -> float:
    lower = (1 - conf) / 2
    return beta.ppf(lower, alpha_post, beta_post)


def make_mastery_grid(max_attempts: int, alpha_prior: float, beta_prior: float,
                      familiar: float, proficient: float, confidence: float):
    grid = np.empty((max_attempts + 1, max_attempts + 1), dtype=object)
    for n in range(0, max_attempts + 1):
        for k in range(0, n + 1):
            a_post, b_post = posterior_params(alpha_prior, beta_prior, n, k)
            low = ci_low(a_post, b_post, confidence)
            if low >= proficient:
                label = "Proficient"
            elif low >= familiar:
                label = "Familiar"
            else:
                label = "Attempted"
            grid[n, k] = label
    return grid


def plot_grid(grid, max_attempts: int, out_path: str):
    label_to_val = {"Attempted": 0, "Familiar": 1, "Proficient": 2}
    mat = np.full_like(grid, fill_value=-1, dtype=float)
    for n in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            if grid[n, k] is not None:
                mat[n, k] = label_to_val[grid[n, k]]
    cmap = plt.get_cmap('viridis', 3)
    plt.figure(figsize=(6, 6))
    plt.imshow(mat.T, origin='lower', extent=[0, max_attempts, 0, max_attempts], aspect='equal', cmap=cmap, vmin=0, vmax=2)
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Attempted", "Familiar", "Proficient"])  # type: ignore
    plt.xlabel("Attempts")
    plt.ylabel("Successes")
    plt.title("Beta–Binomial Mastery Map")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot attempts×successes mastery map for Beta–Binomial classifier")
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--alpha_prior", type=float, default=2.0)
    parser.add_argument("--beta_prior", type=float, default=2.0)
    parser.add_argument("--familiar", type=float, default=0.3)
    parser.add_argument("--proficient", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--out", type=str, default="figures/mastery_map_placeholder.png")
    args = parser.parse_args()

    grid = make_mastery_grid(args.max_attempts, args.alpha_prior, args.beta_prior,
                             args.familiar, args.proficient, args.confidence)
    plot_grid(grid, args.max_attempts, args.out)
    print(f"Saved mastery map to {args.out}")


if __name__ == "__main__":
    main()
