"""
Compare multiple training runs (methods/baselines) from wandb CSV exports.

Examples
--------
# Compare 4 methods, EMA smoothing (half-life=20)
python compare_runs.py \
    --runs "FAIR=1:fair1.csv" "FAIR=0:fair0.csv" "Greedy:greedy.csv" "PPO:ppo.csv" \
    --metric mean_reward --half-life 20 --save compare.png

# A metric where LOWER is better (price / latency)
python compare_runs.py \
    --runs "FAIR=1:fair1.csv" "FAIR=0:fair0.csv" \
    --metric price --lower-better --half-life 20 --save price.png

# Seed aggregation: repeat the SAME label for multiple files -> mean ± std band
python compare_runs.py \
    --runs "PPO:ppo_s0.csv" "PPO:ppo_s1.csv" "PPO:ppo_s2.csv" "Greedy:greedy.csv" \
    --metric mean_reward --band std --save compare.png

You can also edit the RUNS block below and just run `python compare_runs.py`.
"""

import argparse
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Edit this if you don't want to pass --runs on the CLI.
# label -> path  (repeat a label across entries to aggregate seeds)
# ============================================================
RUNS = [
    # ("FAIR=1", "fair1.csv"),
    # ("FAIR=0", "fair0.csv"),
    # ("Greedy", "greedy.csv"),
    # ("PPO",    "ppo.csv"),
]

# Qualitative palette (distinct up to ~10 lines)
PALETTE = [
    "#2a9d8f",  # teal
    "#c03070",  # magenta
    "#e9a23b",  # amber
    "#3a78b8",  # steel blue
    "#6a4c93",  # purple
    "#e76f51",  # coral
    "#264653",  # dark navy
    "#588157",  # green
    "#9b2226",  # dark red
    "#7f7f7f",  # gray
]


# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------
def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def ema(x: np.ndarray, half_life: float, debias: bool = True) -> np.ndarray:
    """Time-weighted EMA parameterized by half-life (TensorBoard/wandb-style, debiased)."""
    x = np.asarray(x, dtype=np.float64)
    if half_life is None or half_life <= 0 or len(x) == 0:
        return x
    alpha = 1.0 - 2.0 ** (-1.0 / float(half_life))
    weight = 1.0 - alpha
    out = np.zeros_like(x)
    s = 0.0
    db = 0.0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = out[i - 1] if i > 0 else np.nan
            continue
        s = weight * s + alpha * v
        if debias:
            db = weight * db + alpha
            out[i] = s / max(db, 1e-12)
        else:
            out[i] = s
    return out


def smooth(x: np.ndarray, method: str, window: int, half_life: float) -> np.ndarray:
    method = method.lower()
    if method == "ema":
        return ema(x, half_life=half_life, debias=True)
    if method in ("rolling", "mean"):
        return rolling_mean(x, window)
    if method == "none":
        return x
    raise ValueError(f"Unknown smoothing method: {method}")


# ------------------------------------------------------------
# Loading / metric column detection
# ------------------------------------------------------------
def find_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    """Find the column for a metric, handling wandb's 'N - metric' naming and ignoring __MIN/__MAX."""
    cols = [c for c in df.columns if c != "Step" and not c.endswith(("__MIN", "__MAX"))]
    # 1) exact
    for c in cols:
        if c == metric:
            return c
    # 2) "<idx> - metric"
    for c in cols:
        if c.split(" - ", 1)[-1].strip() == metric:
            return c
    # 3) loose contains
    for c in cols:
        if metric.lower() in c.lower():
            return c
    return None


def load_run(path: str, metric: str):
    df = pd.read_csv(path)
    step_col = "Step" if "Step" in df.columns else df.columns[0]
    mcol = find_metric_column(df, metric)
    if mcol is None:
        raise ValueError(
            f"Metric '{metric}' not found in {path}. "
            f"Available: {[c for c in df.columns if c != 'Step']}"
        )
    x = df[step_col].to_numpy(dtype=float)
    y = df[mcol].to_numpy(dtype=float)
    return x, y, mcol


def aggregate_seeds(series_list):
    """
    series_list: list of (steps, values) for one label.
    Returns (steps, mean, std) aligned on common steps.
    For a single seed, std is all zeros.
    """
    # Build a Step-indexed DataFrame, one column per seed.
    cols = {}
    for i, (steps, vals) in enumerate(series_list):
        cols[f"seed{i}"] = pd.Series(vals, index=np.round(steps).astype(int))
    df = pd.DataFrame(cols).sort_index()
    steps = df.index.to_numpy(dtype=float)
    mean = df.mean(axis=1, skipna=True).to_numpy()
    std = df.std(axis=1, ddof=0, skipna=True).fillna(0.0).to_numpy()
    return steps, mean, std


# ------------------------------------------------------------
# Stats
# ------------------------------------------------------------
def summarize(label, steps, mean, lower_better, window=20):
    finite = np.isfinite(mean)
    m = mean[finite]
    s = steps[finite]
    if m.size == 0:
        return None
    best_idx = int(np.argmin(m)) if lower_better else int(np.argmax(m))
    w = min(window, m.size)
    return {
        "label": label,
        "n": m.size,
        "mean": float(m.mean()),
        "best": float(m[best_idx]),
        "best_ep": int(s[best_idx]),
        "final": float(m[-1]),
        "last_avg": float(m[-w:].mean()),
    }


def print_table(stats, lower_better):
    stats = [s for s in stats if s is not None]
    if not stats:
        return
    direction = "min" if lower_better else "max"
    print("\n" + "=" * 84)
    print(f"{'method':<16}{'episodes':<10}{'mean':<10}{f'best({direction})':<18}{'final':<10}{'last20':<10}")
    print("-" * 84)
    # order by quality
    order = sorted(stats, key=lambda s: s["last_avg"], reverse=not lower_better)
    for s in order:
        print(f"{s['label']:<16}{s['n']:<10}{s['mean']:<10.4f}"
              f"{s['best']:.4f} @ep{s['best_ep']:<8}{s['final']:<10.4f}{s['last_avg']:<10.4f}")
    print("=" * 84 + "\n")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot(runs, metric, smooth_method="ema", window=10, half_life=20.0,
         band="std", lower_better=False, save=None, show=True, title=None):
    # group files by label (preserve first-seen order)
    grouped = OrderedDict()
    for label, path in runs:
        grouped.setdefault(label, []).append(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    stats = []
    detected_cols = set()

    for i, (label, paths) in enumerate(grouped.items()):
        color = PALETTE[i % len(PALETTE)]

        series_list = []
        for p in paths:
            sx, sy, mcol = load_run(p, metric)
            detected_cols.add(mcol)
            series_list.append((sx, sy))

        steps, mean, std = aggregate_seeds(series_list)
        n_seeds = len(series_list)

        # raw faded (only meaningful for single seed; for multi-seed we show the band)
        if n_seeds == 1:
            ax.plot(steps, mean, color=color, alpha=0.22, linewidth=1.0)

        mean_s = smooth(mean, smooth_method, window, half_life)

        seed_tag = f" (n={n_seeds})" if n_seeds > 1 else ""
        ax.plot(steps, mean_s, color=color, linewidth=2.4, label=f"{label}{seed_tag}")

        # uncertainty band across seeds
        if n_seeds > 1 and band != "none":
            spread = std.copy()
            if band == "sem":
                spread = std / np.sqrt(max(n_seeds, 1))
            spread_s = smooth(spread, smooth_method, window, half_life)
            ax.fill_between(steps, mean_s - spread_s, mean_s + spread_s,
                            color=color, alpha=0.15, linewidth=0)

        stats.append(summarize(label, steps, mean, lower_better))

    metric_label = detected_cols.pop() if len(detected_cols) == 1 else metric
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    sm_tag = (f"EMA half-life={half_life:g}" if smooth_method == "ema"
              else (f"rolling w={window}" if smooth_method == "rolling" else "raw"))
    ax.set_title(title or f"{metric} comparison  ({sm_tag})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.95, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    print_table(stats, lower_better)

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save}")
    if show:
        plt.show()
    return fig, ax


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_run_spec(spec: str):
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--runs entry must be 'label:path', got '{spec}'")
    label, path = spec.split(":", 1)
    return label.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description="Compare multiple runs from wandb CSVs.")
    ap.add_argument("--runs", nargs="*", type=parse_run_spec, default=None,
                    help="Entries like 'FAIR=1:fair1.csv'. Repeat a label for seed aggregation.")
    ap.add_argument("--metric", type=str, default="mean_reward",
                    help="Metric to plot (e.g. mean_reward, returns, price, latencies).")
    ap.add_argument("--smooth-method", type=str, default="ema",
                    choices=["ema", "rolling", "none"])
    ap.add_argument("--half-life", type=float, default=20.0, help="EMA half-life in episodes.")
    ap.add_argument("--window", type=int, default=10, help="Rolling-mean window (if rolling).")
    ap.add_argument("--band", type=str, default="std", choices=["std", "sem", "none"],
                    help="Uncertainty band for multi-seed labels.")
    ap.add_argument("--lower-better", action="store_true",
                    help="Set for metrics where smaller is better (price, latency).")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    runs = args.runs if args.runs else RUNS
    if not runs:
        print("ERROR: no runs. Pass --runs 'label:path' ... or edit the RUNS block.", file=sys.stderr)
        sys.exit(1)

    for _, p in runs:
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    plot(runs, metric=args.metric, smooth_method=args.smooth_method,
         window=args.window, half_life=args.half_life, band=args.band,
         lower_better=args.lower_better, save=args.save,
         show=not args.no_show, title=args.title)


if __name__ == "__main__":
    main()
# # 多方法对比（FAIR=1/0、Greedy、PPO…），EMA 平滑
# python draw.py \
#   --runs "FAIR=1:jain/fair1.csv" "FAIR=0:jain/fair0.csv" "Greedy:jain/greedy.csv" \
#   --metric Jain_fairness_index --half-life 20 --save jain_842.png