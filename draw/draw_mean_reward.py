import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== config ======
csv_paths = [
    "data/mean_reward/0_5_5.csv",
    "data/mean_reward/1_5_5.csv",
    "data/mean_reward/avg_5_5.csv",
    "data/mean_reward/greedy_5_5.csv",
    "data/mean_reward/ppo_0_5_5.csv",
    "data/mean_reward/ppo_1_5_5.csv",
]
ema_decay = 0.99
out_png = "Mean.png"
dpi = 300

# same as matplotlib default figsize (same as your previous plt.figure())
figsize = (6.4, 4.8)

raw_alpha = 0.15
raw_lw = 0.8
ema_lw = 1.0
ema_ls = "-"
# ====================

def find_step_col(columns):
    for c in ["Step", "step", "_step", "global_step"]:
        if c in columns:
            return c
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get("step", None)

def find_metric_col(columns, metric_key="mean_reward"):
    candidates = []
    for c in columns:
        cl = c.lower()
        if metric_key.lower() in cl and "__min" not in cl and "__max" not in cl:
            candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates, key=len)[0]
    return None

def load_xy(csv_path, metric_key="mean_reward"):
    df = pd.read_csv(csv_path)
    step_col = find_step_col(df.columns)
    y_col = find_metric_col(df.columns, metric_key=metric_key)

    if step_col is None or y_col is None:
        raise ValueError(
            f"[{os.path.basename(csv_path)}] Missing Step or metric '{metric_key}'.\n"
            f"Columns: {list(df.columns)}"
        )

    x = pd.to_numeric(df[step_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].astype(float).to_numpy()
    y = y[mask].astype(float).to_numpy()

    order = x.argsort()
    return x[order], y[order]

def ema_series(y, decay):
    m = float(y[0])
    out = []
    for v in y:
        m = decay * m + (1.0 - decay) * float(v)
        out.append(m)
    return out

plt.figure(figsize=figsize)

for csv_path in csv_paths:
    x, y = load_xy(csv_path, metric_key="mean_reward")
    ema = ema_series(y, ema_decay)

    tag = os.path.splitext(os.path.basename(csv_path))[0]
    pretty = tag.replace("_", " ")

    line_raw, = plt.plot(x, y, label=pretty, alpha=raw_alpha, linewidth=raw_lw)
    plt.plot(x, ema, linestyle=ema_ls, color=line_raw.get_color(), linewidth=ema_lw, label="_nolegend_")

plt.title("Mean Reward")
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.legend(title="color", ncol=2)
plt.tight_layout()
plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_png}")
