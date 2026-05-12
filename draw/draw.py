import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== config ======
csv_paths = [
    "data/return/0_5_5.csv",
    "data/return/1_5_5.csv",
    "data/return/avg_5_5.csv",
    "data/return/greedy_5_5.csv",
    "data/return/ppo_0_5_5.csv",
    "data/return/ppo_1_5_5.csv",
]
ema_decay = 0.99
out_png = "return.png"
dpi = 300

# make trend line stand out
raw_alpha = 0.25          # lighter raw line
raw_lw = 1.0
ema_lw = 1.0              # thicker EMA
ema_zorder = 3
# ====================

def pick_col(columns, cands):
    for c in cands:
        if c in columns:
            return c
    lower_map = {c.lower(): c for c in columns}
    for c in cands:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    cands_low = [c.lower() for c in cands]
    for col in columns:
        cl = col.lower()
        if any(k in cl for k in cands_low):
            return col
    return None

def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    step_col = pick_col(df.columns, ["Step", "step", "_step", "global_step"])
    ret_col  = pick_col(df.columns, ["Return", "return", "episode_return", "train/return", "eval/return", "returns"])
    if step_col is None or ret_col is None:
        raise ValueError(f"[{os.path.basename(csv_path)}] Missing Step/Return. Columns: {list(df.columns)}")

    x = pd.to_numeric(df[step_col], errors="coerce")
    y = pd.to_numeric(df[ret_col], errors="coerce")
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

plt.figure()

for csv_path in csv_paths:
    x, y = load_xy(csv_path)
    ema = ema_series(y, ema_decay)

    tag = os.path.splitext(os.path.basename(csv_path))[0]  # "0_5_5"
    pretty = tag.replace("_", " ")

    # raw (faded) + EMA (bold), same color
    line_raw, = plt.plot(x, y, label=pretty, alpha=raw_alpha, linewidth=raw_lw, zorder=1)
    plt.plot(
        x, ema,
        linestyle="-",          # solid for stronger visibility
        color=line_raw.get_color(),
        linewidth=ema_lw,
        zorder=ema_zorder,
        label="_nolegend_"
    )

plt.title("Return")
plt.xlabel("Step")
plt.ylabel("Return")
plt.legend(title="color")
plt.tight_layout()
plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
plt.show()

print(f"Saved to: {out_png}")
