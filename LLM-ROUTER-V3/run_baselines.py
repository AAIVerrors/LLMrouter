#!/usr/bin/env python3
"""Run each prompt-blind baseline under the CURRENT config, one after another.

The flags are read via getattr(Config, ...) at decision time, so mutating the
Config class here is equivalent to editing config.py -- without leaving the
file half-edited between runs.

Do NOT importlib.reload(config): every module does `from config import Config`
and holds a reference to that one class object, so a reload would hand this
script a fresh copy that the trainer never sees. Mutate the shared object and
reset the flags between runs instead.

Episode count is lower than a training run on purpose: none of these learn, so
episodes only buy statistical stability, not convergence.

CAP_WEIGHTED_JSQ and the weighted power-of-d are not optional extras. Plain
JSQ/P2C rank by raw queue length and are heterogeneity-UNAWARE, and this fleet
spans 4.6x in service rate, where that is known to degrade. Comparing only
against the unweighted variants invites the objection that the baseline was
handicapped.
"""
import sys, time
from datetime import datetime

from config import Config

# Ordered most-informative first, so stopping early still leaves the
# comparisons the paper actually needs.
BASELINES = {
    "p2c":           {"P2C": True},
    "cap_jsq":       {"CAP_WEIGHTED_JSQ": True},
    "pod3_weighted": {"POWER_OF_D": True, "POWER_OF_D_CHOICES": 3,
                      "POWER_OF_D_WEIGHTED": True},
    "jsq":           {"JSQ": True},
    "round_robin":   {"ROUND_ROBIN": True},
    "random":        {"RANDOM_SELECT": True},
}
ALL_FLAGS = {"P2C", "JSQ", "CAP_WEIGHTED_JSQ", "POWER_OF_D",
             "POWER_OF_D_WEIGHTED", "ROUND_ROBIN", "RANDOM_SELECT",
             "GREEDY_UTILITY"}


def reset_flags() -> None:
    for f in ALL_FLAGS:
        setattr(Config, f, False)


def run_one(name: str, flags: dict, episodes: int) -> None:
    reset_flags()
    for k, v in flags.items():
        setattr(Config, k, v)
    Config.MAX_EPISODES = episodes

    active = [f for f in sorted(ALL_FLAGS) if getattr(Config, f, False)]
    print(f"\n{'='*70}\n[{datetime.now():%H:%M:%S}] BASELINE {name}"
          f"   active={active}   episodes={episodes}\n{'='*70}", flush=True)

    from trainer import EnhancedLLMRouterTrainer
    t = EnhancedLLMRouterTrainer()
    try:
        t.train()
    finally:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from dotenv import load_dotenv
    load_dotenv()

    want = sys.argv[1].split(",") if len(sys.argv) > 1 else list(BASELINES)
    eps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    bad = [w for w in want if w not in BASELINES]
    if bad:
        sys.exit(f"unknown baseline(s): {bad}\navailable: {list(BASELINES)}")

    t0 = time.time()
    for name in want:
        run_one(name, BASELINES[name], eps)
        print(f"[{datetime.now():%H:%M:%S}] {name} done "
              f"({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
    reset_flags()
    print(f"\nall done in {(time.time()-t0)/60:.0f} min")
