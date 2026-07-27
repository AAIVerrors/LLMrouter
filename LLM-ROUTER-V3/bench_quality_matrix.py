#!/usr/bin/env python3
"""
Measure the (endpoint x task-type) quality matrix and decide whether
prompt-aware routing has anything to exploit.

The premise behind beating a prompt-blind baseline like P2C is that endpoint
quality RANK changes with the prompt: some model is best on math and a
different one is best on multi-hop QA. If the rank is the same everywhere,
knowing the prompt buys nothing and no amount of RL tuning closes the gap --
the fleet has to change instead.

Two things are reported, and they answer different questions:

  1. CROSSOVER -- does the best endpoint differ by task type? Without one,
     prompt-aware routing is provably no better than prompt-blind on quality.

  2. VARIANCE DECOMPOSITION -- of the total spread in per-request quality, how
     much comes from WHICH ENDPOINT served it (routable signal) versus HOW HARD
     THE PROMPT WAS (noise the router cannot control)? This is the signal-to-
     noise ratio the policy gradient actually sees. A high difficulty share
     means the interval reward is dominated by which prompts happened to
     arrive, and PPO is largely fitting noise.

Every endpoint answers the SAME prompts, so the comparison is paired -- the
difficulty term cancels in the within-prompt contrast, which is what makes the
decomposition meaningful.

Usage:
    python bench_quality_matrix.py                 # 12 prompts per task type
    python bench_quality_matrix.py -n 20
    python bench_quality_matrix.py --workers 12    # more parallelism
"""

import argparse
import json
import os
import queue
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env):
        for _line in open(_env):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                k, v = _line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

import numpy as np

from config import Config
from PoissonPromptGenerator import PoissonPromptGenerator
from environment import match_quality_score  # reuse the trainer's exact scoring
from bench_fleet_service_rate import Endpoint


def collect_prompts(n_per_task: int) -> Dict[str, List[Dict[str, Any]]]:
    """Draw n prompts per ACTIVE task type from the configured dataset mix."""
    gen = PoissonPromptGenerator(
        arrival_rate=1.0,
        prompt_queue=queue.Queue(),
        max_queue_size=n_per_task * 40,
        dataset_name=Config.DATASET_NAME,
        dataset_config=Config.DATASET_CONFIG,
        dataset_split=Config.DATASET_SPLIT,
        prompt_style=Config.QA_PROMPT_STYLE,
        qa_include_context=Config.QA_INCLUDE_CONTEXT,
        qa_max_context_docs=Config.QA_MAX_CONTEXT_DOCS,
        qa_max_context_chars=Config.QA_MAX_CONTEXT_CHARS,
        force_final_tag=Config.QA_FORCE_FINAL_TAG,
        final_tag=Config.FINAL_ANSWER_TAG,
        shuffle_dataset=Config.SHUFFLE_DATASET,
        dataset_seed=Config.DATASET_SEED,
        mixed_datasets=Config.MIXED_DATASETS if Config.USE_MIXED_DATASET else None,
        dataset_levels=Config.DATASET_LEVELS,
        dataset_filter=Config.DATASET_FILTER,
        mcq_cot=bool(getattr(Config, "MCQ_COT", True)),
        math_brief=bool(getattr(Config, "MATH_BRIEF_REASONING", True)),
        mcq_brief=bool(getattr(Config, "MCQ_BRIEF_REASONING", True)),
    )
    want = {
        str(d["task_type"])
        for d in (Config.MIXED_DATASETS or [])
        if float(d.get("weight", 0)) > 0
    } or {"any"}

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # The mixture is sampled by weight, so oversample and stop once every
    # active task type is full rather than assuming a fixed draw order.
    for _ in range(n_per_task * 40):
        item = gen.get_next_prompt()
        tt = str(item.get("task_type", "any"))
        if "any" in want:
            tt = "any"
        if tt in want and len(buckets[tt]) < n_per_task:
            buckets[tt].append(item)
        if all(len(buckets[t]) >= n_per_task for t in want):
            break
    return dict(buckets)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-n", "--per-task", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=Config.GEN_MAX_NEW_TOKENS)
    ap.add_argument(
        "--dump",
        default=None,
        help="write per-(endpoint,prompt) scores/latency/out_tok to this JSON, "
             "so fleet-subset comparisons can be redone without re-calling the APIs",
    )
    args = ap.parse_args()

    print(f"Sampling {args.per_task} prompts per task type ...")
    buckets = collect_prompts(args.per_task)
    tasks = sorted(buckets)
    for t in tasks:
        print(f"  {t:14} {len(buckets[t])}")
    jobs = [(t, i, it) for t in tasks for i, it in enumerate(buckets[t])]
    endpoints = [Endpoint(i, m) for i, m in enumerate(Config.MODEL_NAMES)]
    total = len(jobs) * len(endpoints)
    print(f"\n{len(endpoints)} endpoints x {len(jobs)} prompts = {total} calls\n")

    # scores[endpoint_idx][(task, prompt_idx)] = quality in [0,1]
    scores: Dict[int, Dict[Any, float]] = {e.idx: {} for e in endpoints}
    lat: Dict[int, List[float]] = {e.idx: [] for e in endpoints}
    fails: Dict[int, int] = {e.idx: 0 for e in endpoints}

    otok: Dict[int, Dict[Any, int]] = {e.idx: {} for e in endpoints}

    def work(job):
        ep, (task, pi, item) = job
        try:
            r = ep.call(item["prompt"], args.max_tokens)
            q = float(match_quality_score(r["text"], item.get("output")))
            return ep.idx, (task, pi), q, r["latency"], r.get("out_tok"), None
        except Exception as e:
            return ep.idx, (task, pi), None, None, None, f"{type(e).__name__}"

    all_jobs = [(ep, j) for ep in endpoints for j in jobs]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, key, q, la, ot, err in pool.map(work, all_jobs):
            done += 1
            if err:
                fails[idx] += 1
            else:
                scores[idx][key] = q
                lat[idx].append(la)
                if ot is not None:
                    otok[idx][key] = int(ot)
            if done % 40 == 0 or done == total:
                print(f"  {done}/{total} ({time.time()-t0:.0f}s)", end="\r", flush=True)
    print(" " * 40, end="\r")
    print(f"Done in {time.time()-t0:.0f}s\n")

    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump(
                {
                    "models": list(Config.MODEL_NAMES),
                    "price": [list(p) for p in Config.PRICE[: len(Config.MODEL_NAMES)]],
                    "max_tokens": args.max_tokens,
                    "mcq_cot": bool(getattr(Config, "MCQ_COT", True)),
                    # keys are "task|prompt_idx"; JSON cannot hold tuple keys
                    "scores": {
                        str(i): {f"{k[0]}|{k[1]}": v for k, v in d.items()}
                        for i, d in scores.items()
                    },
                    "out_tok": {
                        str(i): {f"{k[0]}|{k[1]}": v for k, v in d.items()}
                        for i, d in otok.items()
                    },
                    "latency": {str(i): v for i, v in lat.items()},
                },
                fh,
            )
        print(f"raw per-request data -> {args.dump}\n")

    names = [m.split("/")[-1][:22] for m in Config.MODEL_NAMES]

    # ---- quality matrix -------------------------------------------------
    print("=" * 78)
    print("QUALITY MATRIX  (mean score per endpoint x task type)")
    print("=" * 78)
    hdr = f"{'idx':>3}  {'endpoint':24}" + "".join(f"{t[:12]:>13}" for t in tasks) + f"{'overall':>9}"
    print(hdr)
    print("-" * len(hdr))
    per_task_mean: Dict[str, Dict[int, float]] = {t: {} for t in tasks}
    for e in endpoints:
        row = f"{e.idx:>3}  {names[e.idx]:24}"
        allv = []
        for t in tasks:
            v = [scores[e.idx][k] for k in scores[e.idx] if k[0] == t]
            if v:
                per_task_mean[t][e.idx] = float(np.mean(v))
                row += f"{np.mean(v):13.3f}"
                allv += v
            else:
                row += f"{'--':>13}"
        row += f"{np.mean(allv):9.3f}" if allv else f"{'--':>9}"
        if fails[e.idx]:
            row += f"   ({fails[e.idx]} failed)"
        print(row)

    # ---- crossover ------------------------------------------------------
    print("\n" + "=" * 78)
    print("CROSSOVER  -- does the best endpoint change with the task?")
    print("=" * 78)
    winners = {}
    for t in tasks:
        if not per_task_mean[t]:
            continue
        ranked = sorted(per_task_mean[t].items(), key=lambda kv: -kv[1])
        winners[t] = ranked[0][0]
        top3 = ", ".join(f"{names[i]}({v:.2f})" for i, v in ranked[:3])
        print(f"  {t:14} best: {top3}")
    uniq = set(winners.values())
    print()
    if len(uniq) > 1:
        print(f"  >>> CROSSOVER PRESENT: {len(uniq)} different endpoints win different tasks.")
        print("      Prompt-aware routing has quality headroom a prompt-blind policy cannot reach.")
    else:
        w = names[list(uniq)[0]] if uniq else "?"
        print(f"  >>> NO CROSSOVER: '{w}' wins every task type.")
        print("      On quality alone, knowing the prompt buys nothing over always picking")
        print("      that endpoint -- the gap to a prompt-blind baseline cannot come from here.")

    # Rank correlation between task types: how similar are the orderings?
    if len(tasks) > 1:
        print("\n  Rank agreement between task types (Spearman; 1.0 = identical ordering):")
        for a in range(len(tasks)):
            for b in range(a + 1, len(tasks)):
                ta, tb = tasks[a], tasks[b]
                common = sorted(set(per_task_mean[ta]) & set(per_task_mean[tb]))
                if len(common) < 3:
                    continue
                ra = np.argsort(np.argsort([-per_task_mean[ta][i] for i in common]))
                rb = np.argsort(np.argsort([-per_task_mean[tb][i] for i in common]))
                rho = float(np.corrcoef(ra, rb)[0, 1])
                flag = "  <- orderings differ" if rho < 0.7 else ""
                print(f"    {ta[:12]:>12} vs {tb[:12]:<12} rho = {rho:+.2f}{flag}")

    # ---- variance decomposition ----------------------------------------
    print("\n" + "=" * 78)
    print("SIGNAL vs NOISE  -- what drives the spread in per-request quality?")
    print("=" * 78)
    keys = sorted(set().union(*[set(scores[e.idx]) for e in endpoints]))
    mat = np.array(
        [[scores[e.idx].get(k, np.nan) for k in keys] for e in endpoints], dtype=float
    )
    ok_cols = ~np.isnan(mat).any(axis=0)
    mat = mat[:, ok_cols]
    if mat.shape[1] < 3:
        print("  not enough complete prompts for a decomposition")
        return 0

    grand = mat.mean()
    prompt_mean = mat.mean(axis=0)          # difficulty of each prompt
    endpoint_mean = mat.mean(axis=1)        # capability of each endpoint
    var_prompt = float(np.var(prompt_mean))
    var_endpoint = float(np.var(endpoint_mean))
    resid = mat - prompt_mean[None, :] - endpoint_mean[:, None] + grand
    var_inter = float(np.var(resid))        # endpoint x prompt interaction
    tot = var_prompt + var_endpoint + var_inter

    print(f"  complete prompts: {mat.shape[1]}   mean quality: {grand:.3f}\n")
    print(f"  prompt difficulty        {var_prompt/tot:6.1%}   <- NOISE: same for every endpoint")
    print(f"  endpoint capability      {var_endpoint/tot:6.1%}   <- routable, but prompt-BLIND can use it too")
    print(f"  endpoint x prompt        {var_inter/tot:6.1%}   <- the ONLY term prompt-aware routing owns")

    print()
    if var_inter / tot < 0.05:
        print("  >>> The interaction term is negligible. Which endpoint is best does not")
        print("      depend on the prompt, so a prompt-aware router can at best match a")
        print("      policy that always picks the globally strongest endpoint.")
    elif var_prompt / tot > 0.6:
        print("  >>> Difficulty dominates. The routing decision moves the reward far less")
        print("      than which prompts happen to arrive, so the interval-level advantage")
        print("      is mostly noise -- a per-request baseline that controls for prompt")
        print("      difficulty (V^task) is what makes this learnable.")
    else:
        print("  >>> Workable: the interaction term is large enough for prompt-aware")
        print("      routing to have real headroom over a prompt-blind policy.")

    # Best-per-prompt oracle vs the best single endpoint: the headroom, in quality units.
    oracle = mat.max(axis=0).mean()
    best_fixed = endpoint_mean.max()
    print(f"\n  per-prompt oracle {oracle:.3f}  vs  best single endpoint {best_fixed:.3f}"
          f"   -> headroom {oracle - best_fixed:+.3f}")
    print("      (upper bound on what ANY prompt-aware policy can gain on quality)")

    print("\n  latency (mean s): " + "  ".join(
        f"{names[e.idx][:10]}={statistics.mean(lat[e.idx]):.1f}" for e in endpoints if lat[e.idx]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
