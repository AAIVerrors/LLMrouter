#!/usr/bin/env python3
"""
Measure each fleet endpoint's effective service rate on the ACTUAL training
workload, and emit a config-ready SERVICE_RATE block.

Why this exists
---------------
Config.SERVICE_RATE is not just a warm start for the online EMA: with
QUOTA_NORMALIZE_BY="service_rate" and QUOTA_USE_FROZEN_MU=True it IS the
fairness entitlement -- the quota that decides how much load each endpoint
"should" carry. A stale value silently distorts every fairness number, and a
value measured on a different prompt mix distorts it in a direction you cannot
see. Re-run this whenever the fleet, the dataset mix, or GEN_MAX_NEW_TOKENS
changes.

It samples real prompts from the configured (mixed) dataset rather than a fixed
string, because service rate depends on how long the answers are, and answer
length depends on the task: an MMLU letter and a GSM8K derivation are not the
same amount of work for the same endpoint.

Usage
-----
    python bench_fleet_service_rate.py                  # 10 prompts per server
    python bench_fleet_service_rate.py -n 20            # more samples
    python bench_fleet_service_rate.py --models a,b     # only these models
    python bench_fleet_service_rate.py --concurrent 4   # measure under load

Sequential (default) measures each endpoint in isolation -- the right input for
a capacity constant. --concurrent measures throughput under contention, which
is lower; do not paste that into SERVICE_RATE unless you intend the entitlement
to bake in queueing effects.
"""

import argparse
import os
import queue
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# .env before any client construction -- the SDKs read keys at init.
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

from config import Config
from PoissonPromptGenerator import PoissonPromptGenerator


# --------------------------------------------------------------------------
# Provider dispatch -- mirrors environment.py's routing so the measurement
# exercises the same code path the trainer will.
# --------------------------------------------------------------------------
def _is_together(name: str) -> bool:
    return (name or "").lower().startswith("together/")


def _strip_together(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def _is_mistral(name: str) -> bool:
    m = (name or "").lower()
    return ("/" not in m) and any(
        k in m for k in ("mistral", "mixtral", "ministral", "magistral", "codestral")
    )


def _is_openai(name: str) -> bool:
    return any(k in (name or "").lower() for k in ("gpt", "o1", "o3"))


class Endpoint:
    """One fleet slot, with a lazily built client for its provider."""

    def __init__(self, idx: int, model_name: str):
        self.idx = idx
        self.model_name = model_name
        self.provider = (
            "together" if _is_together(model_name)
            else "mistral" if _is_mistral(model_name)
            else "openai" if _is_openai(model_name)
            else "unknown"
        )
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        if self.provider == "together":
            from together import Together

            self._client = Together(api_key=os.environ["TOGETHER_API_KEY"])
        elif self.provider == "mistral":
            from mistralai.client import Mistral

            self._client = Mistral(
                api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=120000
            )
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(timeout=120.0, max_retries=0)
        else:
            raise RuntimeError(f"unsupported provider for {self.model_name!r}")
        return self._client

    def call(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """One generation. Returns latency, token counts and the raw text."""
        client = self._client_lazy()
        t0 = time.time()
        if self.provider == "mistral":
            r = client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=Config.GEN_TEMPERATURE,
            )
        else:
            name = _strip_together(self.model_name) if self.provider == "together" else self.model_name
            kw = {}
            if self.provider == "together":
                # Must match the training path (environment.py). Without this,
                # a hybrid-thinking model such as Qwen3.5 spends the whole token
                # budget on a hidden `reasoning` field and returns content="",
                # so the benchmark measures a configuration the trainer never
                # uses: measured mu drops ~40% and every quality score reads 0.
                kw["reasoning"] = {"enabled": False}
                kw["top_p"] = Config.GEN_TOP_P
            r = client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=Config.GEN_TEMPERATURE,
                **kw,
            )
        latency = time.time() - t0
        text = (r.choices[0].message.content or "")
        usage = getattr(r, "usage", None)
        return {
            "latency": latency,
            "text": text,
            "in_tok": int(getattr(usage, "prompt_tokens", 0) or 0),
            "out_tok": int(getattr(usage, "completion_tokens", 0) or 0),
        }


def sample_prompts(n: int) -> List[str]:
    """Draw n prompts from the SAME dataset mix the trainer uses."""
    gen = PoissonPromptGenerator(
        arrival_rate=1.0,
        prompt_queue=queue.Queue(),
        max_queue_size=n * 4,
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
    out = []
    for _ in range(n):
        item = gen.get_next_prompt()
        out.append(item["prompt"] if isinstance(item, dict) else str(item))
    return out


def looks_like_reasoning(texts: List[str], out_toks: List[int]) -> Optional[str]:
    """Flag endpoints that break the fleet's non-reasoning assumption.

    A reasoning model inflates both latency and cost in a way that has nothing
    to do with the endpoint's throughput, so it distorts SERVICE_RATE and the
    price term at once.
    """
    joined = " ".join(texts[:5]).lower()
    if re.search(r"<think>|</think>|<reasoning>|<\|thinking\|>", joined):
        return "emits explicit thinking tags"
    if out_toks and statistics.median(out_toks) > 300:
        return f"median {statistics.median(out_toks):.0f} output tokens (verbose)"
    return None


def bench_one(ep: Endpoint, prompts: List[str], max_tokens: int, workers: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    def run(p):
        try:
            return ep.call(p, max_tokens)
        except Exception as e:  # keep going: one bad prompt shouldn't void the endpoint
            errors.append(f"{type(e).__name__}: {str(e)[:120]}")
            return None

    if workers <= 1:
        for p in prompts:
            r = run(p)
            if r:
                results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(run, prompts):
                if r:
                    results.append(r)

    if not results:
        return {"ok": False, "errors": errors[:3]}

    lat = [r["latency"] for r in results]
    out_tok = [r["out_tok"] for r in results]
    return {
        "ok": True,
        "n": len(results),
        "n_fail": len(errors),
        "errors": errors[:2],
        "mean_lat": statistics.mean(lat),
        "median_lat": statistics.median(lat),
        "p90_lat": sorted(lat)[max(0, int(0.9 * len(lat)) - 1)],
        # mu = 1 / E[service time]: the rate a single busy worker sustains.
        "mu": 1.0 / statistics.mean(lat),
        "mean_out_tok": statistics.mean(out_tok) if out_tok else 0.0,
        "mean_in_tok": statistics.mean([r["in_tok"] for r in results]),
        "warn": looks_like_reasoning([r["text"] for r in results], out_tok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num-prompts", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=Config.GEN_MAX_NEW_TOKENS)
    ap.add_argument("--concurrent", type=int, default=1,
                    help="requests in flight per endpoint (1 = isolated capacity, the SERVICE_RATE input)")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated substrings; only matching models are benched")
    args = ap.parse_args()

    endpoints = [Endpoint(i, m) for i, m in enumerate(Config.MODEL_NAMES)]
    if args.models:
        pats = [s.strip().lower() for s in args.models.split(",") if s.strip()]
        endpoints = [e for e in endpoints if any(p in e.model_name.lower() for p in pats)]
        if not endpoints:
            print("no model matched --models", file=sys.stderr)
            return 1

    print(f"Sampling {args.num_prompts} prompts from the configured dataset mix ...")
    prompts = sample_prompts(args.num_prompts)
    print(f"  got {len(prompts)}; median length {statistics.median(len(p) for p in prompts):.0f} chars\n")

    mode = "sequential (isolated)" if args.concurrent <= 1 else f"{args.concurrent}-way concurrent"
    print(f"Benchmarking {len(endpoints)} endpoints, {mode}, max_tokens={args.max_tokens}\n")
    print(f"{'idx':>3}  {'model':38} {'mu':>7} {'mean':>7} {'p90':>7} {'out_tok':>8}  status")
    print("-" * 92)

    rows: Dict[int, Dict[str, Any]] = {}
    for ep in endpoints:
        res = bench_one(ep, prompts, args.max_tokens, args.concurrent)
        rows[ep.idx] = res
        short = ep.model_name.split("/")[-1][:38]
        if not res["ok"]:
            print(f"{ep.idx:>3}  {short:38} {'--':>7} {'--':>7} {'--':>7} {'--':>8}  FAILED: {res['errors'][0] if res['errors'] else '?'}")
            continue
        status = f"{res['n']}/{args.num_prompts} ok"
        if res["n_fail"]:
            status += f", {res['n_fail']} err"
        if res["warn"]:
            status += f"  [!] {res['warn']}"
        print(f"{ep.idx:>3}  {short:38} {res['mu']:7.4f} {res['mean_lat']:7.2f} {res['p90_lat']:7.2f} "
              f"{res['mean_out_tok']:8.0f}  {status}")

    ok = {i: r for i, r in rows.items() if r.get("ok")}
    if not ok:
        print("\nNo endpoint produced a measurement.", file=sys.stderr)
        return 1

    print("\n" + "=" * 92)
    if len(ok) == len(Config.MODEL_NAMES):
        total = sum(r["mu"] for r in ok.values())
        print(f"Fleet total mu = {total:.3f} req/s")
        print("Pick POISSON_ARRIVAL_RATE from the load you want:")
        for rho in (0.85, 0.90, 0.95):
            print(f"    rho={rho:.2f}  ->  POISSON_ARRIVAL_RATE = {rho * total:.2f}")
        print()
        print("Paste into config.py (this list is also the frozen fairness entitlement):")
        print("    SERVICE_RATE = [")
        for i, m in enumerate(Config.MODEL_NAMES):
            print(f"        {ok[i]['mu']:.4f}, # {i} {m.split('/')[-1]}")
        print(f"    ]   # total {total:.3f} req/s")
    else:
        print("Partial run -- measured rates (rerun the full fleet before editing SERVICE_RATE):")
        for i in sorted(ok):
            print(f"    [{i}] {Config.MODEL_NAMES[i].split('/')[-1]:34} mu = {ok[i]['mu']:.4f}")

    if args.concurrent > 1:
        print("\n[!] Measured under concurrency: these rates include queueing and are")
        print("    lower than isolated capacity. Use --concurrent 1 for SERVICE_RATE.")
    if any(r.get("warn") for r in ok.values()):
        print("\n[!] Endpoints flagged above look like reasoning/verbose models. They inflate")
        print("    latency and output cost independently of throughput, which distorts both")
        print("    SERVICE_RATE and the price term -- consider replacing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
