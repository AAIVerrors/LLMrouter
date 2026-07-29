"""Frozen price-window calibration: 1000 uniform-random (prompt, server)
pairs from the live v2' mix; cost computed exactly like environment.py's
dollar_cost (_rough_token_count on prompt and raw text)."""
import os, sys, json, random, time
sys.path.insert(0, "/home/hshu3770/LLMrouter/LLM-ROUTER-V3")
os.chdir("/home/hshu3770/LLMrouter/LLM-ROUTER-V3")
from dotenv import load_dotenv; load_dotenv("/home/hshu3770/LLMrouter/LLM-ROUTER-V3/.env")
from config import Config
from bench_fleet_service_rate import Endpoint, sample_prompts
from environment import _rough_token_count
from concurrent.futures import ThreadPoolExecutor
import numpy as np

N = 1000
prompts = sample_prompts(N)
rng = random.Random(123)
assign = [rng.randrange(len(Config.MODEL_NAMES)) for _ in prompts]
eps = [Endpoint(i, m) for i, m in enumerate(Config.MODEL_NAMES)]

def one(i):
    p, s = prompts[i], assign[i]
    try:
        r = eps[s].call(p, Config.GEN_MAX_NEW_TOKENS)
        cost = (Config.PRICE[s][0] * _rough_token_count(p)
                + Config.PRICE[s][1] * _rough_token_count(r["text"]))
        return {"server": s, "cost": float(cost)}
    except Exception as e:
        return {"server": s, "error": f"{type(e).__name__}: {str(e)[:120]}"}

t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    rows = list(ex.map(one, range(N)))
ok = [r for r in rows if "cost" in r]
errs = [r for r in rows if "error" in r]
costs = [r["cost"] for r in ok]
sq = np.sqrt(np.maximum(np.asarray(costs), 0.0))
lo, hi = np.percentile(sq, 5), np.percentile(sq, 95)
out = {
    "costs": costs,
    "servers": [r["server"] for r in ok],
    "meta": {
        "date": "2026-07-30", "n_requested": N, "n_ok": len(ok),
        "mix": "v2p (drop/logiqa/competition_math)", "assignment": "uniform, seed 123",
        "cost_definition": "PRICE[s] x _rough_token_count (matches env dollar_cost)",
        "sqrt_p5": float(lo), "sqrt_p95": float(hi),
    },
}
json.dump(out, open("price_window_v2p.json", "w"))
per = {}
for r in ok: per.setdefault(r["server"], []).append(r["cost"])
print(f"done in {time.time()-t0:.0f}s: {len(ok)} ok / {len(errs)} failed")
for s in sorted(per): print(f"  server {s}: n={len(per[s])}  mean=${np.mean(per[s])*1e6:.0f}e-6")
print(f"sqrt bounds p5={lo:.5f} p95={hi:.5f}")
if errs[:3]: print("sample errors:", errs[:3])
