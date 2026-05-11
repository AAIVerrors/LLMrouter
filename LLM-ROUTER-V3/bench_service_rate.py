#!/usr/bin/env python3
"""
One-file benchmark:
- HF local models (transformers): prefill + KV-cache decode loop
- OpenAI / Anthropic / Mistral / Together APIs: TTFT + e2e timing

Adds:
- Multiple prompts (txt/jsonl/inline), optional weighted sampling
- Service rate estimate per model: mu ~= 1 / E[e2e_s] across prompt mix
- Per-run CSV + per-model summary CSV
- Optional plot PNG (service rate and mean service time)
- Optional per-run cost estimate when token usage is available

Together notes:
- Use model names prefixed with `together/`, e.g. `together/Qwen/Qwen3.5-9B`
- Requires environment variable `TOGETHER_API_KEY`
- Uses OpenAI-compatible API with base_url `https://api.together.xyz/v1`
"""

import os
import gc
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# --------------------------
# Default models
# --------------------------
# Keep this order aligned with your router Config.MODEL_NAMES.
MODEL_NAMES = [
    "mistral-medium-3.5",
]

# HF-ish / legacy aliases -> API model IDs.
# The left side is what you may pass in MODEL_NAMES; the right side is sent to API.
MISTRAL_NAME_MAP = {
    "mistral-7b-instruct-v0.2": "open-mistral-7b",
    "mixtral-8x7b-instruct-v0.1": "open-mixtral-8x7b",
    "open-mistral-nemo": "open-mistral-nemo-2407",
    "open-mistral-nemo-2407": "open-mistral-nemo-2407",
}

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


# --------------------------
# Optional pricing metadata
# --------------------------
# USD per token, matching the MODEL_NAMES order above.
# Prices are expressed as dollars / token, so $0.15 / 1M = 0.00000015.
MODEL_PRICE_USD_PER_TOKEN = {
    "mistral-small-2603":      (0.00000015, 0.00000060),
    "mistral-large-2512":      (0.00000050, 0.00000150),
    "mistral-medium-2508":     (0.00000040, 0.00000200),
    "mistral-medium-3.5":      (0.00000150, 0.00000750),

    "ministral-14b-2512":      (0.00000020, 0.00000020),
    "ministral-8b-2512":       (0.00000015, 0.00000015),
    "ministral-3b-2512":       (0.00000010, 0.00000010),

    "magistral-medium-2509":   (0.00000200, 0.00000500),
    "mistral-medium-2505":     (0.00000040, 0.00000200),
    "open-mistral-nemo":       (0.00000015, 0.00000015),
    "open-mistral-nemo-2407":  (0.00000015, 0.00000015),
    "codestral-2508":          (0.00000030, 0.00000090),

    "gpt-4.1-2025-04-14":      (0.00000200, 0.00000800),
    "gpt-4.1-mini-2025-04-14": (0.00000040, 0.00000160),
    "gpt-4.1-nano-2025-04-14": (0.00000010, 0.00000040),
}


# --------------------------
# Utils
# --------------------------
def now() -> float:
    return time.perf_counter()


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return None
    return float(a) / float(b)


def median_or_none(xs: List[Optional[float]]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs2:
        return None
    return float(np.median(xs2))


def weighted_mean_or_none(xs: List[Optional[float]], ws: List[float]) -> Optional[float]:
    vals, wts = [], []
    for x, w in zip(xs, ws):
        if x is None or not np.isfinite(x):
            continue
        if w is None or not np.isfinite(w) or w <= 0:
            continue
        vals.append(float(x))
        wts.append(float(w))
    if not vals:
        return None
    vals = np.array(vals, dtype=np.float64)
    wts = np.array(wts, dtype=np.float64)
    return float((vals * wts).sum() / wts.sum())


def is_together_model(model: str) -> bool:
    return str(model).startswith("together/")


def strip_together_prefix(model: str) -> str:
    return model[len("together/"):] if is_together_model(model) else model


def infer_backend(model: str) -> str:
    if is_together_model(model):
        return "together"
    m = model.lower()
    if "/" in model:
        return "hf"
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if (
        ("mistral" in m)
        or ("mixtral" in m)
        or ("ministral" in m)
        or ("magistral" in m)
        or ("codestral" in m)
    ):
        return "mistral"
    return "hf"


def short_hash(s: str, n: int = 10) -> str:
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def price_for_model(model: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (input_price_per_token, output_price_per_token), if known."""
    candidates = [
        model,
        MISTRAL_NAME_MAP.get(model, model),
        strip_together_prefix(model),
    ]
    for key in candidates:
        if key in MODEL_PRICE_USD_PER_TOKEN:
            return MODEL_PRICE_USD_PER_TOKEN[key]
    return None, None


def estimate_cost_usd(
    model: str,
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return pin, pout, input_cost, output_cost, total_cost."""
    pin, pout = price_for_model(model)
    if pin is None or pout is None:
        return None, None, None, None, None

    input_cost = None
    output_cost = None
    if prompt_tokens is not None and np.isfinite(prompt_tokens):
        input_cost = float(prompt_tokens) * float(pin)
    if output_tokens is not None and np.isfinite(output_tokens):
        output_cost = float(output_tokens) * float(pout)

    total = None
    if input_cost is not None or output_cost is not None:
        total = float(input_cost or 0.0) + float(output_cost or 0.0)

    return float(pin), float(pout), input_cost, output_cost, total


# --------------------------
# Prompt loading
# --------------------------
@dataclass
class PromptItem:
    prompt_id: int
    prompt: str
    weight: float = 1.0
    name: str = ""


def load_prompts(prompt_fallback: str, prompts_file: Optional[str], prompt_sep: str) -> List[PromptItem]:
    if not prompts_file:
        return [PromptItem(prompt_id=0, prompt=prompt_fallback, weight=1.0, name="default")]

    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"--prompts_file not found: {prompts_file}")

    items: List[PromptItem] = []
    if prompts_file.endswith(".jsonl"):
        pid = 0
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj.get("prompt", "")
                if not isinstance(p, str) or not p.strip():
                    continue
                w = float(obj.get("weight", 1.0))
                name = str(obj.get("name", "") or "")
                items.append(PromptItem(prompt_id=pid, prompt=p, weight=w, name=name))
                pid += 1
    else:
        with open(prompts_file, "r", encoding="utf-8") as f:
            text = f.read()

        if prompt_sep and prompt_sep in text:
            blocks = [b.strip() for b in text.split(prompt_sep)]
        else:
            blocks = [b.strip() for b in text.split("\n\n")]

        pid = 0
        for b in blocks:
            if not b:
                continue
            items.append(PromptItem(prompt_id=pid, prompt=b, weight=1.0, name=""))
            pid += 1

    if not items:
        return [PromptItem(prompt_id=0, prompt=prompt_fallback, weight=1.0, name="default")]
    return items


def sample_prompts(items: List[PromptItem], n: int, seed: int) -> List[PromptItem]:
    if n <= 0 or n >= len(items):
        return items
    rng = np.random.default_rng(seed)
    weights = np.array([max(0.0, float(it.weight)) for it in items], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(items), dtype=np.float64)
    probs = weights / weights.sum()
    idxs = rng.choice(len(items), size=n, replace=True, p=probs)

    out: List[PromptItem] = []
    for j, i in enumerate(idxs.tolist()):
        src = items[i]
        out.append(PromptItem(prompt_id=j, prompt=src.prompt, weight=src.weight, name=src.name))
    return out


# --------------------------
# Result
# --------------------------
@dataclass
class BenchResult:
    model: str
    backend: str
    device: str

    prompt_id: int = 0
    prompt_name: str = ""
    prompt_hash: str = ""
    prompt_len_chars: int = 0
    prompt_weight: float = 1.0

    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    prefill_s: Optional[float] = None
    decode_s: Optional[float] = None
    decode_toks_per_s: Optional[float] = None

    ttft_s: Optional[float] = None
    e2e_s: Optional[float] = None
    e2e_toks_per_s: Optional[float] = None

    service_rate_rps: Optional[float] = None  # approx 1/e2e_s

    chars: Optional[int] = None
    chars_per_s: Optional[float] = None

    price_in_per_token: Optional[float] = None
    price_out_per_token: Optional[float] = None
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None

    note: str = ""


# --------------------------
# HF benchmark
# --------------------------
def bench_hf(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    warmup: int,
    repeats: int,
) -> BenchResult:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    res = BenchResult(model=model_id, backend="hf", device=device, note=f"dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    first_param = next(model.parameters())
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}

    res.prompt_tokens = int(inputs["input_ids"].shape[-1])
    res.output_tokens = int(max_new_tokens)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.generate(
                **inputs,
                max_new_tokens=min(16, max_new_tokens),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
        if device != "cpu":
            torch.cuda.synchronize()

    prefill_s_list, decode_s_list, e2e_s_list, ttft_s_list = [], [], [], []

    with torch.inference_mode():
        for _ in range(repeats):
            if device != "cpu":
                torch.cuda.synchronize()

            t_start = now()

            t0 = now()
            out = model(**inputs, use_cache=True)
            if device != "cpu":
                torch.cuda.synchronize()
            t1 = now()

            prefill_s = t1 - t0
            past = out.past_key_values

            next_tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            t_first = now()

            t2 = now()
            cur = next_tok
            for _step in range(max(0, max_new_tokens - 1)):
                o2 = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o2.past_key_values
                cur = o2.logits[:, -1].argmax(dim=-1, keepdim=True)

            if device != "cpu":
                torch.cuda.synchronize()
            t_end = now()

            prefill_s_list.append(prefill_s)
            ttft_s_list.append(t_first - t_start)
            decode_s_list.append(t_end - t2)
            e2e_s_list.append(t_end - t_start)

    res.prefill_s = median_or_none(prefill_s_list)
    res.ttft_s = median_or_none(ttft_s_list)
    res.decode_s = median_or_none(decode_s_list)
    res.e2e_s = median_or_none(e2e_s_list)

    decode_tokens = max(0, max_new_tokens - 1)
    res.decode_toks_per_s = safe_div(decode_tokens, res.decode_s)
    res.e2e_toks_per_s = safe_div(max_new_tokens, res.e2e_s)

    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return res


# --------------------------
# OpenAI-compatible helper
# --------------------------
def _bench_openai_compatible(
    client,
    model: str,
    prompt: str,
    max_new_tokens: int,
    repeats: int,
    backend_name: str,
    note: str = "",
) -> BenchResult:
    ttft_list, e2e_list, out_tok_list, prompt_tok_list = [], [], [], []
    chars_list, decode_tps_list, e2e_tps_list = [], [], []

    for _ in range(repeats):
        t0 = now()
        first_content_time = None
        text = ""
        usage = None

        try:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception as e1:
                msg = str(e1)
                if (
                    ("max_tokens" in msg and "max_completion_tokens" in msg)
                    or ("Unsupported parameter" in msg and "max_tokens" in msg)
                ):
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_new_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                else:
                    raise

            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                for ch in choices:
                    delta = getattr(ch, "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        if first_content_time is None:
                            first_content_time = now()
                        text += content

                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage

        except Exception as e:
            return BenchResult(
                model=model,
                backend="error",
                device="api",
                note=f"{backend_name} request failed: {e}",
            )

        t1 = now()
        if first_content_time is None:
            first_content_time = t1

        ttft = first_content_time - t0
        e2e = t1 - t0
        decode_window = max(1e-9, t1 - first_content_time)

        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ot = getattr(usage, "completion_tokens", None) if usage else None
        if usage and ot is None:
            ot = getattr(usage, "output_tokens", None)

        prompt_tok_list.append(pt)
        out_tok_list.append(ot)
        ttft_list.append(ttft)
        e2e_list.append(e2e)
        chars_list.append(len(text))

        if ot is not None:
            decode_tps_list.append(ot / decode_window)
            e2e_tps_list.append(ot / e2e)
        else:
            decode_tps_list.append(None)
            e2e_tps_list.append(None)

    res = BenchResult(model=model, backend=backend_name, device="api", note=note or "stream include_usage")
    res.ttft_s = median_or_none(ttft_list)
    res.e2e_s = median_or_none(e2e_list)

    res.prompt_tokens = (
        int(np.median([x for x in prompt_tok_list if isinstance(x, (int, float))]))
        if any(isinstance(x, (int, float)) for x in prompt_tok_list)
        else None
    )
    res.output_tokens = (
        int(np.median([x for x in out_tok_list if isinstance(x, (int, float))]))
        if any(isinstance(x, (int, float)) for x in out_tok_list)
        else None
    )
    res.decode_toks_per_s = median_or_none(decode_tps_list)
    res.e2e_toks_per_s = median_or_none(e2e_tps_list)

    res.chars = int(np.median(chars_list)) if chars_list else None
    res.chars_per_s = safe_div(res.chars, res.e2e_s)
    if res.output_tokens is None:
        res.note += "; token usage missing -> chars/sec fallback"

    return res


# --------------------------
# OpenAI benchmark
# --------------------------
def bench_openai(model: str, prompt: str, max_new_tokens: int, repeats: int) -> BenchResult:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return BenchResult(model=model, backend="skipped", device="api", note="OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
    except Exception as e:
        return BenchResult(model=model, backend="error", device="api", note=f"openai init failed: {e}")

    return _bench_openai_compatible(
        client,
        model,
        prompt,
        max_new_tokens,
        repeats,
        backend_name="openai",
        note="stream include_usage",
    )


# --------------------------
# Together benchmark
# --------------------------
def bench_together(model: str, prompt: str, max_new_tokens: int, repeats: int) -> BenchResult:
    key = os.getenv("TOGETHER_API_KEY", "")
    if not key:
        return BenchResult(model=model, backend="skipped", device="api", note="TOGETHER_API_KEY not set")

    api_model = strip_together_prefix(model)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=TOGETHER_BASE_URL)
    except Exception as e:
        return BenchResult(model=model, backend="error", device="api", note=f"together init failed: {e}")

    return _bench_openai_compatible(
        client,
        api_model,
        prompt,
        max_new_tokens,
        repeats,
        backend_name="together",
        note=f"openai-compatible; api_model={api_model}",
    )


# --------------------------
# Anthropic benchmark
# --------------------------
def bench_anthropic(model: str, prompt: str, max_new_tokens: int, repeats: int) -> BenchResult:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return BenchResult(model=model, backend="skipped", device="api", note="ANTHROPIC_API_KEY not set")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
    except Exception as e:
        return BenchResult(model=model, backend="error", device="api", note=f"anthropic init failed: {e}")

    ttft_list, e2e_list, out_tok_list, in_tok_list, chars_list = [], [], [], [], []

    for _ in range(repeats):
        t0 = now()
        first_content_time = None
        text = ""
        out_tok = None
        in_tok = None

        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                for event in s:
                    if getattr(event, "type", "") == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        piece = getattr(delta, "text", None) if delta else None
                        if piece:
                            if first_content_time is None:
                                first_content_time = now()
                            text += piece

                final = s.get_final_message()
                usage = getattr(final, "usage", None)
                in_tok = getattr(usage, "input_tokens", None) if usage else None
                out_tok = getattr(usage, "output_tokens", None) if usage else None

        except Exception as e:
            return BenchResult(model=model, backend="error", device="api", note=f"anthropic request failed: {e}")

        t1 = now()
        if first_content_time is None:
            first_content_time = t1

        ttft_list.append(first_content_time - t0)
        e2e_list.append(t1 - t0)
        chars_list.append(len(text))
        in_tok_list.append(in_tok)
        out_tok_list.append(out_tok)

    res = BenchResult(model=model, backend="anthropic", device="api", note="tokens from final usage")
    res.ttft_s = median_or_none(ttft_list)
    res.e2e_s = median_or_none(e2e_list)

    res.prompt_tokens = (
        int(np.median([x for x in in_tok_list if isinstance(x, (int, float))]))
        if any(isinstance(x, (int, float)) for x in in_tok_list)
        else None
    )
    res.output_tokens = (
        int(np.median([x for x in out_tok_list if isinstance(x, (int, float))]))
        if any(isinstance(x, (int, float)) for x in out_tok_list)
        else None
    )

    if res.output_tokens is not None and res.ttft_s is not None and res.e2e_s is not None:
        decode_window = max(1e-9, res.e2e_s - res.ttft_s)
        res.decode_toks_per_s = res.output_tokens / decode_window
        res.e2e_toks_per_s = res.output_tokens / res.e2e_s

    res.chars = int(np.median(chars_list)) if chars_list else None
    res.chars_per_s = safe_div(res.chars, res.e2e_s)
    return res


# --------------------------
# Mistral benchmark
# --------------------------
def bench_mistral(model: str, prompt: str, max_new_tokens: int, repeats: int, stream_ttft: bool) -> BenchResult:
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return BenchResult(model=model, backend="skipped", device="api", note="MISTRAL_API_KEY not set")

    api_model = MISTRAL_NAME_MAP.get(model, model)

    try:
        try:
            from mistralai.client import Mistral
        except Exception:
            from mistralai import Mistral
        client = Mistral(api_key=key)
    except Exception as e:
        return BenchResult(model=model, backend="error", device="api", note=f"mistral init failed: {e}")

    e2e_list, out_tok_list, prompt_tok_list, chars_list = [], [], [], []
    ttft_list = []

    for _ in range(repeats):
        t0 = now()
        try:
            resp = client.chat.complete(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
        except Exception as e:
            return BenchResult(model=model, backend="error", device="api", note=f"mistral request failed: {e}")
        t1 = now()

        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ot = getattr(usage, "completion_tokens", None) if usage else None

        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            pass

        e2e_list.append(t1 - t0)
        prompt_tok_list.append(pt)
        out_tok_list.append(ot)
        chars_list.append(len(text))

        if stream_ttft:
            tS = now()
            first = None
            try:
                stream = client.chat.stream(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=min(32, max_new_tokens),
                )
                for _chunk in stream:
                    if first is None:
                        first = now()
                        break
            except Exception:
                first = None
            tE = now()
            if first is None:
                first = tE
            ttft_list.append(first - tS)

    res = BenchResult(model=model, backend="mistral", device="api", note=f"api_model={api_model}")
    res.e2e_s = median_or_none(e2e_list)

    res.prompt_tokens = (
        int(np.median([x for x in prompt_tok_list if isinstance(x, (int, float))]))
        if any(isinstance(x, (int, float)) for x in prompt_tok_list)
        else None
    )
    res.output_tokens = (
        int(np.median([x for x in out_tok_list if isinstance(x, (int, float))]))
        if any(isinstance(x, (int, float)) for x in out_tok_list)
        else None
    )

    if res.output_tokens is not None and res.e2e_s is not None:
        res.e2e_toks_per_s = res.output_tokens / res.e2e_s
        if res.decode_toks_per_s is None:
            res.decode_toks_per_s = res.e2e_toks_per_s

    res.chars = int(np.median(chars_list)) if chars_list else None
    res.chars_per_s = safe_div(res.chars, res.e2e_s)

    if stream_ttft:
        res.ttft_s = median_or_none(ttft_list)
        if res.output_tokens is not None and res.ttft_s is not None and res.e2e_s is not None:
            decode_window = max(1e-9, res.e2e_s - res.ttft_s)
            res.decode_toks_per_s = res.output_tokens / decode_window

    return res


# --------------------------
# Summary: service rate per model
# --------------------------
def compute_service_rate_summary(df_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, backend), g in df_runs.groupby(["model", "backend"], dropna=False):
        if str(backend) == "error":
            continue

        e2e_col = pd.to_numeric(g["e2e_s"], errors="coerce")
        w_col = pd.to_numeric(g.get("prompt_weight", 1.0), errors="coerce").fillna(1.0)

        mask = e2e_col.notna() & np.isfinite(e2e_col)
        if int(mask.sum()) == 0:
            continue

        e2e = e2e_col[mask].astype(float).tolist()
        w = w_col[mask].astype(float).tolist()

        mean_s = weighted_mean_or_none(e2e, w)
        mu = (1.0 / mean_s) if (mean_s is not None and mean_s > 0) else None

        service_times = [x for x in e2e if x is not None and np.isfinite(x)]
        p50_s = float(np.median(service_times)) if service_times else None
        p90_s = float(np.quantile(service_times, 0.90)) if service_times else None

        mu_list = [(1.0 / x) for x in service_times if x > 0]
        mu_p50 = float(np.median(mu_list)) if mu_list else None

        e2e_tps_list = [
            x for x in pd.to_numeric(g.loc[mask, "e2e_toks_per_s"], errors="coerce").tolist()
            if x is not None and np.isfinite(x)
        ]
        decode_tps_list = [
            x for x in pd.to_numeric(g.loc[mask, "decode_toks_per_s"], errors="coerce").tolist()
            if x is not None and np.isfinite(x)
        ]
        ttft_list = [
            x for x in pd.to_numeric(g.loc[mask, "ttft_s"], errors="coerce").tolist()
            if x is not None and np.isfinite(x)
        ]
        cost_list = [
            x for x in pd.to_numeric(g.loc[mask, "total_cost_usd"], errors="coerce").tolist()
            if x is not None and np.isfinite(x)
        ]

        rows.append({
            "model": model,
            "backend": backend,
            "n_prompts": int(g.loc[mask, "prompt_id"].nunique()) if "prompt_id" in g.columns else None,
            "n_rows": int(mask.sum()),
            "mean_service_time_s_weighted": mean_s,
            "service_rate_mu_rps": mu,
            "service_time_p50_s": p50_s,
            "service_time_p90_s": p90_s,
            "service_rate_p50_rps": mu_p50,
            "ttft_p50_s": float(np.median(ttft_list)) if ttft_list else None,
            "e2e_toks_per_s_p50": float(np.median(e2e_tps_list)) if e2e_tps_list else None,
            "decode_toks_per_s_p50": float(np.median(decode_tps_list)) if decode_tps_list else None,
            "total_cost_usd_p50": float(np.median(cost_list)) if cost_list else None,
            "total_cost_usd_mean": float(np.mean(cost_list)) if cost_list else None,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["backend", "service_rate_mu_rps"], ascending=[True, False])
    return out


def maybe_plot_summary(summary_csv: str, out_png: str, topk: int):
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    df["service_rate_mu_rps"] = pd.to_numeric(df["service_rate_mu_rps"], errors="coerce")
    df["mean_service_time_s_weighted"] = pd.to_numeric(df["mean_service_time_s_weighted"], errors="coerce")
    df = df.dropna(subset=["service_rate_mu_rps"]).sort_values("service_rate_mu_rps", ascending=False).head(topk)

    if df.empty:
        print("Plot skipped: no valid service_rate_mu_rps rows.")
        return

    plt.figure()
    plt.barh(df["model"], df["service_rate_mu_rps"])
    plt.xlabel("Estimated service rate μ (req/s)  ~  1 / E[e2e_s]")
    plt.ylabel("Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot: {out_png}")

    out2 = out_png.replace(".png", "_service_time.png")
    plt.figure()
    plt.barh(df["model"], df["mean_service_time_s_weighted"])
    plt.xlabel("Weighted mean service time E[e2e_s] (seconds)")
    plt.ylabel("Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print(f"Saved plot: {out2}")


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--prompt", type=str, default="Write a short paragraph about queueing theory in LLM serving.")
    ap.add_argument("--prompts_file", type=str, default=None)
    ap.add_argument("--prompt_sep", type=str, default="---")
    ap.add_argument("--prompt_samples", type=int, default=0, help="If >0, sample this many prompts (weighted) from prompts_file.")
    ap.add_argument("--prompt_seed", type=int, default=123)

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2, help="Repeats per (model,prompt).")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--mistral_stream_ttft", action="store_true")

    ap.add_argument("--out", type=str, default="decode_speed_runs.csv")
    ap.add_argument("--summary_out", type=str, default="service_rate_summary.csv")

    ap.add_argument("--plot_png", type=str, default="", help="If set, save plots to this PNG path.")
    ap.add_argument("--plot_topk", type=int, default=30)

    args = ap.parse_args()

    prompt_items = load_prompts(args.prompt, args.prompts_file, args.prompt_sep)
    if args.prompt_samples and args.prompt_samples > 0:
        prompt_items = sample_prompts(prompt_items, args.prompt_samples, args.prompt_seed)

    results: List[BenchResult] = []

    for m in args.models:
        backend = infer_backend(m)
        for p_item in prompt_items:
            prompt = p_item.prompt

            try:
                if backend == "hf":
                    r = bench_hf(
                        model_id=m,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        device=args.device,
                        dtype=args.dtype,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                elif backend == "openai":
                    r = bench_openai(m, prompt, args.max_new_tokens, args.repeats)
                elif backend == "together":
                    r = bench_together(m, prompt, args.max_new_tokens, args.repeats)
                elif backend == "anthropic":
                    r = bench_anthropic(m, prompt, args.max_new_tokens, args.repeats)
                elif backend == "mistral":
                    r = bench_mistral(m, prompt, args.max_new_tokens, args.repeats, args.mistral_stream_ttft)
                else:
                    r = BenchResult(model=m, backend="skipped", device="n/a", note="unknown backend")
            except Exception as e:
                r = BenchResult(model=m, backend="error", device="n/a", note=str(e))

            # Preserve the original model name used in args, even if API alias was used internally.
            # This keeps rows aligned with MODEL_NAMES / Config.MODEL_NAMES order.
            r.model = m
            r.prompt_id = int(p_item.prompt_id)
            r.prompt_name = (p_item.name or "").strip()
            r.prompt_hash = short_hash(prompt)
            r.prompt_len_chars = int(len(prompt))
            r.prompt_weight = float(p_item.weight) if p_item.weight is not None else 1.0
            r.service_rate_rps = safe_div(1.0, r.e2e_s)

            (
                r.price_in_per_token,
                r.price_out_per_token,
                r.input_cost_usd,
                r.output_cost_usd,
                r.total_cost_usd,
            ) = estimate_cost_usd(r.model, r.prompt_tokens, r.output_tokens)

            results.append(r)

            print(
                f"[{r.backend}] {r.model} prompt#{r.prompt_id} "
                f"e2e_s={r.e2e_s} mu_rps={r.service_rate_rps} "
                f"e2e_toks/s={r.e2e_toks_per_s} decode_toks/s={r.decode_toks_per_s} "
                f"ttft={r.ttft_s} cost_usd={r.total_cost_usd} note={r.note}"
            )

    df = pd.DataFrame([asdict(r) for r in results])

    if {"decode_toks_per_s", "output_tokens", "e2e_s", "ttft_s"}.issubset(df.columns):
        mask = (
            df["decode_toks_per_s"].isna()
            & df["output_tokens"].notna()
            & df["e2e_s"].notna()
            & df["ttft_s"].notna()
        )
        df.loc[mask, "decode_toks_per_s"] = (
            df.loc[mask, "output_tokens"]
            / (df.loc[mask, "e2e_s"] - df.loc[mask, "ttft_s"]).clip(lower=1e-9)
        )

    if {"decode_toks_per_s", "output_tokens", "e2e_s"}.issubset(df.columns):
        mask2 = df["decode_toks_per_s"].isna() & df["output_tokens"].notna() & df["e2e_s"].notna()
        df.loc[mask2, "decode_toks_per_s"] = (
            df.loc[mask2, "output_tokens"] / df.loc[mask2, "e2e_s"].clip(lower=1e-9)
        )
        if "note" in df.columns:
            df.loc[mask2, "note"] = (
                df.loc[mask2, "note"].fillna("").astype(str)
                + "; decode_tps≈out/e2e(no_ttft)"
            ).str.lstrip("; ")

    df.to_csv(args.out, index=False)
    print(f"\nSaved per-run CSV: {args.out}")

    summary = compute_service_rate_summary(df)
    summary.to_csv(args.summary_out, index=False)
    print(f"Saved summary CSV: {args.summary_out}\n")

    print("=== Summary (top 30 by service_rate_mu_rps) ===")
    if not summary.empty and "service_rate_mu_rps" in summary.columns:
        tmp = summary.copy()
        tmp["service_rate_mu_rps"] = pd.to_numeric(tmp["service_rate_mu_rps"], errors="coerce")
        tmp = tmp.sort_values("service_rate_mu_rps", ascending=False).head(30)
        print(tmp.replace({np.nan: ""}).to_string(index=False))
    else:
        print(summary.replace({np.nan: ""}).to_string(index=False))

    if args.plot_png:
        maybe_plot_summary(args.summary_out, args.plot_png, args.plot_topk)


if __name__ == "__main__":
    main()
