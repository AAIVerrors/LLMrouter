#!/usr/bin/env python3
"""
Decode speed benchmark for:
- Local HF models (transformers)  -> prefill + KV-cache decode loop (tokens/sec)
- OpenAI / Anthropic / Mistral APIs -> TTFT + e2e tokens/sec (and decode tokens/sec when TTFT+usage available)

Fixes included:
✅ OpenAI streaming final chunk can have choices=[] (prevents "list index out of range")
✅ Mistral HF-style names mapped to API model IDs (open-mistral-7b / open-mixtral-8x7b)
✅ Robust metrics: TTFT, e2e_s, e2e_toks/s, decode_toks/s (= output_tokens / (e2e - ttft)) when possible
"""

import os
import gc
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# --------------------------
# Your models (edit as needed)
# --------------------------
MODEL_NAMES = [
    "allenai/Llama-3.1-Tulu-3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-1.1-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "gpt-4o-mini-2024-07-18",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
    "ministral-8b-2410",
    "mistral-7b-instruct-v0.2",
    "mixtral-8x7b-instruct-v0.1",
    "mistral-medium",
]

# HF-ish -> Mistral API model IDs
MISTRAL_NAME_MAP = {
    "mistral-7b-instruct-v0.2": "open-mistral-7b",
    "mixtral-8x7b-instruct-v0.1": "open-mixtral-8x7b",
}


# --------------------------
# Utils
# --------------------------
def now() -> float:
    return time.perf_counter()

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if b <= 0:
        return None
    return float(a) / float(b)

def median_or_none(xs: List[Optional[float]]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs2:
        return None
    return float(np.median(xs2))

def infer_backend(model: str) -> str:
    m = model.lower()
    if "/" in model:
        return "hf"
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if ("mistral" in m) or ("mixtral" in m) or ("ministral" in m):
        return "mistral"
    return "hf"


@dataclass
class BenchResult:
    model: str
    backend: str
    device: str

    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Local HF timing
    prefill_s: Optional[float] = None
    decode_s: Optional[float] = None
    decode_toks_per_s: Optional[float] = None

    # General timing
    ttft_s: Optional[float] = None
    e2e_s: Optional[float] = None
    e2e_toks_per_s: Optional[float] = None

    # API fallback
    chars: Optional[int] = None
    chars_per_s: Optional[float] = None

    note: str = ""


# --------------------------
# HF benchmark (KV cache decode)
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

    # Put inputs on same device as first param (device_map="auto" safe)
    first_param = next(model.parameters())
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}

    res.prompt_tokens = int(inputs["input_ids"].shape[-1])
    res.output_tokens = int(max_new_tokens)

    # Warmup with generate()
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

            # Prefill: prompt forward
            t0 = now()
            out = model(**inputs, use_cache=True)
            if device != "cpu":
                torch.cuda.synchronize()
            t1 = now()

            prefill_s = t1 - t0
            past = out.past_key_values

            # First token from prefill logits (counts as TTFT-ish for local)
            next_tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            t_first = now()

            # Decode loop (KV cache): remaining (max_new_tokens-1)
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

    # decode_toks/s counts the KV loop tokens (max_new_tokens-1)
    decode_tokens = max(0, max_new_tokens - 1)
    res.decode_toks_per_s = safe_div(decode_tokens, res.decode_s)
    res.e2e_toks_per_s = safe_div(max_new_tokens, res.e2e_s)

    # cleanup
    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return res


# --------------------------
# OpenAI benchmark (stream, handle choices=[])
# --------------------------
def bench_openai(
    model: str,
    prompt: str,
    max_new_tokens: int,
    repeats: int,
) -> BenchResult:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return BenchResult(model=model, backend="skipped", device="api", note="OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
    except Exception as e:
        return BenchResult(model=model, backend="error", device="api", note=f"openai init failed: {e}")

    ttft_list, e2e_list, out_tok_list, prompt_tok_list, chars_list, decode_tps_list, e2e_tps_list = [], [], [], [], [], [], []

    for _ in range(repeats):
        t0 = now()
        first_content_time = None
        text = ""
        usage = None

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},  # final chunk may have choices=[]
            )

            for chunk in stream:
                # content chunks (choices may be empty on final usage chunk!)
                choices = getattr(chunk, "choices", None) or []
                for ch in choices:
                    delta = getattr(ch, "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        if first_content_time is None:
                            first_content_time = now()
                        text += content

                # usage appears on final chunk if include_usage=True
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage

        except Exception as e:
            return BenchResult(model=model, backend="error", device="api", note=f"openai request failed: {e}")

        t1 = now()
        if first_content_time is None:
            first_content_time = t1  # fallback if no streamed content

        ttft = first_content_time - t0
        e2e = t1 - t0
        decode_window = max(1e-9, t1 - first_content_time)

        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ot = getattr(usage, "completion_tokens", None) if usage else None

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

    res = BenchResult(model=model, backend="openai", device="api", note="stream include_usage; fixed empty choices")
    res.ttft_s = median_or_none(ttft_list)
    res.e2e_s = median_or_none(e2e_list)
    res.prompt_tokens = int(np.median([x for x in prompt_tok_list if isinstance(x, (int, float))])) if any(isinstance(x, (int, float)) for x in prompt_tok_list) else None
    res.output_tokens = int(np.median([x for x in out_tok_list if isinstance(x, (int, float))])) if any(isinstance(x, (int, float)) for x in out_tok_list) else None
    res.decode_toks_per_s = median_or_none(decode_tps_list)
    res.e2e_toks_per_s = median_or_none(e2e_tps_list)

    # chars/sec fallback (use e2e window)
    res.chars = int(np.median(chars_list)) if chars_list else None
    res.chars_per_s = safe_div(res.chars, res.e2e_s)
    if res.output_tokens is None:
        res.note += "; token usage missing -> chars/sec fallback"

    return res


# --------------------------
# Anthropic benchmark (stream; usage in final message)
# --------------------------
def bench_anthropic(
    model: str,
    prompt: str,
    max_new_tokens: int,
    repeats: int,
) -> BenchResult:
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
                temperature=0.0,
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

    res = BenchResult(model=model, backend="anthropic", device="api", note="tokens from final.usage if available")
    res.ttft_s = median_or_none(ttft_list)
    res.e2e_s = median_or_none(e2e_list)

    res.prompt_tokens = int(np.median([x for x in in_tok_list if isinstance(x, (int, float))])) if any(isinstance(x, (int, float)) for x in in_tok_list) else None
    res.output_tokens = int(np.median([x for x in out_tok_list if isinstance(x, (int, float))])) if any(isinstance(x, (int, float)) for x in out_tok_list) else None

    # decode speed if both output_tokens and ttft/e2e exist
    if res.output_tokens is not None and res.ttft_s is not None and res.e2e_s is not None:
        decode_window = max(1e-9, res.e2e_s - res.ttft_s)
        res.decode_toks_per_s = res.output_tokens / decode_window
        res.e2e_toks_per_s = res.output_tokens / res.e2e_s

    res.chars = int(np.median(chars_list)) if chars_list else None
    res.chars_per_s = safe_div(res.chars, res.e2e_s)
    return res


# --------------------------
# Mistral benchmark (non-stream for usage; optional stream TTFT)
# --------------------------
def bench_mistral(
    model: str,
    prompt: str,
    max_new_tokens: int,
    repeats: int,
    stream_ttft: bool,
) -> BenchResult:
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        return BenchResult(model=model, backend="skipped", device="api", note="MISTRAL_API_KEY not set")

    api_model = MISTRAL_NAME_MAP.get(model, model)

    try:
        from mistralai import Mistral
        client = Mistral(api_key=key)
    except Exception as e:
        return BenchResult(model=model, backend="error", device="api", note=f"mistral init failed: {e}")

    e2e_list, out_tok_list, prompt_tok_list, chars_list = [], [], [], []
    ttft_list = []

    for _ in range(repeats):
        # Non-stream (reliable usage)
        t0 = now()
        try:
            resp = client.chat.complete(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
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

        # Optional: separate streaming call only to measure TTFT
        if stream_ttft:
            tS = now()
            first = None
            try:
                stream = client.chat.stream(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=min(32, max_new_tokens),
                    temperature=0.0,
                )
                for chunk in stream:
                    if first is None:
                        # any first chunk arrival is a TTFT proxy
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

    res.prompt_tokens = int(np.median([x for x in prompt_tok_list if isinstance(x, (int, float))])) if any(isinstance(x, (int, float)) for x in prompt_tok_list) else None
    res.output_tokens = int(np.median([x for x in out_tok_list if isinstance(x, (int, float))])) if any(isinstance(x, (int, float)) for x in out_tok_list) else None

    if res.output_tokens is not None and res.e2e_s is not None:
        res.e2e_toks_per_s = res.output_tokens / res.e2e_s

    res.chars = int(np.median(chars_list)) if chars_list else None
    res.chars_per_s = safe_div(res.chars, res.e2e_s)

    if stream_ttft:
        res.ttft_s = median_or_none(ttft_list)
        if res.output_tokens is not None and res.ttft_s is not None and res.e2e_s is not None:
            # approximate decode tps using (e2e - ttft)
            decode_window = max(1e-9, res.e2e_s - res.ttft_s)
            res.decode_toks_per_s = res.output_tokens / decode_window

    return res


# --------------------------
# Run all + save CSV
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--prompt", type=str, default="Write a short paragraph about queueing theory in LLM serving.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--mistral_stream_ttft", action="store_true", help="Extra stream call to estimate TTFT for Mistral")
    ap.add_argument("--out", type=str, default="decode_speed_results.csv")
    args = ap.parse_args()

    results: List[BenchResult] = []

    for m in args.models:
        backend = infer_backend(m)
        try:
            if backend == "hf":
                r = bench_hf(
                    model_id=m,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    dtype=args.dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            elif backend == "openai":
                r = bench_openai(m, args.prompt, args.max_new_tokens, args.repeats)
            elif backend == "anthropic":
                r = bench_anthropic(m, args.prompt, args.max_new_tokens, args.repeats)
            elif backend == "mistral":
                r = bench_mistral(m, args.prompt, args.max_new_tokens, args.repeats, args.mistral_stream_ttft)
            else:
                r = BenchResult(model=m, backend="skipped", device="n/a", note="unknown backend")
        except Exception as e:
            r = BenchResult(model=m, backend="error", device="n/a", note=str(e))

        results.append(r)
        print(f"[{r.backend}] {r.model}  e2e_toks/s={r.e2e_toks_per_s}  decode_toks/s={r.decode_toks_per_s}  ttft={r.ttft_s}  note={r.note}")

    df = pd.DataFrame([asdict(r) for r in results])

    # If decode_toks_per_s missing but we have output_tokens, e2e_s, ttft_s -> fill it
    mask = df["decode_toks_per_s"].isna() & df["output_tokens"].notna() & df["e2e_s"].notna() & df["ttft_s"].notna()
    df.loc[mask, "decode_toks_per_s"] = df.loc[mask, "output_tokens"] / (df.loc[mask, "e2e_s"] - df.loc[mask, "ttft_s"]).clip(lower=1e-9)

    df.to_csv(args.out, index=False)
    print(f"\nSaved results to: {args.out}\n")
    cols = ["model","backend","device","prompt_tokens","output_tokens","prefill_s","decode_toks_per_s","e2e_s","e2e_toks_per_s","ttft_s","chars_per_s","note"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
