# environment_no_inflight_fixed_v2.py
# NOTE: "no inflight" reverted version (capacity/load based on request_queue.qsize()).
# Fixes:
#  - Robust routing for Mistral API vs HF local vs HF Ministral-3 (Mistral3Config)
#  - Correct chat_template handling (Tensor vs BatchEncoding) to avoid AttributeError on .device
#  - skip_special_tokens=True + strip <|...|> to avoid <|end_of_text|> leaking into responses
#  - Better error logging with repr(e) + traceback

import torch
import numpy as np
import time
import math
import re
import string
import json
import hashlib
from collections import Counter
from difflib import SequenceMatcher
from together import Together

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from config import Config
from queueMonitor import QueueUpdateMonitor
import torch.multiprocessing as mp
import os
import traceback
from datetime import datetime
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from queue import Empty
from PoissonPromptGenerator import PoissonPromptGenerator
from quality_model import P2LPredictor
import anthropic
from google import genai
from google.genai import types
from openai import OpenAI
from mistralai.client import Mistral

# Optional: these exist only on newer Transformers versions.
try:
    from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration
except Exception:
    MistralCommonBackend = None
    Mistral3ForConditionalGeneration = None

# Set multiprocessing start method
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set


class TransientAPIError(Exception):
    pass


def _get_status_code_from_exception(e):
    for attr in ("status_code", "status"):
        v = getattr(e, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass

    raw = repr(e)
    m = re.search(r"Status\s+(\d+)", raw)
    if m:
        return int(m.group(1))

    return None


def _get_retry_after_from_exception(e, default=5.0):
    headers = getattr(e, "headers", None)
    if headers is not None:
        try:
            v = headers.get("retry-after") or headers.get("Retry-After")
            if v is not None:
                return float(v)
        except Exception:
            pass

    raw = repr(e)
    m = re.search(r'"retry_after"\s*:\s*(\d+)', raw)
    if m:
        return float(m.group(1))

    m = re.search(r"'retry-after':\s*'(\d+)'", raw)
    if m:
        return float(m.group(1))

    return float(default)


def _is_transient_api_error(e):
    status = _get_status_code_from_exception(e)
    return status in {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}


def mistral_chat_complete_with_retry(client, model_name, prompt, max_retries=2):
    """
    Short retry for PPO training.
    Do not sleep retry_after=60 during training because it blocks the worker/episode.
    """
    last_e = None

    for attempt in range(max_retries + 1):
        try:
            return client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 128),
                temperature=getattr(Config, "GEN_TEMPERATURE", 0.1),
                top_p=getattr(Config, "GEN_TOP_P", 1.0),
            )

        except Exception as e:
            last_e = e

            if not _is_transient_api_error(e):
                raise

            retry_after = _get_retry_after_from_exception(e, default=5.0)

            # During RL training, never block for 60 seconds.
            sleep_s = min(float(retry_after), 5.0) * (attempt + 1)

            if attempt < max_retries:
                print(
                    f"[Mistral transient error] model={model_name}, "
                    f"status={_get_status_code_from_exception(e)}, "
                    f"attempt={attempt + 1}/{max_retries}, sleep={sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
            else:
                raise TransientAPIError(
                    f"Mistral transient API error after retries: {repr(last_e)}"
                )

def _try_flash_attn2(device: Optional[str] = None) -> bool:
    """Return True iff it's worth trying flash_attention_2 on this machine."""
    if device is not None and "cuda" not in str(device):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------
# Ground-truth matching
# --------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_answer(s) -> str:
    """Normalize text for string-based matching."""
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


# --------------------------
# Final-answer extraction (for EM/F1)
# --------------------------
_FINAL_LINE_RE = re.compile(r"^(final\s*answer|final|answer)\s*[:\-]\s*(.*)$", flags=re.I)
_CODEBLOCK_RE = re.compile(r"```[\s\S]*?```", flags=re.S)


def extract_final_answer(text: Optional[str]) -> str:
    """Extract a short final answer span from a model output."""
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # 1) <final>...</final>
    tag = getattr(Config, "FINAL_ANSWER_TAG", "final")
    try:
        tag_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.I | re.S)
        m = tag_re.search(s)
        if m:
            return m.group(1).strip()
    except Exception:
        pass

    # Remove fenced code blocks
    s2 = _CODEBLOCK_RE.sub("", s).strip()
    lines = [ln.strip() for ln in s2.splitlines() if ln.strip()]

    # 2) explicit final/answer lines (scan from bottom)
    for ln in reversed(lines):
        mm = _FINAL_LINE_RE.match(ln)
        if mm:
            return mm.group(2).strip()

    # 3) fallback: last non-empty line
    return lines[-1] if lines else s2


def truncate_at_final_tag(text: str, tag: str = "final") -> str:
    """If </tag> appears, truncate output right after it."""
    if not text:
        return ""
    close = f"</{tag}>"
    idx = text.lower().find(close.lower())
    if idx == -1:
        return text
    return text[: idx + len(close)].strip()


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return [str(x)]


def exact_match_score(pred: Optional[str], gold: Any) -> float:
    """Strict EM on normalized strings (0/1)."""
    pred_n = _normalize_answer(pred)
    best = 0.0
    for g in _as_list(gold):
        best = max(best, float(pred_n == _normalize_answer(g)))
    return best


def token_f1_score(pred: Optional[str], gold: Any) -> float:
    """SQuAD-style token F1 (0..1) on normalized whitespace tokens."""
    pred_toks = _normalize_answer(pred).split()
    best = 0.0
    for g in _as_list(gold):
        gold_toks = _normalize_answer(g).split()
        if len(pred_toks) == 0 or len(gold_toks) == 0:
            best = max(best, float(pred_toks == gold_toks))
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        best = max(best, float(f1))
    return best


def sequence_ratio_score(pred: Optional[str], gold: Any) -> float:
    """Character-level similarity ratio (0..1) on normalized strings."""
    pred_n = _normalize_answer(pred)
    best = 0.0
    for g in _as_list(gold):
        gold_n = _normalize_answer(g)
        if not pred_n and not gold_n:
            best = max(best, 1.0)
            continue
        best = max(best, SequenceMatcher(None, pred_n, gold_n).ratio())
    return float(best)


def contains_score(pred: Optional[str], gold: Any) -> float:
    """1.0 if either string contains the other after normalization, else 0.0."""
    pred_n = _normalize_answer(pred)
    best = 0.0
    for g in _as_list(gold):
        gold_n = _normalize_answer(g)
        if not pred_n or not gold_n:
            continue
        if gold_n in pred_n or pred_n in gold_n:
            best = 1.0
            break
    return best

_CHOICE_LETTER_RE = re.compile(r"^\s*([A-Da-d])\s*([\.\)\]:：]|$|\s+$)")

def _extract_choice_letter(x: Any) -> str:
    if x is None:
        return ""

    s = str(x).strip()
    if not s:
        return ""

    # gold may be 0/1/2/3
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < 4:
            return ["A", "B", "C", "D"][idx]

    # direct A/B/C/D
    if len(s) == 1 and s.upper() in ["A", "B", "C", "D"]:
        return s.upper()

    # "A.", "A)", "A: ..."
    m = _CHOICE_LETTER_RE.match(s)
    if m:
        return m.group(1).upper()

    # "answer is A", "Final answer: A"
    m = re.search(r"(answer|final)\s*(is|:|-)?\s*([A-Da-d])\b", s, flags=re.I)
    if m:
        return m.group(3).upper()

    return ""


def mmlu_choice_score(pred: Optional[str], gold: Any) -> float:
    pred_letter = _extract_choice_letter(pred)

    if not pred_letter:
        return 0.0

    for g in _as_list(gold):
        gold_letter = _extract_choice_letter(g)
        if pred_letter == gold_letter:
            return 1.0

    return 0.0


# def match_quality_score(pred: Optional[str], gold: Any) -> float:
#     """Compute a match score for reward."""
#     if getattr(Config, "EXTRACT_FINAL_ANSWER", True):
#         pred = extract_final_answer(pred)

#     metric = getattr(Config, "EM_METRIC", "f1")
#     if metric == "mmlu":
#         score = mmlu_choice_score(pred, gold)
#     elif metric == "em":
#         score = exact_match_score(pred, gold)
#     elif metric == "ratio":
#         score = sequence_ratio_score(pred, gold)
#     elif metric == "contains":
#         score = contains_score(pred, gold)
#     else:
#         score = token_f1_score(pred, gold)

#     if getattr(Config, "EM_BINARIZE", False):
#         thr = float(getattr(Config, "EM_THRESHOLD", 0.5))
#         score = float(score >= thr)

#     return float(max(min(score, 1.0), 0.0))

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

def _normalize_number_string(x: Any) -> str:
    s = str(x or "").strip()
    if getattr(Config, "EXTRACT_FINAL_ANSWER", True):
        s = extract_final_answer(s)
    s = s.replace(",", "")
    nums = _NUM_RE.findall(s)
    if not nums:
        return _normalize_answer(s)
    v = nums[-1].replace(",", "")
    try:
        f = float(v)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return ("%.10f" % f).rstrip("0").rstrip(".")
    except Exception:
        return v


def numeric_exact_score(pred: Optional[str], gold: Any) -> float:
    pred_n = _normalize_number_string(pred)
    for g in _as_list(gold):
        if pred_n == _normalize_number_string(g):
            return 1.0
    return 0.0


# Replace your current match_quality_score with this version.
def match_quality_score(pred: Optional[str], gold: Any) -> float:
    """Compute a match score for reward.

    Supports mixed dataset gold objects emitted by PoissonPromptGenerator_mixed.py:
      gold = {"answers": ..., "metric": "f1" | "em" | "mmlu" | "number" | ...}
    """
    metric = getattr(Config, "EM_METRIC", "f1")
    gold_answers = gold

    if isinstance(gold, dict):
        metric = str(gold.get("metric", metric)).lower()
        gold_answers = (
            gold.get("answers")
            if "answers" in gold
            else gold.get("answer", gold.get("gold", gold.get("target", "")))
        )

    if getattr(Config, "EXTRACT_FINAL_ANSWER", True):
        pred = extract_final_answer(pred)

    if metric in {"mmlu", "choice", "multiple_choice"}:
        score = mmlu_choice_score(pred, gold_answers)
    elif metric in {"number", "numeric", "gsm8k", "math"}:
        score = numeric_exact_score(pred, gold_answers)
    elif metric == "em":
        score = exact_match_score(pred, gold_answers)
    elif metric == "ratio":
        score = sequence_ratio_score(pred, gold_answers)
    elif metric == "contains":
        score = contains_score(pred, gold_answers)
    else:
        score = token_f1_score(pred, gold_answers)

    if getattr(Config, "EM_BINARIZE", False):
        thr = float(getattr(Config, "EM_THRESHOLD", 0.5))
        score = float(score >= thr)

    return float(max(min(score, 1.0), 0.0))


@dataclass
class Request:
    """Represents a single request in the system."""
    id: int
    prompt: str
    arrival_time: float
    ground_truth: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    processing_latency: Optional[float] = None
    processing_latency_raw: Optional[float] = None
    processing_latency_clipped: Optional[float] = None
    quality_score: Optional[float] = None
    server_id: Optional[int] = None
    response: Optional[Dict] = None
    status: str = "pending"
    reward: Optional[float] = 0.0
    episode: Optional[int] = None
    price: Optional[float] = None
    price_raw: Optional[float] = None


def _is_mistral_api_name(model_name: str) -> bool:
    """True for Mistral API model IDs (no HF repo slash)."""
    m = (model_name or "").lower()
    return ("/" not in m) and any(k in m for k in ("mistral", "mixtral", "ministral", "magistral", "codestral"))


def _is_mistral3_hf_model(model_name: str) -> bool:
    """Robustly detect Ministral-3 HF models (Mistral3Config/Ministral3Config)."""
    if "/" not in (model_name or ""):
        return False
    m = model_name.lower()
    if "ministral-3" in m:
        return True
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return cfg.__class__.__name__ in ("Mistral3Config", "Ministral3Config")
    except Exception:
        return False

def _is_together_api_name(model_name: str) -> bool:
    """True for Together-served remote models explicitly prefixed as together/<provider>/<model>."""
    return (model_name or "").lower().startswith("together/")


def _strip_together_prefix(model_name: str) -> str:
    """Convert together/<provider>/<model> -> <provider>/<model>."""
    if _is_together_api_name(model_name):
        return model_name.split("together/", 1)[1]
    return model_name


def _get_openai_like_text_response(client, model_name: str, prompt: str) -> str:
    kwargs = dict(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256),
        temperature=getattr(Config, "GEN_TEMPERATURE", 0.7),
        top_p=getattr(Config, "GEN_TOP_P", 0.95),
    )

    if model_name.startswith("gpt-5.4"):
        kwargs["reasoning_effort"] = "none"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _build_hf_inputs(tokenizer, model, prompt: str) -> Dict[str, torch.Tensor]:
    """Build (input_ids, attention_mask) on model.device. Handles chat templates robustly."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # enc may be Tensor or BatchEncoding
        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            # BatchEncoding(dict-like)
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            return {k: v.to(model.device) for k, v in enc.items() if torch.is_tensor(v)}
    else:
        return tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)



def _pick_hf_dtype() -> torch.dtype:
    """Choose default dtype for local HF models."""
    dt = str(getattr(Config, "LOCAL_HF_DTYPE", "float16")).lower()
    if dt in ("bf16", "bfloat16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dt in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def _pick_attn_impl(prefer_fa2: bool = True) -> Optional[str]:
    """Pick attention implementation: flash_attention_2 -> sdpa/eager -> None."""
    if not torch.cuda.is_available():
        return None
    if prefer_fa2 and bool(getattr(Config, "USE_FLASH_ATTN_2", True)):
        return "flash_attention_2"
    forced = getattr(Config, "FORCE_ATTN_IMPL", None)
    if forced:
        return str(forced)
    return None


def _load_hf_causallm(model_name: str):
    """Load a local HF CausalLM with best-effort FlashAttention-2 and clean fallback."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_kwargs = dict(
        torch_dtype=_pick_hf_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )

    attn = _pick_attn_impl(prefer_fa2=True)
    try:
        if attn is not None:
            model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn, **base_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **base_kwargs)
    except Exception as e:
        fb = getattr(Config, "FLASH_ATTN_FALLBACK", "sdpa")
        fb = str(fb).lower() if fb is not None else ""
        print(f"[{model_name}] attn={attn} failed; fallback={fb}. ({type(e).__name__}: {e})")
        if fb in ("sdpa", "eager"):
            model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=fb, **base_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **base_kwargs)

    model.eval()
    try:
        print(f"[{model_name}] loaded CausalLM attn_impl={getattr(model.config, 'attn_implementation', None)} dtype={getattr(model, 'dtype', None)}")
    except Exception:
        pass
    return tokenizer, model


def _load_hf_seq2seq(model_name: str):
    """Load a local HF Seq2Seq with best-effort FlashAttention-2 and clean fallback."""
    base_kwargs = dict(
        torch_dtype=_pick_hf_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )
    attn = _pick_attn_impl(prefer_fa2=True)
    try:
        if attn is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation=attn, **base_kwargs)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **base_kwargs)
    except Exception as e:
        fb = getattr(Config, "FLASH_ATTN_FALLBACK", "sdpa")
        fb = str(fb).lower() if fb is not None else ""
        print(f"[{model_name}] attn={attn} failed; fallback={fb}. ({type(e).__name__}: {e})")
        if fb in ("sdpa", "eager"):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation=fb, **base_kwargs)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **base_kwargs)

    model.eval()
    try:
        print(f"[{model_name}] loaded Seq2Seq attn_impl={getattr(model.config, 'attn_implementation', None)} dtype={getattr(model, 'dtype', None)}")
    except Exception:
        pass
    return model
def server_worker_process(
    model_name: str,
    capacity: int,
    server_id: int,
    request_queue,
    response_queue,
    running,
    gpu_id: Optional[int] = None,
    queue_monitor: Optional[QueueUpdateMonitor] = None,
    pause_event=None,
):
    """Worker function for server process (no inflight counter)."""
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)

        is_ministral3 = _is_mistral3_hf_model(model_name)

        tokenizer = None
        model = None
        client = None  # for Mistral API branch
        together_client = None

        # -----------------------------
        # Model / tokenizer init
        # -----------------------------
        if "t5" in model_name.lower() and (not _is_together_api_name(model_name)):
            model = _load_hf_seq2seq(model_name)

        elif _is_together_api_name(model_name):
            together_client = Together(api_key=os.environ["TOGETHER_API_KEY"], timeout=30000)

        elif any(k in model_name.lower() for k in ("gpt", "o1", "o3")):
            model = OpenAI(
                timeout=float(getattr(Config, "GEN_REQUEST_TIMEOUT", 60)),
                max_retries=0,  
            )

        elif "gemini" in model_name.lower():
            model = genai.Client(api_key=os.environ["GEMINI_API"])

        elif "claude" in model_name.lower():
            model = anthropic.Anthropic()

        elif _is_mistral_api_name(model_name):
            client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=30000)

        else:
            # -----------------------------
            # Local HF models
            # -----------------------------
            if is_ministral3:
                if MistralCommonBackend is None or Mistral3ForConditionalGeneration is None:
                    raise RuntimeError(
                        f"Ministral-3 local HF requires newer transformers/mistral-common. "
                        f"Missing MistralCommonBackend/Mistral3ForConditionalGeneration for {model_name}."
                    )
                dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                tokenizer = MistralCommonBackend.from_pretrained(model_name)
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True
                )
                model.eval()
            else:
                tokenizer, model = _load_hf_causallm(model_name)

        print(f"Server {server_id} ({model_name}) initialized on GPU {gpu_id}")

        avg_processing_time = 0.0
        n_completed = 0

        while running.value:
            while pause_event is not None and pause_event.is_set():
                time.sleep(0.1)
                continue

            try:
                current_queue_length = request_queue.qsize()
                queue_state_before = {
                    "current_load": current_queue_length,
                    "utilization": current_queue_length / max(float(capacity), 1.0),
                    "pending_completions": 0,
                    "avg_processing_time": avg_processing_time,
                }

                request = request_queue.get_nowait()
                if request is None:
                    continue

                request.status = "processing"
                request.start_time = time.time()

                queue_state_added = {
                    "current_load": current_queue_length + 1,
                    "utilization": (current_queue_length + 1) / max(float(capacity), 1.0),
                    "pending_completions": 0,
                    "avg_processing_time": avg_processing_time,
                }

                if queue_monitor is not None:
                    queue_monitor.log_request_added(
                        server_id=request.server_id,
                        request_id=request.id,
                        prompt=request.prompt,
                        current_time=request.start_time,
                        processing_latency=None,
                        quality_score=request.quality_score,
                        queue_state_before=queue_state_before,
                        queue_state_after=queue_state_added,
                        episode=request.episode,
                    )

                try:
                    response_text = ""
                    gen_tokens = None

                    # -----------------------------
                    # API models
                    # -----------------------------
                    if _is_together_api_name(model_name):
                        response = together_client.chat.completions.create(
                            model=_strip_together_prefix(model_name),
                            messages=[{"role": "user", "content": request.prompt}],
                            max_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256),
                            temperature=getattr(Config, "GEN_TEMPERATURE", getattr(Config, "TEMPERATURE", 0.7)),
                            top_p=getattr(Config, "GEN_TOP_P", 0.7),
                            reasoning={"enabled": False},
                        )
                        response_text = response.choices[0].message.content

                    elif any(k in model_name.lower() for k in ("gpt", "o1", "o3")):
                        response_text = _get_openai_like_text_response(
                            model,
                            model_name,
                            request.prompt,
                        )

                    elif "gemini" in model_name.lower():
                        mname = model_name
                        # if mname == "gemini-1.5-flash-001":
                        #     mname = "gemini-1.5-flash"
                        # elif mname == "gemini-2.0-flash-exp":
                        #     mname = "gemini-2.0-flash-lite"
                        # elif mname == "gemini-1.5-flash-8b-001":
                        #     mname = "gemini-1.5-flash-8b"
                        msg = model.models.generate_content(
                            model=mname,
                            contents=request.prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256)
                            ),
                        )
                        response_text = msg.text

                    elif _is_mistral_api_name(model_name):
                        api_model = model_name
                        # if model_name == "mistral-7b-instruct-v0.2":
                        #     api_model = "open-mistral-7b"
                        # elif model_name == "mistral-medium":
                        #     api_model = "mistral-medium-latest"
                        # elif model_name == "mistral-small-24b-instruct-2501":
                        #     api_model = "mistral-small-2501"
                        # elif model_name == "mixtral-8x7b-instruct-v0.1":
                        #     api_model = "open-mixtral-8x7b"

                        # chat_response = client.chat.complete(
                        #     model=api_model,
                        #     max_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256),
                        #     temperature=getattr(Config, "GEN_TEMPERATURE", getattr(Config, "TEMPERATURE", 0.7)),
                        #     top_p=getattr(Config, "GEN_TOP_P", 0.95),
                        #     messages=[{"role": "user", "content": request.prompt}],
                        # )
                        chat_response = mistral_chat_complete_with_retry(
                            client=client,
                            model_name=model_name,
                            prompt=request.prompt,
                            max_retries=int(getattr(Config, "MISTRAL_MAX_RETRIES", 2)),
                        )
                        response_text = chat_response.choices[0].message.content

                    elif "claude" in model_name.lower():
                        message = model.messages.create(
                            model=model_name,
                            max_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256),
                            messages=[{
                                "role": "user",
                                "content": [{"type": "text", "text": request.prompt}]
                            }],
                        )
                        response_text = message.content[0].text

                    # -----------------------------
                    # Local HF models
                    # -----------------------------
                    else:
                        if is_ministral3:
                            messages = [{"role": "user", "content": [{"type": "text", "text": request.prompt}]}]
                            tokenized = tokenizer.apply_chat_template(
                                messages,
                                return_tensors="pt",
                                return_dict=True,
                            )

                            dev = next(model.parameters()).device
                            for k, v in list(tokenized.items()):
                                if torch.is_tensor(v):
                                    if k == "pixel_values":
                                        tokenized[k] = v.to(device=dev, dtype=model.dtype)
                                    else:
                                        tokenized[k] = v.to(device=dev)

                            input_len = int(tokenized["input_ids"].shape[1])

                            with torch.no_grad():
                                with torch.amp.autocast("cuda", dtype=model.dtype):
                                    outputs = model.generate(
                                        **tokenized,
                                        max_new_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256),
                                        temperature=getattr(Config, "GEN_TEMPERATURE", getattr(Config, "TEMPERATURE", 0.7)),
                                        top_p=getattr(Config, "GEN_TOP_P", 0.95),
                                        do_sample=getattr(Config, "GEN_DO_SAMPLE", True),
                                        use_cache=True,
                                        num_return_sequences=1,
                                    )

                            new_tokens = outputs[0, input_len:]
                            gen_tokens = int(new_tokens.numel())
                            response_text = tokenizer.decode(
                                new_tokens,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                            response_text = re.sub(r"<\|[^|]+\|>", "", response_text).strip()

                        else:
                            inputs = _build_hf_inputs(tokenizer, model, request.prompt)

                            if inputs["input_ids"].size(0) == 0 or inputs["input_ids"].size(1) == 0:
                                raise ValueError("Empty input tensor")

                            with torch.no_grad():
                                with torch.amp.autocast("cuda", dtype=torch.float16):
                                    outputs = model.generate(
                                        input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        max_new_tokens=getattr(Config, "GEN_MAX_NEW_TOKENS", 256),
                                        temperature=getattr(Config, "GEN_TEMPERATURE", getattr(Config, "TEMPERATURE", 0.7)),
                                        top_p=getattr(Config, "GEN_TOP_P", 0.95),
                                        do_sample=getattr(Config, "GEN_DO_SAMPLE", True),
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        use_cache=True,
                                        num_return_sequences=1,
                                    )

                            input_length = inputs["input_ids"].size(1)
                            if outputs.size(1) <= input_length:
                                raise ValueError("No new tokens generated")

                            new_tokens = outputs[0, input_length:]
                            gen_tokens = int(new_tokens.numel())
                            response_text = tokenizer.decode(
                                new_tokens,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                            response_text = re.sub(r"<\|[^|]+\|>", "", response_text).strip()

                            if not response_text.strip():
                                raise ValueError("Empty response after decoding")

                    # -----------------------------
                    # Completion info
                    # -----------------------------
                    request.completion_time = time.time()
                    raw_lat = max(float(request.completion_time - request.arrival_time), 0.0)
                    round_minmax_lat = bool(getattr(Config, "ROUND_MINMAX_NORM_ENABLE", False))
                    request.processing_latency_raw = raw_lat
                    request.processing_latency_clipped = float(min(raw_lat, Config.MAX_LAT))
                    request.processing_latency = raw_lat if round_minmax_lat else request.processing_latency_clipped
                    request.status = "completed"

                    # Post-process outputs for QA routing
                    response_text_raw = response_text
                    try:
                        if getattr(Config, "TRUNCATE_AT_FINAL_TAG", True) and getattr(Config, "QA_FORCE_FINAL_TAG", False):
                            response_text = truncate_at_final_tag(
                                response_text,
                                tag=getattr(Config, "FINAL_ANSWER_TAG", "final"),
                            )
                        final_answer = extract_final_answer(response_text) if getattr(Config, "EXTRACT_FINAL_ANSWER", True) else response_text
                        if getattr(Config, "OUTPUT_FINAL_ONLY", False) and getattr(Config, "QA_FORCE_FINAL_TAG", False):
                            tag = getattr(Config, "FINAL_ANSWER_TAG", "final")
                            response_text = f"<{tag}>{final_answer}</{tag}>"
                        print(response_text)
                    except Exception:
                        final_answer = extract_final_answer(response_text_raw) if getattr(Config, "EXTRACT_FINAL_ANSWER", True) else response_text_raw

                    request.response = {
                        "response_text": response_text,
                        "response_text_raw": response_text_raw,
                        "final_answer": final_answer,
                        "decode_time": request.completion_time - request.start_time,
                        "tokens_generated": gen_tokens if gen_tokens is not None else len(response_text),
                    }

                    n_completed += 1
                    avg_processing_time = ((avg_processing_time * (n_completed - 1)) + float(request.processing_latency)) / float(n_completed)

                    queue_state_final = {
                        "current_load": request_queue.qsize(),
                        "utilization": request_queue.qsize() / max(float(capacity), 1.0),
                        "pending_completions": 0,
                        "avg_processing_time": avg_processing_time,
                    }

                    response_queue.put([request, queue_state_added, queue_state_final])

                except Exception as e:
                    err_msg = (
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[Server {server_id} | {model_name}] "
                        f"Error processing request {request.id}: {repr(e)}\n"
                    )
                    tb_msg = traceback.format_exc()

                    print(err_msg.strip())
                    print(tb_msg)

                    log_dir = "logs"
                    os.makedirs(log_dir, exist_ok=True)
                    log_file = os.path.join(log_dir, "error_log.txt")

                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(err_msg)
                        f.write(tb_msg)
                        f.write("\n" + "=" * 80 + "\n\n")

                    # request.status = "failed"
                    if isinstance(e, TransientAPIError) or _is_transient_api_error(e):
                        request.status = "api_transient_failed"
                    else:
                        request.status = "failed"
                    request.completion_time = time.time()
                    raw_lat = max(float(request.completion_time - request.arrival_time), 0.0)
                    round_minmax_lat = bool(getattr(Config, "ROUND_MINMAX_NORM_ENABLE", False))
                    request.processing_latency_raw = raw_lat
                    request.processing_latency_clipped = float(min(raw_lat, Config.MAX_LAT))
                    request.processing_latency = raw_lat if round_minmax_lat else request.processing_latency_clipped
                    request.response = {"error": repr(e)}
                    response_queue.put([request, queue_state_before, None])
                    continue

            except Empty:
                continue

    except Exception as e:
        print(f"Server {server_id} error: {repr(e)}")
        traceback.print_exc()

    print(f"Server {server_id} shutting down")


class LLMServerWrapper:
    """Wrapper for LLM server with process management (no inflight)."""

    def __init__(
        self,
        model_name: str,
        capacity: int,
        server_id: int,
        manager: mp.Manager,
        response_queue,
        gpu_id: Optional[int] = None,
        queue_monitor: Optional[QueueUpdateMonitor] = None,
    ):
        self.model_name = model_name
        self.capacity = capacity
        self.server_id = server_id
        self.gpu_id = gpu_id
        self.running = mp.Value("b", True)

        self.pause_event = mp.Event()
        self.pause_event.clear()

        self.request_queue = mp.Queue()
        self.response_queue = response_queue

        self.process = mp.Process(
            target=server_worker_process,
            args=(
                model_name,
                capacity,
                server_id,
                self.request_queue,
                self.response_queue,
                self.running,
                gpu_id,
                queue_monitor,
                self.pause_event,
            ),
        )
        self.process.start()

    def clean_queue(self):
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def put_request(self, request: Request):
        self.request_queue.put(request)

    def stop(self):
        self.running.value = False
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()

    def can_accept_request(self) -> bool:
        return self.request_queue.qsize() < int(self.capacity)

    def get_current_load(self) -> int:
        return int(self.request_queue.qsize())

    def get_server_id(self) -> int:
        return self.server_id

    def get_model_name(self) -> str:
        return self.model_name


class QualityScorer:
    """Quality scoring for prompt-model pairs"""

    def __init__(self):
        self.model_elo_scores = Config.MODEL_ELO_SCORES
        self.quality_model = P2LPredictor()

    def compute_quality_score_real(self, prompt: str, model_name: str) -> float:
        coefs = self.quality_model.get_coefficients(prompt)
        if _is_together_api_name(model_name):
            model_name = _strip_together_prefix(model_name)
        if "/" in model_name:
            model_name = model_name.split("/")[-1].lower()
        all_coefs = {
            ((_strip_together_prefix(m).split("/")[-1].lower()) if "/" in _strip_together_prefix(m) else _strip_together_prefix(m)): 
            coefs.get((_strip_together_prefix(m).split("/")[-1].lower()) if "/" in _strip_together_prefix(m) else _strip_together_prefix(m))
            for m in Config.MODEL_NAMES
        }
        score = coefs.get(model_name)
        normalized_score = (score - min(all_coefs.values())) / (max(all_coefs.values()) - min(all_coefs.values()))
        return float(normalized_score)

    def compute_quality_score_all(self, prompt: str) -> float:
        coefs = self.quality_model.get_coefficients(prompt)
        all_coefs = {
            m: coefs.get((_strip_together_prefix(m).split("/")[-1].lower()) if "/" in _strip_together_prefix(m) else _strip_together_prefix(m))
            for m in Config.MODEL_NAMES
        }
        min_score = min(all_coefs.values())
        max_score = max(all_coefs.values())
        normalized_scores = {
            model: (score - min_score) / (max_score - min_score) if max_score > min_score else 0.0
            for model, score in all_coefs.items()
        }
        return normalized_scores

    def compute_quality_score(self, prompt: str, server_id: int) -> float:
        prompt_length = len(prompt.split())
        prompt_complexity = min(prompt_length / 50.0, 2.0)

        base_elo = self.model_elo_scores.get(server_id, 1150)
        base_score = base_elo / 1000.0

        if server_id == 0:
            complexity_adjustment = max(0.8, 1.2 - prompt_complexity * 0.2)
        else:
            complexity_adjustment = min(1.2, 0.9 + prompt_complexity * 0.15)

        noise = np.random.normal(0, 0.05)
        final_score = base_score * complexity_adjustment + noise
        return float(max(0.1, final_score))


class SkyworkRewardJudge:
    """LLM-as-judge using Skywork/Skywork-Reward-V2-Llama-3.1-8B."""

    def __init__(self):
        self.model_name = getattr(Config, "LLM_JUDGE_MODEL_NAME", "Skywork/Skywork-Reward-V2-Llama-3.1-8B")
        self.device = self._resolve_device(getattr(Config, "LLM_JUDGE_DEVICE", "cuda"))
        self.dtype = self._resolve_dtype(getattr(Config, "LLM_JUDGE_DTYPE", "float16"))
        self.attn_impl = getattr(Config, "LLM_JUDGE_ATTN_IMPL", "flash_attention_2")
        self.max_length = int(getattr(Config, "LLM_JUDGE_MAX_LENGTH", 2048))
        self.normalize = getattr(Config, "LLM_JUDGE_NORMALIZE", "sigmoid")
        self.k = float(getattr(Config, "LLM_JUDGE_NORM_K", 1.0))
        self.use_raw = bool(getattr(Config, "LLM_JUDGE_USE_RAW_RESPONSE", False))

        self.cache_in_memory = bool(getattr(Config, "LLM_JUDGE_CACHE_IN_MEMORY", True))
        self.cache_path = getattr(Config, "LLM_JUDGE_CACHE_PATH", None)
        self._cache = {}
        self._loaded = False
        self._tokenizer = None
        self._model = None

        if self.cache_in_memory and self.cache_path:
            try:
                if os.path.exists(self.cache_path):
                    with open(self.cache_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            k = obj.get("key")
                            v = obj.get("score")
                            if k is not None and v is not None:
                                self._cache[str(k)] = float(v)
            except Exception as e:
                print(f"[SkyworkRewardJudge] Warning: failed to preload cache: {e}")

    def _resolve_device(self, dev: str) -> str:
        if dev == "cuda":
            try:
                if isinstance(getattr(Config, "DEVICE", None), torch.device) and Config.DEVICE.type == "cuda":
                    return str(Config.DEVICE)
            except Exception:
                pass
            return "cuda"
        return str(dev)

    def _resolve_dtype(self, s: str):
        s = str(s).lower()
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp32", "float32"):
            return torch.float32
        return torch.float16

    def _load(self):
        if self._loaded:
            return
        print(f"[SkyworkRewardJudge] Loading judge model: {self.model_name} on {self.device} dtype={self.dtype}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        base_kwargs = dict(torch_dtype=self.dtype, device_map=None, trust_remote_code=True, num_labels=1)

        if self.attn_impl:
            try:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    attn_implementation=self.attn_impl,
                    **base_kwargs,
                ).to(self.device)
            except Exception as e:
                print(f"[SkyworkRewardJudge] attn_implementation={self.attn_impl} failed; falling back. ({type(e).__name__}: {e})")
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    **base_kwargs,
                ).to(self.device)
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **base_kwargs,
            ).to(self.device)

        self._model.eval()
        self._loaded = True

    def preload(self):
        self._load()

    def _key(self, prompt: str, answer: str) -> str:
        h = hashlib.sha256()
        h.update((prompt or "").encode("utf-8"))
        h.update(b"\n\n<ANSWER>\n\n")
        h.update((answer or "").encode("utf-8"))
        return h.hexdigest()

    def _normalize_score(self, raw: float) -> float:
        if self.normalize == "none":
            return float(raw) / 100.0
        if self.normalize == "tanh":
            return float(0.5 * (math.tanh(self.k * raw) + 1.0))
        return float(1.0 / (1.0 + math.exp(-self.k * raw)))

    @torch.no_grad()
    def score(self, prompt: str, answer: str) -> float:
        if answer is None:
            answer = ""
        if prompt is None:
            prompt = ""

        key = self._key(prompt, answer)
        if self.cache_in_memory and key in self._cache:
            return float(self._cache[key])

        self._load()

        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        bos = self._tokenizer.bos_token_id
        if input_ids.size(1) >= 2 and bos is not None:
            if int(input_ids[0, 0]) == int(bos) and int(input_ids[0, 1]) == int(bos):
                input_ids = input_ids[:, 1:]

        if self.max_length and input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, -self.max_length:]

        attn = torch.ones_like(input_ids, device=input_ids.device)
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        out = self._model(input_ids=input_ids, attention_mask=attn)
        raw = float(out.logits[0][0].item())
        s = self._normalize_score(raw)

        if self.cache_in_memory:
            self._cache[key] = s
        if self.cache_path:
            try:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "score": s, "raw": raw}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[SkyworkRewardJudge] Warning: failed to append cache: {e}")

        return float(max(min(s, 1.0), 0.0))


def response_collector_worker(
    response_queue,
    total_completed,
    running,
    config_alpha: float,
    config_beta: float,
    config_lambda: float,
    routed_prompts,
    completed_prompts=None,
    queue_monitor=None,
):
    """Worker function for response collection"""
    print("Response collector started")

    judge = SkyworkRewardJudge() if getattr(Config, "USE_LLM_JUDGE", False) else None
    if judge is not None and bool(getattr(Config, "LLM_JUDGE_PRELOAD", True)):
        try:
            judge.preload()
        except Exception as e:
            print(f"[SkyworkRewardJudge] preload failed: {e}")

    while running.value:
        try:
            data = response_queue.get_nowait()
            if data is None:
                continue

            request = data[0]
            queue_state_before = data[1]
            queue_state_after = data[2]

            if request.status == "completed":
                # Quality scoring (post-generation)
                if getattr(Config, "USE_LLM_JUDGE", False) and judge is not None:
                    try:
                        pred = ""
                        if request.response is not None:
                            if bool(getattr(Config, "LLM_JUDGE_USE_RAW_RESPONSE", False)):
                                pred = request.response.get("response_text_raw") or ""
                            else:
                                pred = request.response.get("response_text") or ""
                        pred_for_judge = pred
                        if bool(getattr(Config, "EXTRACT_FINAL_ANSWER", True)):
                            try:
                                pred_for_judge = extract_final_answer(pred_for_judge)
                            except Exception:
                                pred_for_judge = pred
                        request.quality_score = float(judge.score(request.prompt, pred_for_judge))
                    except Exception as e:
                        print(f"[SkyworkRewardJudge] scoring failed: {e}")
                        request.quality_score = 0.0

                elif getattr(Config, "USE_EM_EXACT_MATCH", False):
                    try:
                        pred = None
                        if request.response is not None:
                            pred = request.response.get("response_text")
                        request.quality_score = match_quality_score(pred, getattr(request, "ground_truth", None))
                    except Exception:
                        request.quality_score = 0.0

                if request.processing_latency is not None:
                    latency = request.processing_latency

                    round_minmax = bool(getattr(Config, "ROUND_MINMAX_NORM_ENABLE", False)) and bool(
                        getattr(Config, "ENV_DISABLE_FIXED_NORM_WHEN_MINMAX", True)
                    )
                    defer_lp = bool(getattr(Config, "ENV_DEFER_LAT_PRICE_REWARD_WHEN_MINMAX", True)) and round_minmax

                    normalized_latency = latency if round_minmax else (latency / Config.MAX_LAT)
                    quality_reward = Config.ALPHA * float(request.quality_score or 0.0)
                    latency_penalty = 0.0 if defer_lp else (Config.BETA * normalized_latency)

                    resp_for_price = ""
                    if request.response:
                        if getattr(Config, "PRICE_USE_RAW_RESPONSE", True):
                            resp_for_price = request.response.get("response_text_raw") or ""
                        else:
                            resp_for_price = request.response.get("response_text") or ""
                            
                    prompt_tokens = _rough_token_count(request.prompt)
                    resp_tokens = _rough_token_count(resp_for_price)

                    num = (
                        Config.PRICE[request.server_id][0] * prompt_tokens
                        + Config.PRICE[request.server_id][1] * resp_tokens
                    )

                    all_server_real_costs = [
                        p_in * prompt_tokens
                        for p_in, p_out in Config.PRICE
                    ]

                    min_num = float(np.min(all_server_real_costs))

                    all_server_ref_costs = [
                        p_in * prompt_tokens + p_out * Config.GEN_MAX_NEW_TOKENS
                        for p_in, p_out in Config.PRICE
                    ]

                    den = float(np.percentile(all_server_ref_costs, 80))

                    eps = 1e-12
                    price_before = (num - min_num) / max(den - min_num, eps)
                    price_before = math.sqrt(max(price_before, 0.0))
                    
                    # prompt_tokens = _rough_token_count(request.prompt)
                    # resp_tokens = _rough_token_count(resp_for_price)

                    # price_in, price_out = Config.PRICE[request.server_id]

                    # num = price_in * prompt_tokens + price_out * resp_tokens

                    # all_server_costs = [
                    #     p_in * prompt_tokens + p_out * resp_tokens
                    #     for p_in, p_out in Config.PRICE
                    # ]

                    # den = float(np.percentile(all_server_costs, 80))
                    # den = max(den, 1e-12)

                    # price_before = math.sqrt(num / den)

                    price_raw = float(max(min(price_before, 1.0), 0.0))
                    request.price_raw = price_raw

                    price = price_raw if round_minmax else float(Config.REWARD_GAMMA * price_raw)
                    request.price = float(price)

                    price_penalty = 0.0 if defer_lp else float(price)
                    reward = float(quality_reward - latency_penalty - price_penalty)

                    clip = getattr(Config, "REWARD_CLIP", None)
                    if clip is not None and clip > 0:
                        reward = float(max(min(reward, clip), -clip))

                    print(
                        f"Response completed - ID: {request.id}, "
                        f"Quality: {request.quality_score:.3f}, "
                        f"Latency: {latency:.3f}s, "
                        f"Price: {price_before:.3f}, "
                        f"Reward: {reward:.3f}"
                    )

                    request.reward = reward
                    request.price = price

                    if queue_monitor is not None:
                        queue_monitor.log_request_completed(
                            current_time=request.completion_time,
                            server_id=request.server_id,
                            request_id=request.id,
                            queue_state_before=queue_state_before,
                            queue_state_after=queue_state_after,
                            reward=reward,
                            episode=request.episode,
                        )

                    with total_completed.get_lock():
                        total_completed.value += 1

                    routed_prompts[request.id] = request
                    if completed_prompts is not None:
                        completed_prompts.value += 1

                else:
                    print(f"Warning: Incomplete response data for request {request.id}")

            elif request.status == "api_transient_failed":
                # Provider outage / retryable API failure.
                # Do not punish the routing policy as hard as invalid route.
                penalty = float(getattr(Config, "API_TRANSIENT_FAIL_PENALTY", 0.05))

                request.reward = -penalty
                request.price = 0.0
                request.quality_score = 0.0

                routed_prompts[request.id] = request
                if completed_prompts is not None:
                    completed_prompts.value += 1

                print(f"Response transient-failed - ID: {request.id}, Penalty: {-penalty:.3f}")
            elif request.status == "failed":
                penalty = getattr(Config, "INVALID_ROUTE_PENALTY", None)
                if penalty is None:
                    penalty = float(config_beta) + float(getattr(Config, "REWARD_GAMMA", 0.0))
                reward = -float(penalty)

                clip = getattr(Config, "REWARD_CLIP", None)
                if clip is not None and clip > 0:
                    reward = float(max(min(reward, clip), -clip))

                request.reward = reward
                request.price = 0.0
                routed_prompts[request.id] = request
                if completed_prompts is not None:
                    completed_prompts.value += 1
                print(f"Response failed - ID: {request.id}, Penalty: {reward:.3f}")

        except Empty:
            continue
        except EOFError:
            if not running.value:
                break
            continue
        except Exception as e:
            print(f"Error in response collector: {e}")
            continue

def _to_text(x: Any) -> str:
    """Convert str/list/dict/None to a safe string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(_to_text(v) for v in x if v is not None)
    if isinstance(x, dict):
        # Common API-style response objects
        for k in ["text", "content", "response_text", "response_text_raw", "answer", "value"]:
            if k in x:
                return _to_text(x.get(k))
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _rough_token_count(x: Any) -> int:
    """Whitespace token count, robust to list/dict/None."""
    return len(_to_text(x).split())

class EnhancedRouterEnvironment:
    """Environment for LLM request routing with parallel processing"""

    def __init__(self, enable_monitoring=True):
        self.enable_monitoring = enable_monitoring

        if self.enable_monitoring:
            try:
                self.queue_monitor = QueueUpdateMonitor(wandb_available=False)
            except ImportError:
                print("Queue monitor not available")
                self.enable_monitoring = False

        self.manager = mp.Manager()

        self.interval_routed_counts = np.zeros(len(Config.MODEL_NAMES), dtype=np.float32)

        self.routed_prompts = self.manager.dict()
        self.completed_prompts = self.manager.Value("i", 0)

        self.quality_scorer = QualityScorer()

        self.response_queue = mp.Queue()

        self.prompt_queue = self.manager.Queue()
        self.prompt_generator = PoissonPromptGenerator(
            arrival_rate=Config.POISSON_ARRIVAL_RATE,
            prompt_queue=self.prompt_queue,
            max_queue_size=Config.MAX_PROMPT_QUEUE_SIZE,
            dataset_name=Config.DATASET_NAME,
            dataset_config=getattr(Config, "DATASET_CONFIG", None),
            dataset_split=getattr(Config, "DATASET_SPLIT", "train"),
            prompt_style=getattr(Config, "QA_PROMPT_STYLE", "instruction"),
            qa_include_context=getattr(Config, "QA_INCLUDE_CONTEXT", True),
            qa_max_context_docs=getattr(Config, "QA_MAX_CONTEXT_DOCS", 8),
            qa_max_context_chars=getattr(Config, "QA_MAX_CONTEXT_CHARS", 2500),
            force_final_tag=getattr(Config, "QA_FORCE_FINAL_TAG", True),
            final_tag=getattr(Config, "FINAL_ANSWER_TAG", "final"),
            shuffle_dataset=getattr(Config, "SHUFFLE_DATASET", True),
            dataset_seed=getattr(Config, "DATASET_SEED", 42),
            mixed_datasets=(
                getattr(Config, "MIXED_DATASETS", None)
                if getattr(Config, "USE_MIXED_DATASET", False)
                else None
            ),
        )
        self.prompt_generator.start()

        self.response_collector_running = mp.Value("b", True)
        self.total_completed = mp.Value("i", 0)

        self.response_collector = mp.Process(
            target=response_collector_worker,
            args=(
                self.response_queue,
                self.total_completed,
                self.response_collector_running,
                Config.ALPHA,
                Config.BETA,
                Config.LAMBDA,
                self.routed_prompts,
                self.completed_prompts,
                self.queue_monitor if self.enable_monitoring else None,
            ),
        )
        self.response_collector.daemon = True
        self.response_collector.start()

        print(f"Initializing {len(Config.MODEL_NAMES)} servers...")
        self.servers = []
        self.gpu_ids = Config.GPU_LIST

        for i in range(len(Config.MODEL_NAMES)):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)] if self.gpu_ids else None
            print(f"Creating server {i} with model {Config.MODEL_NAMES[i]} on GPU {gpu_id}")
            server = LLMServerWrapper(
                model_name=Config.MODEL_NAMES[i],
                capacity=Config.SERVER_CAPACITIES[i],
                server_id=i,
                manager=self.manager,
                response_queue=self.response_queue,
                gpu_id=gpu_id,
                queue_monitor=self.queue_monitor if self.enable_monitoring else None,
            )
            self.servers.append(server)

        self.request_counter = 0
        self.current_episode = 0

        print("Waiting for servers to initialize...")
        time.sleep(10)

        self.reset()
        print("Environment initialization complete!")

    def reset(self) -> np.ndarray:
        self.request_counter = 0
        with self.total_completed.get_lock():
            self.total_completed.value = 0

        if self.enable_monitoring and hasattr(self, "queue_monitor"):
            self.queue_monitor.reset()

        return self.get_state()

    def get_prompt_generator(self):
        return self.prompt_generator

    def get_state(self) -> np.ndarray:
        state = []
        for server in self.servers:
            load = server.get_current_load()
            state.extend([load])
        return np.array(state, dtype=np.float32)

    def get_action_mask(self) -> np.ndarray:
        mask = torch.tensor([server.can_accept_request() for server in self.servers], dtype=torch.float32)
        if float(mask.sum().item()) == 0.0:
            mask = torch.ones_like(mask)
        return mask

    def get_episode_data(self) -> List[Dict[str, Any]]:
        episode = [
            self.routed_prompts[key].__dict__
            for key in sorted(self.routed_prompts.keys())
            if self.routed_prompts[key].episode == self.current_episode
        ]
        return episode

    def clean_episode_completed(self):
        self.routed_prompts.clear()

    def check_get_episode_completed(self):
        print(f"Completed prompts: {self.completed_prompts.value}, Total routed: {len(self.routed_prompts)}")
        return self.completed_prompts.value == len(self.routed_prompts)

    def get_next_prompt(self) -> str:
        prompt = None
        try:
            prompt = self.prompt_queue.get_nowait()
        except Empty:
            pass
        return prompt

    def step(self, action: int, prompt: str, ground_truth: Optional[Any] = None) -> Tuple[np.ndarray, bool]:
        with self.total_completed.get_lock():
            self.total_completed.value = 0

        request_id = self.request_counter
        self.request_counter += 1

        request = Request(
            id=request_id,
            prompt=prompt,
            ground_truth=ground_truth,
            arrival_time=time.time(),
            server_id=action,
            episode=self.current_episode,
        )

        server = self.servers[action]

        self.interval_routed_counts[action] += 1

        queue_len_before = server.get_current_load()
        request.queue_len_at_dispatch = queue_len_before
        try:
            request.queue_util_at_dispatch = queue_len_before / max(float(server.capacity), 1.0)
        except Exception:
            request.queue_util_at_dispatch = None

        try:
            request.queue_length = self.get_state().tolist()
        except Exception:
            request.queue_length = None

        if not server.can_accept_request():
            if self.enable_monitoring and hasattr(self, "queue_monitor"):
                self.queue_monitor.log_request_failed(
                    server_id=action,
                    request_id=request_id,
                    prompt=prompt,
                    current_time=time.time(),
                    reason="Server at capacity",
                    episode=self.current_episode,
                )

            request.status = "failed"
            penalty = getattr(Config, "INVALID_ROUTE_PENALTY", None)
            if penalty is None:
                penalty = float(Config.BETA) + float(getattr(Config, "REWARD_GAMMA", 0.0))
            request.reward = -float(penalty)
            request.completion_time = time.time()
            fail_lat = getattr(Config, "FAIL_LATENCY_CAP", Config.MAX_LAT)
            request.processing_latency = float(fail_lat)
            request.quality_score = 0.0
            request.price = 0.0

            self.response_queue.put([request, None, None])
            print(f"Server {action} at capacity. Request {request_id} failed.")
            return self.get_state(), False

        if getattr(Config, "USE_EM_EXACT_MATCH", False) or getattr(Config, "USE_LLM_JUDGE", False):
            request.quality_score = 0.0
        else:
            request.quality_score = self.quality_scorer.compute_quality_score_real(prompt, server.get_model_name())

        self.routed_prompts[request.id] = request

        server.put_request(request)
        return self.get_state(), False

    def set_episode(self, episode: int):
        self.current_episode = episode

    def pause_prompt_generator(self):
        self.prompt_generator.stop()

    def clean_response_queue(self):
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except Empty:
                break

    def clean_prompt_queue(self):
        while not self.prompt_queue.empty():
            try:
                self.prompt_queue.get_nowait()
            except Empty:
                break

    def pause_all_servers(self):
        print("Pausing all servers...")
        for server in self.servers:
            server.pause()
        self.prompt_generator.stop()
        print("All servers paused.")

    def resume_all_servers(self):
        print("Resuming all servers...")
        for server in self.servers:
            server.resume()
        self.prompt_generator.start()
        print("All servers resumed.")

    def clean_all_queues(self):
        print("Cleaning all server queues...")
        for server in self.servers:
            server.clean_queue()
        self.clean_prompt_queue()
        self.clean_response_queue()
        self.clean_episode_completed()
        self.completed_prompts.value = 0
        print("All queues cleaned.")

    def __del__(self):
        if hasattr(self, "response_collector_running"):
            self.response_collector_running.value = False

        if hasattr(self, "response_collector") and self.response_collector.is_alive():
            self.response_collector.join(timeout=2)
            if self.response_collector.is_alive():
                self.response_collector.terminate()

        if hasattr(self, "servers"):
            for server in self.servers:
                server.stop()

        if hasattr(self, "prompt_generator"):
            self.prompt_generator.stop()
