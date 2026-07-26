import numpy as np
import time
import threading
from typing import Optional, Dict, Any, List, Tuple
import random
import re

from datasets import load_dataset


class PoissonPromptGenerator:
    """Generates prompts according to a Poisson arrival process.

    Supports:
      - single dataset mode, using DATASET_NAME / DATASET_CONFIG / DATASET_SPLIT
      - mixed dataset mode, using Config.MIXED_DATASETS passed as `mixed_datasets`

    It always emits dicts with at least:
      - prompt: str
      - output: gold answer object, either str/list or {"answers": ..., "metric": ...}
      - dataset: source dataset name
      - task_type: optional task label
    """

    def __init__(
        self,
        arrival_rate: float,
        prompt_queue,
        max_queue_size: int,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        prompt_style: str = "instruction",
        qa_include_context: bool = True,
        qa_max_context_docs: int = 8,
        qa_max_context_chars: int = 2500,
        force_final_tag: bool = True,
        final_tag: str = "final",
        shuffle_dataset: bool = True,
        dataset_seed: int = 42,
        mixed_datasets: Optional[List[Dict[str, Any]]] = None,
        dataset_levels: Optional[List[str]] = None,
        dataset_filter: Optional[str] = None,
        mcq_cot: bool = True,
        math_brief: bool = True,
        mcq_brief: bool = True,
    ):
        self.shuffle_dataset = bool(shuffle_dataset)
        self.dataset_seed = int(dataset_seed)
        # Dedicated stream for the mixture draw. It used to call np.random,
        # i.e. the process-global generator, which every prompt-blind baseline
        # also draws from once or more per routing decision (P2C samples d
        # servers, epsilon-greedy flips a coin, ...). The learned policy samples
        # actions through torch instead, so it consumes a different number of
        # numpy draws -- and the workload a run sees therefore depended on the
        # policy being evaluated. Runs were not comparable prompt-for-prompt:
        # the per-episode task mix swings about +/-5%, and with per-task mean
        # quality spanning 0.39 to 0.79 that alone moves episode quality by
        # ~0.02. Seeding a private generator makes every run replay the exact
        # same prompt stream, which turns method-vs-baseline into a PAIRED
        # comparison and cancels the prompt-difficulty term -- the same term
        # the variance decomposition puts at 76% of per-request quality
        # variance -- in the difference.
        self._mix_rng = np.random.default_rng(self.dataset_seed)
        # Same argument for the inter-arrival draw, which had the same defect.
        # Seeding it is necessary but not sufficient: the thread free-runs on
        # cumulative time.sleep(), so OS jitter accumulates and the k-th
        # request lands at a different offset in every run. begin_episode()
        # replaces that with a pre-computed trace replayed against absolute
        # deadlines, so episode e is byte-identical across algorithms.
        self._arr_rng = np.random.default_rng(self.dataset_seed + 7919)
        self._trace = None      # [(offset_seconds, prompt_entry)], or None = free-run
        self._trace_t0 = None
        self.mcq_cot = bool(mcq_cot)
        self.math_brief = bool(math_brief)
        self.mcq_brief = bool(mcq_brief)
        self.dataset_levels = dataset_levels
        self.dataset_filter = dataset_filter
        self.arrival_rate = float(arrival_rate)
        self.prompt_queue = prompt_queue
        self.max_queue_size = int(max_queue_size)

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.prompt_style = prompt_style
        self.qa_include_context = bool(qa_include_context)
        self.qa_max_context_docs = int(qa_max_context_docs)
        self.qa_max_context_chars = int(qa_max_context_chars)
        self.force_final_tag = bool(force_final_tag)
        self.final_tag = str(final_tag)

        self.running = False
        self.thread = None
        self.total_generated = 0
        self.start_time = None

        self.rng = random.Random(self.dataset_seed)
        np.random.seed(self.dataset_seed)
        random.seed(self.dataset_seed)

        # Single-dataset state
        self.dataset = None
        self.dataset_index = 0

        # Mixed-dataset state
        self.mixed_datasets_config = mixed_datasets or []
        self.use_mixed_dataset = len(self.mixed_datasets_config) > 0
        self.dataset_pools: List[Dict[str, Any]] = []
        self.dataset_weights: List[float] = []

        if self.use_mixed_dataset:
            self.load_mixed_datasets(self.mixed_datasets_config)
        else:
            self.load_dataset(self.dataset_name, self.dataset_config, self.dataset_split)

    # -------------------------
    # Dataset loading
    # -------------------------
    def _load_one_dataset(self, dataset_name: str, dataset_config: Optional[str], dataset_split: str) -> List[Dict[str, Any]]:
        # streaming=True does not support slice syntax in split strings (e.g. "auxiliary_train[:40000]"),
        # which several configured splits rely on; load eagerly instead.
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
        else:
            ds = load_dataset(dataset_name, split=dataset_split)
        data = list(ds)
        return data

    def load_dataset(self, dataset_name: str, dataset_config: Optional[str], dataset_split: str):
        """Load one dataset into memory."""
        try:
            self.dataset = self._load_one_dataset(dataset_name, dataset_config, dataset_split)
            # Optional single-dataset filters, mirroring the mixed-mode
            # "levels" / "filter" hooks (Config.DATASET_LEVELS / DATASET_FILTER,
            # passed in as constructor arguments).
            if self.dataset_levels:
                keep = {str(x) for x in self.dataset_levels}
                self.dataset = [s for s in self.dataset if str(s.get("level", "")) in keep]
            flt = str(self.dataset_filter or "")
            if flt == "numeric_boxed":
                self.dataset = [
                    s for s in self.dataset
                    if self._extract_boxed_final(s.get("solution", "")) is not None
                ]
            elif flt == "boxed":
                self.dataset = [
                    s for s in self.dataset
                    if self._extract_boxed_raw(s.get("solution", "")) is not None
                ]
            if self.shuffle_dataset:
                rng = random.Random(self.dataset_seed)
                rng.shuffle(self.dataset)
                print(f"Shuffled dataset with seed={self.dataset_seed}")
            print(f"Loaded {len(self.dataset)} samples from {dataset_name} (config={dataset_config}, split={dataset_split})")
        except Exception as e:
            print(f"Error loading dataset {dataset_name} (config={dataset_config}, split={dataset_split}): {e}")
            self._create_dummy_dataset()

    def load_mixed_datasets(self, mixed_datasets: List[Dict[str, Any]]):
        """Load multiple datasets and keep separate sampling pools.

        Expected config item format:
          {
            "name": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "weight": 0.35,
            "metric": "f1",
            "task_type": "multihop_qa"
          }
        """
        pools = []
        weights = []
        for i, cfg in enumerate(mixed_datasets):
            name = cfg.get("name") or cfg.get("dataset_name")
            config = cfg.get("config", None)
            split = cfg.get("split", "train")
            weight = float(cfg.get("weight", 1.0))
            metric = str(cfg.get("metric", "f1"))
            task_type = str(cfg.get("task_type", self._infer_task_type(name)))
            max_samples = cfg.get("max_samples", None)

            try:
                data = self._load_one_dataset(name, config, split)
                # Optional difficulty filter (e.g., MATH "Level 3"-"Level 5").
                levels = cfg.get("levels", None)
                if levels:
                    keep = {str(x) for x in levels}
                    data = [s for s in data if str(s.get("level", "")) in keep]
                # Optional sample filter:
                #   "numeric_boxed": only purely numeric \boxed answers
                #                    (exact "number"-metric scoring);
                #   "boxed":         any sample with a \boxed answer
                #                    (scored via math-verify equivalence).
                flt = str(cfg.get("filter", "") or "")
                if flt == "numeric_boxed":
                    data = [
                        s for s in data
                        if self._extract_boxed_final(s.get("solution", "")) is not None
                    ]
                elif flt == "boxed":
                    data = [
                        s for s in data
                        if self._extract_boxed_raw(s.get("solution", "")) is not None
                    ]
                if max_samples is not None:
                    data = data[: int(max_samples)]
                if self.shuffle_dataset:
                    rng = random.Random(self.dataset_seed + i)
                    rng.shuffle(data)
                if len(data) == 0:
                    print(f"[MixedDataset] Skipped empty dataset {name}")
                    continue
                pools.append({
                    "name": name,
                    "config": config,
                    "split": split,
                    "weight": weight,
                    "metric": metric,
                    "task_type": task_type,
                    "data": data,
                    "index": 0,
                })
                weights.append(max(weight, 0.0))
                print(
                    f"[MixedDataset] Loaded {len(data)} samples from {name} "
                    f"(config={config}, split={split}, weight={weight}, metric={metric}, task={task_type})"
                )
            except Exception as e:
                print(f"[MixedDataset] Error loading {name} (config={config}, split={split}): {e}")

        if not pools:
            print("[MixedDataset] No valid datasets loaded; using dummy dataset.")
            self._create_dummy_dataset()
            self.use_mixed_dataset = False
            return

        weights = np.asarray(weights, dtype=np.float64)
        if weights.sum() <= 0:
            weights[:] = 1.0
        weights = weights / weights.sum()

        self.dataset_pools = pools
        self.dataset_weights = weights.tolist()
        self.dataset = []  # not used in mixed mode
        print("[MixedDataset] Normalized weights:")
        for p, w in zip(self.dataset_pools, self.dataset_weights):
            print(f"  - {p['name']} ({p['task_type']}, metric={p['metric']}): {w:.3f}")

    def _create_dummy_dataset(self):
        self.dataset = [
            {
                "instruction": "Explain artificial intelligence in one sentence.",
                "input": "",
                "output": "Artificial intelligence is the field of creating systems that can perform tasks requiring human-like intelligence.",
            },
            {
                "instruction": "What is the capital of Australia?",
                "input": "",
                "output": "Canberra",
            },
        ]
        self.dataset_index = 0

    @staticmethod
    def _infer_task_type(name: Optional[str]) -> str:
        n = (name or "").lower()
        if "mmlu" in n:
            return "mmlu"
        if "gsm8k" in n:
            return "math"
        if "competition_math" in n or "hendrycks_math" in n:
            return "math_hard"
        if "hotpot" in n:
            return "multihop_qa"
        if "squad" in n or "trivia" in n:
            return "qa"
        if "mbpp" in n or "humaneval" in n or "code" in n:
            return "code"
        return "qa"

    # -------------------------
    # Prompt extraction helpers
    # -------------------------
    @staticmethod
    def _truncate(s: str, max_chars: int) -> str:
        s = s or ""
        if max_chars is None or max_chars <= 0:
            return s
        return s[:max_chars]

    _CHOICE_LETTERS = "ABCDEFGHIJ"

    @staticmethod
    def _choice_label(i: int) -> str:
        return PoissonPromptGenerator._CHOICE_LETTERS[int(i)]

    @staticmethod
    def _mmlu_choices(sample: Dict[str, Any]) -> List[Any]:
        # MMLU uses "choices"; MMLU-Pro uses "options" (up to 10, A-J).
        c = sample.get("choices")
        if not isinstance(c, (list, tuple)):
            c = sample.get("options")
        return list(c) if isinstance(c, (list, tuple)) else []

    def _is_mmlu_sample(self, sample: Dict[str, Any]) -> bool:
        return (
            isinstance(sample, dict)
            and isinstance(sample.get("question"), str)
            and len(self._mmlu_choices(sample)) > 0
            and (sample.get("answer") is not None or sample.get("answer_index") is not None)
        )

    def _mmlu_gold_letter(self, sample: Dict[str, Any]) -> str:
        ans = sample.get("answer")
        if ans is None:
            ans = sample.get("answer_index")
        if isinstance(ans, (int, np.integer)):
            return self._choice_label(int(ans))
        if isinstance(ans, str):
            s = ans.strip()
            if s.isdigit():
                return self._choice_label(int(s))
            if len(s) == 1 and s.upper() in self._CHOICE_LETTERS:
                return s.upper()
        return str(ans).strip()

    def _build_mmlu_prompt(self, sample: Dict[str, Any]) -> str:
        question = str(sample.get("question", "")).strip()
        choices = self._mmlu_choices(sample)
        subject = str(sample.get("subject") or sample.get("category") or "").replace("_", " ").strip()
        choice_lines = [f"{self._choice_label(i)}. {str(c).strip()}" for i, c in enumerate(choices)]
        subject_line = f"Subject: {subject}\n" if subject else ""
        # Letter range follows the option count (A-D for MMLU, up to A-J for
        # MMLU-Pro). No <final> tag required either way; extraction reads the
        # closing "Answer: X" line.
        last = self._choice_label(max(len(choices) - 1, 0))
        body = subject_line + f"Question: {question}\nChoices:\n" + "\n".join(choice_lines)
        if self.mcq_cot:
            # MMLU-Pro questions are built to need several reasoning steps, so
            # a bare-letter answer collapses the whole fleet onto the random
            # floor and leaves nothing for the router to discriminate on.
            # Bounded when mcq_brief: an unbounded chain can eat the token cap
            # before the "Answer:" line is emitted, and a truncated answer
            # scores 0 no matter how good the reasoning was.
            lead = ("Think briefly: at most 3 short steps, no restating the "
                    "question. Then close" if bool(getattr(self, "mcq_brief", False))
                    else "Reason step by step, then close")
            return body + (
                f"\n{lead} with a final line of exactly "
                f'"Answer: X", where X is one option letter (A-{last}).\n'
            )
        return body + f"\nAnswer:\nAnswer with only one option letter (A-{last}). Do not explain.\n"

    def _is_gsm8k_sample(self, sample: Dict[str, Any]) -> bool:
        return isinstance(sample, dict) and isinstance(sample.get("question"), str) and isinstance(sample.get("answer"), str) and "####" in sample.get("answer", "")

    @staticmethod
    def _extract_gsm8k_final(answer: str) -> str:
        # GSM8K gold usually ends with "#### 42".
        s = str(answer or "")
        if "####" in s:
            s = s.split("####")[-1]
        s = s.strip().replace(",", "")
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        if not m:
            return s
        num = m[-1]
        try:
            f = float(num)
            if abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
        except Exception:
            pass
        return num

    def _build_gsm8k_prompt(self, sample: Dict[str, Any]) -> str:
        # Standard math-reasoning output format: close with \boxed{...};
        # no <final> tag required (same convention as competition math).
        question = str(sample.get("question", "")).strip()
        return f"Question: {question}\nAnswer:{self._math_suffix()}"

    def _is_math_sample(self, sample: Dict[str, Any]) -> bool:
        # Competition-math (MATH) rows: {"problem", "solution", "level", "type"}.
        return (
            isinstance(sample, dict)
            and isinstance(sample.get("problem"), str)
            and isinstance(sample.get("solution"), str)
        )

    @staticmethod
    def _extract_boxed_raw(solution: str) -> Optional[str]:
        """Extract the raw content of the last \\boxed{...} in a solution,
        handling nested braces. Returns None when no boxed answer exists."""
        s = str(solution or "")
        i = s.rfind("\\boxed")
        if i < 0:
            return None
        j = s.find("{", i)
        if j < 0:
            return None
        depth, k = 1, j + 1
        while k < len(s) and depth > 0:
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
            k += 1
        if depth != 0:
            return None
        return s[j + 1:k - 1].strip()

    @staticmethod
    def _extract_boxed_final(solution: str) -> Optional[str]:
        """Extract the last \\boxed{...} answer and normalize it to a plain
        number string. Returns None when the boxed answer is not purely
        numeric (fractions, radicals, pi, intervals, ...), so callers can
        filter those samples out and keep exact numeric scoring."""
        raw = PoissonPromptGenerator._extract_boxed_raw(solution)
        if raw is None:
            return None
        b = raw.strip().strip("$")
        b = b.replace("\\!", "").replace("\\,", "").replace(" ", "")
        b = re.sub(r"(\\text\{[^}]*\}|\\%|\\?\^\\?circ|\\?\^\{\\circ\})$", "", b).strip()
        if not re.match(r"^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^[-+]?\d+(?:\.\d+)?$", b):
            return None
        b = b.replace(",", "")
        try:
            f = float(b)
            if abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
        except Exception:
            pass
        return b

    def _build_math_prompt(self, sample: Dict[str, Any]) -> str:
        # MATH-standard output format: the model closes with \boxed{...};
        # no <final> tag is required (math-verify extracts boxed answers).
        problem = str(sample.get("problem", "")).strip()
        return f"Problem: {problem}\nAnswer:{self._math_suffix()}"

    def _math_suffix(self) -> str:
        """Closing instruction shared by both math builders.

        The brevity clause is not a cost-saving tweak -- it RAISES accuracy
        here. Measured over the fleet at GEN_MAX_NEW_TOKENS=1024: unbounded
        answers averaged 607 tokens and hit the cap 36% of the time, and a
        capped answer loses its \\boxed{} entirely and scores 0. Bounding the
        working took mean quality from 0.61 to 0.74 while cutting tokens 44%
        and latency 26%, and every endpoint improved. The dominant effect at
        this cap is truncation, not reasoning depth.
        Note the opposite holds for MMLU-Pro, where the same kind of bound
        costs 0.12 accuracy (0.67 -> 0.55) -- see MCQ_COT. The two tasks are
        deliberately handled differently.
        """
        if bool(getattr(self, "math_brief", True)):
            return ("\nSolve it concisely: show at most 3 short steps, no restating "
                    "the problem. Put your final answer within \\boxed{}.\n")
        return "\nPut your final answer within \\boxed{}.\n"

    def _extract_question(self, sample: Dict[str, Any]) -> str:
        if isinstance(sample.get("instruction"), str) and sample.get("instruction").strip():
            return sample.get("instruction").strip()
        for k in ["question", "query", "prompt", "title", "text"]:
            v = sample.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Answer the question."

    def _extract_context(self, sample: Dict[str, Any]) -> str:
        if isinstance(sample.get("input"), str) and sample.get("input").strip():
            return sample.get("input").strip()
        ctx = sample.get("context")
        if isinstance(ctx, str) and ctx.strip():
            return self._truncate(ctx.strip(), self.qa_max_context_chars)
        if isinstance(ctx, dict):
            titles = ctx.get("title") or ctx.get("titles")
            sents = ctx.get("sentences") or ctx.get("sentence") or ctx.get("sent")
            if isinstance(titles, list) and isinstance(sents, list):
                parts = []
                for i, (t, ss) in enumerate(zip(titles, sents)):
                    if i >= self.qa_max_context_docs:
                        break
                    title = str(t).strip()
                    para = " ".join(str(x).strip() for x in ss if str(x).strip()) if isinstance(ss, list) else str(ss).strip()
                    parts.append(f"{title}: {para}".strip() if title else para)
                return self._truncate("\n\n".join(p for p in parts if p), self.qa_max_context_chars).strip()
        for k in ["passage", "paragraph", "article", "document"]:
            v = sample.get(k)
            if isinstance(v, str) and v.strip():
                return self._truncate(v.strip(), self.qa_max_context_chars)
        if isinstance(ctx, list) and ctx:
            parts = []
            for item in ctx[: self.qa_max_context_docs]:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    for kk in ["text", "context", "paragraph", "sentence"]:
                        if isinstance(item.get(kk), str) and item.get(kk).strip():
                            parts.append(item.get(kk).strip())
                            break
            return self._truncate("\n\n".join(p for p in parts if p), self.qa_max_context_chars).strip()
        return ""

    def _extract_gold_answers(self, sample: Dict[str, Any]) -> List[str]:
        answers: List[str] = []
        if "output" in sample and sample.get("output") is not None:
            v = sample.get("output")
            if isinstance(v, str) and v.strip():
                answers.append(v.strip())
            elif isinstance(v, list):
                answers.extend([str(x).strip() for x in v if str(x).strip()])
        if not answers and sample.get("answer") is not None:
            a = sample.get("answer")
            if isinstance(a, str) and a.strip():
                answers.append(a.strip())
            elif isinstance(a, list):
                answers.extend([str(x).strip() for x in a if str(x).strip()])
            elif isinstance(a, dict):
                if a.get("value") is not None:
                    answers.append(str(a.get("value")).strip())
                aliases = a.get("aliases") or a.get("alias") or []
                if isinstance(aliases, list):
                    answers.extend([str(x).strip() for x in aliases if str(x).strip()])
        if sample.get("answers") is not None:
            a = sample.get("answers")
            if isinstance(a, dict):
                txt = a.get("text") or a.get("texts")
                if isinstance(txt, list):
                    answers.extend([str(x).strip() for x in txt if str(x).strip()])
                elif isinstance(txt, str) and txt.strip():
                    answers.append(txt.strip())
            elif isinstance(a, list):
                answers.extend([str(x).strip() for x in a if str(x).strip()])
        for k in ["target", "label", "gold", "ground_truth"]:
            v = sample.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                answers.append(v.strip())
            elif isinstance(v, list):
                answers.extend([str(x).strip() for x in v if str(x).strip()])
        seen = set()
        uniq = []
        for x in answers:
            if x and x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    def _build_prompt(self, question: str, context: str) -> str:
        style = (self.prompt_style or "instruction").lower().strip()
        if (not self.qa_include_context) or (not context):
            context = ""
        # Standard QA output: the short answer span alone; no <final> tag
        # required (token-F1 scoring rewards concise answers).
        suffix = "\nAnswer with only the short final answer. Do not explain.\n"
        if style in {"instruction", "alpaca"}:
            if context:
                return f"Instruction: {question}\nInput: {context}\nResponse:{suffix}"
            return f"Instruction: {question}\nResponse:{suffix}"
        if style == "plain":
            if context:
                return f"Question: {question}\nContext: {context}\nAnswer:{suffix}"
            return f"Question: {question}\nAnswer:{suffix}"
        if context:
            return f"Instruction: {question}\nInput: {context}\nResponse:{suffix}"
        return f"Instruction: {question}\nResponse:{suffix}"

    # -------------------------
    # Core API
    # -------------------------
    def _sample_from_single_dataset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.dataset:
            return {"prompt": "Instruction: Explain artificial intelligence.\nResponse:", "output": ""}, {
                "name": self.dataset_name,
                "metric": "f1",
                "task_type": "dummy",
            }
        sample = self.dataset[self.dataset_index]
        self.dataset_index = (self.dataset_index + 1) % len(self.dataset)
        return sample, {"name": self.dataset_name, "metric": "f1", "task_type": self._infer_task_type(self.dataset_name)}

    def _sample_from_mixed_dataset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pool_idx = int(
            self._mix_rng.choice(
                len(self.dataset_pools),
                p=np.asarray(self.dataset_weights, dtype=np.float64),
            )
        )
        pool = self.dataset_pools[pool_idx]
        data = pool["data"]
        idx = int(pool["index"])
        sample = data[idx]
        pool["index"] = (idx + 1) % len(data)
        return sample, pool

    def get_next_prompt(self) -> Dict[str, Any]:
        """Get next prompt + ground-truth output from dataset."""
        if self.use_mixed_dataset:
            sample, source = self._sample_from_mixed_dataset()
        else:
            sample, source = self._sample_from_single_dataset()

        if not isinstance(sample, dict):
            sample = {"text": str(sample)}

        dataset_name = source.get("name", self.dataset_name)
        metric = str(source.get("metric", "f1"))
        task_type = str(source.get("task_type", self._infer_task_type(dataset_name)))

        # MMLU special case
        if metric == "mmlu" or self._is_mmlu_sample(sample):
            prompt = self._build_mmlu_prompt(sample)
            output = self._mmlu_gold_letter(sample)
            question = str(sample.get("question", "")).strip()
            choices = self._mmlu_choices(sample)
            subject = str(sample.get("subject") or sample.get("category") or "").strip()
            return {
                "prompt": prompt,
                "output": {"answers": output, "metric": "mmlu", "dataset": dataset_name, "task_type": task_type},
                "instruction": question,
                "input": "\n".join(f"{self._choice_label(i)}. {str(c).strip()}" for i, c in enumerate(choices)),
                "dataset": dataset_name,
                "task_type": task_type,
                "subject": subject,
                "mmlu_answer": output,
            }

        # Competition math (MATH) special case: \boxed{...} answers, scored
        # by symbolic equivalence (math-verify) with numeric fallback.
        if metric in {"math_boxed", "math_verify", "competition_math"} or self._is_math_sample(sample):
            prompt = self._build_math_prompt(sample)
            output = self._extract_boxed_raw(sample.get("solution", "")) or ""
            return {
                "prompt": prompt,
                "output": {"answers": output, "metric": "math_verify", "dataset": dataset_name, "task_type": task_type},
                "instruction": str(sample.get("problem", "")).strip(),
                "input": "",
                "dataset": dataset_name,
                "task_type": task_type,
            }

        # GSM8K / numeric math special case
        if metric in {"number", "numeric", "gsm8k"} or self._is_gsm8k_sample(sample):
            prompt = self._build_gsm8k_prompt(sample)
            output = self._extract_gsm8k_final(sample.get("answer", ""))
            return {
                "prompt": prompt,
                "output": {"answers": output, "metric": "number", "dataset": dataset_name, "task_type": task_type},
                "instruction": str(sample.get("question", "")).strip(),
                "input": "",
                "dataset": dataset_name,
                "task_type": task_type,
            }

        # Generic QA / Alpaca path
        question = self._extract_question(sample)
        context = self._extract_context(sample)
        golds = self._extract_gold_answers(sample)
        prompt = self._build_prompt(question, context)
        output = "" if len(golds) == 0 else (golds[0] if len(golds) == 1 else golds)

        return {
            "prompt": prompt,
            "output": {"answers": output, "metric": metric, "dataset": dataset_name, "task_type": task_type},
            "instruction": question,
            "input": context,
            "dataset": dataset_name,
            "task_type": task_type,
        }

    # Stride between episodes through each pool. Larger than any plausible
    # per-episode draw from one pool (E[N] is ~84 requests split three ways),
    # so consecutive episodes never see overlapping prompts.
    EPISODE_POOL_STRIDE = 200

    def begin_episode(self, episode: int, duration: float) -> int:
        """Pin episode `episode` to a fixed arrival trace and return its length.

        Everything that makes the workload differ between two runs is decided
        here, before the first routing decision: which prompts, in what order,
        and at what offset from episode start. The trace is a pure function of
        (dataset_seed, episode), so FLAIR's episode 7 and P2C's episode 7 are
        the same 84-odd requests at the same 84-odd timestamps. Without this,
        arrivals kept running through a policy-dependent drain phase, so the
        two runs entered episode 8 at different positions in the stream and the
        offset compounded -- per-episode differences were not paired at all.

        Call before starting the generator; `duration` is the arrival window
        (the interval phase), which excludes the drain.
        """
        self._mix_rng = np.random.default_rng(self.dataset_seed + 1000 * int(episode))
        self._arr_rng = np.random.default_rng(self.dataset_seed + 7919 * int(episode))
        for pool in self.dataset_pools:
            n = max(len(pool["data"]), 1)
            pool["index"] = (int(episode) * self.EPISODE_POOL_STRIDE) % n
        self.dataset_index = (int(episode) * self.EPISODE_POOL_STRIDE) % max(
            len(self.dataset) if self.dataset else 1, 1
        )

        trace, t = [], 0.0
        if self.arrival_rate > 0:
            while True:
                t += float(self._arr_rng.exponential(1.0 / self.arrival_rate))
                if t >= duration:
                    break
                trace.append((t, self.get_next_prompt()))
        self._trace = trace
        self._trace_t0 = None
        return len(trace)

    def generate_prompt(self):
        """Generate prompts with Poisson timing and push into queue."""
        if self._trace is not None:
            self._replay_trace()
            return
        while self.running:
            try:
                if hasattr(self.prompt_queue, "qsize") and self.prompt_queue.qsize() >= self.max_queue_size:
                    time.sleep(0.01)
                    continue
                if self.arrival_rate > 0:
                    inter_arrival = float(self._arr_rng.exponential(1.0 / self.arrival_rate))
                    time.sleep(inter_arrival)
                else:
                    time.sleep(0.01)
                prompt_entry = self.get_next_prompt()
                self.prompt_queue.put(prompt_entry)
                self.total_generated += 1
            except Exception as e:
                print(f"Prompt generation error: {e}")
                time.sleep(0.05)

    def _replay_trace(self):
        """Emit the pinned trace against absolute deadlines.

        Deadlines are offsets from t0 rather than cumulative sleeps: a late
        wake-up is absorbed by the next request instead of shifting every
        request after it, so scheduling jitter stays bounded at ~1 sleep
        quantum rather than accumulating over the episode.
        """
        t0 = self._trace_t0 if self._trace_t0 is not None else time.time()
        for offset, entry in self._trace:
            while self.running:
                remaining = (t0 + offset) - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.01))
            if not self.running:
                break
            self.prompt_queue.put(entry)
            self.total_generated += 1
        # Consume the trace. resume_all_servers() restarts the generator after
        # every PPO update, and an exhausted trace has all its deadlines in the
        # past -- without this it would dump the whole episode's requests into
        # the queue at once. begin_episode() installs the next trace.
        self._trace = []

    def start(self, t0: float = None):
        if self.running:
            return self._trace_t0 if self._trace_t0 is not None else self.start_time
        self.running = True
        self.start_time = time.time()
        # Trace offsets are measured from t0; handing the same value back lets
        # the caller use one origin for both the episode clock and arrivals.
        self._trace_t0 = float(t0) if t0 is not None else self.start_time
        self.thread = threading.Thread(target=self.generate_prompt, daemon=True)
        self.thread.start()
        print(f"PoissonPromptGenerator started with rate={self.arrival_rate}")
        return self._trace_t0

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        print("PoissonPromptGenerator stopped.")

    def get_stats(self) -> Dict[str, Any]:
        if self.start_time:
            elapsed = time.time() - self.start_time
            actual_rate = self.total_generated / elapsed if elapsed > 0 else 0.0
        else:
            actual_rate = 0.0
        qsize = self.prompt_queue.qsize() if hasattr(self.prompt_queue, "qsize") else None
        return {
            "total_generated": self.total_generated,
            "queue_size": qsize,
            "configured_rate": self.arrival_rate,
            "actual_rate": actual_rate,
            "is_running": self.running,
            "dataset": "mixed" if self.use_mixed_dataset else self.dataset_name,
            "dataset_config": None if self.use_mixed_dataset else self.dataset_config,
            "dataset_split": None if self.use_mixed_dataset else self.dataset_split,
            "mixed_datasets": [p["name"] for p in self.dataset_pools] if self.use_mixed_dataset else None,
            "prompt_style": self.prompt_style,
        }
