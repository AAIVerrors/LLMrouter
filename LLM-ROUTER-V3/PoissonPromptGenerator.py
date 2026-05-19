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
    ):
        self.shuffle_dataset = bool(shuffle_dataset)
        self.dataset_seed = int(dataset_seed)
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

    @staticmethod
    def _choice_label(i: int) -> str:
        return ["A", "B", "C", "D", "E", "F"][int(i)]

    def _is_mmlu_sample(self, sample: Dict[str, Any]) -> bool:
        return (
            isinstance(sample, dict)
            and isinstance(sample.get("question"), str)
            and isinstance(sample.get("choices"), (list, tuple))
            and sample.get("answer") is not None
        )

    def _mmlu_gold_letter(self, sample: Dict[str, Any]) -> str:
        ans = sample.get("answer")
        if isinstance(ans, (int, np.integer)):
            return self._choice_label(int(ans))
        if isinstance(ans, str):
            s = ans.strip()
            if s.isdigit():
                return self._choice_label(int(s))
            if len(s) == 1 and s.upper() in ["A", "B", "C", "D", "E", "F"]:
                return s.upper()
        return str(ans).strip()

    def _build_mmlu_prompt(self, sample: Dict[str, Any]) -> str:
        question = str(sample.get("question", "")).strip()
        choices = list(sample.get("choices", []))
        subject = str(sample.get("subject", "")).replace("_", " ").strip()
        choice_lines = [f"{self._choice_label(i)}. {str(c).strip()}" for i, c in enumerate(choices)]
        subject_line = f"Subject: {subject}\n" if subject else ""
        tag = self.final_tag
        if self.force_final_tag:
            suffix = (
                f"\nThink briefly and return only one option letter inside <{tag}>...</{tag}>.\n"
                f"The answer must be one of A, B, C, or D.\n"
            )
        else:
            suffix = "\nReturn only one option letter. Do not explain.\n"
        return subject_line + f"Question: {question}\nChoices:\n" + "\n".join(choice_lines) + f"\nAnswer:{suffix}"

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
        question = str(sample.get("question", "")).strip()
        tag = self.final_tag
        if self.force_final_tag:
            suffix = (
                f"\nSolve step by step if needed. At the end, output only the final numeric answer "
                f"inside <{tag}>...</{tag}>. Do not include units or commas inside the tag.\n"
            )
        else:
            suffix = "\nReturn only the final numeric answer.\n"
        return f"Question: {question}\nAnswer:{suffix}"

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
        suffix = ""
        if self.force_final_tag:
            tag = self.final_tag
            suffix = (
                f"\nThink first and then output the final answer inside <{tag}>...</{tag}>.\n"
                f"Do not output an empty tag.\n"
                f"Do not output placeholders like FINAL_ANSWER.\n"
                f"Do not output your thought process inside the tag.\n"
            )
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
        pool_idx = int(np.random.choice(len(self.dataset_pools), p=np.asarray(self.dataset_weights, dtype=np.float64)))
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
            choices = sample.get("choices", [])
            subject = str(sample.get("subject", "")).strip()
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

    def generate_prompt(self):
        """Generate prompts with Poisson timing and push into queue."""
        while self.running:
            try:
                if hasattr(self.prompt_queue, "qsize") and self.prompt_queue.qsize() >= self.max_queue_size:
                    time.sleep(0.01)
                    continue
                if self.arrival_rate > 0:
                    inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
                    time.sleep(float(inter_arrival))
                else:
                    time.sleep(0.01)
                prompt_entry = self.get_next_prompt()
                self.prompt_queue.put(prompt_entry)
                self.total_generated += 1
            except Exception as e:
                print(f"Prompt generation error: {e}")
                time.sleep(0.05)

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.generate_prompt, daemon=True)
        self.thread.start()
        print(f"PoissonPromptGenerator started with rate={self.arrival_rate}")

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
