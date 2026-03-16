import numpy as np
import time
import threading
from typing import Optional, Dict, Any, List, Tuple
import random

from datasets import load_dataset


class PoissonPromptGenerator:
    """Generates prompts according to a Poisson arrival process.

    This generator supports BOTH:
      - Alpaca-style instruction datasets (instruction/input/output)
      - QA datasets (question/context/answer/answers), e.g., HotpotQA, SQuAD, NQ, TriviaQA, etc.

    It always emits dicts with at least:
      - 'prompt': str
      - 'output': gold answer(s) (str or List[str])
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
    ):
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

        np.random.seed(12)
        random.seed(12)

        self.dataset = None
        self.dataset_index = 0
        self.load_dataset(self.dataset_name, self.dataset_config, self.dataset_split)

    # -------------------------
    # Dataset loading
    # -------------------------
    def load_dataset(self, dataset_name: str, dataset_config: Optional[str], dataset_split: str):
        """Load and cache dataset into memory (as a python list)."""
        try:
            if dataset_config:
                ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
            else:
                ds = load_dataset(dataset_name, split=dataset_split)

            # Some datasets return DatasetDict when split is omitted; we don't omit split,
            # but keep a small safeguard.
            if hasattr(ds, "keys") and not isinstance(ds, list) and not hasattr(ds, "__len__"):
                # unexpected type: fall back
                raise ValueError("Unexpected dataset object")

            # Convert to list for fast indexing in a background thread
            self.dataset = list(ds)

            print(f"Loaded {len(self.dataset)} samples from {dataset_name} (config={dataset_config}, split={dataset_split})")
        except Exception as e:
            print(f"Error loading dataset {dataset_name} (config={dataset_config}, split={dataset_split}): {e}")
            self._create_dummy_dataset()

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

    # -------------------------
    # Prompt extraction helpers
    # -------------------------
    @staticmethod
    def _truncate(s: str, max_chars: int) -> str:
        s = s or ""
        if max_chars is None or max_chars <= 0:
            return s
        return s[:max_chars]

    def _extract_question(self, sample: Dict[str, Any]) -> str:
        # Alpaca-style
        if isinstance(sample.get("instruction"), str) and sample.get("instruction").strip():
            return sample.get("instruction").strip()

        # QA-style
        for k in ["question", "query", "prompt", "title"]:
            v = sample.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Fallback
        return "Answer the question."

    def _extract_context(self, sample: Dict[str, Any]) -> str:
        # Alpaca-style
        if isinstance(sample.get("input"), str) and sample.get("input").strip():
            return sample.get("input").strip()

        # SQuAD-style (context is a single passage string)
        ctx = sample.get("context")
        if isinstance(ctx, str) and ctx.strip():
            return ctx.strip()

        # HotpotQA-style: context is a dict with title/sentences
        ctx = sample.get("context")
        if isinstance(ctx, dict):
            titles = ctx.get("title") or ctx.get("titles")
            sents = ctx.get("sentences") or ctx.get("sentence") or ctx.get("sent")
            if isinstance(titles, list) and isinstance(sents, list):
                parts = []
                for i, (t, ss) in enumerate(zip(titles, sents)):
                    if i >= self.qa_max_context_docs:
                        break
                    title = str(t).strip()
                    if isinstance(ss, list):
                        para = " ".join(str(x).strip() for x in ss if str(x).strip())
                    else:
                        para = str(ss).strip()
                    if title:
                        parts.append(f"{title}: {para}".strip())
                    else:
                        parts.append(para)
                joined = "\n\n".join(p for p in parts if p)
                return self._truncate(joined, self.qa_max_context_chars).strip()

        # Some datasets store passages/docs
        for k in ["passage", "paragraph", "article", "document"]:
            v = sample.get(k)
            if isinstance(v, str) and v.strip():
                return self._truncate(v.strip(), self.qa_max_context_chars)

        # List contexts: join
        if isinstance(ctx, list) and ctx:
            parts = []
            for item in ctx[: self.qa_max_context_docs]:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    # try common keys
                    for kk in ["text", "context", "paragraph", "sentence"]:
                        if isinstance(item.get(kk), str) and item.get(kk).strip():
                            parts.append(item.get(kk).strip())
                            break
            joined = "\n\n".join(p for p in parts if p)
            return self._truncate(joined, self.qa_max_context_chars).strip()

        return ""

    def _extract_gold_answers(self, sample: Dict[str, Any]) -> List[str]:
        answers: List[str] = []

        # Alpaca-style
        if "output" in sample and sample.get("output") is not None:
            v = sample.get("output")
            if isinstance(v, str) and v.strip():
                answers.append(v.strip())
            elif isinstance(v, list):
                answers.extend([str(x).strip() for x in v if str(x).strip()])

        # Common QA-style fields
        if not answers and sample.get("answer") is not None:
            a = sample.get("answer")
            if isinstance(a, str) and a.strip():
                answers.append(a.strip())
            elif isinstance(a, list):
                answers.extend([str(x).strip() for x in a if str(x).strip()])
            elif isinstance(a, dict):
                # TriviaQA uses {value, aliases}
                if a.get("value") is not None:
                    answers.append(str(a.get("value")).strip())
                aliases = a.get("aliases") or a.get("alias") or []
                if isinstance(aliases, list):
                    answers.extend([str(x).strip() for x in aliases if str(x).strip()])

        if sample.get("answers") is not None:
            a = sample.get("answers")
            if isinstance(a, dict):
                # SQuAD: answers['text'] is list
                txt = a.get("text") or a.get("texts")
                if isinstance(txt, list):
                    answers.extend([str(x).strip() for x in txt if str(x).strip()])
                elif isinstance(txt, str) and txt.strip():
                    answers.append(txt.strip())
            elif isinstance(a, list):
                answers.extend([str(x).strip() for x in a if str(x).strip()])

        # Some datasets use target/label
        for k in ["target", "label", "gold", "ground_truth"]:
            v = sample.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                answers.append(v.strip())
            elif isinstance(v, list):
                answers.extend([str(x).strip() for x in v if str(x).strip()])

        # De-duplicate, keep order
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
            suffix = ("\nFirstly, give your thought and then return the final answer in the following XML tag format:\n"
                      f"<{tag}>FINAL_ANSWER</{tag}>")

        if style in {"instruction", "alpaca"}:
            if context:
                return f"Instruction: {question}\nInput: {context}\nResponse:" + suffix
            return f"Instruction: {question}\nResponse:" + suffix

        if style == "plain":
            if context:
                return f"Question: {question}\nContext: {context}\nAnswer:" + suffix
            return f"Question: {question}\nAnswer:" + suffix

        # default fallback
        if context:
            return f"Instruction: {question}\nInput: {context}\nResponse:" + suffix
        return f"Instruction: {question}\nResponse:" + suffix

    # -------------------------
    # Core API
    # -------------------------
    def get_next_prompt(self) -> Dict[str, Any]:
        """Get next prompt + ground-truth output from dataset."""
        if not self.dataset:
            return {"prompt": "Instruction: Explain artificial intelligence.\nResponse:", "output": ""}

        sample = self.dataset[self.dataset_index]
        self.dataset_index = (self.dataset_index + 1) % len(self.dataset)

        if not isinstance(sample, dict):
            # HF datasets always yield dicts; be safe
            sample = {"text": str(sample)}

        question = self._extract_question(sample)
        context = self._extract_context(sample)
        golds = self._extract_gold_answers(sample)

        prompt = self._build_prompt(question, context)

        # For compatibility with your trainer/env logic:
        # - output can be a string or a list; environment.match_quality_score supports lists via _as_list().
        output: Any
        if len(golds) == 0:
            output = ""
        elif len(golds) == 1:
            output = golds[0]
        else:
            output = golds

        return {
            "prompt": prompt,
            "output": output,
            "instruction": question,
            "input": context,
            "dataset": self.dataset_name,
        }

    def generate_prompt(self):
        """Generate prompts with Poisson timing and push into queue."""
        while self.running:
            try:
                # Respect max queue size
                if hasattr(self.prompt_queue, "qsize") and self.prompt_queue.qsize() >= self.max_queue_size:
                    time.sleep(0.01)
                    continue

                # Wait for the next Poisson arrival
                if self.arrival_rate > 0:
                    inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
                    time.sleep(float(inter_arrival))
                else:
                    time.sleep(0.01)

                prompt_entry = self.get_next_prompt()
                self.prompt_queue.put(prompt_entry)
                self.total_generated += 1
            except Exception as e:
                # Don't crash the background thread
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
            "dataset": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "prompt_style": self.prompt_style,
        }
