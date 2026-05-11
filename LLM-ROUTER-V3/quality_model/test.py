# pip install -U "torch" "transformers>=4.40.0" "huggingface_hub"

import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Qwen2Model, Qwen2PreTrainedModel
from transformers.utils import ModelOutput


REPO_ID = "lmarena-ai/p2l-3b-grk-01112025"

# -------------------------
# Load model list (maps coef index -> model name)
# -------------------------
model_list_path = hf_hub_download(repo_id=REPO_ID, filename="model_list.json", repo_type="model")
with open(model_list_path, "r") as f:
    model_list = json.load(f)

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

# Some tokenizers may not define pad_token; safe default:
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

assert tokenizer.cls_token is not None, "This model expects a CLS token. Check tokenizer.cls_token in the repo."


# -------------------------
# Model definition (per HF model card)
# -------------------------
@dataclass
class HeadOutputs(ModelOutput):
    coefs: torch.FloatTensor = None
    eta: Optional[torch.FloatTensor] = None
    gamma: Optional[torch.FloatTensor] = None

@dataclass
class P2LOutputs(ModelOutput):
    coefs: torch.FloatTensor = None
    eta: Optional[torch.FloatTensor] = None
    gamma: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None

class RKHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **kwargs) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, output_dim, bias=True)
        self.eta_head = nn.Linear(input_dim, 1, bias=True)

    def forward(self, cls_hidden: torch.Tensor) -> HeadOutputs:
        coefs = self.head(cls_hidden)
        eta = self.eta_head(cls_hidden)
        return HeadOutputs(coefs=coefs, eta=eta)

class P2LModel(Qwen2PreTrainedModel):
    def __init__(self, config, CLS_id: int, num_models: int, head_kwargs=None, **kwargs):
        super().__init__(config)
        self.num_models = num_models
        self.cls_token_id = CLS_id
        self.model = Qwen2Model(config)
        self.head = RKHead(input_dim=config.hidden_size, output_dim=self.num_models, **(head_kwargs or {}))
        self.post_init()

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_mask = (input_ids == self.cls_token_id)
        cls_hidden = hidden[cls_mask]  # expects exactly 1 CLS token per sample
        assert cls_hidden.shape[0] == input_ids.shape[0], "Need exactly one CLS token per input."
        head_out = self.head(cls_hidden)
        return P2LOutputs(coefs=head_out.coefs, eta=head_out.eta, last_hidden_state=cls_hidden)

# -------------------------
# Load weights
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32

model = P2LModel.from_pretrained(
    REPO_ID,
    CLS_id=tokenizer.cls_token_id,
    num_models=len(model_list),
    torch_dtype=dtype,
).to(device).eval()

# -------------------------
# Try it: provide user turns (NOT a batch) — e.g., ["hi!", "what's 1+1?"]
# -------------------------
user_turns = ["Give a novel about a friendly dragon and a brave knight.",]

# Simple formatting: put CLS token explicitly so the model can find it
text = tokenizer.cls_token + "\n" + "\n".join(user_turns)

enc = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    add_special_tokens=False,  # we already injected CLS
)

enc = {k: v.to(device) for k, v in enc.items()}

with torch.inference_mode():
    out = model(**enc)
    coefs = out.coefs[0].float().cpu()          # (num_models,)
    probs = torch.softmax(coefs, dim=-1)        # optional: turn into a routing distribution

topk = 10
vals, idx = torch.topk(probs, k=topk)
print(f"Top-{topk} recommended models (by softmax(coef)):")
for rank, (p, j) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
    print(f"{rank:>2}. p={p:.4f}  coef={coefs[j]:+.3f}  model={model_list[j]}")
print("eta (tie parameter raw):", out.eta[0].item())
