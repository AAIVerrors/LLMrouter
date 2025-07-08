'''
@misc{frick2025prompttoleaderboard,
      title={Prompt-to-Leaderboard}, 
      author={Evan Frick and Connor Chen and Joseph Tennyson and Tianle Li and Wei-Lin Chiang and Anastasios N. Angelopoulos and Ion Stoica},
      year={2025},
      eprint={2502.14855},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.14855}, 
}
'''

import torch
from transformers import (
    Qwen2Model,
    Qwen2PreTrainedModel,
    LlamaModel,
    LlamaPreTrainedModel,
    PreTrainedModel,
    AutoTokenizer,
)
from transformers import AutoTokenizer
from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Callable, Optional
from huggingface_hub import hf_hub_download
import json


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
    def __init__(
        self,
        input_dim,
        output_dim,
        **kwargs,
    ) -> None:
        super().__init__()
        self.head = nn.Linear(
            in_features=input_dim, out_features=output_dim, bias=True
        )
        self.eta_head = nn.Linear(
            in_features=input_dim, out_features=1, bias=True
        )

    def forward(self, last_hidden_dim: torch.Tensor):
        coefs = self.head(last_hidden_dim)
        eta = self.eta_head(last_hidden_dim)

        return HeadOutputs(coefs=coefs, eta=eta)

class P2LModel(Qwen2PreTrainedModel):
    def __init__(
        self,
        config,
        CLS_id,
        num_models,
        head_kwargs={},
        **kwargs,
    ):
        super().__init__(config)

        self.num_models = num_models
        self.cls_token_id = CLS_id

        self.model = Qwen2Model(config)

        self.head = RKHead(
            input_dim=config.hidden_size,
            output_dim=self.num_models,
            **head_kwargs,
        )

        self.post_init()

    def freeze_transformer(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids, attention_mask, labels=None, weights=None):
        batch_size = input_ids.shape[0]

        hidden_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        ).last_hidden_state  # (bs, num_token, embed_dim)

        cls_mask = input_ids == self.cls_token_id

        # double check this is getting the current CLS token
        cls_hidden_dim = hidden_outputs[cls_mask]

        assert (
            cls_hidden_dim.shape[0] == batch_size
        ), f"input ids {input_ids.shape}, cls_mask {cls_mask.shape}, cls_logit {cls_hidden_dim.shape}"

        head_output = self.head(cls_hidden_dim)

    
        outputs = P2LOutputs(
            coefs=head_output.coefs,
            last_hidden_state=cls_hidden_dim,
            eta=head_output.eta,
            gamma=head_output.gamma,
        )

        return outputs



