# local_llm/models/bert.py
from __future__ import annotations

import math # for sqrt and constants
from dataclasses import dataclass # to declare `BertConfig` as a dataclass, enabling default values, type hints, a __dict__ that's JSON-friendly
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Controls what gets imported when someone does from local_llm.models.bert import *
__all__ = [
    "BertConfig",
    "BertEmbeddings",
    "BertSelfAttention",
    "BertSelfOutput",
    "BertAttention",
    "BertIntermediate",
    "BertOutput",
    "BertLayer",
    "BertEncoder",
    "BertPooler",
    "BertModel",
    "masked_mean_pool",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BertConfig:
    """
    Minimal configuration for a BERT encoder.

    Fields are aligned with Google's `bert_config.json` schema so we can
    round-trip TF checkpoints without using HuggingFace.
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "new_gelu"  # "gelu" | "relu" | "tanh" | "new_gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02 # included for completeness, but _init_weights hardcodes std=0.02

# classic GELU formula
def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# approximate GELU (used in some transformer variants)
def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# a lookup map from string name in config -> actual function
# used by `BertIntermediate`
ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "new_gelu": new_gelu, 
}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # word_embeddings: maps token IDs → hidden vectors.
        # padding_idx=0 ensures the pad token’s embedding is zeroed in initialization.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # position_embeddings: maps positions (0..max_position_embeddings) → vectors.
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # token_type_embeddings: segment IDs (0/1 for sentence A/B).
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # LayerNorm + dropout as in original BERT.
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # Sanity check: prevent feeding sequences longer than the learned positional table
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.position_embeddings.num_embeddings}."
            )
        # If caller doesn’t provide explicit positions, create [0,1,...,seq_len-1] and broadcast to the batch.
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(bsz, seq_len)
        # Default all tokens to segment 0 if not specified.
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # Sum the three embeddings → [batch, seq_len, hidden_size].
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # guard that head dimension is an integer
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # scale implments th 1/sqrt(d_k) factor from scaled dot-product attention.
        self.scale = self.attention_head_size ** -0.5

    # Reshape from [B, T, H] → [B, heads, T, head_dim].
    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    # scores shape: [B, heads, T, T].
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._shape(self.query(hidden_states))
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attnetion_mask is expected to be broadcastable mask with large negative values on padded
        # positions so softmax ~ 0 there. 
        if attention_mask is not None:
            scores = scores + attention_mask

        # weight sum over values -> context tensor
        # reshape back to [B, T, H]
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        return ctx.view(hidden_states.size(0), -1, self.all_head_size)


class BertSelfOutput(nn.Module):
    # standard "post-attention" block
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # residual connection + LayerNorm
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.dropout(x)
        return self.LayerNorm(x + input_tensor)

# wrapper to combine self-attention with its output projects + residual norm
class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn = self.self(x, attention_mask)
        return self.output(attn, x)

# the "inner" FFN layer and linear + nonlinearity.
class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(x))

# second FFN layer back down to hidden size + residual + norm
class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        return self.LayerNorm(x + residual)

# one full transfomer block
class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(x, attention_mask)
        inter = self.intermediate(attn_out)
        return self.output(inter, attn_out)

# stack of num_hidden_layers BERT Layers. Simple loop; not outputs-per-layer, just final hidden state.
class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x

# Takes the hidden state at postion 0 (the [CLS] token) and runs it through a dense + tanh.
# This gives the "pooled output" used by classification heads. 
class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(x[:, 0]))


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        """
        - Store config
        - build embedding stack, encoder stack, pooler
        - calls self.apply(..) which recursively applies _init_weights to all submodules
        """
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        - BERT-Style initialization:
            - Linear + embedding weights ~ N(0, 0.02).
            - Biases zero
            - LayerNorm gamma = 1, beta = 0
            - padding embedding row zeroed out
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _extend_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        extended = attention_mask[:, None, None, :].to(dtype=dtype)
        extended = (1.0 - extended) * -10000.0
        return extended
    # Input attention_mask is [B, T] with 1 for valid tokens, 0 for padding.
    # Expand to [B, 1, 1, T] so it broadcasts over heads and query positions.
    # Transform:
        # valid tokens: 1 → (1-1)*-10000 = 0
        # padding: 0 → (1-0)*-10000 = -10000
    # Adding this to scores before softmax masks out padding.

    def set_finetune_policy(
        self,
        policy: str = "none",  # "none", "last_n", "full"
        last_n: int = 0,
        train_embeddings: bool = False,
    ) -> None:
        """
        Apply a simple finetuning policy to encoder parameters.

        - "none"  : encoder + embeddings are frozen
        - "full"  : everything trainable
        - "last_n": last N transformer layers trainable (embeddings frozen unless train_embeddings=True)
        """
        policy = policy.lower().strip()
        if policy not in {"none", "last_n", "full"}:
            raise ValueError(f"Unknown finetune policy: {policy}")

        # Start by freezing everything in encoder + embeddings (set requires_grad=False)
        for p in self.embeddings.parameters():
            p.requires_grad = False
        for layer in self.encoder.layer:
            for p in layer.parameters():
                p.requires_grad = False

        # if policy is none: leave everything frozen
        if policy == "none":
            return
        
        # if policy is full: 
            # encoder layer have requires_grad=True.
            # embeddings trainable only if train_embeddings=True.
        if policy == "full":
            for p in self.embeddings.parameters():
                p.requires_grad = train_embeddings
            for layer in self.encoder.layer:
                for p in layer.parameters():
                    p.requires_grad = True
            return
        
        # if policy is last_n:
            # compute n = min(last_n, num_layers)
            # optionally unfreeze embeddings
            # unfreeze parameters only in the last n encoder layers. 
        if policy == "last_n":
            n = max(0, min(last_n, len(self.encoder.layer)))
            if train_embeddings:
                for p in self.embeddings.parameters():
                    p.requires_grad = True
            for layer in self.encoder.layer[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            return
    
    # Perform end-to-end forward pass:
        # embeddings -> encoder -> pooler
        # return a dict, BERT-style:
            # "last_hidden_state": [B, T, H]
            # "pooled_output"    : [B, H]
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.embeddings(input_ids, token_type_ids, position_ids)

        if attention_mask is not None:
            extended_mask = self._extend_attention_mask(attention_mask, dtype=x.dtype)
        else:
            extended_mask = None

        seq = self.encoder(x, extended_mask)
        pooled = self.pooler(seq)

        return {"last_hidden_state": seq, "pooled_output": pooled}

    # Convenience utilities

    def freeze_encoder(self) -> None:
        for p in self.embeddings.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        if n <= 0:
            return
        n = min(n, len(self.encoder.layer))
        for layer in self.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

def masked_mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    This is for mean pooling over non-padded tokens:
        - attention_mask ([B, H]) -> add last dimension ([B, T, 1]) and cast to same dtype.
        - multiply hidden states by mask so padded tokens become zero. 
        - sum across sequence dimension → [B, H].
        - divide by number of valid tokens per example.
        - clamp(min=1e-9) prevents divide-by-zero if someone passes all-zeros mask.
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom
