"""
Tests for local_llm.models.bert

Covers:
- BertConfig
- gelu / new_gelu / ACT2FN
- BertEmbeddings
- BertSelfAttention
- BertSelfOutput
- BertAttention
- BertIntermediate
- BertOutput
- BertLayer
- BertEncoder
- BertPooler
- BertModel (including _extend_attention_mask, freeze/unfreeze, finetune policy)
- masked_mean_pool
"""

from __future__ import annotations

import math

import pytest
import torch

import local_llm.models.bert as bm
from local_llm.models.bert import (
    BertConfig,
    BertEmbeddings,
    BertSelfAttention,
    BertSelfOutput,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayer,
    BertEncoder,
    BertPooler,
    BertModel,
    masked_mean_pool,
)


# ---------------------------------------------------------------------------
# BertConfig + activations
# ---------------------------------------------------------------------------


def test_bert_config_defaults_and_overrides():
    cfg = BertConfig()
    assert cfg.vocab_size == 30522
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 12
    assert cfg.intermediate_size == 3072
    assert cfg.hidden_act == "new_gelu"

    cfg2 = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
    )
    assert cfg2.vocab_size == 1000
    assert cfg2.hidden_size == 64
    assert cfg2.num_hidden_layers == 3
    assert cfg2.num_attention_heads == 4
    assert cfg2.intermediate_size == 128
    assert cfg2.hidden_act == "gelu"


def test_gelu_and_new_gelu_shape_and_sensible_ordering():
    x = torch.linspace(-3, 3, steps=13)
    y_gelu = bm.gelu(x)
    y_new_gelu = bm.new_gelu(x)

    # Shapes preserved
    assert y_gelu.shape == x.shape
    assert y_new_gelu.shape == x.shape

    # Basic sanity: for large negative < near zero < large positive
    mid = len(x) // 2  # index of 0
    assert x[mid].abs() < 1e-6  # confirm it's ~0

    # gelu
    assert y_gelu[0] < y_gelu[mid] < y_gelu[-1]
    # new_gelu
    assert y_new_gelu[0] < y_new_gelu[mid] < y_new_gelu[-1]

    # For positive inputs, outputs should be between 0 and x (GELU-like behavior)
    pos_mask = x > 0
    assert torch.all(y_gelu[pos_mask] >= 0)
    assert torch.all(y_gelu[pos_mask] <= x[pos_mask] + 1e-6)
    assert torch.all(y_new_gelu[pos_mask] >= 0)
    assert torch.all(y_new_gelu[pos_mask] <= x[pos_mask] + 1e-6)


def test_act2fn_mappings_and_behavior():
    assert set(bm.ACT2FN.keys()) == {"gelu", "relu", "tanh", "new_gelu"}

    x = torch.randn(4, 5)

    # gelu
    y1 = bm.ACT2FN["gelu"](x)
    assert y1.shape == x.shape

    # relu
    y2 = bm.ACT2FN["relu"](x)
    assert y2.shape == x.shape
    assert torch.all(y2 >= 0)

    # tanh
    y3 = bm.ACT2FN["tanh"](x)
    assert y3.shape == x.shape
    assert torch.all((y3 >= -1.0) & (y3 <= 1.0))

    # new_gelu: mapping is to the new_gelu *function*
    assert bm.ACT2FN["new_gelu"] is bm.new_gelu

    # Call to verify it behaves sensibly
    y4 = bm.ACT2FN["new_gelu"](x)
    assert y4.shape == x.shape
    # Optional: basic sanity check that it's not crazy
    assert torch.isfinite(y4).all()


# ---------------------------------------------------------------------------
# BertEmbeddings
# ---------------------------------------------------------------------------


def _small_cfg() -> BertConfig:
    return BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=2,
        hidden_act="gelu",
    )


def test_bert_embeddings_basic_forward():
    cfg = _small_cfg()
    emb = BertEmbeddings(cfg)
    emb.eval()  # disable dropout for deterministic behavior

    bsz, seq_len = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
    out = emb(input_ids)

    assert out.shape == (bsz, seq_len, cfg.hidden_size)
    # Check that different tokens produce different embeddings at least somewhere
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_bert_embeddings_uses_default_token_type_and_position_ids():
    cfg = _small_cfg()
    emb = BertEmbeddings(cfg)
    emb.eval()

    bsz, seq_len = 1, 4
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
    out_default = emb(input_ids)

    # Provide explicit token_type_ids and position_ids that match defaults
    token_type_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(input_ids)
    out_explicit = emb(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    assert torch.allclose(out_default, out_explicit, atol=1e-5)


def test_bert_embeddings_raises_if_seq_exceeds_max_position():
    cfg = _small_cfg()
    emb = BertEmbeddings(cfg)
    bsz, seq_len = 1, cfg.max_position_embeddings + 1
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))

    with pytest.raises(ValueError):
        _ = emb(input_ids)


# ---------------------------------------------------------------------------
# BertSelfAttention + BertSelfOutput + BertAttention
# ---------------------------------------------------------------------------


def test_bert_self_attention_hidden_size_must_divide_heads():
    cfg = _small_cfg()
    bad_cfg = BertConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=30,  # not divisible by 4
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    with pytest.raises(ValueError):
        _ = BertSelfAttention(bad_cfg)


def test_bert_self_attention_shapes_and_mask_application():
    cfg = _small_cfg()
    sa = BertSelfAttention(cfg)

    bsz, seq_len = 2, 5
    hidden_states = torch.randn(bsz, seq_len, cfg.hidden_size)

    # Attention mask with shape [bsz, 1, 1, seq_len]
    # Mask out last token strongly
    attention_mask = torch.zeros(bsz, 1, 1, seq_len)
    attention_mask[:, :, :, -1] = -10000.0

    out = sa(hidden_states, attention_mask=attention_mask)
    assert out.shape == (bsz, seq_len, cfg.hidden_size)

    # Without mask should also work
    out2 = sa(hidden_states, attention_mask=None)
    assert out2.shape == (bsz, seq_len, cfg.hidden_size)


def test_bert_self_output_residual_connection():
    cfg = _small_cfg()
    so = BertSelfOutput(cfg)
    so.eval()

    bsz, seq_len = 2, 4
    hidden_states = torch.randn(bsz, seq_len, cfg.hidden_size)
    residual = torch.randn_like(hidden_states)

    out = so(hidden_states, residual)
    assert out.shape == hidden_states.shape

    # With zero dense weights, output ~ residual (after LN)
    with torch.no_grad():
        so.dense.weight.zero_()
        so.dense.bias.zero_()
        out2 = so(hidden_states, residual)
        assert out2.shape == residual.shape


def test_bert_attention_composition():
    cfg = _small_cfg()
    att = BertAttention(cfg)
    att.eval()

    bsz, seq_len = 2, 6
    x = torch.randn(bsz, seq_len, cfg.hidden_size)
    out = att(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# BertIntermediate + BertOutput + BertLayer
# ---------------------------------------------------------------------------


def test_bert_intermediate_uses_config_activation():
    cfg = _small_cfg()
    cfg.hidden_act = "relu"
    inter = BertIntermediate(cfg)
    inter.eval()

    x = torch.randn(2, 4, cfg.hidden_size)
    out = inter(x)
    assert out.shape == (2, 4, cfg.intermediate_size)
    # ReLU: no negatives when run on dense output
    assert torch.all(out >= 0)


def test_bert_output_shapes_and_residual():
    cfg = _small_cfg()
    out_layer = BertOutput(cfg)
    out_layer.eval()

    bsz, seq_len = 2, 3
    inter = torch.randn(bsz, seq_len, cfg.intermediate_size)
    residual = torch.randn(bsz, seq_len, cfg.hidden_size)

    out = out_layer(inter, residual)
    assert out.shape == (bsz, seq_len, cfg.hidden_size)


def test_bert_layer_end_to_end():
    cfg = _small_cfg()
    layer = BertLayer(cfg)
    layer.eval()

    bsz, seq_len = 2, 4
    x = torch.randn(bsz, seq_len, cfg.hidden_size)
    out = layer(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# BertEncoder
# ---------------------------------------------------------------------------


def test_bert_encoder_stacks_layers():
    cfg = _small_cfg()
    cfg.num_hidden_layers = 3
    enc = BertEncoder(cfg)
    enc.eval()

    bsz, seq_len = 2, 5
    x = torch.randn(bsz, seq_len, cfg.hidden_size)

    out = enc(x)
    assert out.shape == (bsz, seq_len, cfg.hidden_size)
    assert len(enc.layer) == cfg.num_hidden_layers


# ---------------------------------------------------------------------------
# BertPooler
# ---------------------------------------------------------------------------


def test_bert_pooler_pools_first_token_only():
    cfg = _small_cfg()
    pool = BertPooler(cfg)
    pool.eval()

    bsz, seq_len = 2, 5
    hidden = torch.randn(bsz, seq_len, cfg.hidden_size)
    out = pool(hidden)

    assert out.shape == (bsz, cfg.hidden_size)

    # If we zero out the first token and keep others nonzero, the result should change
    hidden2 = hidden.clone()
    hidden2[:, 0, :] = 0.0
    out2 = pool(hidden2)
    assert not torch.allclose(out, out2)


# ---------------------------------------------------------------------------
# BertModel
# ---------------------------------------------------------------------------


def test_bert_model_init_and_forward_basic():
    cfg = _small_cfg()
    model = BertModel(cfg)
    model.eval()

    bsz, seq_len = 2, 6
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )

    assert "last_hidden_state" in out and "pooled_output" in out
    last_hidden = out["last_hidden_state"]
    pooled = out["pooled_output"]

    assert last_hidden.shape == (bsz, seq_len, cfg.hidden_size)
    assert pooled.shape == (bsz, cfg.hidden_size)


def test_bert_model_forward_without_attention_mask():
    cfg = _small_cfg()
    model = BertModel(cfg)
    model.eval()

    bsz, seq_len = 1, 4
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
    out = model(input_ids=input_ids)

    assert out["last_hidden_state"].shape == (bsz, seq_len, cfg.hidden_size)
    assert out["pooled_output"].shape == (bsz, cfg.hidden_size)


def test_bert_model_extend_attention_mask_values_and_dtype():
    bsz, seq_len = 2, 5
    attn = torch.tensor([[1, 1, 0, 1, 0],
                         [1, 0, 0, 1, 1]], dtype=torch.long)
    dtype = torch.float32

    extended = BertModel._extend_attention_mask(attn, dtype=dtype)
    assert extended.shape == (bsz, 1, 1, seq_len)
    assert extended.dtype == dtype

    # Squeeze to [bsz, seq_len] so it matches `attn`
    ext2 = extended.squeeze(1).squeeze(1)
    assert ext2.shape == attn.shape

    # Positions with 1 in the mask → 0; 0 → -10000
    assert torch.all(ext2[attn == 1] == 0.0)
    assert torch.all(ext2[attn == 0] == -10000.0)



def test_bert_model_init_weights_properties():
    cfg = _small_cfg()
    model = BertModel(cfg)
    # Check some modules for proper initialization
    linear = None
    emb = None
    ln = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and linear is None:
            linear = m
        if isinstance(m, torch.nn.Embedding) and emb is None:
            emb = m
        if isinstance(m, torch.nn.LayerNorm) and ln is None:
            ln = m
        if linear and emb and ln:
            break

    assert linear is not None and emb is not None and ln is not None

    # Linear weights are normal with mean near 0
    mean_val = float(linear.weight.detach().mean())
    assert abs(mean_val) < 0.1
    # LayerNorm params
    assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))


# ---------------------------------------------------------------------------
# Finetune policy and freeze/unfreeze helpers
# ---------------------------------------------------------------------------


def test_freeze_encoder_sets_requires_grad_false():
    cfg = _small_cfg()
    model = BertModel(cfg)

    model.freeze_encoder()

    for p in model.embeddings.parameters():
        assert not p.requires_grad
    for p in model.encoder.parameters():
        assert not p.requires_grad
    # Pooler should remain trainable
    for p in model.pooler.parameters():
        assert p.requires_grad


def test_unfreeze_last_n_layers_behavior():
    cfg = _small_cfg()
    cfg.num_hidden_layers = 4
    model = BertModel(cfg)
    model.freeze_encoder()

    # Unfreeze last 2 layers
    model.unfreeze_last_n_layers(2)
    assert len(model.encoder.layer) == 4

    # First two layers remain frozen
    for p in model.encoder.layer[0].parameters():
        assert not p.requires_grad
    for p in model.encoder.layer[1].parameters():
        assert not p.requires_grad

    # Last two layers unfrozen
    for p in model.encoder.layer[2].parameters():
        assert p.requires_grad
    for p in model.encoder.layer[3].parameters():
        assert p.requires_grad

    # n <= 0 does nothing
    model.freeze_encoder()
    model.unfreeze_last_n_layers(0)
    for p in model.encoder.parameters():
        assert not p.requires_grad


def test_set_finetune_policy_none_freezes_everything():
    cfg = _small_cfg()
    cfg.num_hidden_layers = 3
    model = BertModel(cfg)

    model.set_finetune_policy(policy="none")

    for p in model.embeddings.parameters():
        assert not p.requires_grad
    for layer in model.encoder.layer:
        for p in layer.parameters():
            assert not p.requires_grad


def test_set_finetune_policy_full_with_and_without_embeddings():
    cfg = _small_cfg()
    cfg.num_hidden_layers = 2
    model = BertModel(cfg)

    # full, train_embeddings=False
    model.set_finetune_policy(policy="full", train_embeddings=False)
    for p in model.embeddings.parameters():
        assert not p.requires_grad
    for layer in model.encoder.layer:
        for p in layer.parameters():
            assert p.requires_grad

    # full, train_embeddings=True
    model.set_finetune_policy(policy="full", train_embeddings=True)
    for p in model.embeddings.parameters():
        assert p.requires_grad
    for layer in model.encoder.layer:
        for p in layer.parameters():
            assert p.requires_grad


def test_set_finetune_policy_last_n_layers_and_bounds():
    cfg = _small_cfg()
    cfg.num_hidden_layers = 3
    model = BertModel(cfg)

    # last_n = 2, embeddings frozen
    model.set_finetune_policy(policy="last_n", last_n=2, train_embeddings=False)
    # embeddings frozen
    for p in model.embeddings.parameters():
        assert not p.requires_grad
    # first layer frozen, last 2 layers trainable
    for p in model.encoder.layer[0].parameters():
        assert not p.requires_grad
    for p in model.encoder.layer[1].parameters():
        assert p.requires_grad
    for p in model.encoder.layer[2].parameters():
        assert p.requires_grad

    # last_n > num_hidden_layers should clamp to all layers
    model.set_finetune_policy(policy="last_n", last_n=10, train_embeddings=True)
    for p in model.embeddings.parameters():
        assert p.requires_grad
    for layer in model.encoder.layer:
        for p in layer.parameters():
            assert p.requires_grad


def test_set_finetune_policy_invalid_raises():
    cfg = _small_cfg()
    model = BertModel(cfg)

    with pytest.raises(ValueError):
        model.set_finetune_policy(policy="invalid_mode")


# ---------------------------------------------------------------------------
# masked_mean_pool
# ---------------------------------------------------------------------------


def test_masked_mean_pool_regular_case():
    bsz, seq_len, hidden = 2, 4, 8
    last_hidden = torch.arange(bsz * seq_len * hidden, dtype=torch.float32).reshape(
        bsz, seq_len, hidden
    )
    attention_mask = torch.tensor([[1, 1, 0, 0],
                                   [1, 0, 1, 0]], dtype=torch.float32)

    pooled = masked_mean_pool(last_hidden, attention_mask)

    assert pooled.shape == (bsz, hidden)

    # For first example: mean of first two positions
    expected0 = (last_hidden[0, 0] + last_hidden[0, 1]) / 2.0
    assert torch.allclose(pooled[0], expected0, atol=1e-6)


def test_masked_mean_pool_all_padded_rows_safe():
    bsz, seq_len, hidden = 1, 3, 5
    last_hidden = torch.randn(bsz, seq_len, hidden)
    attention_mask = torch.zeros(bsz, seq_len)

    pooled = masked_mean_pool(last_hidden, attention_mask)
    assert pooled.shape == (bsz, hidden)
    # Numerator will be zero, denom clamped, so result should be finite and ~0
    assert torch.all(torch.isfinite(pooled))
    assert torch.allclose(pooled, torch.zeros_like(pooled), atol=1e-6)
