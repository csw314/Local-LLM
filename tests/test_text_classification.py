"""
Tests for local_llm.pipelines.text_classification

Covers:
- ClassifierHeadConfig
- BertClassifierHead
- BertTextClassifier (init, from_pretrained, save_pretrained, _pool, forward)
- build_bert_input_encoder

We rely on the real BertModel/BertConfig but with small configs to keep tests light.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest
import torch
import torch.nn as nn

from local_llm.models.bert import BertConfig, BertModel
from local_llm.pipelines import text_classification as tc
from local_llm.tokenization.bert_wordpiece import BertInputEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_bert_config() -> BertConfig:
    """Tiny config to keep tests fast."""
    return BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        type_vocab_size=2,
    )


def _build_dummy_assets_dir(tmp_path: Path) -> Path:
    """
    Create a temporary assets directory with:
      - config.json for BertConfig
      - pytorch_model.bin for a small BertModel state dict
    """
    cfg = _small_bert_config()
    bert = BertModel(cfg)

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    cfg_path = assets_dir / "config.json"
    weights_path = assets_dir / "pytorch_model.bin"

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f)

    torch.save(bert.state_dict(), weights_path)
    return assets_dir


def _make_vocab_file(dir_path: Path) -> Path:
    """
    Write a minimal BERT-style vocab.txt compatible with load_vocab.
    """
    vocab_path = dir_path / "vocab.txt"
    lines = [
        "[PAD]\n",
        "[UNK]\n",
        "[CLS]\n",
        "[SEP]\n",
        "[MASK]\n",
        "this\n",
        "is\n",
        "a\n",
        "test\n",
        ".\n",
    ]
    vocab_path.write_text("".join(lines), encoding="utf-8")
    return vocab_path


# ---------------------------------------------------------------------------
# ClassifierHeadConfig
# ---------------------------------------------------------------------------

def test_classifier_head_config_defaults_and_customization():
    # Defaults
    cfg = tc.ClassifierHeadConfig()
    assert isinstance(cfg.hidden_sizes, Sequence)
    assert cfg.hidden_sizes == (768,)
    assert cfg.dropouts == (0.15, 0.20)
    assert cfg.use_layer_norm is True
    assert cfg.activation == "gelu"

    # Override some fields
    cfg2 = tc.ClassifierHeadConfig(
        hidden_sizes=(128, 64),
        dropouts=(0.1,),
        use_layer_norm=False,
        activation="relu",
    )
    assert cfg2.hidden_sizes == (128, 64)
    assert cfg2.dropouts == (0.1,)
    assert cfg2.use_layer_norm is False
    assert cfg2.activation == "relu"


# ---------------------------------------------------------------------------
# BertClassifierHead
# ---------------------------------------------------------------------------

def test_bert_classifier_head_builds_network_and_forward_shapes():
    hidden_size = 16
    num_labels = 3
    cfg = tc.ClassifierHeadConfig(
        hidden_sizes=(8,),
        dropouts=(0.1, 0.2),
        use_layer_norm=True,
        activation="gelu",
    )
    head = tc.BertClassifierHead(hidden_size=hidden_size, num_labels=num_labels, cfg=cfg)
    assert isinstance(head.net, nn.Sequential)

    batch = 4
    x = torch.randn(batch, hidden_size)
    logits = head(x)
    assert logits.shape == (batch, num_labels)


def test_bert_classifier_head_no_hidden_layers_still_works():
    # hidden_sizes empty -> just a dropout + final Linear
    hidden_size = 10
    num_labels = 2
    cfg = tc.ClassifierHeadConfig(
        hidden_sizes=(),
        dropouts=(0.3,),
        use_layer_norm=False,
        activation="tanh",
    )
    head = tc.BertClassifierHead(hidden_size=hidden_size, num_labels=num_labels, cfg=cfg)
    x = torch.randn(5, hidden_size)
    logits = head(x)
    assert logits.shape == (5, num_labels)


def test_bert_classifier_head_extra_dropouts_do_not_crash():
    # More dropouts than hidden layers; extras should simply be unused.
    hidden_size = 8
    num_labels = 2
    cfg = tc.ClassifierHeadConfig(
        hidden_sizes=(4,),
        dropouts=(0.1, 0.2, 0.3),  # third dropout should be ignored
        use_layer_norm=True,
        activation="relu",
    )
    head = tc.BertClassifierHead(hidden_size=hidden_size, num_labels=num_labels, cfg=cfg)
    x = torch.randn(3, hidden_size)
    logits = head(x)
    assert logits.shape == (3, num_labels)


def test_bert_classifier_head_unknown_activation_defaults_to_gelu():
    hidden_size = 6
    num_labels = 2
    cfg = tc.ClassifierHeadConfig(
        hidden_sizes=(4,),
        dropouts=(),
        use_layer_norm=False,
        activation="not_a_real_act",
    )
    head = tc.BertClassifierHead(hidden_size=hidden_size, num_labels=num_labels, cfg=cfg)

    # There should be at least one GELU module present
    acts = [m for m in head.net if isinstance(m, nn.GELU)]
    assert len(acts) >= 1


# ---------------------------------------------------------------------------
# BertTextClassifier: construction and pooling
# ---------------------------------------------------------------------------

def test_bert_text_classifier_init_and_forward_cls_pooling():
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    num_labels = 4

    model = tc.BertTextClassifier(bert=bert, num_labels=num_labels, pooling="cls")
    assert model.pooling == "cls"
    assert isinstance(model.classifier, nn.Module)

    batch_size, seq_len = 2, 7
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, num_labels, (batch_size,))

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
    )

    assert "logits" in out
    assert out["logits"].shape == (batch_size, num_labels)
    assert "loss" in out
    assert out["loss"].shape == ()  # scalar


def test_bert_text_classifier_init_invalid_pooling_raises():
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    with pytest.raises(ValueError):
        tc.BertTextClassifier(bert=bert, num_labels=2, pooling="max")  # invalid pooling


def test_bert_text_classifier_mean_pooling_respects_mask():
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    num_labels = 3

    model = tc.BertTextClassifier(bert=bert, num_labels=num_labels, pooling="mean")

    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    # Mask: first sample uses all tokens; second masks last two
    attention_mask = torch.tensor([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 0, 0]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"]
    assert logits.shape == (batch_size, num_labels)


def test_bert_text_classifier_pool_uses_pooled_output_or_fallback():
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    model = tc.BertTextClassifier(bert=bert, num_labels=2, pooling="cls")

    seq_len = 4
    last_hidden = torch.randn(1, seq_len, cfg.hidden_size)
    pooled = torch.randn(1, cfg.hidden_size)

    # Case 1: pooled_output present
    bert_out = {"last_hidden_state": last_hidden, "pooled_output": pooled}
    pooled1 = model._pool(bert_out, attention_mask=None)
    assert torch.allclose(pooled1, pooled)

    # Case 2: pooled_output missing: should fall back to first token
    bert_out2 = {"last_hidden_state": last_hidden}
    pooled2 = model._pool(bert_out2, attention_mask=None)
    assert pooled2.shape == (1, cfg.hidden_size)
    # Should match the first token of last_hidden_state
    assert torch.allclose(pooled2, last_hidden[:, 0, :])


def test_bert_text_classifier_custom_head_is_respected():
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    num_labels = 3

    class IdentityHead(nn.Module):
        def __init__(self, in_dim, num_labels):
            super().__init__()
            self.linear = nn.Linear(in_dim, num_labels)

        def forward(self, x):
            return self.linear(x)

    head = IdentityHead(cfg.hidden_size, num_labels)
    model = tc.BertTextClassifier(bert=bert, num_labels=num_labels, pooling="cls", head=head)

    assert model.classifier is head


# ---------------------------------------------------------------------------
# BertTextClassifier.from_pretrained & save_pretrained
# ---------------------------------------------------------------------------

def test_bert_text_classifier_from_pretrained_happy_path(tmp_path: Path):
    assets_dir = _build_dummy_assets_dir(tmp_path)
    num_labels = 5

    model = tc.BertTextClassifier.from_pretrained(
        assets_dir=assets_dir,
        num_labels=num_labels,
        pooling="cls",
    )

    assert isinstance(model.bert, BertModel)
    assert model.num_labels == num_labels
    assert model.pooling == "cls"


def test_bert_text_classifier_from_pretrained_missing_files_raises(tmp_path: Path):
    assets_dir = tmp_path / "assets_missing"
    assets_dir.mkdir()

    # Only write config.json, no pytorch_model.bin
    cfg = _small_bert_config()
    (assets_dir / "config.json").write_text(json.dumps(cfg.__dict__), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        tc.BertTextClassifier.from_pretrained(assets_dir=assets_dir, num_labels=2)


def test_bert_text_classifier_from_pretrained_with_custom_head(tmp_path: Path):
    assets_dir = _build_dummy_assets_dir(tmp_path)
    cfg = _small_bert_config()
    num_labels = 4

    class DummyHead(nn.Module):
        def __init__(self, in_dim, num_labels):
            super().__init__()
            self.lin = nn.Linear(in_dim, num_labels)

        def forward(self, x):
            return self.lin(x)

    head = DummyHead(cfg.hidden_size, num_labels)

    model = tc.BertTextClassifier.from_pretrained(
        assets_dir=assets_dir,
        num_labels=num_labels,
        head=head,
    )

    assert model.classifier is head


def test_bert_text_classifier_save_pretrained_round_trip_meta(tmp_path: Path):
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    num_labels = 3
    model = tc.BertTextClassifier(bert=bert, num_labels=num_labels, pooling="mean")

    out_dir = tmp_path / "saved_model"
    model.save_pretrained(out_dir)

    full_path = out_dir / "classifier_full.pt"
    meta_path = out_dir / "classifier_meta.json"

    assert full_path.is_file()
    assert meta_path.is_file()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["num_labels"] == num_labels
    assert meta["pooling"] == "mean"
    assert isinstance(meta["bert_config"], dict)
    assert meta["bert_config"]["hidden_size"] == cfg.hidden_size


def test_bert_text_classifier_forward_without_labels_returns_logits_only():
    cfg = _small_bert_config()
    bert = BertModel(cfg)
    model = tc.BertTextClassifier(bert=bert, num_labels=2, pooling="cls")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(input_ids=input_ids)
    assert "logits" in out
    assert "loss" not in out


# ---------------------------------------------------------------------------
# build_bert_input_encoder
# ---------------------------------------------------------------------------

def test_build_bert_input_encoder_happy_path(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    _make_vocab_file(assets_dir)

    encoder = tc.build_bert_input_encoder(
        assets_dir=assets_dir,
        max_len=8,
        lowercase=True,
    )
    assert isinstance(encoder, BertInputEncoder)

    # Encode some text and check length
    out = encoder.encode("This is a test.")
    assert len(out.input_ids) == 8
    assert len(out.attention_mask) == 8
    assert len(out.token_type_ids) == 8


def test_build_bert_input_encoder_missing_vocab_raises(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    # Do not create vocab.txt

    with pytest.raises(FileNotFoundError):
        tc.build_bert_input_encoder(assets_dir=assets_dir)


def test_build_bert_input_encoder_uses_lowercase_flag(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    _make_vocab_file(assets_dir)

    # lowercase=False: BasicTokenizer will preserve case,
    # and since vocab only has lowercase tokens ("this", etc.),
    # WordPiece should produce [UNK] for "This".
    encoder = tc.build_bert_input_encoder(
        assets_dir=assets_dir,
        max_len=6,
        lowercase=False,
    )

    out = encoder.encode("This")
    # CLS + payload + SEP + padding
    # payload will be [UNK] (id 1) given our vocab
    unk_id = 1
    assert unk_id in out.input_ids
