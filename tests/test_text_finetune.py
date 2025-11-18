# tests/test_text_finetune.py
"""
Unit tests for local_llm.training.text_finetune

Covers:
- FineTuneConfig
- set_seed
- prepare_label_mapping
- concat_text
- stratified_split_indices
- build_input_encoder / encode_dataframe / encode_splits
- TensorDictDataset
- build_dataloaders
- build_bert_text_classifier_from_assets
- train_text_classifier
- evaluate_on_split
- save_finetuned_classifier
- export_predictions_csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from local_llm.models.bert import BertConfig, BertModel
from local_llm.training import text_finetune as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_bert_assets(tmp_path: Path) -> Path:
    """
    Create a tiny BERT config + weights in a temp directory and return that dir.

    This keeps models very small so training/eval tests are fast.
    """
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cfg = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=16,
        max_position_embeddings=32,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    model = BertModel(cfg)

    # config.json
    cfg_path = assets_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # pytorch_model.bin
    weights_path = assets_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), weights_path)

    # vocab.txt (needed by tokenizer encoder helpers)
    vocab_path = assets_dir / "vocab.txt"
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "."]
    vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    return assets_dir


def _small_dataframe() -> pd.DataFrame:
    """
    Small synthetic dataset with two text columns and a label column.
    """
    return pd.DataFrame(
        {
            "col1": ["alpha", "beta", "gamma", "delta", None],
            "col2": ["one", "two", None, "four", "five"],
            "label": ["A", "B", "A", "B", "A"],
        }
    )


# ---------------------------------------------------------------------------
# FineTuneConfig
# ---------------------------------------------------------------------------

def test_finetune_config_resolved_creates_output_dir(tmp_path: Path):
    cfg = tf.FineTuneConfig(
        text_cols=("col1", "col2"),
        label_col="label",
        assets_dir=tmp_path / "assets",
        output_dir=tmp_path / "out",
    )
    assert not cfg.output_dir.exists()
    resolved = cfg.resolved()
    assert isinstance(resolved.assets_dir, Path)
    assert isinstance(resolved.output_dir, Path)
    assert resolved.output_dir.exists()
    assert resolved.output_dir.is_dir()


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def test_set_seed_reproducible_numpy_and_torch():
    tf.set_seed(123)
    a1 = np.random.randn(5)
    b1 = torch.randn(5)

    tf.set_seed(123)
    a2 = np.random.randn(5)
    b2 = torch.randn(5)

    assert np.allclose(a1, a2)
    assert torch.allclose(b1, b2)


# ---------------------------------------------------------------------------
# prepare_label_mapping & concat_text
# ---------------------------------------------------------------------------

def test_prepare_label_mapping_basic():
    df = _small_dataframe()
    df2, label_to_id, id_to_label = tf.prepare_label_mapping(df, "label")

    # label_to_id/id_to_label consistent
    assert set(label_to_id.keys()) == {"A", "B"}
    assert set(id_to_label.keys()) == set(label_to_id.values())
    for lab, idx in label_to_id.items():
        assert id_to_label[idx] == lab

    # label_id column present and integer
    assert "label_id" in df2.columns
    assert pd.api.types.is_integer_dtype(df2["label_id"])
    assert set(df2["label_id"].unique()).issubset(set(label_to_id.values()))


def test_prepare_label_mapping_missing_column_raises():
    df = _small_dataframe()
    with pytest.raises(KeyError):
        tf.prepare_label_mapping(df, "not_there")


def test_concat_text_with_strings_nans_and_non_strings():
    row = pd.Series(
        {
            "a": " Hello ",
            "b": None,
            "c": float("nan"),
            "d": 123,
            "e": "",
        }
    )
    text = tf.concat_text(row, ["a", "b", "c", "d", "e", "missing"])
    # Expected: "Hello [SEP] 123"
    assert "[SEP]" in text
    assert "Hello" in text
    assert "123" in text
    assert "nan" not in text  # NaNs should be ignored
    assert "None" not in text


def test_concat_text_all_empty_returns_empty_string():
    row = pd.Series({"a": None, "b": "", "c": float("nan")})
    text = tf.concat_text(row, ["a", "b", "c"])
    assert text == ""


# ---------------------------------------------------------------------------
# stratified_split_indices
# ---------------------------------------------------------------------------

def test_stratified_split_indices_covers_all_indices_and_is_disjoint():
    labels = np.array([0, 0, 0, 1, 1, 2])
    train_idx, val_idx, test_idx = tf.stratified_split_indices(
        labels, train_frac=0.6, val_frac=0.2, seed=42
    )

    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert sorted(all_idx.tolist()) == list(range(len(labels)))

    # disjoint
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # each label appears somewhere
    for lab in np.unique(labels):
        assert np.isin(lab, labels[train_idx]).any() or np.isin(lab, labels[val_idx]).any() or np.isin(lab, labels[test_idx]).any()


def test_stratified_split_indices_single_label_edge_case():
    labels = np.array([1, 1, 1, 1])
    train_idx, val_idx, test_idx = tf.stratified_split_indices(
        labels, train_frac=0.5, val_frac=0.25, seed=0
    )
    # No index lost
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert sorted(all_idx.tolist()) == [0, 1, 2, 3]
    # Train not empty if possible
    assert len(train_idx) >= 1


# ---------------------------------------------------------------------------
# build_input_encoder / encode_dataframe / encode_splits
# ---------------------------------------------------------------------------

def test_build_input_encoder_uses_assets_vocab(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    cfg = tf.FineTuneConfig(
        text_cols=("col1",),
        label_col="label",
        assets_dir=assets_dir,
        max_len=8,
    )
    enc = tf.build_input_encoder(cfg)
    # smoke test encode
    out = enc.encode("hello world.")
    assert len(out.input_ids) == cfg.max_len
    assert len(out.attention_mask) == cfg.max_len


def test_encode_dataframe_produces_tensors_and_uses_label_id(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    cfg = tf.FineTuneConfig(
        text_cols=("col1", "col2"),
        label_col="label",
        assets_dir=assets_dir,
        max_len=8,
    )
    df = _small_dataframe()
    df2, _, _ = tf.prepare_label_mapping(df, "label")
    enc = tf.build_input_encoder(cfg)
    tensors = tf.encode_dataframe(df2, cfg.text_cols, enc)

    assert set(tensors.keys()) == {"input_ids", "token_type_ids", "attention_mask", "labels"}
    n = len(df2)
    for v in tensors.values():
        assert isinstance(v, torch.Tensor)
        assert v.size(0) == n

    seq_len = tensors["input_ids"].size(1)
    assert seq_len == cfg.max_len

    # labels reflect label_id
    assert torch.equal(tensors["labels"], torch.tensor(df2["label_id"].tolist(), dtype=torch.long))


def test_encode_splits_uses_indices_and_produces_all_three(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    cfg = tf.FineTuneConfig(
        text_cols=("col1", "col2"),
        label_col="label",
        assets_dir=assets_dir,
        max_len=8,
    )
    df = _small_dataframe()
    df2, _, _ = tf.prepare_label_mapping(df, "label")
    labels = df2["label_id"].values
    train_idx, val_idx, test_idx = tf.stratified_split_indices(labels, 0.6, 0.2, seed=123)

    splits = tf.encode_splits(df2, train_idx, val_idx, test_idx, cfg)
    assert set(splits.keys()) == {"train", "val", "test"}

    total = (
        splits["train"]["labels"].size(0)
        + splits["val"]["labels"].size(0)
        + splits["test"]["labels"].size(0)
    )
    assert total == len(df2)


# ---------------------------------------------------------------------------
# TensorDictDataset
# ---------------------------------------------------------------------------

def test_tensor_dict_dataset_len_and_getitem():
    data = {
        "input_ids": torch.zeros(3, 5, dtype=torch.long),
        "token_type_ids": torch.zeros(3, 5, dtype=torch.long),
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1, 2], dtype=torch.long),
    }
    ds = tf.TensorDictDataset(data)
    assert len(ds) == 3
    item = ds[1]
    assert len(item) == 4
    assert item[0].shape == (5,)
    assert item[3].item() == 1


def test_tensor_dict_dataset_raises_on_mismatched_shapes():
    data = {
        "input_ids": torch.zeros(3, 5, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 5, dtype=torch.long),  # mismatched
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1, 2], dtype=torch.long),
    }
    with pytest.raises(ValueError):
        tf.TensorDictDataset(data)


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------

def test_build_dataloaders_returns_three_loaders():
    data = {
        "input_ids": torch.zeros(10, 4, dtype=torch.long),
        "token_type_ids": torch.zeros(10, 4, dtype=torch.long),
        "attention_mask": torch.ones(10, 4, dtype=torch.long),
        "labels": torch.randint(0, 2, (10,), dtype=torch.long),
    }
    splits = {"train": data, "val": data, "test": data}
    cfg = tf.FineTuneConfig(text_cols=("c1",), label_col="label", batch_size=3)

    loaders = tf.build_dataloaders(splits, cfg)
    assert set(loaders.keys()) == {"train", "val", "test"}

    # iterate a couple of batches
    for phase in ("train", "val", "test"):
        batches = list(loaders[phase])
        assert len(batches) >= 1
        x0, _, _, y = batches[0]
        assert x0.dim() == 2
        assert y.dim() == 1


# ---------------------------------------------------------------------------
# build_bert_text_classifier_from_assets
# ---------------------------------------------------------------------------

def test_build_bert_text_classifier_from_assets_uses_finetune_policy(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    cfg = tf.FineTuneConfig(
        text_cols=("c1",),
        label_col="label",
        assets_dir=assets_dir,
        finetune_policy="last_n",
        finetune_last_n=1,
        train_embeddings=False,
        pooling="cls",
    )
    num_labels = 3
    model = tf.build_bert_text_classifier_from_assets(cfg, num_labels=num_labels)

    # Some encoder params should be trainable, some not
    # embeddings should remain frozen if train_embeddings=False
    emb_requires_grad = {p.requires_grad for p in model.bert.embeddings.parameters()}
    assert emb_requires_grad == {False}

    # At least some encoder layer params should be trainable
    enc_requires_grad = [p.requires_grad for p in model.bert.encoder.parameters()]
    assert any(enc_requires_grad)

    # classifier head should be trainable
    head_requires_grad = {p.requires_grad for p in model.classifier.parameters()}
    assert head_requires_grad == {True}


def test_build_bert_text_classifier_from_assets_missing_files_raises(tmp_path: Path):
    assets_dir = tmp_path / "missing_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    cfg = tf.FineTuneConfig(
        text_cols=("c1",),
        label_col="label",
        assets_dir=assets_dir,
    )
    with pytest.raises(FileNotFoundError):
        tf.build_bert_text_classifier_from_assets(cfg, num_labels=2)


# ---------------------------------------------------------------------------
# train_text_classifier (integration-ish, small)
# ---------------------------------------------------------------------------

def test_train_text_classifier_runs_and_returns_history_and_state(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    df = _small_dataframe()
    df2, label_to_id, id_to_label = tf.prepare_label_mapping(df, "label")

    cfg = tf.FineTuneConfig(
        text_cols=("col1", "col2"),
        label_col="label",
        assets_dir=assets_dir,
        max_len=8,
        batch_size=2,
        epochs=2,            # keep small for speed
        finetune_policy="none",  # head-only training to keep it light
    )
    tf.set_seed(cfg.seed)

    labels = df2["label_id"].values
    train_idx, val_idx, test_idx = tf.stratified_split_indices(
        labels, train_frac=0.6, val_frac=0.2, seed=cfg.seed
    )
    splits = tf.encode_splits(df2, train_idx, val_idx, test_idx, cfg)
    loaders = tf.build_dataloaders(splits, cfg)

    model = tf.build_bert_text_classifier_from_assets(cfg, num_labels=len(label_to_id))
    history, best_state = tf.train_text_classifier(model, loaders, cfg)

    assert len(history) == cfg.epochs
    for row in history:
        assert {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}.issubset(row.keys())

    assert isinstance(best_state, dict)
    assert all(isinstance(k, str) for k in best_state.keys())
    assert all(isinstance(v, torch.Tensor) for v in best_state.values())


# ---------------------------------------------------------------------------
# evaluate_on_split
# ---------------------------------------------------------------------------

def test_evaluate_on_split_produces_metrics_and_predictions(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    df = _small_dataframe()
    df2, label_to_id, id_to_label = tf.prepare_label_mapping(df, "label")

    cfg = tf.FineTuneConfig(
        text_cols=("col1", "col2"),
        label_col="label",
        assets_dir=assets_dir,
        max_len=8,
        batch_size=2,
        epochs=1,
        finetune_policy="none",
    )
    tf.set_seed(cfg.seed)

    labels = df2["label_id"].values
    train_idx, val_idx, test_idx = tf.stratified_split_indices(
        labels, train_frac=0.6, val_frac=0.2, seed=cfg.seed
    )
    splits = tf.encode_splits(df2, train_idx, val_idx, test_idx, cfg)
    loaders = tf.build_dataloaders(splits, cfg)

    model = tf.build_bert_text_classifier_from_assets(cfg, num_labels=len(label_to_id))
    _, best_state = tf.train_text_classifier(model, loaders, cfg)
    model.load_state_dict(best_state)

    metrics, preds_df = tf.evaluate_on_split(model, splits["test"], cfg)
    assert set(metrics.keys()) == {"loss", "acc"}
    assert 0.0 <= metrics["acc"] <= 1.0
    assert len(preds_df) == splits["test"]["labels"].size(0)
    assert {"label_id", "pred_label_id", "pred_confidence"}.issubset(preds_df.columns)


# ---------------------------------------------------------------------------
# save_finetuned_classifier & export_predictions_csv
# ---------------------------------------------------------------------------

def test_save_finetuned_classifier_writes_files(tmp_path: Path):
    assets_dir = _tiny_bert_assets(tmp_path)
    df = _small_dataframe()
    df2, label_to_id, id_to_label = tf.prepare_label_mapping(df, "label")

    cfg = tf.FineTuneConfig(
        text_cols=("col1", "col2"),
        label_col="label",
        assets_dir=assets_dir,
        output_dir=tmp_path / "out",
        max_len=8,
        batch_size=2,
        epochs=1,
        finetune_policy="none",
    )
    tf.set_seed(cfg.seed)

    labels = df2["label_id"].values
    train_idx, val_idx, test_idx = tf.stratified_split_indices(
        labels, train_frac=0.6, val_frac=0.2, seed=cfg.seed
    )
    splits = tf.encode_splits(df2, train_idx, val_idx, test_idx, cfg)
    loaders = tf.build_dataloaders(splits, cfg)

    model = tf.build_bert_text_classifier_from_assets(cfg, num_labels=len(label_to_id))
    _, best_state = tf.train_text_classifier(model, loaders, cfg)

    tf.save_finetuned_classifier(model, best_state, cfg, label_to_id, id_to_label)

    cfg_resolved = cfg.resolved()
    assert (cfg_resolved.output_dir / "classifier_full.pt").exists()
    assert (cfg_resolved.output_dir / "pytorch_model_finetuned.bin").exists()
    meta_path = cfg_resolved.output_dir / "finetune_meta.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["num_labels"] == len(label_to_id)
    assert meta["pooling"] == cfg.pooling


def test_export_predictions_csv_creates_csv_with_expected_columns(tmp_path: Path):
    cfg = tf.FineTuneConfig(
        text_cols=("c1",),
        label_col="label",
        output_dir=tmp_path / "out_preds",
    ).resolved()

    raw_df = pd.DataFrame(
        {
            "c1": ["foo", "bar", "baz"],
            "label": ["A", "B", "A"],
        }
    )
    preds_df = pd.DataFrame(
        {
            "label_id": [0, 1, 0],
            "pred_label_id": [1, 1, 0],
            "pred_confidence": [0.7, 0.8, 0.6],
        }
    )
    id_to_label = {0: "A", 1: "B"}

    path = tf.export_predictions_csv(preds_df, raw_df, id_to_label, cfg, split_name="test")
    assert path.exists()

    out = pd.read_csv(path)
    assert {"c1", "label", "label_id", "pred_label_id", "pred_label", "pred_confidence"}.issubset(out.columns)
    # mapping check on a couple rows
    assert out.loc[0, "pred_label"] in {"A", "B"}
