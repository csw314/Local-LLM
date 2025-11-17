# local-llm

**Local LLM Utilities for Offline NLP Pipelines**

`local-llm` is a lightweight Python library for running open-source language models completely offline.
It includes:

* **BERT checkpoint conversion** (TensorFlow → PyTorch)
* A dependency-minimal **WordPiece tokenizer**
* A **BERT-style input encoder** that produces `input_ids`, `token_type_ids`, and `attention_mask`
* A simple PyTorch **classification head** for downstream training
* Full test coverage to ensure reliability and long-term maintainability

This project is designed for restricted or air-gapped environments where:

* Internet access is not allowed
* Model downloads must be avoided
* Reproducibility and transparency are required
* External ML ecosystems (HF Hub, cloud APIs) cannot be used

---

## Features

### ✔ Offline BERT checkpoint conversion

Convert standard **TensorFlow 1.x** BERT checkpoints to PyTorch format:

```python
from local_llm.convert import setup_bert_base

assets_dir = setup_bert_base(
    checkpoints="path/to/checkpoint_dir",
    vocab="path/to/vocab.txt",
    config="path/to/bert_config.json",
    output_dir="path/to/output_assets",
)
```

This produces:

```
pytorch_model.bin
config.json
vocab.txt
```

### ✔ Clean WordPiece tokenizer (no HuggingFace required)

* Accurate greedy-longest-match WordPiece implementation
* Proper handling of `[CLS]`, `[SEP]`, `[MASK]`, etc.
* BasicTokenizer handles case folding, accents, punctuation splitting

```python
from local_llm.tokenization import build_bert_input_encoder

encoder = build_bert_input_encoder(assets_dir, max_len=256)
encoded = encoder.encode("This is an example.")
```

### ✔ Lightweight PyTorch inference & training

```python
from local_llm import BertTextClassifier

model = BertTextClassifier.from_pretrained(assets_dir, num_labels=8)
```

Supports:

* CLS pooling or mean pooling
* GPU acceleration when available
* Fine-tuning with standard PyTorch optimizers

---

## Installation

```bash
pip install -e .
```

### Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* TensorFlow ≥ 2.12 (only for checkpoint conversion)
* NumPy ≥ 1.23

GPU support requires installing a **CUDA-enabled PyTorch build**.

---

## Project Structure

```
local-llm/
│
├── src/local_llm/
│   ├── convert.py              # TF → PyTorch conversion logic
│   ├── tokenization/           # Basic + WordPiece tokenizers
│   ├── models/                 # PyTorch wrappers/classifiers
│   └── __init__.py
│
├── tests/
│   ├── test_convert.py
│   ├── test_bert_wordpiece.py
│   └── ...
│
├── assets/                     # Example assets (optional)
├── README.md
└── pyproject.toml
```

---

## Usage

### 1. Convert a TensorFlow BERT checkpoint

```python
from local_llm.convert import setup_bert_base

assets = setup_bert_base(
    checkpoints="./uncased_L-12_H-768_A-12",
    vocab="./uncased_L-12_H-768_A-12/vocab.txt",
    config="./uncased_L-12_H-768_A-12/bert_config.json",
)
```

---

### 2. Encode text

```python
from local_llm.tokenization import build_bert_input_encoder

encoder = build_bert_input_encoder(assets, max_len=128)
encoded = encoder.encode("Example sentence for encoding.")
```

The result contains:

* `input_ids`
* `token_type_ids`
* `attention_mask`

---

### 3. Run inference

```python
from local_llm import BertTextClassifier
import torch

model = BertTextClassifier.from_pretrained(assets, num_labels=4)
model.eval()

batch = torch.tensor([encoded.input_ids])
logits = model(input_ids=batch)["logits"]
```

---

### 4. Fine-tune the classifier

```python
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss = model(
    input_ids=batch,
    labels=torch.tensor([1])
)["loss"]

loss.backward()
optimizer.step()
```

GPU is used automatically if available.

---

## Testing

This project includes extensive tests for:

* Vocabulary loading
* Basic tokenization
* WordPiece segmentation
* Input encoding
* TF → PyTorch conversion
* Classification head

Run all tests:

```bash
pytest
```

All tests pass on Python 3.13 and PyTorch 2.9.

---

## Why This Project Exists

Many enterprise and government environments:

* Restrict external dependencies
* Require fully offline model execution
* Need transparent, auditable ML systems
* Cannot fetch models from HuggingFace or cloud APIs

`local-llm` provides a **fully compliant**, **offline**, and **self-contained** BERT pipeline for such environments.

---

## License

MIT License (modify as needed).

---

## Contact / Maintainer

**Cameron Webster**
*([cameron.webster@nnsa.doe.gov](mailto:cameron.webster@nnsa.doe.gov))*

