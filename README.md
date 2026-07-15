# BERTuneClassifier

A library for hyperparameter optimization and fine-tuning of BERT-based classification models. It integrates **Optuna** for efficient search and **MLflow** for experiment tracking.

Supports both classic 512-token encoders (BERT, RoBERTa, DistilBERT, ELECTRA) and long-context models such as **ModernBERT** (8192 tokens).

## Installation

```bash
pip install bertuner            # training + inference, batteries included
```

From source (development):

```bash
git clone https://github.com/elemets/bertuner && cd bertuner
pip install -r requirements.txt
```

MLflow tracking works in two modes:

```bash
# Option A: run a tracking server (default, expects port 9090)
mlflow server --port 9090
```

```python
# Option B: no server — log to a local directory instead
classifier = BERTuneClassifier(..., mlflow_tracking_uri="./mlruns")
```

## Training

```python
from bertuner.BERTuner import BERTuneClassifier

# 1. Initialize
classifier = BERTuneClassifier(
    data_path="../data/dataset.csv",   # or dataframe=my_df
    models_dir="../models/",
    text_feature="text_col",           # column containing the text
    target_cols=["label_col"],         # one column = single-label
    max_length=512,
)

# 2. Configure (optional: uses defaults if called without arguments)
classifier.initialize_model_choices()
classifier.initialize_search_space()

# 3. Optimize — runs Optuna trials and logs to MLflow
best_value = classifier.optimize(
    n_trials=20,
    optimize_metric="avg_precision",
    study_name="bert_experiment_v1",
)

# 4. Train final model — retrains on best params, optimises the decision
#    threshold on the validation set, evaluates on the test set, and saves
#    model + tokenizer + bertuner_config.json under models_dir/final_model/model
metrics, model, test_ds = classifier.train_final_model()
print(metrics)
```

Multi-label classification: pass several target columns — `target_cols=["l1", "l2", "l3"]`. The loss switches to BCE-with-logits and one decision threshold is optimised per label.

Grouped data (e.g. multiple notes per patient): pass `group_key="patient_id"` and the train/val/test split guarantees no group leaks across splits.

## Customizing the hyperparameter search

Two things are configurable: **which models** are searched and **which hyperparameters** with what ranges.

`initialize_model_choices` maps short names to HuggingFace model paths:

```python
classifier.initialize_model_choices({
    "bert-base": "bert-base-uncased",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "my-domain-model": "allenai/scibert_scivocab_uncased",
})
```

`initialize_search_space` takes a dict where the value type decides the Optuna suggestion:

- **list** → categorical choice, e.g. `"batch_size": [8, 16, 32]`
- **dict with int `low`/`high`** → integer range, e.g. `{"low": 3, "high": 8}` (optional `"step"`)
- **dict with float `low`/`high`** → float range, e.g. `{"low": 1e-6, "high": 5e-5, "log": True}` (`"log"` samples on a log scale — use it for learning rates)

```python
classifier.initialize_search_space({
    "model": ["bert-base", "my-domain-model"],   # keys from model_choices
    "learning_rate": {"low": 1e-6, "high": 5e-5, "log": True},
    "batch_size": [8, 16, 32],
    "gradient_accumulation_steps": [1, 2, 4],    # optional, defaults to 1
    "loss_type": ["weighted", "focal", "label_smoothing"],
    "weight_decay": {"low": 0.0, "high": 0.2},
    "warmup_ratio": {"low": 0.0, "high": 0.2},
    "scheduler": ["linear", "cosine"],
    "dropout": {"low": 0.0, "high": 0.3},
    "early_stopping_patience": {"low": 3, "high": 8},
})
```

Required keys: `model`, `learning_rate`, `batch_size`, `weight_decay`, `warmup_ratio`, `scheduler`, `dropout`, `early_stopping_patience`. Optional: `loss_type` (single-label only; defaults to `weighted`) and `gradient_accumulation_steps`.

Ready-made spaces live in `bertuner.constants`: `DEFAULT_SEARCH_SPACE_SINGLELABEL`, `DEFAULT_SEARCH_SPACE_MULTILABEL`, and `DEFAULT_SEARCH_SPACE_LONGCONTEXT`. Tweak one instead of starting from scratch:

```python
from bertuner.constants import DEFAULT_SEARCH_SPACE_SINGLELABEL

classifier.initialize_search_space({
    **DEFAULT_SEARCH_SPACE_SINGLELABEL,
    "model": ["bert-base"],                       # pin a single model
    "learning_rate": {"low": 1e-5, "high": 3e-5, "log": True},
})
```

## Long documents (ModernBERT, 8192 tokens)

```python
from bertuner.BERTuner import BERTuneClassifier
from bertuner.constants import DEFAULT_SEARCH_SPACE_LONGCONTEXT

classifier = BERTuneClassifier(
    data_path="../data/long_docs.csv",
    models_dir="../models/",
    text_feature="text_col",
    target_cols=["label_col"],
    max_length=8192,
)
classifier.initialize_model_choices()
classifier.initialize_search_space(DEFAULT_SEARCH_SPACE_LONGCONTEXT)
classifier.optimize(n_trials=10, study_name="long_context_v1")
```

`DEFAULT_SEARCH_SPACE_LONGCONTEXT` searches over ModernBERT base/large with small per-device batches and `gradient_accumulation_steps`, keeping the effective batch size in the usual range without exhausting GPU memory. Mixing 512-token models into the same search space is safe — `max_length` is clamped per model.

## Loading a trained model and predicting

`train_final_model()` saves everything the predictor needs (weights, tokenizer, optimised thresholds, `max_length`) under `models_dir/final_model/model`:

```python
from bertuner.Predictor import BERTunePredictor

predictor = BERTunePredictor("../models/final_model/model")

# Hard class predictions, using the threshold(s) optimised during training
preds = predictor.predict(["some clinical note", "another document"])
# single-label → array of 0/1 (binary) or class ids (multiclass)
# multi-label  → array of shape (N, num_labels) with 0/1 per label

# Probabilities
probs = predictor.predict_proba(["some clinical note"])
# single-label → softmax over classes, shape (N, num_classes)
# multi-label  → sigmoid per label,    shape (N, num_labels)

# Predictions as a DataFrame with one column per target
df = predictor.predict_df(["some clinical note", "another document"])
```

Options: `BERTunePredictor(model_dir, device="cuda", batch_size=64)` — device defaults to CUDA when available, batch size to 32. Texts longer than the trained `max_length` are truncated.
