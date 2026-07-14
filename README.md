# BERTuneClassifier

A library for hyperparameter optimization and fine-tuning of BERT-based classification models. It integrates **Optuna** for efficient search and **MLflow** for experiment tracking.

Supports both classic 512-token encoders (BERT, RoBERTa, DistilBERT, ELECTRA) and long-context models such as **ModernBERT** (8192 tokens). Per-architecture dropout is applied automatically, `max_length` is clamped to each model's real context window, precision is bf16 where the GPU supports it, and gradient checkpointing switches on automatically for sequences longer than 1024 tokens (override with `gradient_checkpointing=True/False`).

## Setup
Ensure you have a local MLflow server running before execution:
```bash
mlflow server --port 9090
```


## Usage Like so 

``` python

from bertuner.BERTuner import BERTuneClassifier

# 1. Initialize
classifier = BERTuneClassifier(
    data_path="../data/dataset.csv", 
    models_dir="../models/", 
    text_feature="text_col", 
    target_col="label_col"
)

# 2. Configure (Optional: uses defaults if skipped)
classifier.initialize_model_choices()
classifier.initialize_search_space()

# 3. Optimize
# Runs Optuna trials and logs to MLflow
best_params = classifier.optimize(
    n_trials=20, 
    optimize_metric="avg_precision", 
    study_name="bert_experiment_v1"
)

# 4. Train Final Model
# Retrains on best params, evaluates on test set, saves config
metrics, model, test_ds = classifier.train_final_model()

print(metrics)
```