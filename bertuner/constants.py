SEED = 42
DEFAULT_MODEL_CHOICES = {
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "bert-base": "bert-base-uncased",
    "electra-small": "google/electra-small-discriminator",
    "electra-base": "google/electra-base-discriminator",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
}

# Dropout attribute names per architecture (config.model_type).
# "default" covers BERT/RoBERTa/ELECTRA-style configs.
MODEL_DROPOUT_ATTRS = {
    "distilbert": ("dropout", "attention_dropout", "seq_classif_dropout"),
    "modernbert": (
        "attention_dropout",
        "mlp_dropout",
        "classifier_dropout",
        "embedding_dropout",
    ),
    "default": (
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "classifier_dropout",
    ),
}

DEFAULT_SEARCH_SPACE_SINGLELABEL = {
    "model": ["bioclinicalbert", "bert-base", "roberta-base", "distilbert"],
    "learning_rate": {"low": 1e-6, "high": 5e-5, "log": True},
    "batch_size": [8, 16, 32],
    "loss_type": ["weighted", "focal", "label_smoothing"],
    "label_smoothing": {"low": 0.0, "high": 0.1},
    "focal_gamma": {"low": 1.0, "high": 3.0},
    "weight_decay": {"low": 0.0, "high": 0.2},
    "warmup_ratio": {"low": 0.0, "high": 0.2},
    "scheduler": ["linear", "cosine"],
    "dropout": {"low": 0.0, "high": 0.3},
    "early_stopping_patience": {"low": 3, "high": 8},
}

DEFAULT_SEARCH_SPACE_MULTILABEL = {
    "model": ["bioclinicalbert", "bert-base", "roberta-base", "distilbert"],
    "learning_rate": {"low": 1e-6, "high": 5e-5, "log": True},
    "batch_size": [8, 16, 32],
    "weight_decay": {"low": 0.0, "high": 0.2},
    "warmup_ratio": {"low": 0.0, "high": 0.2},
    "scheduler": ["linear", "cosine"],
    "dropout": {"low": 0.0, "high": 0.3},
    "early_stopping_patience": {"low": 3, "high": 8},
}

# Long-context models (e.g. ModernBERT, 8192 tokens): tiny per-device batches with
# gradient accumulation so the effective batch size stays in the usual 8-32 range.
DEFAULT_SEARCH_SPACE_LONGCONTEXT = {
    **DEFAULT_SEARCH_SPACE_SINGLELABEL,
    "model": ["modernbert-base", "modernbert-large"],
    "batch_size": [1, 2, 4],
    "gradient_accumulation_steps": [4, 8, 16],
}

DEFAULT_SEARCH_SPACE = DEFAULT_SEARCH_SPACE_SINGLELABEL  # backwards-compatible default