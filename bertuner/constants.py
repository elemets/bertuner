SEED = 42
DEFAULT_MODEL_CHOICES = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "bert-base": "bert-base-uncased",
    "electra-small": "google/electra-small-discriminator",
    "electra-base": "google/electra-base-discriminator",
}

DEFAULT_SEARCH_SPACE = {
    "model": ["deberta-v3-base", "roberta-base", "distilbert"],
    "learning_rate": {"low": 1e-6, "high": 5e-5, "log": True},
    "batch_size": [8, 16, 32],
    "loss_type": ["weighted", "focal", "label_smoothing"],
    "label_smoothing": {"low": 0.0, "high": 0.1},
    "focal_gamma": {"low": 1.0, "high": 3.0},
    "weight_decay": {"low": 0.0, "high": 0.2},
    "warmup_ratio": {"low": 0.0, "high": 0.2},
    "scheduler": ["linear", "cosine"],
    "dropout": {"low": 0.0, "high": 0.3},
    "early_stopping_patience": {"low": 3, "high": 8}, # Int inferred from values
}