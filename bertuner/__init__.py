"""bertuner: hyperparameter optimization and fine-tuning for BERT-style text classifiers."""

__version__ = "0.1.0"

from bertuner.Predictor import BERTunePredictor

__all__ = ["BERTuneClassifier", "BERTunePredictor", "__version__"]


def __getattr__(name):
    # Lazy import: BERTuneClassifier pulls training-only deps (optuna, mlflow,
    # datasets), which are optional extras — inference installs must not need them.
    if name == "BERTuneClassifier":
        from bertuner.BERTuner import BERTuneClassifier

        return BERTuneClassifier
    raise AttributeError(f"module 'bertuner' has no attribute '{name}'")
