import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BERTunePredictor:
    """
    Loads a model directory saved by BERTuneClassifier.train_final_model()
    and runs inference with the thresholds optimised during training.

    Usage:
        predictor = BERTunePredictor("models/BERTModels/final_model/model")
        preds = predictor.predict(["some text", "other text"])
        probs = predictor.predict_proba(["some text"])
    """

    def __init__(self, model_dir: str, device: str = None, batch_size: int = 32):
        with open(f"{model_dir}/bertuner_config.json") as f:
            self.config = json.load(f)

        meta = self.config["model_metadata"]
        self.is_multilabel = meta["is_multilabel"]
        self.target_cols = meta["target_cols"]
        self.max_length = meta.get("max_length", 512)

        threshold = meta["optimal_threshold"]
        self.threshold = np.array(threshold) if isinstance(threshold, list) else threshold

        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        use_fast = "deberta" not in meta.get("model_path", "").lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _logits(self, texts: list[str]) -> torch.Tensor:
        batches = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(
                texts[i : i + self.batch_size],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            batches.append(self.model(**enc).logits.cpu())
        return torch.cat(batches)

    def predict_proba(self, texts: str | list[str]) -> np.ndarray:
        """
        Probabilities per input.
        Single-label → softmax over classes, shape (N, num_classes)
        Multi-label  → sigmoid per label,    shape (N, num_labels)
        """
        if isinstance(texts, str):
            texts = [texts]
        logits = self._logits(texts)
        if self.is_multilabel:
            return torch.sigmoid(logits).numpy()
        return F.softmax(logits, dim=-1).numpy()

    def predict(self, texts: str | list[str]) -> np.ndarray:
        """
        Class predictions using the training-time optimised threshold(s).
        Binary single-label → 0/1 via threshold on positive-class probability
        Multiclass          → argmax (threshold not applicable)
        Multi-label         → per-label 0/1 via per-label thresholds, shape (N, num_labels)
        """
        probs = self.predict_proba(texts)
        if self.is_multilabel:
            return (probs >= self.threshold).astype(int)
        if probs.shape[1] == 2:
            return (probs[:, 1] >= self.threshold).astype(int)
        return probs.argmax(axis=1)

    def predict_df(self, texts: str | list[str]) -> pd.DataFrame:
        """Predictions as a DataFrame with one column per target."""
        preds = self.predict(texts)
        if self.is_multilabel:
            return pd.DataFrame(preds, columns=self.target_cols)
        return pd.DataFrame({self.target_cols[0]: preds})
