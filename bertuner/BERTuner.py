import os
import random
import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mlflow
import optuna
from datasets import Dataset
from optuna.samplers import TPESampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from mlflow.tracking import MlflowClient

from bertuner.CustomTrainer import CustomTrainer
from bertuner.TensorBoardCallback import (
    TensorBoardSyncCallback,
    CleanupCheckpointsCallback,
)
from bertuner.utils import (
    split_group_stratified,
    check_group_leakage,
    train_val_test_split,
)
from bertuner.constants import (
    DEFAULT_MODEL_CHOICES,
    DEFAULT_SEARCH_SPACE,
    DEFAULT_SEARCH_SPACE_SINGLELABEL,
    DEFAULT_SEARCH_SPACE_MULTILABEL,
    MODEL_DROPOUT_ATTRS,
    SEED,
)


class BERTuneClassifier:
    """
    Handles hyperparameter optimization and final training for BERT-based classification models.

    Supports both single-label (binary/multiclass) and multi-label classification.

    Single-label:  target_cols = ['label']          → CrossEntropyLoss, argmax predictions
    Multi-label:   target_cols = ['l1', 'l2', ...]  → BCEWithLogitsLoss, sigmoid + threshold
    """

    def __init__(
        self,
        models_dir: str,
        text_feature: str,
        target_cols: list[str],
        data_path: str = None,
        dataframe: pd.DataFrame = None,
        num_labels: int = None,
        group_key: str = None,
        seed: int = SEED,
        mlflow_port: int = 9090,
        mlflow_tracking_uri: str = None,
        log_level: str = "best",
        max_length: int = 512,
        gradient_checkpointing: bool = None,
    ):
        if data_path is None and dataframe is None:
            raise ValueError("Provide either data_path (CSV) or dataframe, not neither.")
        if data_path is not None and dataframe is not None:
            raise ValueError("Provide either data_path (CSV) or dataframe, not both.")

        self.data_path = data_path
        self.models_dir = models_dir
        self.text_feature = text_feature
        self.target_cols = target_cols
        self.seed = seed
        self.df = pd.read_csv(data_path) if data_path is not None else dataframe.copy()
        self.label2id = None
        self.id2label = None
        if not self.is_multilabel:
            self._encode_labels()
        if num_labels is not None:
            self.num_labels = num_labels
        elif self.is_multilabel:
            self.num_labels = len(target_cols)
        else:
            # Single-label: model head needs one logit per class, not per column
            self.num_labels = int(self.df[target_cols[0]].nunique())
        self.group_key = group_key
        if mlflow_tracking_uri is not None:
            # File-based tracking: logs to a local directory, no server required.
            # Accepts a plain path (converted to file: URI) or any mlflow URI.
            if "://" not in mlflow_tracking_uri:
                mlflow_tracking_uri = f"file:{os.path.abspath(mlflow_tracking_uri)}"
            self.mlflow_uri = mlflow_tracking_uri
        else:
            self.mlflow_uri = f"http://127.0.0.1:{mlflow_port}"
        self.log_level = log_level  # 'best' or 'verbose'
        self.best_params = {}
        # Single-label: scalar float. Multi-label: array of per-label floats.
        self.best_threshold = 0.5
        self.max_length = max_length
        # None → auto: enabled when the effective sequence length is long enough
        # that activation memory dominates (see _use_gradient_checkpointing).
        self.gradient_checkpointing = gradient_checkpointing
        self._setup_seed()
        if self.mlflow_uri.startswith("http"):
            print(f"Please make sure an mlflow instance is running on: {self.mlflow_uri}")
        else:
            print(f"Logging mlflow runs locally to: {self.mlflow_uri} (no server needed)")

        if self.is_multilabel and self.group_key:
            print(
                "[WARNING] group_key is set but multi-label mode is active. "
                "StratifiedGroupKFold does not support multi-label targets — "
                "falling back to standard stratify-free train/val/test split."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_multilabel(self) -> bool:
        """True when there are multiple target columns (multi-label classification)."""
        return len(self.target_cols) > 1

    @property
    def is_binary(self) -> bool:
        """True for single-label two-class classification (thresholded predictions)."""
        return not self.is_multilabel and self.num_labels == 2

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def initialize_model_choices(self, model_choices: dict = DEFAULT_MODEL_CHOICES):
        """Initializes the different model choices."""
        self.MODEL_CHOICES = model_choices
        print("Model choices set")

    def initialize_search_space(self, search_space: dict = None):
        """Initializes the search space, auto-selecting based on classification mode if not provided."""
        if search_space is not None:
            self.search_space = search_space
        elif self.is_multilabel:
            self.search_space = DEFAULT_SEARCH_SPACE_MULTILABEL
        else:
            self.search_space = DEFAULT_SEARCH_SPACE_SINGLELABEL
        print(f"Search space set ({'multi-label' if self.is_multilabel else 'single-label'} mode)")

    def _encode_labels(self):
        """
        Maps single-label targets to contiguous ids 0..n-1 (handles string labels
        and non-contiguous ints). Keeps id2label/label2id for the saved model.
        No-op when labels are already 0..n-1 integers.
        """
        col = self.target_cols[0]
        classes = sorted(self.df[col].dropna().unique().tolist())
        if classes == list(range(len(classes))):
            return
        self.label2id = {c: i for i, c in enumerate(classes)}
        self.id2label = {i: str(c) for i, c in enumerate(classes)}
        self.df[col] = self.df[col].map(self.label2id)
        print(f"Encoded target labels: {self.label2id}")

    def _setup_seed(self):
        """Sets reproducible seeds."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        set_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_tokenizer(self, model_path: str) -> AutoTokenizer:
        """Get tokenizer. DeBERTa fast tokenizer has compatibility issues."""
        use_fast = "deberta" not in model_path.lower()
        return AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    def _effective_max_length(self, tokenizer, model_path: str) -> int:
        """Clamps the requested max_length to what the model can actually encode."""
        capacity = getattr(tokenizer, "model_max_length", None)
        # Some tokenizers report a huge sentinel instead of the real limit
        if capacity is None or capacity > 100_000:
            capacity = getattr(
                AutoConfig.from_pretrained(model_path),
                "max_position_embeddings",
                self.max_length,
            )
        if self.max_length > capacity:
            print(
                f"[WARNING] max_length={self.max_length} exceeds the context window "
                f"of {model_path} ({capacity}); using {capacity} for this model."
            )
        return min(self.max_length, capacity)

    def _precision_flags(self) -> dict:
        """bf16 where the GPU supports it (required for ModernBERT), else fp16."""
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return {"bf16": bf16, "fp16": torch.cuda.is_available() and not bf16}

    def _use_gradient_checkpointing(self, max_length: int) -> bool:
        """Auto-enables checkpointing for long sequences unless explicitly overridden."""
        if self.gradient_checkpointing is not None:
            return self.gradient_checkpointing
        return max_length > 1024

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, eval_pred):
        """
        Computes metrics for both single-label and multi-label modes.

        Single-label: accuracy, f1, precision, recall, specificity, AP, AUC-ROC
        Multi-label:  micro/macro/sample-averaged variants of the above
        """
        predictions, labels = eval_pred

        if self.is_multilabel:
            # predictions shape: (N, num_labels) — raw logits
            probs = torch.sigmoid(torch.from_numpy(predictions)).numpy()
            # Use a fixed 0.5 threshold during training eval (optimised later on val set)
            preds = (probs >= 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(labels, preds),
                "precision_micro": precision_score(
                    labels, preds, average="micro", zero_division=0
                ),
                "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
                "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "f1_samples": f1_score(labels, preds, average="samples", zero_division=0),
            }
            # Average-precision and AUC per label, then macro-average

            aps = []
            aucs = []

            for i in range(labels.shape[1]):
                y_i = labels[:, i]
                p_i = probs[:, i]

                if len(np.unique(y_i)) > 1:
                    aps.append(average_precision_score(y_i, p_i))
                    aucs.append(roc_auc_score(y_i, p_i))

            metrics["avg_precision"] = float(np.mean(aps)) if aps else 0.5
            metrics["auc_roc"] = float(np.mean(aucs)) if aucs else 0.5
        else:
            # Single-label: predictions shape (N, num_classes) or (N,)
            if len(predictions.shape) > 1:
                probs = F.softmax(torch.from_numpy(predictions), dim=-1).numpy()
                preds = np.argmax(predictions, axis=1)
            else:
                preds = predictions
                probs = None

            if self.is_binary:
                metrics = {
                    "accuracy": accuracy_score(labels, preds),
                    "precision": precision_score(labels, preds, zero_division=0),
                    "recall": recall_score(labels, preds, zero_division=0),
                    "f1": f1_score(labels, preds, zero_division=0),
                    "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
                }
                if probs is not None:
                    metrics["avg_precision"] = average_precision_score(labels, probs[:, 1])
                    metrics["auc_roc"] = (
                        roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.5
                    )
            else:
                # Multiclass: macro-averaged metrics, one-vs-rest ranking metrics
                metrics = {
                    "accuracy": accuracy_score(labels, preds),
                    "precision": precision_score(labels, preds, average="macro", zero_division=0),
                    "recall": recall_score(labels, preds, average="macro", zero_division=0),
                    "f1": f1_score(labels, preds, average="macro", zero_division=0),
                }
                # Ranking metrics need every class present in the eval split
                if probs is not None and len(np.unique(labels)) == probs.shape[1]:
                    y_bin = label_binarize(labels, classes=np.arange(probs.shape[1]))
                    metrics["avg_precision"] = average_precision_score(
                        y_bin, probs, average="macro"
                    )
                    metrics["auc_roc"] = roc_auc_score(
                        labels, probs, multi_class="ovr", average="macro"
                    )
                elif probs is not None:
                    metrics["avg_precision"] = 0.5
                    metrics["auc_roc"] = 0.5

        return metrics

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_datasets(self, tokenizer, group_key, max_length=512):
        """
        Splits, balances, and tokenizes data.

        Splitting strategy
        ------------------
        Multi-label + group_key  → standard split (group stratification not supported
                                   for multi-label targets; warning shown in __init__)
        Single-label + group_key → StratifiedGroupKFold split
        No group_key             → standard stratified split (single-label) or plain
                                   random split (multi-label)
        """
        use_group_split = (
            not self.is_multilabel and group_key is not None and group_key in self.df.columns
        )

        if use_group_split:
            self.df[group_key] = self.df[group_key].astype(str).str.strip().str.casefold()
            train, val, test = split_group_stratified(
                self.df,
                group_key,
                self.target_cols[0],  # single-label: one column
                seed=self.seed,
                test_size=0.15,
                val_size=0.15,
            )
            check_group_leakage(train, val, test, group_key)
        else:
            # For multi-label we skip stratification on labels; for single-label
            # train_val_test_split handles it internally when stratify_y=True.
            stratify = not self.is_multilabel
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                self.df[[self.text_feature]],
                self.df[self.target_cols],
                random_state=self.seed,
                stratify_y=stratify,
                test_size=0.15,
                validation_size=0.15,
                train_size=0.7,
            )
            train = pd.concat([X_train, y_train], axis=1)
            val = pd.concat([X_val, y_val], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

        def _tokenize(data):
            data = data.copy()
            ds = Dataset.from_pandas(data)
            # No padding here — DataCollatorWithPadding pads each batch dynamically
            ds = ds.map(
                lambda x: tokenizer(
                    x[self.text_feature],
                    truncation=True,
                    max_length=max_length,
                ),
                batched=True,
            )

            if self.is_multilabel:
                # Labels: float vector of length num_labels for BCEWithLogitsLoss
                ds = ds.map(
                    lambda x: {
                        "labels": [
                            [float(x[col][i]) for col in self.target_cols]
                            for i in range(len(x[self.target_cols[0]]))
                        ]
                    },
                    batched=True,
                )
            else:
                # Labels: single integer for CrossEntropyLoss
                col = self.target_cols[0]
                ds = ds.map(
                    lambda x: {"labels": [int(v) for v in x[col]]},
                    batched=True,
                )

            remove_cols = [self.text_feature] + self.target_cols
            ds = ds.remove_columns([c for c in remove_cols if c in ds.column_names])
            ds.set_format("torch")
            return ds

        return _tokenize(train), _tokenize(val), _tokenize(test)

    # ------------------------------------------------------------------
    # Class-weight helpers
    # ------------------------------------------------------------------

    def _compute_class_weights(self, train_ds) -> torch.Tensor:
        """
        Single-label : returns shape (2,)  — weight for [neg, pos] class.
        Multi-label  : returns shape (num_labels,) — pos_weight per label,
                       suitable for BCEWithLogitsLoss(pos_weight=...).
        """
        if self.is_multilabel:
            # Stack all label vectors: shape (N, num_labels)
            all_labels = torch.stack([item["labels"] for item in train_ds]).numpy()
            pos_counts = all_labels.sum(axis=0).clip(min=1)
            neg_counts = (len(all_labels) - all_labels.sum(axis=0)).clip(min=1)
            pos_weight = neg_counts / pos_counts  # shape (num_labels,)
            return torch.tensor(pos_weight, dtype=torch.float)
        else:
            labels = [item["labels"].item() for item in train_ds]
            neg = labels.count(0)
            pos = labels.count(1)
            pos = max(pos, 1)
            return torch.tensor([1.0, neg / pos], dtype=torch.float)

    # ------------------------------------------------------------------
    # Hyperparameter suggestion
    # ------------------------------------------------------------------

    def _suggest_hyperparams(self, trial: optuna.Trial):
        """Infers Optuna method based on search space value types."""
        params = {}
        for name, spec in self.search_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, dict):
                low, high = spec["low"], spec["high"]
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high, step=spec.get("step", 1))
                else:
                    params[name] = trial.suggest_float(name, low, high, log=spec.get("log", False))
        return params

    # ------------------------------------------------------------------
    # Model loading helper
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, dropout: float):
        """
        Loads a model with the correct config for single-label vs multi-label,
        applying dropout robustly across architectures.
        """
        problem_type = (
            "multi_label_classification" if self.is_multilabel else "single_label_classification"
        )
        label_kwargs = {}
        if self.id2label:
            label_kwargs["id2label"] = self.id2label
            label_kwargs["label2id"] = {str(k): v for k, v in self.label2id.items()}
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=self.num_labels,
            problem_type=problem_type,
            **label_kwargs,
        )

        drop = float(dropout)
        attrs = MODEL_DROPOUT_ATTRS.get(config.model_type, MODEL_DROPOUT_ATTRS["default"])
        for attr in attrs:
            if hasattr(config, attr):
                setattr(config, attr, drop)

        # ModernBERT's compiled-MLP fast path crashes under gradient checkpointing
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        return AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def _objective(self, trial):
        """Optuna objective function with conditional logging."""
        params = self._suggest_hyperparams(trial)
        model_path = self.MODEL_CHOICES[params["model"]]
        self._setup_seed()

        tokenizer = self._get_tokenizer(model_path)
        max_length = self._effective_max_length(tokenizer, model_path)
        self.train_ds, self.val_ds, self.test_ds = self._prepare_datasets(
            tokenizer, self.group_key, max_length
        )

        class_weights = self._compute_class_weights(self.train_ds)
        model = self._load_model(model_path, params["dropout"])

        output_dir = f"{self.models_dir}/optuna_trial_{trial.number}"
        optimize_metric_key = f"eval_{self.optimize_metric}"

        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["batch_size"],
            gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
            gradient_checkpointing=self._use_gradient_checkpointing(max_length),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            metric_for_best_model=optimize_metric_key,
            greater_is_better=self.greater_is_better,
            lr_scheduler_type=params["scheduler"],
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=6,
            save_total_limit=2,
            load_best_model_at_end=True,
            seed=self.seed,
            remove_unused_columns=True,
            report_to=["none"],
            **self._precision_flags(),
        )

        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=params["early_stopping_patience"])
            ],
            loss_type=params.get("loss_type", "weighted"),
            class_weights=class_weights,
        )

        if self.log_level == "verbose":
            with mlflow.start_run(run_name=f"trial_{trial.number}"):
                mlflow.log_params(params)
                trainer.train()
                metrics = trainer.evaluate()
                mlflow.log_metrics(metrics)
        else:
            trainer.train()
            metrics = trainer.evaluate()

        shutil.rmtree(output_dir, ignore_errors=True)
        return metrics.get(optimize_metric_key, 0.0)

    # ------------------------------------------------------------------
    # Public API: optimize
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_trials: int = 10,
        optimize_metric: str = "avg_precision",
        study_name: str = "bert_optimization",
        greater_is_better: bool = True,
    ):

        self.greater_is_better = greater_is_better

        """Runs Optuna optimization."""
        mlflow.set_tracking_uri(self.mlflow_uri)
        client = MlflowClient()
        exp_name = f"enhanced_expert_model_optimization_{study_name}"
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            client.create_experiment(exp_name)
        elif exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)

        mlflow.set_experiment(experiment_name=exp_name)

        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)
        self.optimize_metric = optimize_metric
        study.optimize(self._objective, n_trials=n_trials)

        self.best_params = study.best_params
        self._cleanup_trials(study.best_trial.number)
        return study.best_value

    # ------------------------------------------------------------------
    # Public API: train_final_model
    # ------------------------------------------------------------------

    def train_final_model(self, run_name: str = "final_model_run"):
        """Trains the model with best parameters and evaluates on test set."""
        if not self.best_params:
            raise ValueError("Run optimize() before training final model.")

        self._setup_seed()
        model_path = self.MODEL_CHOICES[self.best_params["model"]]

        tokenizer = self._get_tokenizer(model_path)
        max_length = self._effective_max_length(tokenizer, model_path)
        train_ds, val_ds, test_ds = self._prepare_datasets(
            tokenizer, self.group_key, max_length
        )

        class_weights = self._compute_class_weights(train_ds)
        model = self._load_model(model_path, self.best_params["dropout"])

        final_dir = f"{self.models_dir}/final_model"
        args = TrainingArguments(
            output_dir=final_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=6,
            learning_rate=self.best_params["learning_rate"],
            per_device_train_batch_size=self.best_params["batch_size"],
            per_device_eval_batch_size=self.best_params["batch_size"],
            gradient_accumulation_steps=self.best_params.get("gradient_accumulation_steps", 1),
            gradient_checkpointing=self._use_gradient_checkpointing(max_length),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            weight_decay=self.best_params["weight_decay"],
            warmup_ratio=self.best_params["warmup_ratio"],
            metric_for_best_model=f"eval_{self.optimize_metric}",
            greater_is_better=self.greater_is_better,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to=["tensorboard", "mlflow"],
            logging_dir=f"{final_dir}/logs",
            seed=self.seed,
            **self._precision_flags(),
        )

        with mlflow.start_run(run_name=run_name):
            trainer = CustomTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=self._compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.best_params["early_stopping_patience"]
                    ),
                    TensorBoardSyncCallback(f"{final_dir}/logs"),
                    CleanupCheckpointsCallback,
                ],
                loss_type=self.best_params.get("loss_type", "weighted"),
                class_weights=class_weights,
            )
            trainer.train()

            val_res = trainer.predict(val_ds)
            self.best_threshold = self._optimize_threshold(val_res)

            test_res = trainer.predict(test_ds)
            metrics_df = self._build_metrics_df(
                val_res.label_ids,
                self._get_probs(val_res.predictions),
                test_res.label_ids,
                self._get_probs(test_res.predictions),
                self.best_threshold,
            )

            mlflow.log_params(self.best_params)
            for _, row in metrics_df.iterrows():
                for m in ["Accuracy", "F1", "AUC"]:
                    mlflow.log_metric(f"{row['Split']}_{m}", row[m])

            self._save_model(final_dir, trainer, tokenizer, model_path, max_length)

        return metrics_df, model, test_ds

    # ------------------------------------------------------------------
    # Threshold optimisation
    # ------------------------------------------------------------------

    def _get_probs(self, predictions: np.ndarray) -> np.ndarray:
        """
        Converts raw logits to probabilities.
        Binary       → softmax → positive-class column  shape (N,)
        Multiclass   → softmax over classes             shape (N, num_classes)
        Multi-label  → sigmoid                          shape (N, num_labels)
        """
        if self.is_multilabel:
            return torch.sigmoid(torch.from_numpy(predictions)).numpy()
        probs = F.softmax(torch.from_numpy(predictions), dim=-1).numpy()
        return probs[:, 1] if self.is_binary else probs

    def _optimize_threshold(self, val_res):
        """
        Finds the best classification threshold(s) on the validation set.

        Binary       → one scalar threshold (maximises F1).
        Multiclass   → None (predictions are argmax; thresholds don't apply).
        Multi-label  → one threshold per label (maximises macro-F1);
                       returns np.ndarray of shape (num_labels,).
        """
        if not self.is_multilabel and not self.is_binary:
            return None

        probs = self._get_probs(val_res.predictions)
        labels = val_res.label_ids

        if self.is_multilabel:
            # Optimise each label independently
            best_thresholds = np.full(self.num_labels, 0.5)
            for i in range(self.num_labels):
                best_f1, best_t = 0.0, 0.5
                for thresh in np.linspace(0.1, 0.9, 81):
                    f1 = f1_score(
                        labels[:, i],
                        (probs[:, i] >= thresh).astype(int),
                        zero_division=0,
                    )
                    if f1 > best_f1:
                        best_f1, best_t = f1, thresh
                best_thresholds[i] = best_t
            return best_thresholds
        else:
            best_f1, best_t = 0.0, 0.5
            for thresh in np.linspace(0.1, 0.9, 81):
                f1 = f1_score(
                    labels,
                    (probs >= thresh).astype(int),
                    zero_division=0,
                )
                if f1 > best_f1:
                    best_f1, best_t = f1, thresh
            return best_t

    # ------------------------------------------------------------------
    # Metrics DataFrame
    # ------------------------------------------------------------------

    def _build_metrics_df(self, val_y, val_p, test_y, test_p, thresh):
        """Helper to construct the results DataFrame."""

        def get_row(split, y, p):
            if self.is_multilabel:
                # thresh is shape (num_labels,); p is (N, num_labels)
                preds = (p >= thresh).astype(int)
                return {
                    "Split": split,
                    "Accuracy": accuracy_score(y, preds),
                    "Precision_micro": precision_score(y, preds, average="micro", zero_division=0),
                    "Recall_micro": recall_score(y, preds, average="micro", zero_division=0),
                    "F1": f1_score(y, preds, average="macro", zero_division=0),
                    "F1_micro": f1_score(y, preds, average="micro", zero_division=0),
                    "F1_samples": f1_score(y, preds, average="samples", zero_division=0),
                    "AP": average_precision_score(y, p, average="macro"),
                    "AUC": (
                        roc_auc_score(y, p, average="macro") if len(np.unique(y)) > 1 else 0.5
                    ),
                    "Threshold": str(np.round(thresh, 3).tolist()),
                }
            elif self.is_binary:
                # thresh is a scalar; p is (N,)
                preds = (p >= thresh).astype(int)
                return {
                    "Split": split,
                    "Accuracy": accuracy_score(y, preds),
                    "Precision": precision_score(y, preds, zero_division=0),
                    "Recall": recall_score(y, preds, zero_division=0),
                    "F1": f1_score(y, preds, zero_division=0),
                    "Specificity": recall_score(y, preds, pos_label=0, zero_division=0),
                    "AP": average_precision_score(y, p),
                    "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5,
                    "Threshold": thresh,
                }
            else:
                # Multiclass: thresh is None; p is (N, num_classes), argmax predictions
                preds = p.argmax(axis=1)
                all_classes_present = len(np.unique(y)) == p.shape[1]
                return {
                    "Split": split,
                    "Accuracy": accuracy_score(y, preds),
                    "Precision": precision_score(y, preds, average="macro", zero_division=0),
                    "Recall": recall_score(y, preds, average="macro", zero_division=0),
                    "F1": f1_score(y, preds, average="macro", zero_division=0),
                    "AP": (
                        average_precision_score(
                            label_binarize(y, classes=np.arange(p.shape[1])),
                            p,
                            average="macro",
                        )
                        if all_classes_present
                        else 0.5
                    ),
                    "AUC": (
                        roc_auc_score(y, p, multi_class="ovr", average="macro")
                        if all_classes_present
                        else 0.5
                    ),
                    "Threshold": None,
                }

        return pd.DataFrame(
            [get_row("Validation", val_y, val_p), get_row("Test", test_y, test_p)],
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self, path, trainer, tokenizer, model_path, max_length=None):
        """Saves model + tokenizer via save_pretrained and a JSON config for reloading."""
        save_dir = f"{path}/model"
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

        threshold = self.best_threshold
        # Convert numpy array to list for JSON serialisation
        if isinstance(threshold, np.ndarray):
            threshold = threshold.tolist()

        config = {
            "model_metadata": {
                "model": self.best_params["model"],
                "model_path": model_path,
                "optimal_threshold": threshold,
                "is_multilabel": self.is_multilabel,
                "target_cols": self.target_cols,
                "max_length": max_length if max_length is not None else self.max_length,
                "id2label": self.id2label,
            },
            "parameters": self.best_params,
        }
        with open(f"{save_dir}/bertuner_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def _cleanup_trials(self, keep_trial_id):
        """Removes non-best trial directories."""
        for entry in os.listdir(self.models_dir):
            if entry.startswith("optuna_trial_") and f"_{keep_trial_id}" not in entry:
                shutil.rmtree(
                    os.path.join(self.models_dir, entry),
                    ignore_errors=True,
                )
