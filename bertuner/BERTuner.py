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
    AutoModelForSequenceClassification,
    TrainingArguments,
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
from mlflow.tracking import MlflowClient

from model_tuner import train_val_test_split
from bertuner.CustomTrainer import CustomTrainer
from bertuner.TensorBoardCallback import TensorBoardSyncCallback, CleanupCheckpointsCallback
from bertuner.utils import enhanced_balance_data
from bertuner.constants import (
    DEFAULT_MODEL_CHOICES, DEFAULT_SEARCH_SPACE, SEED, 
)

class BERTuneClassifier:
    """
    Handles hyperparameter optimization and final training for BERT-based classification models.
    """

    def __init__(self, data_path: str, models_dir: str, text_feature: str, target_col: str, group_key: str = None, balance_type: str = None, seed: int = SEED, mlflow_port: int = 9090, log_level: str = "best"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.text_feature = text_feature
        self.target_col = target_col
        self.seed = seed
        self.df = pd.read_csv(data_path)
        self.group_key = group_key
        self.balance_type = balance_type
        self.mlflow_uri = f"http://127.0.0.1:{mlflow_port}"
        self.log_level = log_level # 'best' or 'verbose'
        self.best_params = {}
        self.best_threshold = 0.5
        self._setup_seed()
        print(f"Please make sure an mlflow instance is running on: {self.mlflow_uri}")

    def initialize_model_choices(self, model_choices: dict = DEFAULT_MODEL_CHOICES):
        """Initializes the different model choices."""
        self.MODEL_CHOICES = model_choices
        print("Model choices set")
        
    def initialize_search_space(self, search_space: dict = DEFAULT_SEARCH_SPACE):
        """Initializes the search space."""
        self.search_space = search_space
        print("Search space set")

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

    @staticmethod
    def _compute_metrics(eval_pred):
        """Computes accuracy, f1, precision, recall, specificity, AP, and AUC."""
        predictions, labels = eval_pred
        if len(predictions.shape) > 1:
            probs = F.softmax(torch.from_numpy(predictions), dim=-1).numpy()
            preds = np.argmax(predictions, axis=1)
        else:
            preds = predictions
            probs = None

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        }

        if probs is not None:
            metrics["avg_precision"] = average_precision_score(labels, probs[:, 1])
            metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.5

        return metrics

    def _prepare_datasets(self, tokenizer, group_key, balance_type, max_length=512):
        """Splits, balances, and tokenizes data."""
        if group_key in self.df.columns:
            self.df[group_key] = self.df[group_key].str.strip().str.casefold()
            group_data = self.df.groupby(group_key)[self.target_col].agg(lambda x: x.mode().iloc[0]).reset_index()
            
            X_train_q, X_val_q, X_test_q, _, _, _ = train_val_test_split(
                group_data[[group_key]], group_data[self.target_col],
                random_state=self.seed, stratify_y=True, test_size=0.15, validation_size=0.15, train_size=0.7
            )
            
            train = self.df[self.df[group_key].isin(X_train_q[group_key])]
            val = self.df[self.df[group_key].isin(X_val_q[group_key])]
            test = self.df[self.df[group_key].isin(X_test_q[group_key])]
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                self.df[[self.text_feature]], self.df[self.target_col],
                random_state=self.seed, stratify_y=True, test_size=0.15, validation_size=0.15, train_size=0.7
            )
            train = pd.concat([X_train, y_train], axis=1)
            val = pd.concat([X_val, y_val], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

        if balance_type:
            train = enhanced_balance_data(train, balance_type, self.seed)

        def _tokenize(data):
            ds = Dataset.from_pandas(data)
            ds = ds.map(
                lambda x: tokenizer(x[self.text_feature], truncation=True, padding="max_length", max_length=max_length),
                batched=True
            )
            ds = ds.remove_columns([self.text_feature]).rename_column(self.target_col, "labels")
            ds.set_format("torch")
            return ds

        return _tokenize(train), _tokenize(val), _tokenize(test)

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

    def _objective(self, trial):
        """Optuna objective function with conditional logging."""
        params = self._suggest_hyperparams(trial)
        model_path = self.MODEL_CHOICES[params["model"]]
        self._setup_seed()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_ds, val_ds, _ = self._prepare_datasets(tokenizer, self.group_key, self.balance_type)

        labels = [item["labels"].item() for item in train_ds]
        class_weights = torch.tensor([1.0, labels.count(0) / labels.count(1)], dtype=torch.float)

        model_kwargs = {"num_labels": 2}
        if "distilbert" not in params["model"]:
            model_kwargs["hidden_dropout_prob"] = params['dropout']
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)

        output_dir = f"{self.models_dir}/optuna_trial_{trial.number}"
        
        # IMPORTANT: metric_for_best_model needs the 'eval_' prefix to work reliably in TrainingArguments
        optimize_metric_key = f"eval_{self.optimize_metric}"

        args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps", eval_steps=5,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["batch_size"],
            num_train_epochs=10,
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            metric_for_best_model=optimize_metric_key, 
            greater_is_better=True,
            lr_scheduler_type=params["scheduler"],
            save_strategy="steps", save_steps=5, save_total_limit=2,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            seed=self.seed,
            remove_unused_columns=True,
            report_to=["none"] 
        )

        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=params["early_stopping_patience"])],
            loss_type=params["loss_type"],
            class_weights=class_weights,
        )

        # Logic to handle Verbose vs Best logging
        if self.log_level == "verbose":
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                trainer.train()
                metrics = trainer.evaluate()
                mlflow.log_metrics(metrics)
        else:
            trainer.train()
            metrics = trainer.evaluate()
        
        shutil.rmtree(output_dir, ignore_errors=True)

        # FIX: Retrieve the metric using the correct 'eval_' prefix key
        return metrics.get(optimize_metric_key, 0.0)

    def optimize(self, n_trials: int = 10, optimize_metric: str = "avg_precision", study_name: str = "bert_optimization"):
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

    def train_final_model(self):
        """Trains the model with best parameters and evaluates on test set."""
        if not self.best_params:
            raise ValueError("Run optimize() before training final model.")

        self._setup_seed()
        model_path = self.MODEL_CHOICES[self.best_params["model"]]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_ds, val_ds, test_ds = self._prepare_datasets(tokenizer, self.group_key, self.balance_type)

        labels = [item["labels"].item() for item in train_ds]
        class_weights = torch.tensor([1.0, labels.count(0) / labels.count(1)], dtype=torch.float)

        model_kwargs = {"num_labels": 2, "attention_probs_dropout_prob": self.best_params["dropout"]}
        if "distilbert" not in self.best_params["model"]:
            model_kwargs["hidden_dropout_prob"] = self.best_params["dropout"]
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)

        final_dir = f"{self.models_dir}/final_model"
        args = TrainingArguments(
            output_dir=final_dir,
            eval_strategy="steps", eval_steps=5,
            learning_rate=self.best_params["learning_rate"],
            per_device_train_batch_size=self.best_params["batch_size"],
            per_device_eval_batch_size=self.best_params["batch_size"],
            num_train_epochs=10,
            weight_decay=self.best_params["weight_decay"],
            warmup_ratio=self.best_params["warmup_ratio"],
            metric_for_best_model=f"eval_{self.optimize_metric}", # Added prefix here too
            save_strategy="steps", save_steps=5, save_total_limit=2,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            report_to=["tensorboard", "mlflow"], 
            logging_dir=f"{final_dir}/logs",
            seed=self.seed
        )

        with mlflow.start_run(run_name="final_model_run", nested=True):
            trainer = CustomTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=self._compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=self.best_params["early_stopping_patience"]),
                    TensorBoardSyncCallback(f"{final_dir}/logs"),
                    CleanupCheckpointsCallback
                ],
                loss_type=self.best_params["loss_type"],
                class_weights=class_weights,
            )
            trainer.train()

            val_res = trainer.predict(val_ds)
            val_probs = F.softmax(torch.from_numpy(val_res.predictions), dim=-1).numpy()[:, 1]
            
            best_f1 = 0
            for thresh in np.linspace(0.1, 0.9, 81):
                f1 = f1_score(val_res.label_ids, (val_probs >= thresh).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, self.best_threshold = f1, thresh

            test_res = trainer.predict(test_ds)
            test_probs = F.softmax(torch.from_numpy(test_res.predictions), dim=-1).numpy()[:, 1]
            
            metrics_df = self._build_metrics_df(
                val_res.label_ids, val_probs, 
                test_res.label_ids, test_probs, 
                self.best_threshold
            )

            mlflow.log_params(self.best_params)
            for _, row in metrics_df.iterrows():
                for m in ["Accuracy", "F1", "AUC"]:
                    mlflow.log_metric(f"{row['Split']}_{m}", row[m])
            
            self._save_config(final_dir, model)
            
        return metrics_df, model, test_ds

    def _build_metrics_df(self, val_y, val_p, test_y, test_p, thresh):
        """Helper to construct the results DataFrame."""
        def get_row(split, y, p):
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
        return pd.DataFrame([get_row("Validation", val_y, val_p), get_row("Test", test_y, test_p)])

    def _save_config(self, path, model):
        """Saves model state and JSON config."""
        save_path = f"{path}/final_optimized_model.pth"
        torch.save(model.state_dict(), save_path)
        config = {
            "model_metadata": {
                "model": self.best_params["model"],
                "optimal_threshold": float(self.best_threshold),
            },
            "parameters": self.best_params,
        }
        with open(save_path.replace(".pth", "_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _cleanup_trials(self, keep_trial_id):
        """Removes non-best trial directories."""
        for entry in os.listdir(self.models_dir):
            if entry.startswith("optuna_trial_") and f"_{keep_trial_id}" not in entry:
                shutil.rmtree(os.path.join(self.models_dir, entry), ignore_errors=True)