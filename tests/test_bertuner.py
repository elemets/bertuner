"""Unit tests for BERTuneClassifier — construction, metrics, thresholds, saving."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import optuna
import mlflow
from mlflow.tracking import MlflowClient

from bertuner.BERTuner import BERTuneClassifier
from bertuner.constants import (
    DEFAULT_SEARCH_SPACE_LONGCONTEXT,
    DEFAULT_SEARCH_SPACE_SINGLELABEL,
)


def make_df(n=40, num_classes=2, multilabel_cols=None, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"text_feature": [f"sample text {i}" for i in range(n)]})
    if multilabel_cols:
        for col in multilabel_cols:
            df[col] = rng.integers(0, 2, n)
    else:
        df["target"] = np.arange(n) % num_classes
    return df


def make_classifier(tmp_path, **kwargs):
    defaults = dict(
        models_dir=str(tmp_path),
        text_feature="text_feature",
        target_cols=["target"],
        dataframe=make_df(),
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
    )
    defaults.update(kwargs)
    return BERTuneClassifier(**defaults)


# ------------------------------------------------------------------
# Construction / num_labels
# ------------------------------------------------------------------


class TestConstruction:
    def test_requires_data(self, tmp_path):
        with pytest.raises(ValueError):
            BERTuneClassifier(
                models_dir=str(tmp_path), text_feature="t", target_cols=["y"]
            )

    def test_rejects_both_data_sources(self, tmp_path):
        with pytest.raises(ValueError):
            BERTuneClassifier(
                models_dir=str(tmp_path),
                text_feature="t",
                target_cols=["y"],
                data_path="some.csv",
                dataframe=make_df(),
            )

    def test_binary_single_label_gets_two_labels(self, tmp_path):
        clf = make_classifier(tmp_path)
        assert clf.num_labels == 2
        assert not clf.is_multilabel

    def test_multiclass_infers_class_count(self, tmp_path):
        clf = make_classifier(tmp_path, dataframe=make_df(num_classes=4))
        assert clf.num_labels == 4

    def test_multilabel_uses_column_count(self, tmp_path):
        cols = ["l1", "l2", "l3"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        assert clf.num_labels == 3
        assert clf.is_multilabel

    def test_explicit_num_labels_wins(self, tmp_path):
        clf = make_classifier(tmp_path, num_labels=7)
        assert clf.num_labels == 7


class TestMlflowUri:
    def test_plain_path_becomes_file_uri(self, tmp_path):
        clf = make_classifier(tmp_path, mlflow_tracking_uri="./mlruns")
        assert clf.mlflow_uri.startswith("file:")
        assert clf.mlflow_uri.endswith("/mlruns")

    def test_full_uri_passthrough(self, tmp_path):
        clf = make_classifier(tmp_path, mlflow_tracking_uri="sqlite:///db.sqlite")
        assert clf.mlflow_uri == "sqlite:///db.sqlite"

    def test_default_is_http_server(self, tmp_path):
        clf = make_classifier(tmp_path, mlflow_tracking_uri=None, mlflow_port=5001)
        assert clf.mlflow_uri == "http://127.0.0.1:5001"


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class TestComputeMetrics:
    def test_single_label_perfect_predictions(self, tmp_path):
        clf = make_classifier(tmp_path)
        labels = np.array([0, 1, 0, 1])
        logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]])
        m = clf._compute_metrics((logits, labels))
        assert m["accuracy"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["mcc"] == 1.0
        assert m["auc_roc"] == 1.0
        assert m["brier_score"] < 0.001
        assert m["log_loss"] < 0.001

    def test_multilabel_perfect_predictions(self, tmp_path):
        cols = ["l1", "l2"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        logits = np.where(labels == 1, 5.0, -5.0)
        m = clf._compute_metrics((logits, labels))
        assert m["f1_micro"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["hamming_loss"] == 0.0
        assert m["jaccard_samples"] == 0.75
        assert m["mcc_micro"] == 1.0
        assert m["avg_precision_micro"] == 1.0
        assert m["auc_roc"] == 1.0


class TestGetProbs:
    def test_single_label_returns_positive_column(self, tmp_path):
        clf = make_classifier(tmp_path)
        logits = np.array([[0.0, 0.0], [10.0, -10.0]])
        probs = clf._get_probs(logits)
        assert probs.shape == (2,)
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.0, abs=1e-6)

    def test_multilabel_returns_sigmoid_matrix(self, tmp_path):
        cols = ["l1", "l2"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        logits = np.zeros((3, 2))
        probs = clf._get_probs(logits)
        assert probs.shape == (3, 2)
        assert np.allclose(probs, 0.5)


class TestOptimizeThreshold:
    def test_single_label_finds_separating_threshold(self, tmp_path):
        clf = make_classifier(tmp_path)
        # Positives cluster at high prob, negatives at low → best threshold between
        labels = np.array([0] * 10 + [1] * 10)
        pos_logit = np.concatenate([np.full(10, -2.0), np.full(10, 2.0)])
        logits = np.stack([-pos_logit, pos_logit], axis=1)
        val_res = SimpleNamespace(predictions=logits, label_ids=labels)
        t = clf._optimize_threshold(val_res)
        assert isinstance(t, float)
        assert 0.1 <= t <= 0.9

    def test_multilabel_returns_per_label_thresholds(self, tmp_path):
        cols = ["l1", "l2"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        labels = np.array([[0, 1], [1, 0], [1, 1], [0, 0]] * 5)
        logits = np.where(labels == 1, 3.0, -3.0).astype(float)
        t = clf._optimize_threshold(SimpleNamespace(predictions=logits, label_ids=labels))
        assert isinstance(t, np.ndarray)
        assert t.shape == (2,)


# ------------------------------------------------------------------
# Class weights
# ------------------------------------------------------------------


class TestClassWeights:
    def test_single_label_pos_weight_ratio(self, tmp_path):
        import torch

        clf = make_classifier(tmp_path)
        # 6 negatives, 2 positives → pos weight = 3
        ds = [{"labels": torch.tensor(0)}] * 6 + [{"labels": torch.tensor(1)}] * 2
        w = clf._compute_class_weights(ds)
        assert w.shape == (2,)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(3.0)

    def test_multilabel_per_label_pos_weight(self, tmp_path):
        import torch

        cols = ["l1", "l2"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        # l1: 2 pos / 2 neg → 1.0 ; l2: 1 pos / 3 neg → 3.0
        ds = [
            {"labels": torch.tensor([1.0, 1.0])},
            {"labels": torch.tensor([1.0, 0.0])},
            {"labels": torch.tensor([0.0, 0.0])},
            {"labels": torch.tensor([0.0, 0.0])},
        ]
        w = clf._compute_class_weights(ds)
        assert w.shape == (2,)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(3.0)


# ------------------------------------------------------------------
# Hyperparameter suggestion
# ------------------------------------------------------------------


class TestSuggestHyperparams:
    def test_fixed_trial_round_trip(self, tmp_path):
        clf = make_classifier(tmp_path)
        clf.search_space = DEFAULT_SEARCH_SPACE_SINGLELABEL
        fixed = {
            "model": "bert-base",
            "learning_rate": 1e-5,
            "batch_size": 16,
            "loss_type": "weighted",
            "label_smoothing": 0.05,
            "focal_gamma": 2.0,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
            "dropout": 0.1,
            "early_stopping_patience": 4,
        }
        params = clf._suggest_hyperparams(optuna.trial.FixedTrial(fixed))
        assert params == fixed

    def test_int_ranges_suggest_ints(self, tmp_path):
        clf = make_classifier(tmp_path)
        clf.search_space = {"early_stopping_patience": {"low": 3, "high": 8}}
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        trial = study.ask()
        params = clf._suggest_hyperparams(trial)
        assert isinstance(params["early_stopping_patience"], int)
        assert 3 <= params["early_stopping_patience"] <= 8

    def test_optimize_uses_minimize_for_lower_is_better_metric(
        self, tmp_path, monkeypatch
    ):
        clf = make_classifier(tmp_path)
        client = MagicMock()
        client.get_experiment_by_name.return_value = SimpleNamespace(
            lifecycle_stage="active"
        )
        study = MagicMock(
            best_params={"model": "bert-base"},
            best_value=0.1,
            best_trial=SimpleNamespace(number=0),
        )
        create_study = MagicMock(return_value=study)

        monkeypatch.setattr("bertuner.BERTuner.MlflowClient", lambda: client)
        monkeypatch.setattr("bertuner.BERTuner.mlflow.set_tracking_uri", MagicMock())
        monkeypatch.setattr("bertuner.BERTuner.mlflow.set_experiment", MagicMock())
        monkeypatch.setattr("bertuner.BERTuner.optuna.create_study", create_study)

        clf.optimize(
            n_trials=2,
            optimize_metric="log_loss",
            greater_is_better=False,
        )

        assert create_study.call_args.kwargs["direction"] == "minimize"
        study.optimize.assert_called_once_with(clf._objective, n_trials=2)


# ------------------------------------------------------------------
# Metrics DataFrame
# ------------------------------------------------------------------


class TestBuildMetricsDf:
    def test_single_label_rows_and_columns(self, tmp_path):
        clf = make_classifier(tmp_path)
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])
        df = clf._build_metrics_df(y, p, y, p, thresh=0.5)
        assert list(df["Split"]) == ["Validation", "Test"]
        assert df.iloc[0]["F1"] == 1.0
        assert df.iloc[0]["Balanced_Accuracy"] == 1.0
        assert df.iloc[0]["MCC"] == 1.0
        assert df.iloc[0]["Average_Precision"] == 1.0
        assert df.iloc[0]["AUROC"] == 1.0
        assert df.iloc[0]["Brier_Score"] < 0.05
        assert df.iloc[0]["Threshold"] == 0.5

    def test_multilabel_uses_array_threshold(self, tmp_path):
        cols = ["l1", "l2"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        p = np.where(y == 1, 0.9, 0.1)
        df = clf._build_metrics_df(y, p, y, p, thresh=np.array([0.5, 0.5]))
        assert df.iloc[0]["F1"] == 1.0
        assert "F1_micro" in df.columns
        assert df.iloc[0]["Hamming_Loss"] == 0.0
        assert df.iloc[0]["Average_Precision_macro"] == 1.0
        assert df.iloc[0]["AUROC_micro"] == 1.0


class TestLossCurveLogging:
    def test_logs_average_precision_and_auroc(self, tmp_path, monkeypatch):
        clf = make_classifier(tmp_path)
        logged = MagicMock()
        logged_param = MagicMock()
        monkeypatch.setattr("bertuner.BERTuner.mlflow.log_metric", logged)
        monkeypatch.setattr("bertuner.BERTuner.mlflow.log_param", logged_param)
        metrics = pd.DataFrame(
            [
                {
                    "Split": "Validation",
                    "Accuracy": 0.7,
                    "Balanced_Accuracy": 0.68,
                    "Precision": 0.72,
                    "F1": 0.6,
                    "Average_Precision": 0.8,
                    "AUROC": 0.9,
                    "Threshold": 0.42,
                },
                {
                    "Split": "Test",
                    "Accuracy": 0.65,
                    "Balanced_Accuracy": 0.63,
                    "Precision": 0.67,
                    "F1": 0.55,
                    "Average_Precision": 0.75,
                    "AUROC": 0.85,
                    "Threshold": 0.42,
                },
            ]
        )

        clf._log_final_metrics(metrics)

        calls = {call.args[0]: call.args[1] for call in logged.call_args_list}
        assert calls["Validation_Average_Precision"] == 0.8
        assert calls["Validation_AUROC"] == 0.9
        assert calls["Test_Average_Precision"] == 0.75
        assert calls["Test_AUROC"] == 0.85
        assert calls["Validation_Balanced_Accuracy"] == 0.68
        assert calls["Test_Precision"] == 0.67
        logged_param.assert_called_once_with("decision_threshold", 0.42)

    def test_logs_training_and_eval_loss_figure(self, tmp_path, monkeypatch):
        clf = make_classifier(tmp_path)
        logged = MagicMock()
        monkeypatch.setattr("bertuner.BERTuner.mlflow.log_figure", logged)

        clf._log_loss_curve(
            [
                {"loss": 0.8, "epoch": 1.0},
                {"eval_loss": 0.9, "epoch": 1.0},
                {"loss": 0.5, "epoch": 2.0},
                {"eval_loss": 0.7, "epoch": 2.0},
            ]
        )

        logged.assert_called_once()
        assert logged.call_args.args[1] == "plots/training_vs_evaluation_loss.png"

    def test_loss_figure_is_stored_as_mlflow_artifact(self, tmp_path):
        clf = make_classifier(tmp_path)
        previous_uri = mlflow.get_tracking_uri()
        tracking_uri = (tmp_path / "artifact-test-mlruns").as_uri()
        artifact_location = (tmp_path / "artifact-test-files").as_uri()

        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_id = mlflow.create_experiment(
                "loss-curve-artifact-test",
                artifact_location=artifact_location,
            )
            with mlflow.start_run(experiment_id=experiment_id) as run:
                clf._log_loss_curve(
                    [
                        {"loss": 0.8, "epoch": 1.0},
                        {"eval_loss": 0.9, "epoch": 1.0},
                    ]
                )

            artifact_path = MlflowClient().download_artifacts(
                run.info.run_id,
                "plots/training_vs_evaluation_loss.png",
                str(tmp_path / "downloaded-artifacts"),
            )
            assert Path(artifact_path).is_file()
        finally:
            mlflow.end_run()
            mlflow.set_tracking_uri(previous_uri)

    def test_skips_figure_when_either_loss_series_is_missing(
        self, tmp_path, monkeypatch
    ):
        clf = make_classifier(tmp_path)
        logged = MagicMock()
        monkeypatch.setattr("bertuner.BERTuner.mlflow.log_figure", logged)

        clf._log_loss_curve([{"eval_loss": 0.9, "epoch": 1.0}])

        logged.assert_not_called()


# ------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------


class TestSaveModel:
    def test_saves_model_tokenizer_and_config(self, tmp_path):
        clf = make_classifier(tmp_path)
        clf.best_params = {"model": "bert-base", "learning_rate": 1e-5}
        clf.best_threshold = 0.42

        trainer, tokenizer = MagicMock(), MagicMock()
        clf._save_model(str(tmp_path), trainer, tokenizer, "bert-base-uncased")

        save_dir = str(tmp_path / "model")
        trainer.save_model.assert_called_once_with(save_dir)
        tokenizer.save_pretrained.assert_called_once_with(save_dir)

        with open(tmp_path / "model" / "bertuner_config.json") as f:
            config = json.load(f)
        meta = config["model_metadata"]
        assert meta["model_path"] == "bert-base-uncased"
        assert meta["optimal_threshold"] == 0.42
        assert meta["is_multilabel"] is False
        assert meta["target_cols"] == ["target"]
        assert meta["max_length"] == 512
        assert config["parameters"] == clf.best_params

    def test_multilabel_threshold_serialised_as_list(self, tmp_path):
        cols = ["l1", "l2"]
        clf = make_classifier(
            tmp_path, target_cols=cols, dataframe=make_df(multilabel_cols=cols)
        )
        clf.best_params = {"model": "bert-base"}
        clf.best_threshold = np.array([0.3, 0.6])

        clf._save_model(str(tmp_path), MagicMock(), MagicMock(), "bert-base-uncased")

        with open(tmp_path / "model" / "bertuner_config.json") as f:
            config = json.load(f)
        assert config["model_metadata"]["optimal_threshold"] == [0.3, 0.6]

    def test_explicit_max_length_overrides_requested(self, tmp_path):
        clf = make_classifier(tmp_path, max_length=8192)
        clf.best_params = {"model": "bert-base"}
        clf.best_threshold = 0.5

        clf._save_model(str(tmp_path), MagicMock(), MagicMock(), "bert-base-uncased", 512)

        with open(tmp_path / "model" / "bertuner_config.json") as f:
            config = json.load(f)
        assert config["model_metadata"]["max_length"] == 512


# ------------------------------------------------------------------
# Long-context support
# ------------------------------------------------------------------


class TestLongContext:
    def test_max_length_clamped_to_model_capacity(self, tmp_path):
        clf = make_classifier(tmp_path, max_length=8192)
        tok = SimpleNamespace(model_max_length=512)
        assert clf._effective_max_length(tok, "bert-base-uncased") == 512

    def test_requested_max_length_kept_when_model_allows(self, tmp_path):
        clf = make_classifier(tmp_path, max_length=4096)
        tok = SimpleNamespace(model_max_length=8192)
        assert clf._effective_max_length(tok, "answerdotai/ModernBERT-base") == 4096

    def test_gradient_checkpointing_auto_by_length(self, tmp_path):
        clf = make_classifier(tmp_path)
        assert clf._use_gradient_checkpointing(8192) is True
        assert clf._use_gradient_checkpointing(512) is False

    def test_gradient_checkpointing_explicit_override(self, tmp_path):
        clf = make_classifier(tmp_path, gradient_checkpointing=False)
        assert clf._use_gradient_checkpointing(8192) is False

    def test_longcontext_search_space_round_trip(self, tmp_path):
        clf = make_classifier(tmp_path)
        clf.search_space = DEFAULT_SEARCH_SPACE_LONGCONTEXT
        fixed = {
            "model": "modernbert-large",
            "learning_rate": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "loss_type": "weighted",
            "label_smoothing": 0.05,
            "focal_gamma": 2.0,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
            "dropout": 0.1,
            "early_stopping_patience": 4,
        }
        params = clf._suggest_hyperparams(optuna.trial.FixedTrial(fixed))
        assert params == fixed

    def test_modernbert_dropout_attrs_applied(self, tmp_path, monkeypatch):
        clf = make_classifier(tmp_path)
        cfg = SimpleNamespace(
            model_type="modernbert",
            attention_dropout=0.0,
            mlp_dropout=0.0,
            classifier_dropout=0.0,
            embedding_dropout=0.0,
        )
        monkeypatch.setattr(
            "bertuner.BERTuner.AutoConfig",
            SimpleNamespace(from_pretrained=lambda *a, **k: cfg),
        )
        monkeypatch.setattr(
            "bertuner.BERTuner.AutoModelForSequenceClassification",
            SimpleNamespace(from_pretrained=MagicMock()),
        )
        clf._load_model("answerdotai/ModernBERT-large", 0.2)
        assert (
            cfg.attention_dropout
            == cfg.mlp_dropout
            == cfg.classifier_dropout
            == cfg.embedding_dropout
            == 0.2
        )

    def test_bert_style_dropout_attrs_applied(self, tmp_path, monkeypatch):
        clf = make_classifier(tmp_path)
        cfg = SimpleNamespace(
            model_type="bert",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            classifier_dropout=None,
        )
        monkeypatch.setattr(
            "bertuner.BERTuner.AutoConfig",
            SimpleNamespace(from_pretrained=lambda *a, **k: cfg),
        )
        monkeypatch.setattr(
            "bertuner.BERTuner.AutoModelForSequenceClassification",
            SimpleNamespace(from_pretrained=MagicMock()),
        )
        clf._load_model("bert-base-uncased", 0.25)
        assert cfg.hidden_dropout_prob == 0.25
        assert cfg.attention_probs_dropout_prob == 0.25
        assert cfg.classifier_dropout == 0.25


# ------------------------------------------------------------------
# Multiclass (3+ classes, single-label)
# ------------------------------------------------------------------


class TestMulticlass:
    def make_multiclass(self, tmp_path, n_classes=3):
        return make_classifier(tmp_path, dataframe=make_df(num_classes=n_classes))

    def test_is_binary_flags(self, tmp_path):
        assert make_classifier(tmp_path).is_binary
        assert not self.make_multiclass(tmp_path).is_binary

    def test_get_probs_returns_full_matrix(self, tmp_path):
        clf = self.make_multiclass(tmp_path)
        logits = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2]])
        probs = clf._get_probs(logits)
        assert probs.shape == (2, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    def test_optimize_threshold_returns_none(self, tmp_path):
        clf = self.make_multiclass(tmp_path)
        res = SimpleNamespace(
            predictions=np.array([[2.0, 0.5, -1.0]]), label_ids=np.array([0])
        )
        assert clf._optimize_threshold(res) is None

    def test_compute_metrics_does_not_crash(self, tmp_path):
        clf = self.make_multiclass(tmp_path)
        logits = np.array(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]
        )
        labels = np.array([0, 1, 2, 0])
        metrics = clf._compute_metrics((logits, labels))
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["f1_weighted"] == 1.0
        assert metrics["mcc"] == 1.0
        assert metrics["auc_roc"] == 1.0
        assert metrics["auc_roc_weighted"] == 1.0
        assert metrics["log_loss"] < 0.1
        assert "specificity" not in metrics

    def test_compute_metrics_missing_class_falls_back(self, tmp_path):
        clf = self.make_multiclass(tmp_path)
        logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        labels = np.array([0, 1])  # class 2 absent from eval split
        metrics = clf._compute_metrics((logits, labels))
        assert metrics["auc_roc"] == 0.5

    def test_class_weights_one_per_class(self, tmp_path):
        import torch

        clf = self.make_multiclass(tmp_path)
        ds = [{"labels": torch.tensor(i % 3)} for i in range(12)]
        w = clf._compute_class_weights(ds)
        assert w.shape == (3,)
        assert w[0] == pytest.approx(1.0)  # balanced classes → uniform weights

    def test_build_metrics_df_argmax(self, tmp_path):
        clf = self.make_multiclass(tmp_path)
        y = np.array([0, 1, 2, 1])
        p = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.7, 0.1],
            ]
        )
        df = clf._build_metrics_df(y, p, y, p, thresh=None)
        assert df.iloc[0]["Accuracy"] == 1.0
        assert df.iloc[0]["Balanced_Accuracy"] == 1.0
        assert df.iloc[0]["F1"] == 1.0
        assert df.iloc[0]["F1_weighted"] == 1.0
        assert df.iloc[0]["MCC"] == 1.0
        assert df.iloc[0]["Average_Precision_macro"] == 1.0
        assert df.iloc[0]["AUROC_weighted"] == 1.0
        assert df.iloc[0]["Threshold"] is None


# ------------------------------------------------------------------
# Multiclass weighted loss (regression: pre-0.1.0 code always built
# 2-element class weights, crashing cross_entropy on 3+ classes)
# ------------------------------------------------------------------


class TestMulticlassWeightedLoss:
    """Weighted CE with 3 classes must not raise the shape-[2] weight error."""

    def _trainer(self, class_weights, loss_type="weighted"):
        import torch
        from bertuner.CustomTrainer import CustomTrainer

        t = CustomTrainer.__new__(CustomTrainer)
        t.loss_type = loss_type
        t.class_weights = class_weights
        return t

    def test_weighted_ce_three_classes(self, tmp_path):
        import torch

        clf = make_classifier(tmp_path, dataframe=make_df(num_classes=3))
        ds = [{"labels": torch.tensor(i % 3)} for i in range(9)]
        weights = clf._compute_class_weights(ds)
        assert weights.shape == (3,)

        trainer = self._trainer(weights)
        logits = torch.randn(6, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = trainer._singlelabel_loss(logits, labels, torch.device("cpu"))
        assert torch.isfinite(loss)

    def test_two_element_weights_reproduce_reported_error(self):
        import torch

        trainer = self._trainer(torch.tensor([1.0, 3.0]))
        logits = torch.randn(6, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        with pytest.raises(RuntimeError, match="weight tensor"):
            trainer._singlelabel_loss(logits, labels, torch.device("cpu"))

    def test_focal_and_label_smoothing_three_classes(self, tmp_path):
        import torch

        logits = torch.randn(6, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        for loss_type in ("focal", "label_smoothing", "plain"):
            trainer = self._trainer(None, loss_type=loss_type)
            loss = trainer._singlelabel_loss(logits, labels, torch.device("cpu"))
            assert torch.isfinite(loss)

    def test_end_to_end_training_three_classes(self, tmp_path):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            TrainingArguments,
        )
        from bertuner.CustomTrainer import CustomTrainer

        model_path = "prajjwal1/bert-tiny"
        clf = make_classifier(tmp_path, dataframe=make_df(n=60, num_classes=3))
        assert clf.num_labels == 3

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_ds, val_ds, _ = clf._prepare_datasets(tokenizer, None, max_length=32)
        class_weights = clf._compute_class_weights(train_ds)
        assert class_weights.shape == (3,)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
        )
        args = TrainingArguments(
            output_dir=str(tmp_path / "e2e"),
            per_device_train_batch_size=8,
            num_train_epochs=1,
            eval_strategy="no",
            save_strategy="no",
            report_to=["none"],
            seed=42,
        )
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            loss_type="weighted",
            class_weights=class_weights,
        )
        result = trainer.train()
        assert np.isfinite(result.training_loss)
