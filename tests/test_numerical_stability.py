"""Deterministic tests for NaN/Inf detection and FP32 recovery."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from bertuner.BERTuner import BERTuneClassifier
from bertuner.CustomTrainer import CustomTrainer, NonFiniteGradientCallback
from bertuner.exceptions import NonFiniteTrainingError, NoStableTrialError


def make_classifier(tmp_path, **kwargs):
    defaults = {
        "models_dir": str(tmp_path),
        "text_feature": "text",
        "target_cols": ["label"],
        "dataframe": pd.DataFrame(
            {"text": [f"sample {i}" for i in range(20)], "label": [0, 1] * 10}
        ),
        "mlflow_tracking_uri": str(tmp_path / "mlruns"),
    }
    defaults.update(kwargs)
    return BERTuneClassifier(**defaults)


def sampled_params():
    return {
        "model": "tiny",
        "learning_rate": 2e-5,
        "batch_size": 4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "scheduler": "linear",
        "dropout": 0.1,
        "early_stopping_patience": 3,
        "loss_type": "weighted",
    }


def bare_trainer(precision="bf16"):
    trainer = CustomTrainer.__new__(CustomTrainer)
    trainer.loss_type = "plain"
    trainer.class_weights = None
    trainer.training_precision = precision
    trainer.state = SimpleNamespace(global_step=7, epoch=1.5)
    trainer.optimizer = None
    return trainer


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self._logits = logits

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self._logits + self.anchor * 0)


def test_nonfinite_logits_raise_before_loss():
    trainer = bare_trainer()
    model = FixedLogitModel(torch.tensor([[float("nan"), 0.0]]))

    with pytest.raises(NonFiniteTrainingError) as caught:
        trainer.compute_loss(
            model,
            {"input_ids": torch.tensor([[1]]), "labels": torch.tensor([0])},
        )

    assert caught.value.stage == "forward"
    assert caught.value.tensor_name == "logits"
    assert caught.value.precision == "bf16"
    assert caught.value.nonfinite_kind == "nan"


def test_nonfinite_loss_raises_with_finite_logits():
    trainer = bare_trainer()
    trainer._singlelabel_loss = lambda *args: torch.tensor(float("inf"))
    model = FixedLogitModel(torch.tensor([[1.0, 0.0]]))

    with pytest.raises(NonFiniteTrainingError) as caught:
        trainer.compute_loss(
            model,
            {"input_ids": torch.tensor([[1]]), "labels": torch.tensor([0])},
        )

    assert caught.value.stage == "loss"
    assert caught.value.nonfinite_kind == "inf"


def test_nonfinite_gradient_raises_before_optimizer_step():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("nan"), 0.0]])
    model.bias.grad = torch.tensor([0.0])
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)
    callback = NonFiniteGradientCallback("fp16")

    with pytest.raises(NonFiniteTrainingError) as caught:
        callback.on_pre_optimizer_step(
            None,
            SimpleNamespace(global_step=4, epoch=0.5),
            SimpleNamespace(),
            model=model,
            optimizer=optimizer,
        )

    assert caught.value.stage == "backward"
    assert caught.value.tensor_name == "gradient:weight"
    assert caught.value.learning_rate == pytest.approx(2e-5)


def test_evaluation_logits_fail_before_sklearn(tmp_path):
    clf = make_classifier(tmp_path)
    logits = np.array([[0.0, float("nan")], [1.0, 0.0]])

    with pytest.raises(NonFiniteTrainingError, match="stage=evaluation"):
        clf._compute_metrics((logits, np.array([0, 1])))


def test_large_class_weight_warns_without_clamping(tmp_path):
    clf = make_classifier(tmp_path, class_weight_warning_threshold=100.0)
    dataset = (
        [{"labels": torch.tensor(0)} for _ in range(201)]
        + [{"labels": torch.tensor(1)}]
    )

    with pytest.warns(RuntimeWarning, match="weight=201"):
        weights = clf._compute_class_weights(dataset)

    assert weights.tolist() == pytest.approx([1.0, 201.0])


def test_invalid_multilabel_weights_fail_without_precision_retry(tmp_path):
    clf = make_classifier(
        tmp_path,
        target_cols=["a", "b"],
        dataframe=pd.DataFrame(
            {"text": ["x", "y"], "a": [0, 1], "b": [1, 0]}
        ),
    )
    dataset = [
        {"labels": torch.tensor([1.0, float("nan")])},
        {"labels": torch.tensor([0.0, 1.0])},
    ]

    with pytest.raises(ValueError, match="labels contain NaN or Inf"):
        clf._compute_class_weights(dataset)


def configure_objective(clf, monkeypatch, runner):
    params = sampled_params()
    clf.MODEL_CHOICES = {"tiny": "local/tiny"}
    clf.optimize_metric = "avg_precision"
    clf.greater_is_better = True
    monkeypatch.setattr(clf, "_suggest_hyperparams", lambda trial: params)
    monkeypatch.setattr(clf, "_get_tokenizer", lambda model_path: object())
    monkeypatch.setattr(clf, "_effective_max_length", lambda *args: 32)
    monkeypatch.setattr(clf, "_prepare_datasets", lambda *args: ("train", "val", "test"))
    monkeypatch.setattr(clf, "_compute_class_weights", lambda dataset: torch.ones(2))
    monkeypatch.setattr(clf, "_resolve_precision", lambda: "bf16")
    monkeypatch.setattr(clf, "_run_training_attempt", runner)
    return params


def test_trial_retries_same_params_in_fp32_and_cleans_attempts(tmp_path, monkeypatch):
    clf = make_classifier(tmp_path)
    calls = []

    def runner(params, *args, **kwargs):
        precision, output_dir = args[6], args[7]
        calls.append((params, precision, output_dir))
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "checkpoint.bin").write_text("synthetic")
        if precision == "bf16":
            raise NonFiniteTrainingError("loss", "loss", precision=precision)
        trainer = SimpleNamespace(
            evaluate=lambda: {"eval_avg_precision": 0.75},
            state=SimpleNamespace(log_history=[]),
        )
        return trainer, object()

    params = configure_objective(clf, monkeypatch, runner)
    study = optuna.create_study()
    trial = study.ask()

    assert clf._objective(trial) == pytest.approx(0.75)
    assert [precision for _, precision, _ in calls] == ["bf16", "fp32"]
    assert all(attempt_params is params for attempt_params, _, _ in calls)
    assert calls[0][2].endswith("/mixed")
    assert calls[1][2].endswith("/fp32")
    assert not (tmp_path / f"optuna_trial_{trial.number}").exists()
    assert trial.user_attrs["precision_fallback"] is True
    assert trial.user_attrs["effective_precision"] == "fp32"


def test_trial_prunes_after_mixed_and_fp32_fail(tmp_path, monkeypatch):
    clf = make_classifier(tmp_path)
    calls = []

    def runner(params, *args, **kwargs):
        precision, output_dir = args[6], args[7]
        calls.append(precision)
        Path(output_dir).mkdir(parents=True)
        raise NonFiniteTrainingError("backward", "gradient:test", precision=precision)

    configure_objective(clf, monkeypatch, runner)
    trial = optuna.create_study().ask()

    with pytest.raises(optuna.TrialPruned, match="precision=fp32"):
        clf._objective(trial)

    assert calls == ["bf16", "fp32"]
    assert "fp32_failure_reason" in trial.user_attrs
    assert not (tmp_path / f"optuna_trial_{trial.number}").exists()


def test_non_numerical_error_never_retries(tmp_path, monkeypatch):
    clf = make_classifier(tmp_path)
    calls = []

    def runner(params, *args, **kwargs):
        calls.append(args[6])
        raise ValueError("invalid labels")

    configure_objective(clf, monkeypatch, runner)

    with pytest.raises(ValueError, match="invalid labels"):
        clf._objective(optuna.create_study().ask())

    assert calls == ["bf16"]


def test_final_training_retries_fp32_and_persists_effective_precision(
    tmp_path, monkeypatch
):
    clf = make_classifier(tmp_path)
    clf.MODEL_CHOICES = {"tiny": "local/tiny"}
    clf.best_params = sampled_params()
    clf.best_precision = "bf16"
    clf.optimize_metric = "avg_precision"
    clf.greater_is_better = True
    calls = []

    monkeypatch.setattr(clf, "_get_tokenizer", lambda model_path: object())
    monkeypatch.setattr(clf, "_effective_max_length", lambda *args: 32)
    monkeypatch.setattr(clf, "_prepare_datasets", lambda *args: ("train", "val", "test"))
    monkeypatch.setattr(clf, "_compute_class_weights", lambda dataset: torch.ones(2))

    prediction = SimpleNamespace(
        predictions=np.array([[3.0, 0.0], [0.0, 3.0]]),
        label_ids=np.array([0, 1]),
    )

    def runner(params, *args, **kwargs):
        precision, output_dir = args[6], args[7]
        calls.append(precision)
        Path(output_dir).mkdir(parents=True)
        if precision == "bf16":
            raise NonFiniteTrainingError("forward", "logits", precision=precision)
        trainer = SimpleNamespace(
            predict=lambda dataset: prediction,
            state=SimpleNamespace(log_history=[]),
        )
        return trainer, "fresh-fp32-model"

    monkeypatch.setattr(clf, "_run_training_attempt", runner)
    monkeypatch.setattr(clf, "_save_model", MagicMock())
    monkeypatch.setattr(clf, "_log_final_metrics", MagicMock())
    monkeypatch.setattr(clf, "_log_loss_curve", MagicMock())
    monkeypatch.setattr("bertuner.BERTuner.mlflow.start_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr("bertuner.BERTuner.mlflow.log_params", MagicMock())
    monkeypatch.setattr("bertuner.BERTuner.mlflow.set_tag", MagicMock())

    _, model, _ = clf.train_final_model()

    assert calls == ["bf16", "fp32"]
    assert model == "fresh-fp32-model"
    assert clf.best_precision == "fp32"
    assert clf.best_precision_fallback is True
    assert not (tmp_path / "final_model" / "mixed").exists()
    assert not (tmp_path / "final_model" / "fp32").exists()


def test_no_completed_trials_raise_summary_error(tmp_path, monkeypatch):
    clf = make_classifier(tmp_path, precision="fp32")
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(lifecycle_stage="active")

    def prune(trial):
        trial.set_user_attr("initial_failure_reason", "synthetic non-finite loss")
        raise optuna.TrialPruned("synthetic")

    monkeypatch.setattr(clf, "_objective", prune)
    monkeypatch.setattr("bertuner.BERTuner.MlflowClient", lambda: client)
    monkeypatch.setattr("bertuner.BERTuner.mlflow.set_tracking_uri", MagicMock())
    monkeypatch.setattr("bertuner.BERTuner.mlflow.set_experiment", MagicMock())

    with pytest.raises(NoStableTrialError, match="No numerically stable trial"):
        clf.optimize(n_trials=2)


def test_training_arguments_make_clipping_and_nan_logging_explicit(tmp_path):
    clf = make_classifier(tmp_path, precision="fp32", max_grad_norm=0.5)
    clf.optimize_metric = "avg_precision"
    clf.greater_is_better = True

    args = clf._build_training_arguments(
        sampled_params(), str(tmp_path / "attempt"), 32, "fp32"
    )

    assert args.max_grad_norm == pytest.approx(0.5)
    assert args.logging_nan_inf_filter is False
    assert args.bf16 is False
    assert args.fp16 is False


def test_exported_config_records_precision_recovery(tmp_path):
    clf = make_classifier(tmp_path, precision="auto")
    clf.best_params = {"model": "tiny"}
    clf.best_threshold = 0.5
    clf.best_precision = "fp32"
    clf.best_precision_fallback = True
    trainer, tokenizer = MagicMock(), MagicMock()

    clf._save_model(str(tmp_path), trainer, tokenizer, "local/tiny", 32)

    with open(tmp_path / "model" / "bertuner_config.json") as config_file:
        metadata = json.load(config_file)["model_metadata"]
    assert metadata["requested_precision"] == "auto"
    assert metadata["effective_precision"] == "fp32"
    assert metadata["precision_fallback"] is True
