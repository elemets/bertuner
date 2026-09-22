"""Run unchanged against the oldest supported Transformers and current releases."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from transformers import AutoTokenizer, TrainingArguments

from bertuner.CustomTrainer import CustomTrainer
from test_bertuner import make_classifier, make_df
from test_numerical_stability import sampled_params


@pytest.mark.parametrize("ratio", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("final", [False, True])
def test_warmup_and_scheduler(tmp_path, ratio, final):
    clf = make_classifier(tmp_path)
    clf.optimize_metric = "avg_precision"
    clf.greater_is_better = True
    params = dict(sampled_params(), warmup_ratio=ratio, scheduler="cosine")
    args = clf._build_training_arguments(
        params, str(tmp_path / "output"), 32, "fp32", final=final,
        logging_dir=str(tmp_path / "logs"),
    )
    assert args.get_warmup_steps(100) == int(100 * ratio)
    assert args.lr_scheduler_type == "cosine"


def test_final_training_logs_and_reloads(tmp_path, tiny_model_path):
    clf = make_classifier(tmp_path, dataframe=make_df(n=30, num_classes=3))
    clf.optimize_metric = "f1"
    clf.greater_is_better = True
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
    train, val, _ = clf._prepare_datasets(tokenizer, None, max_length=32)
    params = sampled_params()
    log_dir = str(tmp_path / "logs")
    args = clf._build_training_arguments(
        params, str(tmp_path / "output"), 32, "fp32", final=True, logging_dir=log_dir,
    )
    args.num_train_epochs = 1
    args.use_cpu = True
    trainer = clf._build_trainer(
        clf._load_model(tiny_model_path, 0.0), args, train, val, tokenizer,
        params, clf._compute_class_weights(train), "fp32", final=True,
        logging_dir=log_dir,
    )
    assert np.isfinite(trainer.train().training_loss)
    assert trainer.state.best_model_checkpoint is not None
    assert np.isfinite(trainer.evaluate()["eval_loss"])
    events = EventAccumulator(log_dir).Reload()
    assert "train/loss" in events.Tags()["scalars"]
    assert "eval/loss" in events.Tags()["scalars"]


class KwargsClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.accepts_loss_kwargs = True

    def forward(self, features, labels=None, **kwargs):
        return SimpleNamespace(logits=self.linear(features))


def test_accumulation_matches_full_batch_update(tmp_path):
    torch.manual_seed(42)
    initial = KwargsClassifier()
    dataset = Dataset.from_dict({"features": [[1., 2.]] * 4, "labels": [1] * 4})
    models = []
    for batch, accumulation in [(4, 1), (2, 2)]:
        model = copy.deepcopy(initial)
        args = TrainingArguments(
            output_dir=str(tmp_path / str(batch)), use_cpu=True,
            per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
            max_steps=1, max_grad_norm=0., report_to="none", save_strategy="no",
            lr_scheduler_type="constant", disable_tqdm=True,
        )
        trainer = CustomTrainer(
            model=model, args=args, train_dataset=dataset, loss_type="plain",
            optimizers=(torch.optim.SGD(model.parameters(), lr=0.1), None),
        )
        trainer.train()
        models.append(model)
    for full, accumulated in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(full, accumulated)
