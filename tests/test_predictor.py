"""Integration tests for BERTunePredictor using a small locally-cached model."""
import json

import numpy as np
import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bertuner.Predictor import BERTunePredictor

MODEL_PATH = "google/electra-small-discriminator"


def _save_model_dir(tmp_dir, num_labels, is_multilabel, threshold, target_cols, id2label=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=num_labels,
            problem_type=(
                "multi_label_classification"
                if is_multilabel
                else "single_label_classification"
            ),
            local_files_only=True,
        )
    except OSError:
        pytest.skip(f"{MODEL_PATH} not in local HF cache")

    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    config = {
        "model_metadata": {
            "model": "electra-small",
            "model_path": MODEL_PATH,
            "optimal_threshold": threshold,
            "is_multilabel": is_multilabel,
            "target_cols": target_cols,
            "max_length": 128,
            "id2label": id2label,
        },
        "parameters": {"model": "electra-small"},
    }
    with open(f"{tmp_dir}/bertuner_config.json", "w") as f:
        json.dump(config, f)
    return str(tmp_dir)


@pytest.fixture(scope="module")
def binary_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("binary_model")
    return _save_model_dir(d, 2, False, 0.42, ["target"])


@pytest.fixture(scope="module")
def multilabel_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("multilabel_model")
    return _save_model_dir(d, 3, True, [0.3, 0.5, 0.7], ["l1", "l2", "l3"])


@pytest.fixture(scope="module")
def multiclass_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("multiclass_model")
    return _save_model_dir(
        d, 3, False, None, ["target"], id2label={"0": "low", "1": "mid", "2": "high"}
    )


class TestBinaryPredictor:
    def test_loads_config(self, binary_model_dir):
        p = BERTunePredictor(binary_model_dir, device="cpu")
        assert p.threshold == 0.42
        assert not p.is_multilabel
        assert p.max_length == 128

    def test_predict_proba_shape_and_sum(self, binary_model_dir):
        p = BERTunePredictor(binary_model_dir, device="cpu")
        probs = p.predict_proba(["hello world", "another text"])
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_applies_threshold(self, binary_model_dir):
        p = BERTunePredictor(binary_model_dir, device="cpu")
        texts = ["hello world", "another text", "third text"]
        preds = p.predict(texts)
        probs = p.predict_proba(texts)
        assert preds.shape == (3,)
        np.testing.assert_array_equal(preds, (probs[:, 1] >= 0.42).astype(int))

    def test_accepts_single_string(self, binary_model_dir):
        p = BERTunePredictor(binary_model_dir, device="cpu")
        assert p.predict("just one text").shape == (1,)

    def test_batching_matches_single_pass(self, binary_model_dir):
        p = BERTunePredictor(binary_model_dir, device="cpu", batch_size=2)
        texts = [f"text number {i}" for i in range(5)]
        batched = p.predict_proba(texts)
        p.batch_size = 32
        single = p.predict_proba(texts)
        assert np.allclose(batched, single, atol=1e-4)

    def test_predict_df_columns(self, binary_model_dir):
        p = BERTunePredictor(binary_model_dir, device="cpu")
        df = p.predict_df(["hello", "world"])
        assert list(df.columns) == ["target"]
        assert len(df) == 2


class TestMultilabelPredictor:
    def test_per_label_thresholds(self, multilabel_model_dir):
        p = BERTunePredictor(multilabel_model_dir, device="cpu")
        texts = ["hello world", "another text"]
        probs = p.predict_proba(texts)
        preds = p.predict(texts)
        assert probs.shape == (2, 3)
        assert preds.shape == (2, 3)
        np.testing.assert_array_equal(
            preds, (probs >= np.array([0.3, 0.5, 0.7])).astype(int)
        )

    def test_sigmoid_probs_independent(self, multilabel_model_dir):
        p = BERTunePredictor(multilabel_model_dir, device="cpu")
        probs = p.predict_proba(["some text"])
        assert ((probs > 0) & (probs < 1)).all()

    def test_predict_df_columns(self, multilabel_model_dir):
        p = BERTunePredictor(multilabel_model_dir, device="cpu")
        df = p.predict_df(["hello"])
        assert list(df.columns) == ["l1", "l2", "l3"]


class TestMulticlassPredictor:
    def test_loads_none_threshold_and_id2label(self, multiclass_model_dir):
        p = BERTunePredictor(multiclass_model_dir, device="cpu")
        assert p.threshold is None
        assert p.id2label == {0: "low", 1: "mid", 2: "high"}

    def test_predict_is_argmax(self, multiclass_model_dir):
        p = BERTunePredictor(multiclass_model_dir, device="cpu")
        texts = ["one text", "another text"]
        probs = p.predict_proba(texts)
        np.testing.assert_array_equal(p.predict(texts), probs.argmax(axis=1))

    def test_predict_df_maps_ids_to_labels(self, multiclass_model_dir):
        p = BERTunePredictor(multiclass_model_dir, device="cpu")
        df = p.predict_df(["hello", "world"])
        assert list(df.columns) == ["target"]
        assert set(df["target"]) <= {"low", "mid", "high"}
