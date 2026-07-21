"""Example: tune and train 8K-context ModernBERT classifiers.

Run from the repository root:

    python scripts/bertuner_long_context_test.py

ModernBERT-large at an 8,192-token sequence length requires a substantial amount
of GPU memory. The long-context search space therefore uses small per-device
batches and gradient accumulation. Reduce the model list to ``modernbert-base``
or reduce ``max_length`` if the run does not fit on your GPU.
"""

from bertuner.BERTuner import BERTuneClassifier
from bertuner.constants import DEFAULT_SEARCH_SPACE_LONGCONTEXT


# Both of these ModernBERT checkpoints support an 8,192-token context window.
MODEL_CHOICES = {
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
}


def main():
    classifier_opt = BERTuneClassifier(
        data_path="./test_data/question_data_qa_sep.csv",
        models_dir="./models/ModernBERTModels",
        text_feature="text_feature",
        target_cols=["target"],
        mlflow_tracking_uri="./mlruns",
        max_length=8192,
        # Long sequences automatically enable checkpointing, but setting this
        # explicitly makes the memory-saving behavior clear in this example.
        gradient_checkpointing=True,
    )

    classifier_opt.initialize_model_choices(MODEL_CHOICES)

    # Uses per-device batches of 1/2/4 and gradient accumulation of 4/8/16.
    # Deriving the names from MODEL_CHOICES also makes it safe to remove one
    # checkpoint above when you want to test only base or only large.
    search_space = {
        **DEFAULT_SEARCH_SPACE_LONGCONTEXT,
        "model": list(MODEL_CHOICES),
    }
    classifier_opt.initialize_search_space(search_space)

    best_value = classifier_opt.optimize(
        n_trials=1,
        optimize_metric="avg_precision",
        study_name="modernbert_8k_test",
        greater_is_better=True,
    )
    print(f"Best validation average precision: {best_value:.4f}")

    # Retrain the selected model and parameters, evaluate it, and save the result.
    metrics, _, _ = classifier_opt.train_final_model(
        run_name="modernbert_8k_final"
    )
    print(metrics)


if __name__ == "__main__":
    main()
