import sys
from pathlib import Path

# Running ``python scripts/bertuner_test.py`` otherwise puts ``scripts/`` first
# on sys.path and can import a stale site-packages copy instead of this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bertuner.BERTuner import BERTuneClassifier


classifier_opt = BERTuneClassifier(
    data_path="./test_data/question_data_qa_sep.csv",
    models_dir="./models/BERTModels",
    text_feature="text_feature",
    target_cols=["target"],
    # Use the running MLflow server so its mlflow-artifacts proxy stores and
    # serves plots correctly. Start it on port 9090 before running this script.
    mlflow_tracking_uri="http://127.0.0.1:9090"
)
## initializing model choices with default options
classifier_opt.initialize_model_choices()
## initializing search space with default options
classifier_opt.initialize_search_space()
## Optimizing will find the best parameters and save them to the model object
classifier_opt.optimize(
    n_trials=1,
    optimize_metric="avg_precision",
    study_name="bertclass_test",
    greater_is_better=True
)
## Here we train the model on these best parameters
classifier_opt.train_final_model(run_name="ufo_run")
