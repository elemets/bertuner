"""Example: load a model trained by BERTuneClassifier and run inference with BERTunePredictor."""
import pandas as pd

from bertuner.Predictor import BERTunePredictor

## Directory written by classifier_opt.train_final_model()
predictor = BERTunePredictor("./models/BERTModels/final_model/model")

## Single string
print(predictor.predict("What is the investigator manual?"))

## Batch of texts
texts = [
    "What submission system does the IRB use?",
    "Where do I submit documents to the IRB?",
    "I want to know about pasta",
]
print(predictor.predict(texts))

## Probabilities (softmax for single-label, per-label sigmoid for multi-label)
print(predictor.predict_proba(texts))

## DataFrame output, one column per target
print(predictor.predict_df(texts))

## Scoring a whole CSV
df = pd.read_csv("./test_data/question_data_qa_sep.csv")
df["prediction"] = predictor.predict(df["text_feature"].tolist())
print(df[["text_feature", "prediction"]].head())
