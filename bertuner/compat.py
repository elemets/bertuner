"""Small adapters for Transformers APIs shared by supported 4.x and 5.x."""

import math

from transformers import TrainingArguments as HFTrainingArguments


class TrainingArguments(HFTrainingArguments):
    def get_warmup_steps(self, num_training_steps):
        ratio = getattr(self, "_bertuner_warmup_ratio", None)
        if ratio is not None:
            # New warmup_steps treats 1.0 as one step, whereas the old
            # warmup_ratio treats it as the entire training run.
            return math.ceil(num_training_steps * ratio)
        return super().get_warmup_steps(num_training_steps)
