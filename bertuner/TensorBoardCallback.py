import os
import shutil
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter


class CleanupCheckpointsCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        best_ckpt = state.best_model_checkpoint
        for name in os.listdir(args.output_dir):
            path = os.path.join(args.output_dir, name)
            if name.startswith("checkpoint‐") and path != best_ckpt:
                shutil.rmtree(path)


class TensorBoardSyncCallback(TrainerCallback):
    def __init__(self, log_dir, writer_cls=None):
        # Allow dependency injection for custom writers (e.g., testing or alternative loggers).
        writer_cls = writer_cls or SummaryWriter
        self.writer = writer_cls(log_dir)
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step

        # remember most recent train loss
        if "loss" in logs:
            self.last_train_loss = logs["loss"]

        # when eval happens, log both at once
        if "eval_loss" in logs and self.last_train_loss is not None:
            self.writer.add_scalars(
                "Loss",  # main tag
                {"train": self.last_train_loss, "eval": logs["eval_loss"]},
                step,
            )

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
