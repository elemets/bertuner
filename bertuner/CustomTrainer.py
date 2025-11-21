from transformers import Trainer
import torch
import torch.nn.functional as F


class CustomTrainer(Trainer):
    """Enhanced trainer with multiple loss functions and advanced features."""

    def __init__(self, loss_type="focal", class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """Focal loss for handling imbalanced datasets."""
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def label_smoothing_loss(self, logits, labels, smoothing=0.1):
        """Label smoothing for better generalization."""
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction="none")
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        return loss.mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_type == "focal":
            loss = self.focal_loss(logits, labels)
        elif self.loss_type == "weighted":
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                loss = loss_fct(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)
        elif self.loss_type == "label_smoothing":
            loss = self.label_smoothing_loss(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
