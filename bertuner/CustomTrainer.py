from transformers import Trainer
import torch
import torch.nn.functional as F


class CustomTrainer(Trainer):
    """Trainer supporting single-label (CE/focal/label-smoothing) and multi-label (BCE) classification."""

    def __init__(self, loss_type="focal", class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights

    def _is_multilabel(self, labels: torch.Tensor) -> bool:
        """Multi-label labels are 2D float vectors; single-label are 1D integer class indices."""
        return labels.dim() == 2

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """Focal loss with -100 ignore-index handling."""
        num_classes = logits.size(-1)

        # Flatten in case of token-level output [B, seq_len, C] → [B*seq_len, C]
        logits = logits.view(-1, num_classes)
        labels = labels.view(-1)

        ignore_mask = labels == -100
        safe_labels = labels.clone()
        safe_labels[ignore_mask] = 0  # temp placeholder, masked out below

        ce_loss = F.cross_entropy(logits, safe_labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = alpha * (1 - pt) ** gamma * ce_loss

        focal = focal.masked_fill(ignore_mask, 0.0)
        n_valid = (~ignore_mask).sum().clamp(min=1)
        return focal.sum() / n_valid

    def label_smoothing_loss(self, logits, labels, smoothing=0.1):
        """Label smoothing with -100 ignore-index handling."""
        num_classes = logits.size(-1)

        logits = logits.view(-1, num_classes)
        labels = labels.view(-1)

        ignore_mask = labels == -100
        safe_labels = labels.clone()
        safe_labels[ignore_mask] = 0

        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, safe_labels, reduction="none")
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss

        loss = loss.masked_fill(ignore_mask, 0.0)
        n_valid = (~ignore_mask).sum().clamp(min=1)
        return loss.sum() / n_valid

    def _multilabel_loss(self, logits, labels, device):
        """BCEWithLogitsLoss with optional per-label pos_weight for class imbalance."""
        pos_weight = self.class_weights.to(device) if self.class_weights is not None else None
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fct(logits, labels.float())

    def _singlelabel_loss(self, logits, labels, device):
        """Dispatches to the configured single-label loss."""
        if self.loss_type == "focal":
            return self.focal_loss(logits, labels)
        elif self.loss_type == "weighted":
            weight = self.class_weights.to(device) if self.class_weights is not None else None
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        elif self.loss_type == "label_smoothing":
            return self.label_smoothing_loss(logits, labels)
        else:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        labels = inputs.pop("labels")
        if torch.is_tensor(labels):
            labels = labels.to(device)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        # ── Label sanity check (catches mismatches early, CPU-side) ──────────
        if not self._is_multilabel(labels):
            valid_mask = labels != -100
            if valid_mask.any():
                label_max = labels[valid_mask].max().item()
                num_classes = logits.size(-1)
                if label_max >= num_classes:
                    raise ValueError(
                        f"Label value {label_max} is out of range for "
                        f"num_labels={num_classes}. Check your label encoding."
                    )
        # ─────────────────────────────────────────────────────────────────────

        if self._is_multilabel(labels):
            loss = self._multilabel_loss(logits, labels, device)
        else:
            loss = self._singlelabel_loss(logits, labels, device)

        return (loss, outputs) if return_outputs else loss
