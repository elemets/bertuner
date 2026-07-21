import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

from bertuner.exceptions import NonFiniteTrainingError


class NonFiniteGradientCallback(TrainerCallback):
    """Abort before optimizer.step() when any clipped gradient is NaN or Inf."""

    def __init__(self, precision: str | None = None):
        self.precision = precision

    @staticmethod
    def _gradient_values(gradient: torch.Tensor) -> torch.Tensor:
        return gradient.coalesce().values() if gradient.is_sparse else gradient

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        # One host synchronization per device during healthy training. Individual
        # gradients are inspected only after their device-level check fails.
        gradients_by_device = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            values = self._gradient_values(parameter.grad.detach())
            gradients_by_device.setdefault(values.device, []).append((name, values))

        for entries in gradients_by_device.values():
            checks = torch.stack([torch.isfinite(values).all() for _, values in entries])
            if checks.all().item():
                continue
            bad_name = next(
                name for name, values in entries if not torch.isfinite(values).all().item()
            )
            bad_values = next(values for name, values in entries if name == bad_name)
            nonfinite_kind = "nan" if torch.isnan(bad_values).any().item() else "inf"
            optimizer = kwargs.get("optimizer")
            learning_rate = (
                optimizer.param_groups[0].get("lr")
                if optimizer is not None and optimizer.param_groups
                else None
            )
            raise NonFiniteTrainingError(
                "backward",
                f"gradient:{bad_name}",
                step=state.global_step,
                epoch=state.epoch,
                precision=self.precision,
                learning_rate=learning_rate,
                nonfinite_kind=nonfinite_kind,
            )

        return control


class CustomTrainer(Trainer):
    """Trainer supporting single-label (CE/focal/label-smoothing) and multi-label (BCE) classification."""

    def __init__(
        self,
        loss_type="focal",
        class_weights=None,
        training_precision: str | None = None,
        **kwargs,
    ):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(NonFiniteGradientCallback(training_precision))
        super().__init__(callbacks=callbacks, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.training_precision = training_precision

    def _require_finite(self, tensor: torch.Tensor, stage: str, tensor_name: str):
        if torch.isfinite(tensor.detach()).all().item():
            return
        detached = tensor.detach()
        nonfinite_kind = "nan" if torch.isnan(detached).any().item() else "inf"
        optimizer = getattr(self, "optimizer", None)
        learning_rate = (
            optimizer.param_groups[0].get("lr")
            if optimizer is not None and optimizer.param_groups
            else None
        )
        raise NonFiniteTrainingError(
            stage,
            tensor_name,
            step=self.state.global_step,
            epoch=self.state.epoch,
            precision=self.training_precision,
            learning_rate=learning_rate,
            nonfinite_kind=nonfinite_kind,
        )

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
        self._require_finite(logits, "forward", "logits")

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

        self._require_finite(loss, "loss", "loss")

        return (loss, outputs) if return_outputs else loss
