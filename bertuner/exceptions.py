"""Public exceptions for numerical-stability failures."""


class NonFiniteTrainingError(RuntimeError):
    """Raised when training or evaluation produces a verified NaN or Inf."""

    def __init__(
        self,
        stage: str,
        tensor_name: str,
        *,
        step: int | None = None,
        epoch: float | None = None,
        precision: str | None = None,
        learning_rate: float | None = None,
        nonfinite_kind: str | None = None,
    ):
        self.stage = stage
        self.tensor_name = tensor_name
        self.step = step
        self.epoch = epoch
        self.precision = precision
        self.learning_rate = learning_rate
        self.nonfinite_kind = nonfinite_kind
        super().__init__(self._message())

    def add_context(self, *, precision: str | None = None):
        """Add attempt context that was unavailable at detection time."""
        if self.precision is None:
            self.precision = precision
        self.args = (self._message(),)
        return self

    def _message(self) -> str:
        context = [f"stage={self.stage}", f"tensor={self.tensor_name}"]
        if self.step is not None:
            context.append(f"step={self.step}")
        if self.epoch is not None:
            context.append(f"epoch={self.epoch:.6g}")
        if self.precision is not None:
            context.append(f"precision={self.precision}")
        if self.learning_rate is not None:
            context.append(f"learning_rate={self.learning_rate:.6g}")
        if self.nonfinite_kind is not None:
            context.append(f"kind={self.nonfinite_kind}")
        return "Non-finite value detected (" + ", ".join(context) + ")."


class NoStableTrialError(RuntimeError):
    """Raised when an Optuna study finishes without a numerically stable trial."""
