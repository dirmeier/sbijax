import dataclasses
import math


# pylint: disable=missing-function-docstring
@dataclasses.dataclass
class EarlyStopping:
    """Early stopping of neural network training."""

    min_delta: float = 0
    patience: int = 0
    best_metric: float = float("inf")
    patience_count: int = 0
    should_stop: bool = False

    def reset(self):
        """Reset the object.

        Returns:
          self
        """
        self.best_metric = float("inf")
        self.patience_count = 0
        self.should_stop = False
        return self

    def update(self, metric):
        """Update the stopping criterion.

        Args:
            metric: the early stopping criterion metric as float

        Returns:
            tuple
        """
        if (
            math.isinf(self.best_metric)
            or self.best_metric - metric > self.min_delta
        ):
            self.best_metric = metric
            self.patience_count = 0
            return True, self

        should_stop = self.patience_count >= self.patience or self.should_stop
        self.should_stop = should_stop
        self.patience_count = self.patience_count + 1
        return False, self
