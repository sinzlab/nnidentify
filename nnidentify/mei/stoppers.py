from __future__ import annotations
from mei.stoppers import OptimizationStopper
from typing import Tuple, Optional, Any

from .domain import State

import matplotlib.pyplot as plt

class EarlyStopping(OptimizationStopper):
    """Callable that stops the optimization process after validation and evaluation outputs start to differ
       or after a specified number of steps."""

    def __init__(self, num_iterations: int, patience: int = 5):
        """Initializes EarlyStopping.

        Args:
            num_iterations: The number of optimization steps before the process is stopped.
            patience: (default: 3) Number of steps that the validation is allowed to increase.
        """
        self.history = {
            'validation': [],
            'evaluation': []
        }

        self.num_iterations = num_iterations
        self.patience = patience

        self.last_validation = -float('inf')  # previous value of the validation
        self.n_steps_decrease = 0  # number of steps for which the validation has been decreasing

    def __call__(self, current_state: State) -> Tuple[bool, Optional[Any]]:
        """Stops the optimization process after a set number of steps by returning True
        or when the patience threshold is reached."""
        self.history['evaluation'].append(current_state.evaluation)
        self.history['validation'].append(current_state.validation)

        if self.last_validation > current_state.validation:
            self.n_steps_decrease += 1
        else:
            self.n_steps_decrease = 0

        self.last_validation = current_state.validation

        # if self.n_steps_decrease >= self.patience and current_state.i_iter > 20:
        #     print(f'Stopping early after {current_state.i_iter} steps')
        #     plt.plot(self.history['evaluation'], c='tab:cyan')
        #     plt.plot(self.history['validation'], c='tab:purple')
        #     plt.show()
        #
        #     return True, None

        if current_state.i_iter == self.num_iterations:
            print(f'Stopping after {current_state.i_iter} steps')
            plt.plot(self.history['evaluation'], c='tab:cyan')
            plt.plot(self.history['validation'], c='tab:purple')
            plt.show()
            return True, None
        return False, None

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.num_iterations})"
