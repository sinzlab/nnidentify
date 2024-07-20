"""This module contains PyTorch modules used in the MEI optimization process."""

from typing import Dict, Any

import torch
from torch import Tensor
from torch.nn import Module, ModuleList


class EnsembleValidationModel(Module):
    """A ensemble model consisting of several individual ensemble members, where some are used for validation

    Attributes:
        *members: PyTorch modules representing the members of the ensemble.
        *n_validation_models: int -> number of models that are to be used as validation (at most len(members) - 1)
    """

    _module_container_cls = ModuleList

    def __init__(self, *members: Module, n_validation_models: int = 3):
        """Initializes EnsembleModel."""
        super().__init__()
        self.members = self._module_container_cls(members)

        if n_validation_models > len(self.members) - 1:
            raise Exception(f'The number of validation models ({n_validation_models}) should not exceed the number of '
                            f'members - 1 ({len(members) - 1})')

        self.n_validation_models = n_validation_models

    def __call__(self, x: Tensor, *args, **kwargs) -> [Tensor, Tensor]:
        """Calculates the forward pass through the ensemble.

        The input is passed through all individual members of the ensemble and their outputs are averaged.

        Args:
            x: A tensor representing the input to the ensemble.
            *args: Additional arguments will be passed to all ensemble members.
            **kwargs: Additional keyword arguments will be passed to all ensemble members.

        Returns:
            A tensor representing the ensemble's output.
        """
        outputs = [m(x, *args, **kwargs) for m in self.members]

        train_outputs = outputs[:-self.n_validation_models]
        val_outputs = outputs[-self.n_validation_models:]

        mean_output = torch.stack(train_outputs, dim=0).mean(dim=0)
        mean_val_output = torch.stack(val_outputs, dim=0).mean(dim=0)

        return torch.cat((mean_output, mean_val_output))

    def __repr__(self):
        return f"{self.__class__.__qualname__}({', '.join(m.__repr__() for m in self.members)})" \
               f" - val_split {self.n_validation_models}"
