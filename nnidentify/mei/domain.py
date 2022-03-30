from mei.domain import State
from typing import Any, Dict


class StateWithValidation(State):
    def __init__(self, validation, *args, **kwargs):
        self.validation = validation
        super().__init__(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the State."""
        return dict(
            i_iter=self.i_iter,
            evaluation=self.evaluation,
            validation=self.validation,
            reg_term=self.reg_term,
            input_=self.input,
            transformed_input=self.transformed_input,
            post_processed_input=self.post_processed_input,
            grad=self.grad,
            preconditioned_grad=self.preconditioned_grad,
            stopper_output=self.stopper_output,
        )

    @classmethod
    def from_dict(cls, state: Dict[str, Any]):
        """Creates a new State object from a dictionary."""
        return cls(**state)
