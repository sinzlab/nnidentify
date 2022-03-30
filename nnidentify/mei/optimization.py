from __future__ import annotations

from mei.optimization import MEI

from .domain import StateWithValidation


class MEIWithValidation(MEI):
    cls_state = StateWithValidation

    def step(self) -> StateWithValidation:
        """Performs an optimization step."""
        state = dict(i_iter=self.i_iteration, input_=self._current_input.cloned_data)

        self.optimizer.zero_grad()

        output = self.evaluate()
        evaluation, validation = output[0], output[1]
        state["evaluation"] = evaluation.item()
        state["validation"] = validation.item()

        reg_term = self.regularization(self._transformed_input, self.i_iteration)
        state["reg_term"] = reg_term.item()

        state["transformed_input"] = self._transformed_input.data.cpu().clone()

        (-evaluation + reg_term).backward()

        if self._current_input.grad is None:
            raise RuntimeError("Gradient did not reach MEI")

        state["grad"] = self._current_input.cloned_grad

        self._current_input.grad = self.precondition(self._current_input.grad, self.i_iteration)
        state["preconditioned_grad"] = self._current_input.cloned_grad

        self.optimizer.step()

        self._current_input.data = self.postprocessing(self._current_input.data, self.i_iteration)
        state["post_processed_input"] = self._current_input.cloned_data

        self._transformed = None
        self.i_iteration += 1

        return self.cls_state.from_dict(state)
