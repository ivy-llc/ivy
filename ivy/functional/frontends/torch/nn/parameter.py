import ivy
from ivy.functional.frontends.torch.tensor import Tensor
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.ivy.gradients import _variable, _is_variable, _variable_data


class Parameter(Tensor):
    def __init__(self, data=None, device=None, requires_grad=True):
        if data is None:
            data = torch_frontend.empty(0)
        ivy_array = (
            ivy.array(data) if not hasattr(data, "_ivy_array") else data.ivy_array
        )
        ivy_array = _variable(ivy_array) if not _is_variable(data) else ivy_array
        self._ivy_array = ivy.to_device(ivy_array, device) if device else ivy_array
        self._data = Tensor(_variable_data(self._ivy_array), _init_overload=True)
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._grads = None
        self.grad_fn = None

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
