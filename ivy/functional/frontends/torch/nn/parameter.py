from ivy.functional.frontends.torch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, array, device=None, _init_overload=False, requires_grad=True):
        super().__init__(array, device, _init_overload, requires_grad)

    def __deepcopy__(self, memo):
        # TODO: Need to add test. Adding for KLA demo (priority)
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
