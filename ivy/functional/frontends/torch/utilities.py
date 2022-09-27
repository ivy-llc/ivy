# global
import torch

# local


class Utilities:

    def result_type(arg1, arg2):
        return torch.result_type(arg1, arg2)

    def can_cast(dtype1, dtype2):
        return torch.can_cast(dtype1, dtype2)

    def _assert(condition, message):
        return torch._assert(condition, message)
