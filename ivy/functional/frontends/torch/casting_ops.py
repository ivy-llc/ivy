import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_native_arrays,
)
from ivy.func_wrapper import outputs_to_ivy_arrays


def is_autocast_enabled():
    return False

def is_autocast_cpu_enabled():
    return False
