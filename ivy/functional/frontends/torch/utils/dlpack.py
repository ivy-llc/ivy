# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def from_dlpack(ext_tensor):
    return ivy.from_dlpack(ext_tensor)


@to_ivy_arrays_and_back
def to_dlpack(ext_tensor):
    return ivy.to_dlpack(ext_tensor)
