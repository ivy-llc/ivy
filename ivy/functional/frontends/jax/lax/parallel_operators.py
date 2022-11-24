# global
from typing import Any

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def axis_index(axis_name):
    return ivy.axis_index(axis_name)
