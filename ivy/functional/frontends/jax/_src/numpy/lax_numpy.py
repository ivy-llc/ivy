# global
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def _rewriting_take(
    arr, idx, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None
):
    return ivy.get_item(arr, idx)
