# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    inputs_to_ivy_arrays,
    _assert_no_array,
    _assert_array,
)


@inputs_to_ivy_arrays
def copyto(dst, src, /, *, casting="same_kind", where=True):
    # Handle casting
    # Numpy copyto doesn't cast the inputs
    # It just checks casting rules
    ivy.utils.assertions.check_elem_in_list(
        casting,
        ["no", "equiv", "safe", "same_kind", "unsafe"],
        message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
    )

    args = [dst, src]
    args_idxs = ivy.nested_argwhere(args, ivy.is_array)
    args_to_check = ivy.multi_index_nest(args, args_idxs)
    dtype = args_to_check[0].dtype

    if casting in ["no", "equiv"]:
        _assert_no_array(
            args_to_check,
            dtype,
        )
    elif casting in ["same_kind", "safe"]:
        _assert_array(
            args_to_check,
            dtype,
            casting=casting,
        )

    ivy.where(where, src, dst, out=dst)


@inputs_to_ivy_arrays
def shape(array, /):
    return ivy.shape(array)
