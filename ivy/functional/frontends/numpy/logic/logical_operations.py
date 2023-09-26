# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_and(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.logical_and(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_not(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.logical_not(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_or(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.logical_or(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_xor(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.logical_xor(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def my_logical_not(x):
    """
    Custom implementation of logical NOT for various data types.

    Parameters:
    - x: Input value or array to apply the logical NOT operation on.

    Returns:
    - Result of the logical NOT operation applied to the input.

    Note:
    This function handles different data types including integers, floats, NumPy arrays,
    lists, and dictionaries and performs logical NOT accordingly.
    """
    # Check if the input x is an integer
    if isinstance(x, int):
        # If x is an integer, return the logical NOT of x (True for 0, False for non-zero)
        return not x
    # Check if the input x is a float
    elif isinstance(x, float):
        # If x is a float, return the logical NOT of x (True for 0, False for non-zero)
        return not x
    # Check if the input x is a NumPy array
    elif isinstance(x, np.ndarray):
        # Check if the data type of the NumPy array is boolean
        if x.dtype == bool:
            # If it's a boolean array, use NumPy's logical NOT function
            return np.logical_not(x)
        # Check if the data type of the NumPy array is an integer type
        elif np.issubdtype(x.dtype, np.integer):
            # If it's an integer array, return True for 0 and False for non-zero values
            return x == 0
        # Check if the data type of the NumPy array is a float type
        elif np.issubdtype(x.dtype, np.float64):
            # If it's a float array, return True for 0 and False for non-zero values
            return x == 0
    # Check if the input x is a list
    elif isinstance(x, list):
        check = [type(element) for element in x]

        if list in check:
            check_2 = [len(element) for element in x]
            if len(set(check_2)) != 1:
                bool_list = []
                for element in x:
                    bool_list.append(False)
            return bool_list
        else:
            # If it's a list, recursively apply the logical NOT operation to its elements
            return my_logical_not(np.array(x))
    # Check if the input x is a dictionary
    elif isinstance(x, dict):
        # Return a single False value for dictionaries
        return False
