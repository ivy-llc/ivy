"""Collection of general Ivy functions."""

# global
import gc
import inspect
import math
from functools import wraps
from numbers import Number
from typing import (
    Callable,
    Any,
    Union,
    List,
    Tuple,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Literal,
)
import einops
import numpy as np

# local
import ivy
from ivy.utils.backend import current_backend, backend_stack
from ivy.functional.ivy.gradients import _is_variable
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_array_function,
    inputs_to_ivy_arrays,
    inputs_to_native_arrays,
    to_native_arrays_and_back,
    inputs_to_native_shapes,
    outputs_to_ivy_shapes,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_view_indexing,
    handle_device_shifting,
)
from ivy.functional.ivy.device import dev

FN_CACHE = dict()
INF = float("inf")

precise_mode_stack = list()
queue_timeout_stack = list()
array_mode_stack = list()
shape_array_mode_stack = list()
nestable_mode_stack = list()
exception_trace_mode_stack = list()
trace_mode_dict = dict()
trace_mode_dict["frontend"] = "ivy/functional/frontends"
trace_mode_dict["ivy"] = "ivy/"
trace_mode_dict["full"] = ""
trace_mode_dict["none"] = ""
show_func_wrapper_trace_mode_stack = list()
min_denominator_stack = list()
min_base_stack = list()
tmp_dir_stack = list()


# Extra #
# ------#


class PreciseMode:
    """Precise Mode Context Manager."""

    # noinspection PyShadowingNames
    def __init__(self, precise_mode):
        self._precise_mode = precise_mode

    def __enter__(self):
        set_precise_mode(self._precise_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_precise_mode()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


ivy.precise_mode = precise_mode_stack[-1] if precise_mode_stack else True


@handle_exceptions
def set_precise_mode(mode: bool) -> None:
    """
    Set the mode of whether to use a promotion table that avoids any precision loss or a
    compute effecient table that avoids most wider-than-necessary promotions.

    Parameter
    ---------
    mode
        boolean whether to use high precision promtion table

    Examples
    --------
    >>> ivy.set_precise_mode(False)
    >>> ivy.precise_mode
    False

    >>> ivy.set_precise_mode(True)
    >>> ivy.precise_mode
    True
    """
    global precise_mode_stack
    ivy.utils.assertions.check_isinstance(mode, bool)
    precise_mode_stack.append(mode)
    ivy.__setattr__("precise_mode", mode, True)
    _update_promotion_table(precise=mode)


@handle_exceptions
def unset_precise_mode() -> None:
    """
    Reset the mode of whether to use a promotion table that avoids any precision loss or
    a compute effecient table that avoids most wider-than-necessary promotions.

    Examples
    --------
    >>> ivy.set_precise_mode(False)
    >>> ivy.precise_mode
    False

    >>> ivy.unset_precise_mode()
    >>> ivy.precise_mode
    True
    """
    global precise_mode_stack
    if precise_mode_stack:
        precise_mode_stack.pop(-1)
        mode = precise_mode_stack[-1] if precise_mode_stack else True
        ivy.__setattr__("precise_mode", mode, True)
        _update_promotion_table(precise=mode)


def _update_promotion_table(precise):
    """Update the current datatype promotion table."""
    if precise:
        ivy.promotion_table = {
            **ivy.array_api_promotion_table,
            **ivy.common_extra_promotion_table,
            **ivy.precise_extra_promotion_table,
        }

    else:
        ivy.promotion_table = {
            **ivy.array_api_promotion_table,
            **ivy.common_extra_promotion_table,
            **ivy.extra_promotion_table,
        }


class ArrayMode:
    """Array Mode Context Manager."""

    # noinspection PyShadowingNames
    def __init__(self, array_mode):
        self._array_mode = array_mode

    def __enter__(self):
        set_array_mode(self._array_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_array_mode()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


def _parse_ellipsis(so, ndims, ret_inds=False):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    ret = tuple(
        pre
        + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
        + list(reversed(post))
    )
    if ret_inds:
        return ret, (len(pre), ndims - len(post))
    return ret


def _parse_index(indices, shape):
    ind = list()
    for so in indices:
        pre = list()
        for s in so:
            if s == -1:
                pre.append(shape[len(pre) :][0] - 1)
                break
            pre.append(s.numpy())
        post = list()
        for s in reversed(so):
            if s == -1:
                break
            post.append(s.numpy())
        ind.append(
            tuple(
                pre
                + [
                    slice(None, None, None)
                    for _ in range(len(shape) - len(pre) - len(post))
                ]
                + list(reversed(post))
            )
        )
    return ind


def get_referrers_recursive(
    item, depth=0, max_depth=None, seen_set=None, local_set=None
):
    """
    Summary.

    Parameters
    ----------
    item

    depth
         (Default value = 0)
    max_depth
         (Default value = None)
    seen_set
         (Default value = None)
    local_set
         (Default value = None`)
    """
    seen_set = ivy.default(seen_set, set())
    local_set = ivy.default(local_set, set())
    ret_cont = ivy.Container(
        repr=str(item).replace(" ", ""),
        alphabetical_keys=False,
        keyword_color_dict={"repr": "magenta"},
    )
    referrers = [
        ref
        for ref in gc.get_referrers(item)
        if not (
            isinstance(ref, dict)
            and min([k in ref for k in ["depth", "max_depth", "seen_set", "local_set"]])
        )
    ]
    local_set.add(str(id(referrers)))
    for ref in referrers:
        ref_id = str(id(ref))
        if ref_id in local_set or hasattr(ref, "cell_contents"):
            continue
        seen = ref_id in seen_set
        seen_set.add(ref_id)
        refs_rec = lambda: get_referrers_recursive(
            ref, depth + 1, max_depth, seen_set, local_set
        )
        this_repr = "tracked" if seen else str(ref).replace(" ", "")
        if not seen and (not max_depth or depth < max_depth):
            val = ivy.Container(
                repr=this_repr,
                alphabetical_keys=False,
                keyword_color_dict={"repr": "magenta"},
            )
            refs = refs_rec()
            for k, v in refs.items():
                val[k] = v
        else:
            val = this_repr
        ret_cont[str(ref_id)] = val
    return ret_cont


@handle_exceptions
def is_native_array(
    x: Union[ivy.Array, ivy.NativeArray], /, *, exclusive: bool = False
) -> bool:
    """
    Determine whether the input x is an :class:`ivy.NativeArray` instance.

    Parameters
    ----------
    x
        The input to check
    exclusive
        Whether to check if the data type is exclusively an array, rather than a
        variable or traced array.

    Returns
    -------
    ret
        Boolean, whether or not x is an :class:`ivy.NativeArray`.

    Examples
    --------
    >>> x = ivy.array([0, 1, 2])
    >>> ivy.is_native_array(x)
    False

    >>> x = ivy.native_array([9.1, -8.3, 2.8, 3.0])
    >>> ivy.is_native_array(x, exclusive=True)
    True
    """
    try:
        return current_backend(x).is_native_array(x, exclusive=exclusive)
    except ValueError:
        return False


@handle_exceptions
def is_ivy_array(
    x: Union[ivy.Array, ivy.NativeArray], /, *, exclusive: Optional[bool] = False
) -> bool:
    """
    Determine whether the input x is a valid Ivy Array.

    Parameters
    ----------
    x
        The input to check
    exclusive
        Whether to check if the data type is exclusively an array, rather than a
        variable or traced array.

    Returns
    -------
    ret
        Boolean, whether or not x is a valid Ivy Array.

    Examples
    --------
    >>> x = ivy.array([0, 1, 2])
    >>> ivy.is_ivy_array(x)
    True

    >>> x = ivy.native_array([9.1, -8.3, 2.8, 3.0])
    >>> ivy.is_ivy_array(x, exclusive=True)
    False
    """
    return isinstance(x, ivy.Array) and ivy.is_native_array(x.data, exclusive=exclusive)


@handle_exceptions
def is_array(x: Any, /, *, exclusive: bool = False) -> bool:
    """
    Determine whether the input x is either an Ivy Array or a Native Array.

    Parameters
    ----------
    x
        The input to check
    exclusive
        Whether to check if the data type is exclusively an array, rather than a
        variable or traced array.

    Returns
    -------
    ret
        Boolean, whether or not x is an array.

    Examples
    --------
    >>> x = ivy.array([0, 1, 2])
    >>> print(ivy.is_array(x))
    True

    >>> x = ivy.native_array([9.1, -8.3, 2.8, 3.0])
    >>> print(ivy.is_array(x, exclusive=True))
    True

    >>> x = [2, 3]
    >>> print(ivy.is_array(x))
    False
    """
    return ivy.is_ivy_array(x, exclusive=exclusive) or ivy.is_native_array(
        x, exclusive=exclusive
    )


@handle_exceptions
def is_ivy_container(x: Any, /) -> bool:
    """
    Determine whether the input x is an Ivy Container.

    Parameters
    ----------
    x
        The input to check

    Returns
    -------
    ret
        Boolean, whether or not x is an ivy container.

    Examples
    --------
    >>> x = ivy.Container()
    >>> print(ivy.is_ivy_container(x))
    True

    >>> x = [2, 3]
    >>> print(ivy.is_ivy_container(x))
    False
    """
    return isinstance(x, ivy.Container)


ivy.array_mode = array_mode_stack[-1] if array_mode_stack else True


@handle_exceptions
def set_array_mode(mode: bool) -> None:
    """
    Set the mode of whether to convert inputs to ivy.NativeArray, then convert outputs
    back to ivy.Array.

    It Stops the conversion of ivy.NativeArray to ivy.Array in the
    case when it is set to False.

    Parameter
    ---------
    mode
        boolean whether to perform ivy.Array conversions

    Examples
    --------
    >>> ivy.set_array_mode(False)
    >>> ivy.array_mode
    False

    >>> ivy.set_array_mode(True)
    >>> ivy.array_mode
    True
    """
    global array_mode_stack
    ivy.utils.assertions.check_isinstance(mode, bool)
    array_mode_stack.append(mode)
    ivy.__setattr__("array_mode", mode, True)


@handle_exceptions
def unset_array_mode() -> None:
    """
    Reset the mode of converting inputs to ivy.NativeArray, then converting outputs back
    to ivy.Array to the previous state.

    Examples
    --------
    >>> ivy.set_array_mode(False)
    >>> ivy.array_mode
    False

    >>> ivy.unset_shape_array_mode()
    >>> ivy.array_mode
    True
    """
    global array_mode_stack
    if array_mode_stack:
        array_mode_stack.pop(-1)
        mode = array_mode_stack[-1] if array_mode_stack else True
        ivy.__setattr__("array_mode", mode, True)


ivy.nestable_mode = nestable_mode_stack[-1] if nestable_mode_stack else True


@handle_exceptions
def set_nestable_mode(mode: bool) -> None:
    """
    Set the mode of whether to check if function inputs are ivy.Container.

    Parameter
    ---------
    mode
        boolean whether to check if function inputs are ivy.Container

    Examples
    --------
    >>> ivy.set_nestable_mode(False)
    >>> ivy.nestable_mode
    False

    >>> ivy.set_nestable_mode(True)
    >>> ivy.nestable_mode
    True
    """
    global nestable_mode_stack
    ivy.utils.assertions.check_isinstance(mode, bool)
    nestable_mode_stack.append(mode)
    ivy.__setattr__("nestable_mode", mode, True)


@handle_exceptions
def unset_nestable_mode() -> None:
    """
    Reset the mode of whether to check if function inputs are ivy.Container to the
    previous state.

    Examples
    --------
    >>> ivy.set_nestable_mode(False)
    >>> ivy.nestable_mode
    False

    >>> ivy.unset_nestable_mode()
    >>> ivy.nestable_mode
    True
    """
    global nestable_mode_stack
    if nestable_mode_stack:
        nestable_mode_stack.pop(-1)
        mode = nestable_mode_stack[-1] if nestable_mode_stack else True
        ivy.__setattr__("nestable_mode", mode, True)


ivy.exception_trace_mode = (
    exception_trace_mode_stack[-1] if exception_trace_mode_stack else "full"
)


@handle_exceptions
def set_exception_trace_mode(mode: Literal["ivy", "full", "frontend"]) -> None:
    """
    Set the mode of whether to show frontend-truncated exception stack traces, ivy-
    truncated exception stack traces or full exception stack traces.

    Parameter
    ---------
    mode
        str exeption trace mode, one of `ivy`, `full` or `frontend`

    Examples
    --------
    >>> ivy.set_exception_trace_mode("ivy")
    >>> ivy.exception_trace_mode
    'ivy'

    >>> ivy.set_exception_trace_mode("full")
    >>> ivy.exception_trace_mode
    'full'
    """
    global exception_trace_mode_stack
    trace_modes = list(trace_mode_dict.keys())
    ivy.utils.assertions.check_elem_in_list(
        mode, trace_modes, False, "trace mode must be one of {}".format(trace_modes)
    )
    exception_trace_mode_stack.append(mode)
    ivy.__setattr__("exception_trace_mode", mode, True)


@handle_exceptions
def unset_exception_trace_mode() -> None:
    """
    Reset the trace mode to the previously set mode.

    Examples
    --------
    >>> ivy.set_exception_trace_mode("ivy")
    >>> ivy.exception_trace_mode
    'ivy'

    >>> ivy.unset_exception_trace_mode()
    >>> ivy.exception_trace_mode
    'full'
    """
    global exception_trace_mode_stack
    if exception_trace_mode_stack:
        exception_trace_mode_stack.pop(-1)
        mode = exception_trace_mode_stack[-1] if exception_trace_mode_stack else "full"
        ivy.__setattr__("exception_trace_mode", mode, True)


ivy.show_func_wrapper_trace_mode = (
    show_func_wrapper_trace_mode_stack[-1]
    if show_func_wrapper_trace_mode_stack
    else True
)


@handle_exceptions
def set_show_func_wrapper_trace_mode(mode: bool) -> None:
    """
    Set the mode of whether to show the full stack trace with function wrapping traces.

    Parameter
    ---------
    mode
        boolean whether to perform ivy.Array conversions

    Examples
    --------
    >>> ivy.set_show_func_wrapper_trace_mode(False)
    >>> ivy.show_func_wrapper_trace_mode
    False

    >>> ivy.set_show_func_wrapper_trace_mode(True)
    >>> ivy.show_func_wrapper_trace_mode
    True
    """
    global show_func_wrapper_trace_mode_stack
    ivy.utils.assertions.check_isinstance(mode, bool)
    show_func_wrapper_trace_mode_stack.append(mode)
    ivy.__setattr__("show_func_wrapper_trace_mode", mode, True)


@handle_exceptions
def unset_show_func_wrapper_trace_mode() -> None:
    """
    Reset the mode of whether to show the full stack trace with function wrapping
    traces.

    Examples
    --------
    >>> ivy.set_show_func_wrapper_trace_mode(False)
    >>> ivy.show_func_wrapper_trace_mode
    False

    >>> ivy.unset_show_func_wrapper_trace_mode()
    >>> ivy.show_func_wrapper_trace_mode
    True
    """
    global show_func_wrapper_trace_mode_stack
    if show_func_wrapper_trace_mode_stack:
        show_func_wrapper_trace_mode_stack.pop(-1)
        mode = (
            show_func_wrapper_trace_mode_stack[-1]
            if show_func_wrapper_trace_mode_stack
            else True
        )
        ivy.__setattr__("show_func_wrapper_trace_mode", mode, True)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
def array_equal(
    x0: Union[ivy.Array, ivy.NativeArray],
    x1: Union[ivy.Array, ivy.NativeArray],
    /,
) -> bool:
    """
    Determine whether two input arrays are equal across all elements.

    Parameters
    ----------
    x0
        The first input array to compare.
    x1
        The second input array to compare.

    Returns
    -------
    ret
        Boolean, whether or not the input arrays are equal across all elements.

    Examples
    --------
    >>> x = ivy.array([1,0,1])
    >>> y = ivy.array([1,0,-1])
    >>> z = ivy.array_equal(x,y)
    >>> print(z)
    False

    >>> a = ivy.array([1, 2])
    >>> b = ivy.array([1, 2])
    >>> c = ivy.array_equal(a,b)
    >>> print(c)
    True

    >>> i = ivy.array([1, 2])
    >>> j = ivy.array([1, 2, 3])
    >>> k = ivy.array_equal(i,j)
    >>> print(k)
    False
    """
    return current_backend(x0).array_equal(x0, x1)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def all_equal(
    *xs: Iterable[Any], equality_matrix: bool = False
) -> Union[bool, ivy.Array, ivy.NativeArray]:
    """
    Determine whether the inputs are all equal.

    Parameters
    ----------
    xs
        inputs to compare.
    equality_matrix
        Whether to return a matrix of equalities comparing each input with every other.
        Default is ``False``.

    Returns
    -------
    ret
        Boolean, whether or not the inputs are equal, or matrix array of booleans if
        equality_matrix=True is set.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([1, 1, 0, 0, 1, -1])
    >>> x2 = ivy.array([1, 1, 0, 0, 1, -1])
    >>> y = ivy.all_equal(x1, x2)
    >>> print(y)
    True

    >>> x1 = ivy.array([0, 0])
    >>> x2 = ivy.array([0, 0])
    >>> x3 = ivy.array([1, 0])
    >>> y = ivy.all_equal(x1, x2, x3, equality_matrix=True)
    >>> print(y)
    ivy.array([[ True,  True, False],
       [ True,  True, False],
       [False, False,  True]])

    With one :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([0, 0, -1, 1, 0]),
    ...                    b=ivy.array([0, 0, -1, 1, 0]))
    >>> x2 = ivy.array([0, 0, -1, 1, 0])
    >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    {
        a: true,
        b: true
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]),
    ...                    b=ivy.array([1, 0, 0, 1]))
    >>> x2 = ivy.Container(a=ivy.array([1, 0, 1, 1]),
    ...                    b=ivy.array([1, 0, -1, -1]))
    >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    {
        a: true,
        b: false
    }
    """
    equality_fn = ivy.array_equal if ivy.is_array(xs[0]) else lambda a, b: a == b
    if equality_matrix:
        num_arrays = len(xs)
        mat = [[None for _ in range(num_arrays)] for _ in range(num_arrays)]
        for i, xa in enumerate(xs):
            for j_, xb in enumerate(xs[i:]):
                j = j_ + i
                res = equality_fn(xa, xb)
                if ivy.is_native_array(res):
                    # noinspection PyTypeChecker
                    res = ivy.to_scalar(res)
                # noinspection PyTypeChecker
                mat[i][j] = res
                # noinspection PyTypeChecker
                mat[j][i] = res
        return ivy.array(mat)
    x0 = xs[0]
    for x in xs[1:]:
        if not equality_fn(x0, x):
            return False
    return True


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
def to_numpy(
    x: Union[ivy.Array, ivy.NativeArray], /, *, copy: bool = True
) -> np.ndarray:
    """
    Convert an array into a numpy array.

    Parameters
    ----------
    x
        input array
    copy
        whether to copy the array to a new address or not.
        Default is ``True``.

    Returns
    -------
    ret
        a numpy array copying all the element of the array ``x``.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.to_numpy(x, copy=True)
    >>> print(y)
    [-1  0  1]

    >>> x = ivy.array([[-1, 0, 1],[-1, 0, 1], [1,0,-1]])
    >>> y = ivy.to_numpy(x, copy=True)
    >>> print(y)
    [[-1  0  1]
    [-1  0  1]
    [ 1  0 -1]]

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]))
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    {
        a: array([-1, 0, 1], dtype=int32)
    }

    >>> x = ivy.Container(a=ivy.array([[-1.0, 0., 1.], [-1, 0, 1], [1, 0, -1]]),
    ...                   b=ivy.array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    {
        a: array([[-1., 0., 1.],
                  [-1., 0., 1.],
                  [1., 0., -1.]], dtype=float32),
        b: array([[-1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=int32)
    }
    """
    return current_backend(x).to_numpy(x, copy=copy)


@handle_exceptions
@handle_nestable
def isscalar(x: Any, /) -> bool:
    return np.isscalar(x)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
def to_scalar(x: Union[ivy.Array, ivy.NativeArray], /) -> Number:
    """
    Convert an array with a single element into a scalar.

    Parameters
    ----------
    x
        Input array with a single element.

    Returns
    -------
    ret
        a scalar copying the element of the array ``x``.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([3])
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    3

    With a mix of :class:`ivy.Container` and :class:`ivy.Array` input:

    >>> x = ivy.Container(a=ivy.array([-1]), b=ivy.array([3]))
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    {
        a: -1,
        b: 3
    }

    >>> x = ivy.Container(a=ivy.array([1]), b=ivy.array([0]),
    ...                   c=ivy.array([-1]))
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    {
        a: 1,
        b: 0,
        c: -1
    }
    """
    return current_backend(x).to_scalar(x)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
def to_list(x: Union[ivy.Array, ivy.NativeArray], /) -> List:
    """
    Create a (possibly nested) list from input array.

    Parameters
    ----------
    x
        Input array.

    Returns
    -------
    ret
        A list representation of the input array ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [-1, 0, 1]

    >>> x = ivy.array([[ 1.1,  2.2,  3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [[1.100000023841858,2.200000047683716,3.299999952316284],
    [-4.400000095367432,-5.5,-6.599999904632568]]

    >>> x = ivy.array([[[-1,  0,  1],
    ...                 [ 1,  0, -1]],
    ...                [[ 1, -1,  0],
    ...                 [ 1,  0, -1]]])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]

    With a mix of :class:`ivy.Container` and :class:`ivy.Array` input:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [-1, 0, 1]
    }

    >>> x = ivy.Container(a=ivy.array([[-1, 0, 1],
    ...                                [-1, 0, 1],
    ...                                [1, 0, -1]]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [[-1, 0, 1], [-1, 0, 1], [1,0,-1]]
    }

    >>> x = ivy.Container(a=ivy.array([[[-1, 0, 1],[1, 0, -1]],
    ...                                [[1, -1, 0],[1, 0, -1]]]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]
    }
    """
    return current_backend(x).to_list(x)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def clip_vector_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    max_norm: float,
    /,
    *,
    p: float = 2.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Clips (limits) the vector p-norm of an array.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    max_norm
        The maximum value of the array norm.
    p
        The p-value for computing the p-norm.
        Default is 2.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        An array with the vector norm downscaled to the max norm if needed.

    Functional Examples
    ------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.clip_vector_norm(x, 2.0)
    >>> print(y)
    ivy.array([0.   , 0.894, 1.79 ])

    >>> x = ivy.array([0.5, -0.7, 2.4])
    >>> y = ivy.clip_vector_norm(x, 3.0, p=1.0)
    >>> print(y)
    ivy.array([ 0.417, -0.583,  2.   ])

    >>> x = ivy.array([[[0., 0.], [1., 3.], [2., 6.]],
    ...                [[3., 9.], [4., 12.], [5., 15.]]])
    >>> y = ivy.zeros(((2, 3, 2)))
    >>> ivy.clip_vector_norm(x, 4.0, p=1.0, out=y)
    >>> print(y)
    ivy.array([[[0.    , 0.    ],
                [0.0667, 0.2   ],
                [0.133 , 0.4   ]],
               [[0.2   , 0.6   ],
                [0.267 , 0.8   ],
                [0.333 , 1.    ]]])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.clip_vector_norm(x, 1.0, p=3.0, out=x)
    >>> print(x)
    ivy.array([[ 0.131,  0.263,  0.394],
               [-0.526, -0.657, -0.788]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> y = ivy.clip_vector_norm(x, 2.0)
    >>> print(y)
    {
        a: ivy.array([0., 0.894, 1.79]),
        b: ivy.array([0.849, 1.13, 1.41])
    }
    """
    norm = ivy.vector_norm(x, keepdims=True, ord=p)
    ratio = ivy.stable_divide(max_norm, norm)
    if ratio < 1:
        ret = ratio * x
    else:
        ret = ivy.copy_array(x)
    if out is not None:
        ret = ivy.inplace_update(out, ret)
    return ret


@handle_exceptions
@handle_nestable
@handle_array_function
@inputs_to_ivy_arrays
def clip_matrix_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    max_norm: float,
    /,
    *,
    p: float = 2.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Clips (limits) the matrix norm of an array.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    max_norm
        The maximum value of the array norm.
    p
        The p-value for computing the p-norm.
        Default is 2.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the matrix norm downscaled to the max norm if needed.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([[0., 1., 2.]])
    >>> y = ivy.clip_matrix_norm(x, 2.0)
    >>> print(y)
    ivy.array([[0.   , 0.894, 1.79 ]])

    >>> x = ivy.array([[0.1, -1.2, 3.7], [0., 7.3, -0.5]])
    >>> y = ivy.clip_matrix_norm(x, 3.0, p=1.0)
    >>> print(y)
    ivy.array([[ 0.0353, -0.424 ,  1.31  ],
               [ 0.    ,  2.58  , -0.176 ]])

    >>> x = ivy.array([[[5., 4.], [-2., 6.]],
    ...                [[3., 7.], [0., -5.]]])
    >>> y = ivy.empty((2, 2, 2))
    >>> y = ivy.clip_matrix_norm(x, 0.5, p=2.0)
    >>> print(y)
    ivy.array([[[ 0.339,  0.271],
                [-0.135,  0.406]],
               [[ 0.168,  0.391],
                [ 0.   , -0.279]]])

    >>> x = ivy.array([[0., 1.],
    ...                [2., 3.]])
    >>> ivy.clip_matrix_norm(x, 5.0, p=1.0, out=x)
    >>> print(x)
    ivy.array([[0., 1.],
               [2., 3.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]),
    ...                   b=ivy.array([[3., 4., 5.]]))
    >>> y = ivy.clip_matrix_norm(x, 2.0)
    >>> print(y)
    {
        a: ivy.array([[0., 0.894, 1.79]]),
        b: ivy.array([[0.849, 1.13, 1.41]])
    }
    """
    norms = ivy.matrix_norm(x, ord=p, keepdims=True)
    ratios = ivy.minimum(ivy.stable_divide(max_norm, norms), 1.0)
    return ivy.multiply(ratios, x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def fourier_encode(
    x: Union[ivy.Array, ivy.NativeArray],
    max_freq: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    num_bands: int = 4,
    linear: bool = False,
    concat: bool = True,
    flatten: bool = False,
) -> Union[ivy.Array, ivy.NativeArray, Tuple]:
    """
    Pad an array with fourier encodings.

    Parameters
    ----------
    x
        Input array to encode.
    max_freq
        The maximum frequency of the encoding.
    num_bands
        The number of frequency bands for the encoding.
        Default is 4.
    linear
        Whether to space the frequency bands linearly as opposed to geometrically.
        Default is ``False``.
    concat
        Whether to concatenate the position, sin and cos values, or return seperately.
        Default is ``True``.
    flatten
        Whether to flatten the position dimension into the batch dimension.
        Default is False.

    Returns
    -------
    ret
        New array with the final dimension expanded, and the encodings stored in this
        channel.

    Examples
    --------
    >>> x = ivy.array([1,2,3])
    >>> y = 1.5
    >>> z = ivy.fourier_encode(x,y)
    >>> print(z)
    ivy.array([[ 1.0000000e+00, 1.2246468e-16, 0.0000000e+00, 0.0000000e+00,
                 0.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                 1.0000000e+00],
               [ 2.0000000e+00, -2.4492936e-16, 0.0000000e+00, 0.0000000e+00,
                 0.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                 1.0000000e+00],
               [ 3.0000000e+00, 3.6739404e-16, 0.0000000e+00, 0.0000000e+00,
                 0.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                 1.0000000e+00]])


    >>> x = ivy.array([3,10])
    >>> y = 2.5
    >>> z = ivy.fourier_encode(x, y, num_bands=3)
    >>> print(z)
    ivy.array([[ 3.0000000e+00,  3.6739404e-16,  3.6739404e-16, 3.6739404e-16,
                -1.0000000e+00, -1.0000000e+00, -1.0000000e+00],
               [ 1.0000000e+01, -1.2246468e-15, -1.2246468e-15, -1.2246468e-15,
                 1.0000000e+00,  1.0000000e+00,  1.0000000e+00]])
    """
    x_in = x
    dim = x.shape[-1]
    x = ivy.expand_dims(x, axis=-1)
    orig_x = x
    if linear:
        scales = ivy.linspace(1.0, max_freq / 2, num_bands, device=dev(x))
    else:
        if ivy.backend == "torch" and isinstance(max_freq, float):
            scales = ivy.logspace(
                0.0,
                ivy.log(ivy.array(max_freq / 2)) / math.log(10),
                num_bands,
                base=10,
                device=dev(x),
            )
        else:
            scales = ivy.logspace(
                0.0,
                ivy.log(max_freq / 2) / math.log(10),
                num_bands,
                base=10,
                device=dev(x),
            )
    scales = ivy.astype(scales, ivy.dtype(x))
    scales = scales[(*((None,) * (len(x.shape) - len(scales.shape))), Ellipsis)]
    x = x * scales * math.pi
    sin_x = ivy.sin(x)
    cos_x = ivy.cos(x)
    if flatten:
        orig_x = x_in
        sin_x = ivy.reshape(sin_x, [-1, num_bands * dim])
        cos_x = ivy.reshape(cos_x, [-1, num_bands * dim])
    if concat:
        return ivy.concat([orig_x, sin_x, cos_x], axis=-1)
    return sin_x, cos_x


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def value_is_nan(
    x: Union[ivy.Array, ivy.NativeArray, Number],
    /,
    *,
    include_infs: bool = True,
) -> bool:
    """
    Determine whether the single valued array or scalar is of nan type.

    Parameters
    ----------
    x
        The input to check Input array.
    include_infs
        Whether to include infs and -infs in the check.
        Default is ``True``.

    Returns
    -------
    ret
        Boolean as to whether the input value is a nan or not.

    Examples
    --------
    >>> x = ivy.array([451])
    >>> y = ivy.value_is_nan(x)
    >>> print(y)
    False

    >>> x = ivy.array([float('inf')])
    >>> y = ivy.value_is_nan(x)
    >>> print(y)
    True

    >>> x = ivy.array([float('inf')])
    >>> y = ivy.value_is_nan(x, include_infs=False)
    >>> print(y)
    False

    >>> x = ivy.array([float('nan')])
    >>> y = ivy.value_is_nan(x, include_infs=False)
    >>> print(y)
    True

    >>> x = ivy.array([0])
    >>> y = ivy.value_is_nan(x)
    >>> print(y)
    False
    """
    x_scalar = ivy.to_scalar(x) if ivy.is_array(x) else x
    if not x_scalar == x:
        return True
    if include_infs and (x_scalar == INF or x_scalar == -INF):
        return True
    return False


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def has_nans(
    x: Union[ivy.Array, ivy.NativeArray], /, *, include_infs: bool = True
) -> bool:
    """
    Determine whether the array contains any nans, as well as infs or -infs if
    specified.

    Parameters
    ----------
    x
        Input array.
    include_infs
        Whether to include ``+infinity`` and ``-infinity`` in the check.
        Default is ``True``.

    Returns
    -------
    ret
        Boolean as to whether the array contains nans.


    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.has_nans(x)
    >>> print(y)
    False

    >>> x = ivy.array([float('nan'), 2, 3])
    >>> y = ivy.has_nans(x)
    >>> print(y)
    True

    >>> x = ivy.array([float('inf'), 2, 3])
    >>> y = ivy.has_nans(x)
    >>> print(y)
    True

    >>> x = ivy.array([float('inf'), 2, 3])
    >>> y = ivy.has_nans(x, False)
    >>> print(y)
    False

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.has_nans(x)
    >>> print(y)
    {
        a: false,
        b: false
    }
    """
    return ivy.value_is_nan(ivy.sum(x), include_infs=include_infs)


@handle_exceptions
def exists(x: Any) -> bool:
    """
    Check as to whether the input is None or not.

    Parameters
    ----------
    x
        Input to check.

    Returns
    -------
    ret
        True if x is not None, else False.

    Examples
    --------
    With :code:`Any` input:

    >>> x = None
    >>> y = ivy.exists(x)
    >>> print(y)
    False

    >>> x = ""
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = []
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = 1
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = "abc"
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = [1, 0, -1, 1]
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = ivy.array([1, 2, 3, 1.2])
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    With a mix of :class:`ivy.Container` and :code:`Any` input:

    >>> x = ivy.Container(a=None, b=None)
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = ivy.Container(a=None, b="")
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = ivy.Container(a=123, b="")
    >>> y = ivy.exists(x)
    >>> print(y)
    True
    """
    return x is not None


@handle_exceptions
def default(
    x: Any,
    /,
    default_val: Any,
    *,
    catch_exceptions: bool = False,
    rev: bool = False,
    with_callable: bool = False,
) -> Any:
    """
    Return x provided it exists (is not None), else returns default value.

    Parameters
    ----------
    x
        Input which may or may not exist (be None).
    default_val
        The default value.
    catch_exceptions
        Whether to catch exceptions from callable x.
        Default is ``False``.
    rev
        Whether to reverse the input x and default_val.
        Default is ``False``.
    with_callable
        Whether either of the arguments might be callable functions.
        Default is ``False``.

    Returns
    -------
    ret
        x if x exists (is not None), else default.

    Functional Examples
    ------------------
    With :code:`Any` input:

    >>> x = None
    >>> y = ivy.default(x, "default_string")
    >>> print(y)
    default_string

    >>> x = ""
    >>> y = ivy.default(x, "default_string")
    >>> print(y)


    >>> x = ivy.array([4, 5, 6])
    >>> y = ivy.default(x, ivy.array([1, 2, 3]), rev=True)
    >>> print(y)
    ivy.array([1, 2, 3])

    >>> x = lambda: ivy.array([1, 2, 3])
    >>> y = ivy.default(x, ivy.array([4, 5, 6]), with_callable=True)
    >>> print(y)
    ivy.array([1, 2, 3])

    >>> x = lambda: None
    >>> y = ivy.default(x, lambda: ivy.array([1, 2, 3]), with_callable=True)
    >>> print(y)
    ivy.array([1, 2, 3])

    >>> x = lambda: None
    >>> y = ivy.default(x, lambda: ivy.array([1, 2, 3]), catch_exceptions=True)
    >>> print(y)
    ivy.array([1, 2, 3])

    >>> x = lambda a, b: a + b
    >>> y = ivy.default(x, lambda: ivy.array([1, 2, 3]), with_callable=True,
    ...                 catch_exceptions=True)
    >>> print(y)
    ivy.array([1, 2, 3])

    >>> x = lambda a, b: a + b
    >>> y = ivy.default(x, lambda: ivy.array([1, 2, 3]), with_callable=True,
    ...                 catch_exceptions=True, rev=True)
    >>> print(y)
    ivy.array([1, 2, 3])
    """
    with_callable = catch_exceptions or with_callable
    if rev:
        tmp = x
        x = default_val
        default_val = tmp
    if with_callable:
        x_callable = callable(x)
        default_callable = callable(default_val)
    else:
        x_callable = False
        default_callable = False
    if catch_exceptions:
        # noinspection PyBroadException
        try:
            x = x() if x_callable else x
        except Exception:
            return default_val() if default_callable else default_val
    else:
        x = x() if x_callable else x
    return x if exists(x) else default_val() if default_callable else default_val


@handle_exceptions
def to_ivy_shape(shape: Union[ivy.Shape, ivy.NativeShape]) -> ivy.Shape:
    """
    Return the input shape in ivy.Shape form.

    Parameters
    ----------
    shape
        The input to be converted

    Returns
    -------
     ret
        the input in ivy.Shape form
    """
    if isinstance(shape, ivy.Shape):
        return shape
    return ivy.Shape(shape)


@handle_exceptions
def to_native_shape(
    shape: Union[ivy.Array, ivy.Shape, ivy.NativeShape, tuple, int, list]
) -> ivy.NativeShape:
    """
    Return the input shape in its native backend framework form.

    Parameters
    ----------
    shape
        The input to be converted

    Returns
    -------
     ret
        the input in its native framework form
    """
    if len(backend_stack) != 0 and isinstance(shape, ivy.NativeShape):
        return shape
    ivy.utils.assertions.check_isinstance(
        shape, (int, list, tuple, ivy.Array, ivy.NativeArray, ivy.Shape)
    )
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, list):
        shape = tuple(shape)
    elif is_array(shape):
        shape = ivy.to_numpy(shape).tolist()
    elif isinstance(shape, ivy.Shape):
        shape = shape.shape
    ivy.utils.assertions.check_all(
        [isinstance(v, int) for v in shape if not is_array(v)],
        "shape must take integers only",
        as_array=False,
    )
    ivy.utils.assertions.check_true(
        not is_array(shape) or ivy.is_int_dtype(shape), "shape must take integers only"
    )
    return ivy.NativeShape(shape) if len(backend_stack) != 0 else ivy.Shape(shape)


@handle_exceptions
@handle_nestable
def try_else_none(fn: Callable, *args: Any, **kwargs: Any) -> Union[Callable, None]:
    """
    Try and return the function, otherwise return None if an exception was raised during
    function execution.

    Parameters
    ----------
    fn
        Function to try and call and return.
    args
        list of arguments.
    kwargs
        dictionay of keyword arguments

    Returns
    -------
        Either the function itself or None if an exception was raised
        during function execution.

    Examples
    --------
    with a function that is executed without any exception:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = ivy.try_else_none(ivy.add,x, y)
    >>> print(z.__name__)
    add

    with a function that is executed with an exception:

    >>> x = ivy.array([1, 2, 3])
    >>> y = 'hemant'
    >>> z = ivy.try_else_none(ivy.add,x, y)
    >>> print(z)
    None
    """
    try:
        _ = fn(*args, **kwargs)
        return fn
    except Exception:
        return None


@handle_exceptions
def arg_names(receiver):
    """
    Get the expected keyword arguments for a function or class constructor.

    Parameters
    ----------
    receiver
        Function or class constructor

    Returns
    -------
    ret
        List containing the keyword arguments' names for a function or class constructor

    Examples
    --------
    >>> x = ivy.arg_names(ivy.tan)
    >>> print(x)
    ['x', 'out']

    >>> x = ivy.arg_names(ivy.optimizers.Adam)
    >>> print(x)
    ['lr', 'beta1', 'beta2', 'epsilon', 'inplace',
    'stop_gradients', 'compile_on_next_step', 'device']
    """
    return list(inspect.signature(receiver).parameters.keys())


@handle_exceptions
def match_kwargs(
    kwargs: Dict, *receivers: Iterable[Callable], allow_duplicates: bool = False
) -> Union[List[Dict], Dict]:
    """
    Match keyword arguments to either class or function receivers.

    Parameters
    ----------
    kwargs
        Keyword arguments to match.
    receivers
        Functions and/or classes to match the keyword arguments to.
    allow_duplicates
        Whether to allow one keyword argument to be used for multiple receivers.
        Default is ``False``.

    Returns
    -------
    ret
        Sequence of keyword arguments split as best as possible.

    Examples
    --------
    >>> o = ivy.zeros(3)
    >>> kwargs = {'out': o, 'bias': ivy.arange(3)}
    >>> x = ivy.match_kwargs(kwargs, ivy.add, ivy.linear)
    >>> print(x)
    [{'out': ivy.array([0., 0., 0.])}, {'bias': ivy.array([0, 1, 2])}]

    >>> o = ivy.zeros(3)
    >>> kwargs = {'out': o, 'bias': ivy.arange(3)}
    >>> x = ivy.match_kwargs(kwargs, ivy.linear, ivy.add)
    >>> print(x)
    [{'out': ivy.array([0., 0., 0.]), 'bias': ivy.array([0, 1, 2])}, {}]
    """
    split_kwargs = list()
    for receiver in receivers:
        expected_kwargs = arg_names(receiver)
        found_kwargs = {k: v for k, v in kwargs.items() if k in expected_kwargs}
        if not allow_duplicates:
            for k in found_kwargs.keys():
                del kwargs[k]
        split_kwargs.append(found_kwargs)
    if len(split_kwargs) == 1:
        return split_kwargs[0]
    return split_kwargs


@handle_exceptions
def cache_fn(func: Callable) -> Callable:
    """
    Cache function outputs.

    A decorator to wrap a function, such that computed outputs are cached to avoid
    recalculating them later.

    Parameters
    ----------
    func
        The function to wrap, whose output should be cached for later.

    Returns
    -------
    ret
        The newly cache wrapped function.

    Examples
    --------
    With positional arguments only:

    >>> def my_sum(val1:float, val2:float)->float: return val1 + val2
    >>> cached_sum = ivy.cache_fn(my_sum)
    >>> print(cached_sum(3, 5)) # Compute the output
    8

    >>> print(cached_sum(10, 34)) # Compute the output
    44

    >>> print(cached_sum(3, 5)) # Returns the cached value
    8

    >>> print(cached_sum(5, 3)) # Compute the output
    8


    With keyword arguments:

    >>> def line_eq(x:float, /, *, slp:float=2, itc:float=0)->float: return x*slp+itc
    >>> cached_line_eq = ivy.cache_fn(line_eq)
    >>> print(cached_line_eq(3, itc=5, slp=2))
    11

    >>> print(cached_line_eq(3, slp=2, itc=5)) # Returns the cached value
    11


    Note: providing keyword arguments by position, or using the default
    keyword argument values will prevent the cache from being used.

    >>> print(cached_line_eq(5, slp=2)) # Output is re-computed
    10

    >>> print(cached_line_eq(5)) # Output is re-computed
    10
    """
    global FN_CACHE
    if func not in FN_CACHE:
        FN_CACHE[func] = dict()

    @wraps(func)
    def cached_fn(*args, **kwargs):
        key = "".join(
            [str(i) + ", " for i in args]
            + [" kw, "]
            + [str(i) + ", " for i in sorted(kwargs.items())]
        )
        cache = FN_CACHE[func]
        if key in cache:
            return cache[key]
        ret = func(*args, **kwargs)
        cache[key] = ret
        return ret

    return cached_fn


@handle_exceptions
def current_backend_str() -> Union[str, None]:
    """
    Return framework string.

    Returns
    -------
    ret
        The framework string.
    """
    fw = current_backend()
    if not backend_stack:
        return ""
    return fw.current_backend_str()


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def einops_rearrange(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    /,
    *,
    out: Optional[ivy.Array] = None,
    **axes_lengths: Dict[str, int],
) -> ivy.Array:
    """
    Perform einops rearrange operation on input array x.

    Parameters
    ----------
    x
        Input array to be re-arranged.
    pattern
        Rearrangement pattern.
    axes_lengths
        Any additional specifications for dimensions.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array with einops.rearrange having been applied.

    Examples
    --------
    With :class:`ivy.Array` instance method:

    >>> x = ivy.array([[1, 2, 3],
    ...               [-4, -5, -6]])
    >>> y = x.einops_rearrange("height width -> width height")
    >>> print(y)
    ivy.array([[ 1, -4],
        [ 2, -5],
        [ 3, -6]])

    >>> x = ivy.array([[[ 1,  2,  3],
    ...                  [ 4,  5,  6]],
    ...               [[ 7,  8,  9],
    ...                  [10, 11, 12]]])
    >>> y = x.einops_rearrange("c h w -> c (h w)")
    >>> print(y)
    ivy.array([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12]])

    >>> x = ivy.array([[1, 2, 3, 4, 5, 6],
    ...            [7, 8, 9, 10, 11, 12]])
    >>> y = ivy.zeros((4,3))
    >>> x.einops_rearrange("c (h w) -> (c h) w", out=y, h=2, w=3)
    ivy.array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])

    With :class:`ivy.Container` input:

    x = ivy.Container(a=ivy.array([[-4.47, 0.93, -3.34],
    ...                            [3.66, 24.29, 3.64]]),
    ...               b=ivy.array([[4.96, 1.52, -10.67],
    ...                            [4.36, 13.96, 0.3]]))
    y = ivy.einops_rearrange(x, 'a b -> b a')
    print(y)
    {
        a: ivy.array([[-4.46999979, 3.66000009],
                    [0.93000001, 24.29000092],
                    [-3.33999991, 3.6400001]]),
        b: ivy.array([[4.96000004, 4.36000013],
                    [1.51999998, 13.96000004],
                    [-10.67000008, 0.30000001]])
    }

    With varying pattern:

    Suppose we have a set of 32 images in "h w c" format (height-width-channel)
    >>> images = ivy.asarray([ivy.random_normal(shape=(30, 40, 3)) for _ in range(32)])

    Concatenate images along height (vertical axis), 960 = 32 * 30
    >>> x = ivy.einops_rearrange(images, 'b h w c -> (b h) w c')
    >>> print(x.shape)
    (960, 40, 3)

    Concatenate images along horizontal axis, 1280 = 32 * 40
    >>> x = ivy.einops_rearrange(images, 'b h w c -> h (b w) c')
    >>> print(x.shape)
    (30, 1280, 3)

    Reorder axes to "b c h w" format for deep learning
    >>> x = ivy.einops_rearrange(images, 'b h w c -> b c h w')
    >>> print(x.shape)
    (32, 3, 30, 40)

    Flatten each image into a vector, 3600 = 30 * 40 * 3
    >>> x = ivy.einops_rearrange(images, 'b h w c -> b (c h w)')
    >>> print(x.shape)
    (32, 3600)

    Split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right),
    128 = 32 * 2 * 2
    >>> x = ivy.einops_rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c',
    ... h1=2, w1=2)
    >>> print(x.shape)
    (128, 15, 20, 3)

    Space-to-depth operation
    >>> x = ivy.einops_rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2,
    ... w1=2)
    >>> print(x.shape)
    (32, 15, 20, 12)
    """
    ret = einops.rearrange(x._data, pattern, **axes_lengths)
    ret = ivy.array(ret, dtype=x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
def einops_reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    reduction: Union[str, Callable],
    /,
    *,
    out: Optional[ivy.Array] = None,
    **axes_lengths: Dict[str, int],
) -> ivy.Array:
    """
    Perform einops reduce operation on input array x.

    Parameters
    ----------
    x
        Input array to be reduced.
    pattern
        Reduction pattern.
    reduction
        One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or callable.
    axes_lengths
        Any additional specifications for dimensions.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array with einops.reduce having been applied.

    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[-4.47, 0.93, -3.34],
    ...                [3.66, 24.29, 3.64]])
    >>> reduced = ivy.einops_reduce(x, 'a b -> b', 'mean')
    >>> print(reduced)
    ivy.array([-0.40499985, 12.61000061, 0.1500001 ])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[-4.47, 0.93, -3.34],
    ...                                [3.66, 24.29, 3.64]]),
    ...                   b=ivy.array([[4.96, 1.52, -10.67],
    ...                                [4.36, 13.96, 0.3]]))
    >>> reduced = ivy.einops_reduce(x, 'a b -> a', 'mean')
    >>> print(reduced)
    {
        a: ivy.array([-2.29333329, 10.53000069]),
        b: ivy.array([-1.39666676, 6.20666695])
    }
    """
    ret = einops.reduce(x, pattern, reduction, **axes_lengths)
    ret = ivy.array(ret, dtype=x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# IMPORTANT: assign attribute directly to function instead of wrapper here
einops_reduce.unsupported_dtypes = {
    "torch": ("float16",),
    "tensorflow": ("complex",),
    "paddle": ("complex", "uint8", "int8", "int16", "float16"),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def einops_repeat(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    /,
    *,
    out: Optional[ivy.Array] = None,
    **axes_lengths: Dict[str, int],
) -> ivy.Array:
    """
    Perform einops repeat operation on input array x.

    Parameters
    ----------
    x
        Input array to be repeated.
    pattern
        Rearrangement pattern.
    axes_lengths
        Any additional specifications for dimensions.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array with einops.repeat having been applied.

    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3, 4])
    >>> repeated = ivy.einops_repeat(x, 'a -> b a', b=2)
    >>> print(repeated)
    ivy.array([[1, 2, 3, 4],
               [1, 2, 3, 4]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[4,5],
    ...                                [1, 3]]),
    ...                    b=ivy.array([[9, 10],
    ...                                 [4, 2]]))
    >>> repeated = ivy.einops_repeat(x, 'h w -> h (c w)', c=2)
    >>> print(repeated)
    {
        a: ivy.array([[4, 5, 4, 5],
                      [1, 3, 1, 3]]),
        b: ivy.array([[9, 10, 9, 10],
                      [4, 2, 4, 2]])
    }
    """
    ret = einops.repeat(x._data, pattern, **axes_lengths)
    ret = ivy.array(ret, dtype=x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


ivy.min_denominator = min_denominator_stack[-1] if min_denominator_stack else 1e-12


@handle_exceptions
@handle_array_function
def set_min_denominator(val: float) -> None:
    """
    Set the global minimum denominator used by ivy for numerically stable division.

    Parameters
    ----------
    val
        The value to set the global minimum denominator to.

    Examples
    --------
    >>> x = ivy.min_denominator
    >>> print(x)
    1e-12

    >>> ivy.set_min_denominator(1e-13)
    >>> y = ivy.min_denominator
    >>> print(y)
    1e-13
    """
    global min_denominator_stack
    ivy.utils.assertions.check_isinstance(val, (int, float))
    min_denominator_stack.append(val)
    ivy.__setattr__("min_denominator", val, True)


@handle_exceptions
def unset_min_denominator() -> None:
    """
    Reset the global minimum denominator used by ivy for numerically stable division to
    the previous value.

    Examples
    --------
    >>> ivy.set_min_denominator(1e-10)
    >>> y = ivy.min_denominator
    >>> print(y)
    1e-10

    >>> ivy.unset_min_denominator()
    >>> ivy.min_denominator
    1e-12
    """
    global min_denominator_stack
    if min_denominator_stack:
        min_denominator_stack.pop(-1)
        val = min_denominator_stack[-1] if min_denominator_stack else 1e-12
        ivy.__setattr__("min_denominator", val, True)


ivy.min_base = min_base_stack[-1] if min_base_stack else 1e-05


@handle_exceptions
@handle_array_function
def set_min_base(val: float) -> None:
    """
    Set the global minimum base used by ivy for numerically stable power raising.

    Parameters
    ----------
    val
        The new value to set the minimum base to.

    Examples
    --------
    >>> x = ivy.min_base
    >>> print(x)
    1e-05

    >>> ivy.set_min_base(1e-04)
    >>> y = ivy.min_base
    >>> print(y)
    1e-04
    """
    global min_base_stack
    ivy.utils.assertions.check_isinstance(val, (int, float))
    min_base_stack.append(val)
    ivy.__setattr__("min_base", val, True)


@handle_exceptions
def unset_min_base() -> None:
    """
    Reset the global minimum base used by ivy for numerically stable power raising to
    the previous value.

    Examples
    --------
    >>> ivy.set_min_base(1e-07)
    >>> y = ivy.min_base
    >>> print(y)
    1e-07

    >>> ivy.unset_min_base()
    >>> ivy.min_base
    1e-05
    """
    global min_base_stack
    if min_base_stack:
        min_base_stack.pop(-1)
        val = min_base_stack[-1] if min_base_stack else 1e-05
        ivy.__setattr__("min_base", val, True)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def stable_divide(
    numerator: Union[Number, ivy.Array, ivy.NativeArray],
    denominator: Union[Number, ivy.Array, ivy.NativeArray],
    /,
    *,
    min_denominator: Union[Number, ivy.Array, ivy.NativeArray] = None,
) -> Union[Number, ivy.Array]:
    """
    Divide the numerator by the denominator, with min denominator added to the
    denominator for numerical stability.

    Parameters
    ----------
    numerator
        The numerator of the division.
    denominator
        The denominator of the division.
    min_denominator
        The minimum denominator to use, use global ivy._MIN_DENOMINATOR (1e-12)
        by default.

    Returns
    -------
    ret
        The new item following the numerically stable division.

    Examples
    --------
    With :code:`int` input:

    >>> x = ivy.stable_divide(1, 2)
    >>> print(x)
    0.49999999999975

    >>> x = ivy.stable_divide(1, 4, min_denominator=1)
    >>> print(x)
    0.2

    With float input:

    >>> x = ivy.stable_divide(5.0, 3.33)
    >>> print(x)
    1.5015015015010504

    With :code:`complex` input:

    >>> x = ivy.stable_divide(1+1j, 1-1j)
    >>> print(x)
    (5.000444502911705e-13+0.9999999999995j)

    With :class:`ivy.Array` input:

    >>> x = ivy.asarray([[10., 20., 30.],
    ...                  [40., 50., 60.]])
    >>> y = ivy.stable_divide(x, 10.)
    >>> print(y)
    ivy.array([[1., 2., 3.],
              [4., 5., 6.]])


    >>> x = ivy.asarray([1,2,3])
    >>> y = np.array((1., 3., 5.))
    >>> z = ivy.stable_divide(x, y)
    >>> print(z)
    ivy.array([1.   , 0.667, 0.6  ])

    >>> x = ivy.asarray([1., 2., 4.])
    >>> y = ivy.asarray([1., 0.5, 0.25])
    >>> z = ivy.asarray([0.01, 0.02, 0.03])
    >>> w = ivy.stable_divide(x, y, min_denominator=z)
    >>> print(w)
    ivy.array([ 0.99,  3.85, 14.3 ])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.asarray([10., 15.]), b=ivy.asarray([20., 25.]))
    >>> y = ivy.stable_divide(x, 0.5)
    >>> print(y)
    {
        a: ivy.array([20., 30.]),
        b: ivy.array([40., 50.])
    }


    >>> x = ivy.Container(a=ivy.asarray([1., 2.]), b=ivy.asarray([3., 4.]))
    >>> y = ivy.Container(a=ivy.asarray([0.5, 2.5]), b=ivy.asarray([3.5, 0.4]))
    >>> z = ivy.stable_divide(x, y)
    >>> print(z)
    {
        a: ivy.array([2., 0.8]),
        b: ivy.array([0.857, 10.])
    }
    """
    return numerator / (denominator + default(min_denominator, ivy.min_denominator))


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def stable_pow(
    base: Union[Number, ivy.Array, ivy.NativeArray],
    exponent: Union[Number, ivy.Array, ivy.NativeArray],
    /,
    *,
    min_base: float = None,
) -> Any:
    """
    Raise the base by the power, with ivy.min_base added to the base when exponent > 1
    for numerical stability.

    Parameters
    ----------
    base
        The base number.
    exponent
        The exponent number.
    min_base
        The minimum base to use, use global ivy.min_base by default.

    Returns
    -------
    ret
        The new item following the numerically stable power.
    """
    return_dtype = ivy.promote_types(
        ivy.default_dtype(item=base),
        ivy.default_dtype(item=default(min_base, ivy.min_base)),
    )
    return_dtype = ivy.promote_types(return_dtype, ivy.default_dtype(item=exponent))
    ret = (base + default(min_base, ivy.min_base)) ** ivy.array(exponent)
    return ret.astype(return_dtype)


stable_pow.unsupported_dtypes = ("bfloat16",)


@handle_exceptions
def get_all_arrays_in_memory() -> List[Union[ivy.Array, ivy.NativeArray]]:
    """
    Get all arrays which are currently alive.

    Returns
    -------
    ret
        All arrays which are alive.

    Examples
    --------
    >>> ivy.get_all_arrays_in_memory()
    []
    >>> x = ivy.get_all_arrays_in_memory()
    >>> x
    []
    >>> y = ivy.array([0, 1, 2])
    >>> x
    [ivy.array([0, 1, 2])]
    """
    all_arrays = list()
    for obj in gc.get_objects():
        try:
            if ivy.current_backend_str() in ["", "numpy"]:
                if ivy.is_ivy_array(obj):
                    all_arrays.append(obj)
            else:
                if ivy.is_native_array(obj):
                    all_arrays.append(obj)

        except Exception:
            pass
    return all_arrays


@handle_exceptions
def num_arrays_in_memory() -> int:
    """
    Return the number of arrays which are currently alive.

    Returns
    -------
    ret
        Number of all arrays which are alive.

    Examples
    --------
    >>> ivy.num_arrays_in_memory()
    0
    >>> x = ivy.num_arrays_in_memory()
    >>> x
    0
    >>> y = ivy.array([0, 1, 2])
    >>> x
    1
    """
    return len(get_all_arrays_in_memory())


@handle_exceptions
def print_all_arrays_in_memory():
    """
    Print all native Ivy arrays in memory to the console.

    Gets all the native Ivy arrays which are currently alive(in the
    garbage collector) from get_all_arrays_in_memory() function and
    prints them to the console.
    """
    for arr in get_all_arrays_in_memory():
        print(type(arr), arr.shape)


ivy.queue_timeout = queue_timeout_stack[-1] if queue_timeout_stack else 15.0


@handle_exceptions
@handle_array_function
def set_queue_timeout(timeout: float):
    """
    Set a timeout value (in seconds) for the global queue.

    Set the global queue timeout value (in seconds) Default value without this function
    being called is 15 seconds.

    Parameters
    ----------
    timeout
        The timeout when waiting for containers to arrive from the queues.
        To be set in seconds.

    Examples
    --------
    >>> x = ivy.set_queue_timeout(10)
    >>> x = ivy.queue_timeout
    >>> print(x)
    10.0

    >>> ivy.set_queue_timeout(30)
    >>> y = ivy.queue_timeout
    >>> print(y)
    30
    """
    global queue_timeout_stack
    ivy.utils.assertions.check_isinstance(timeout, (int, float))
    queue_timeout_stack.append(timeout)
    ivy.__setattr__("queue_timeout", timeout, True)


@handle_exceptions
def unset_queue_timeout() -> None:
    """
    Reset the global queue timeout value (in seconds) to the previous state.

    Examples
    --------
    >>> ivy.set_queue_timeout(10.0)
    >>> y = ivy.queue_timeout
    >>> print(y)
    10.0

    >>> ivy.unset_queue_timeout()
    >>> ivy.queue_timeout
    15.0
    """
    global queue_timeout_stack
    if queue_timeout_stack:
        queue_timeout_stack.pop(-1)
        timeout = queue_timeout_stack[-1] if queue_timeout_stack else 15.0
        ivy.__setattr__("queue_timeout", timeout, True)


ivy.tmp_dir = tmp_dir_stack[-1] if tmp_dir_stack else "/tmp"


@handle_exceptions
def set_tmp_dir(tmp_dr: str) -> None:
    """
    Set the directory for saving temporary files.

    Parameters
    ----------
    tmp_dr
        The new directory for saving temporary files

    Examples
    --------
    >>> x = ivy.tmp_dir
    >>> print(x)
    /tmp

    >>> ivy.set_tmp_dir("/my_tmp")
    >>> y = ivy.tmp_dir
    >>> print(y)
    /my_tmp
    """
    global tmp_dir_stack
    ivy.utils.assertions.check_isinstance(tmp_dr, str)
    tmp_dir_stack.append(tmp_dr)
    ivy.__setattr__("tmp_dir", tmp_dr, True)


@handle_exceptions
def unset_tmp_dir() -> None:
    """
    Reset the directory for saving temporary files to the previous value.

    Examples
    --------
    >>> ivy.set_tmp_dir("/my_dir")
    >>> y = ivy.tmp_dir
    >>> print(y)
    /my_dir

    >>> ivy.unset_tmp_dir()
    >>> ivy.tmp_dir
    /tmp
    """
    global tmp_dir_stack
    if tmp_dir_stack:
        tmp_dir_stack.pop(-1)
        tmp_dr = tmp_dir_stack[-1] if tmp_dir_stack else "/tmp"
        ivy.__setattr__("tmp_dir", tmp_dr, True)


@handle_exceptions
def container_types():
    """
    Summary.

    Returns
    -------
    ret
        a key-value structure, and exposes public methods .keys(), .values() and
        items().
    """
    # noinspection PyBroadException
    try:
        return current_backend().container_types()
    except ValueError:
        return []


@handle_exceptions
def inplace_arrays_supported() -> bool:
    """
    Determine whether inplace arrays are supported for the current backend framework.

    Returns
    -------
    ret
        Boolean, whether or not inplace arrays are supported.
    """
    return current_backend().inplace_arrays_supported()


@handle_exceptions
def inplace_variables_supported() -> bool:
    """
    Determine whether inplace variables are supported for the current backend framework.

    Returns
    -------
    ret
        Boolean, whether or not inplace variables are supported.
    """
    return current_backend().inplace_variables_supported()


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_array_function
def supports_inplace_updates(x: Union[ivy.Array, ivy.NativeArray], /) -> bool:
    """
    Return if in-place operations are supported for x's data type.

    Determine whether in-place operations are supported for x's data type, by the
    current backend framework setting.

    Parameters
    ----------
    x
        Input variable for whose data type we check whether the current backend
        framework supports in-place operations.

    Returns
    -------
    ret
        Value depends on whether in-place operations are supported for
        data type of x.

    Raises
    ------
    IvyException
        If x isn't a class instance of ivy.Array or ivy.NativeArray, an exception will
        be raised.

    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`ivy.Array` input and default backend set as `numpy`:

    >>> x = ivy.array([0, 1, 2])
    >>> y = ivy.supports_inplace_updates(x)
    >>> print(y)
    True

    With :class:`ivy.Container` input and backend set as `torch`:

    >>> x = ivy.Container(a=ivy.array([5., 6.]), b=ivy.array([7., 8.]))
    >>> y = ivy.supports_inplace_updates(x)
    >>> print(y)
    {
        a: True,
        b: True
    }

    With `ivy.Array` input and backend set as "tensorflow":

    >>> x = ivy.array([1., 4.2, 2.2])
    >>> ret = x.supports_inplace_updates()
    >>> print(ret)
    False
    """
    if _is_variable(x):
        return ivy.inplace_variables_supported()
    elif ivy.is_native_array(x):
        return ivy.inplace_arrays_supported()
    raise ivy.utils.exceptions.IvyException(
        "Input x must be either a variable or an array."
    )


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_array_function
def assert_supports_inplace(x: Union[ivy.Array, ivy.NativeArray], /) -> bool:
    """
    Assert that inplace operations are supported for x.

    Parameters
    ----------
    x
        Input variable or array to check for inplace support for.

    Returns
    -------
    ret
        True if supports, raises IvyBackendException otherwise
    
    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`ivy.Array` input and default backend set as `numpy`:

    >>> x = ivy.array([1, 2, 3])
    >>> print(x.assert_supports_inplace())
    True

    With :class:`ivy.Array` input and default backend set as `jax`:

    >>> x = ivy.array([1, 2, 3])
    >>> print(x.assert_supports_inplace())
    IvyBackendException: jax: assert_supports_inplace: Inplace operations \
    are not supported <class 'jaxlib.xla_extension.DeviceArray'> types with jax backend

    With :class:`ivy.Container` input and default backend set as `numpy`:

    >>> x = ivy.Container(a=ivy.array([5, 6]), b=ivy.array([7, 8]))
    >>> print(x.assert_supports_inplace())
    {
        a: True,
        b: True
    }

    With :class:`ivy.Container` input and default backend set as `jax`:

    >>> x = ivy.Container(a=ivy.array([5, 6]), b=ivy.array([7, 8]))
    >>> print(x.assert_supports_inplace())
    IvyBackendException: jax: assert_supports_inplace: Inplace operations \
    are not supported <class 'jaxlib.xla_extension.DeviceArray'> types with jax backend
    """
    ivy.utils.assertions.check_true(
        ivy.supports_inplace_updates(x),
        "Inplace operations are not supported {} types with {} backend".format(
            type(x), ivy.current_backend_str()
        ),
    )
    return True


@handle_nestable
@handle_view_indexing
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def get_item(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    query: Union[ivy.Array, ivy.NativeArray, Tuple],
    *,
    copy: Optional[bool] = None,
) -> ivy.Array:
    """
    Gather slices from x according to query array, identical to x[query].

    Parameters
    ----------
    x
        array, the array from which to gather values.
    query
        array, index array, integer indices or boolean mask.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        New array with the values gathered at the specified indices.

    Functional Examples
    -------------------

    >>> x = ivy.array([0, -1, 20])
    >>> query = ivy.array([0, 1])
    >>> print(ivy.get_item(x, query))
    ivy.array([ 0, -1])

    >>> x = ivy.array([[4, 5], [20, 128], [-2, -10]])
    >>> query = ivy.array([[True, False], [False, False], [True, True]])
    >>> print(ivy.get_item(x, query))
    ivy.array([  4,  -2, -10])
    """
    if ivy.is_array(query) and ivy.is_bool_dtype(query):
        if not len(query.shape):
            if not query:
                return ivy.array([], shape=(0,), dtype=x.dtype)
            return ivy.expand_dims(x, axis=0)
        query = ivy.nonzero(query, as_tuple=False)
        ret = ivy.gather_nd(x, query)
    else:
        query, target_shape = _parse_query(query, x.shape)
        ret = ivy.gather_nd(x, query)
        ret = ivy.reshape(ret, target_shape) if target_shape != list(ret.shape) else ret
    if copy:
        return ivy.copy_array(ret)
    return ret


get_item.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_nestable
@handle_view_indexing
@inputs_to_ivy_arrays
@handle_array_function
def set_item(
    x: Union[ivy.Array, ivy.NativeArray],
    query: Union[ivy.Array, ivy.NativeArray, Tuple],
    val: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = False,
) -> ivy.Array:
    """
    Replace slices of x (defined by query) with val, identical to x[query] = val.

    Parameters
    ----------
    x
        the array to be updated.
    query
        either an index array, or a tuple of integers or slices.
    val
        the array containing the values to be infused into x
    copy
        boolean indicating whether to copy x.
        If True, the function will update and return a copy of x.
        If False, the function will update x inplace.

    Returns
    -------
    ret
        the array with updated values at the specified indices.

    Functional Examples
    -------------------

    >>> x = ivy.array([0, -1, 20])
    >>> query = ivy.array([0, 1])
    >>> val = ivy.array([10, 10])
    >>> ivy.set_item(x, query, val)
    >>> print(x)
    ivy.array([10, 10, 20])

    >>> x = ivy.array([[0, -1, 20], [5, 2, -8]])
    >>> query = (1, 0:1)
    >>> val = ivy.array([10, 10])
    >>> y = ivy.set_item(x, query, val, copy=True)
    >>> print(y)
    ivy.array([[0, -1, 20], [10, 10, -8]])
    """
    if 0 in x.shape:
        return x
    if ivy.is_array(query) and ivy.is_bool_dtype(query):
        if not len(query.shape):
            query = ivy.tile(query, (x.shape[0],))
        target_shape = ivy.get_item(x, query).shape
        query = ivy.nonzero(query, as_tuple=False)
    else:
        query, target_shape = _parse_query(query, x.shape)
    val = _broadcast_to(val, target_shape).astype(x.dtype)
    if copy:
        x = ivy.copy_array(x)
    return ivy.scatter_nd(query, val, reduction="replace", out=x)


set_item.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


def _int_list_or_array(var):
    # check if var is a list/tuple of integers or a 1-d array of integers
    return (
        isinstance(var, (list, tuple)) and all(isinstance(i, int) for i in var)
    ) or (ivy.is_array(var) and ivy.is_int_dtype(var) and len(var.shape) == 1)


def _parse_query(query, x_shape):
    query = (query,) if not isinstance(query, tuple) else query
    new_axes = [i for i, q in enumerate(query) if q is None]
    query = [q for q in query if q is not None]
    query = [Ellipsis] if query == [] else query
    ellipsis_inds = None
    if any(q is Ellipsis for q in query):
        query, ellipsis_inds = _parse_ellipsis(query, len(x_shape), ret_inds=True)
    query = (
        list(query)
        if isinstance(query, tuple)
        else [query] if not isinstance(query, list) else query
    )
    vectorized_indexing = False
    if all(_int_list_or_array(q) for q in query):
        vectorized_indexing = True
        for i, idx in enumerate(query):
            if ivy.is_array(idx):
                query[i] = ivy.where(idx < 0, idx + x_shape[i], idx)
            else:
                query[i] = [ii + x_shape[i] if ii < 0 else ii for ii in idx]
        query = ivy.array(query)
        query = query.T if len(query.shape) > 1 else query
        target_shape = [query.shape[0], *x_shape[query.shape[1] :]]
        target_shape = [target_shape]
    else:
        for i, idx in enumerate(query):
            s = x_shape[i]
            if isinstance(idx, slice):
                step = 1 if idx.step is None else idx.step
                if step > 0:
                    start = 0 if idx.start is None else idx.start
                    if start >= s:
                        stop = start
                    else:
                        if start <= -s:
                            start = 0
                        elif start < 0:
                            start = start + s
                        stop = s if idx.stop is None else idx.stop
                        if stop > s:
                            stop = s
                        elif start <= -s:
                            stop = 0
                        elif stop < 0:
                            stop = stop + s
                else:
                    start = s - 1 if idx.start is None else idx.start
                    if start <= -s:
                        stop = start
                    else:
                        if start >= s:
                            start = s
                        elif start < 0:
                            start = start + s
                        if idx.stop is None:
                            stop = -1
                        else:
                            stop = idx.stop
                            if stop > s:
                                stop = s
                            elif stop <= -s:
                                stop = 0
                            elif stop < 0:
                                stop = stop + s
                q_i = ivy.arange(start, stop, step).to_list()
                q_i = [q for q in q_i if 0 <= q < s]
                q_i = (
                    ivy.array(q_i)
                    if len(q_i) or start == stop or idx.stop is not None
                    else ivy.arange(0, s, 1)
                )
            elif isinstance(idx, int):
                q_i = ivy.array(idx + s if idx < 0 else idx)
            elif isinstance(idx, (tuple, list)):
                # ToDo: add handling for case of nested tuple/lists
                q_i = ivy.array([ii + s if ii < 0 else ii for ii in idx])
            elif ivy.is_array(idx):
                q_i = ivy.where(idx < 0, idx + s, idx)
            else:
                raise ivy.exceptions.IvyException("unsupported query format")
            query[i] = ivy.astype(q_i, ivy.int64)
        target_shape = [list(q.shape) for q in query]
    if ellipsis_inds is not None:
        target_shape = (
            target_shape[: ellipsis_inds[0]]
            + [target_shape[ellipsis_inds[0] : ellipsis_inds[1]]]
            + target_shape[ellipsis_inds[1] :]
        )
    for ax in new_axes:
        target_shape = [*target_shape[:ax], 1, *target_shape[ax:]]
    target_shape = _deep_flatten(target_shape)
    if vectorized_indexing:
        return query, target_shape
    target_shape += list(x_shape[len(query) :])
    query = [q.reshape((-1,)) if len(q.shape) > 1 else q for q in query]
    grid = ivy.meshgrid(*query, indexing="ij")
    indices = ivy.stack(grid, axis=-1)
    return indices, target_shape


def _deep_flatten(iterable):
    def _flatten_gen(iterable):
        for item in iterable:
            if isinstance(item, list):
                yield from _flatten_gen(item)
            else:
                yield item

    return list(_flatten_gen(iterable))


def _numel(shape):
    return math.prod(shape) if shape != () else 1


def _broadcast_to(input, target_shape):
    input = ivy.squeeze(input)
    if _numel(tuple(input.shape)) == _numel(tuple(target_shape)):
        return ivy.reshape(input, target_shape)
    else:
        input = ivy.expand_dims(input, axis=0) if not len(input.shape) else input
        new_dims = ()
        i_i = len(input.shape) - 1
        for i_t in range(len(target_shape) - 1, -1, -1):
            if len(input.shape) + len(new_dims) >= len(target_shape):
                break
            if i_i < 0 or target_shape[i_t] != input.shape[i_i]:
                new_dims += (i_t,)
            else:
                i_i -= 1
        input = ivy.expand_dims(input, axis=new_dims)
        return ivy.broadcast_to(input, target_shape)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
# @handle_device_shifting
def inplace_update(
    x: Union[ivy.Array, ivy.NativeArray],
    val: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
) -> ivy.Array:
    """
    Perform in-place update for the input array.

    This will always be performed on ivy.Array instances pass in the input, and will
    also be performed on the native array classes in the backend when the backend
    supports this. If the backend does not natively support inplace updates, and x is an
    ivy.NativeArray instance, then an
    exception will be thrown.

    Parameters
    ----------
    x
        The variable to update.
    val
        The array to update the variable with.
    ensure_in_backend
        Whether or not to ensure that the `ivy.NativeArray` is also inplace updated.
        In cases where it should be, backends which do not natively support inplace
        updates will raise an exception.
    keep_input_dtype
        Whether or not to preserve `x` data type after the update, otherwise `val`
        data type will be applied. Defaults to False.

    Returns
    -------
    ret
        The array following the in-place update.

    Raises
    ------
    IvyException
        If backend set doesn't natively support inplace updates and ensure_in_backend is
        True, above exception will be raised.

    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input and default backend set as `numpy`:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([0])
    >>> ivy.inplace_update(x, y)
    >>> print(x)
    ivy.array([0])

    With :class:`ivy.Array` input and default backend set as `numpy`:

    >>> x = ivy.array([1, 2, 3], dtype=ivy.float32)
    >>> y = ivy.array([0, 0, 0], dtype=ivy.int32)
    >>> ivy.inplace_update(x, y, keep_input_dtype=True)
    >>> print(x, x.dtype)
    ivy.array([0., 0., 0.]) float32

    With :class:`ivy.Container` instances:, and backend set as `torch`:

    >>> x = ivy.Container(a=ivy.array([5, 6]), b=ivy.array([7, 8]))
    >>> y = ivy.Container(a=ivy.array([1]), b=ivy.array([2]))
    >>> ivy.inplace_update(x, y)
    >>> print(x)
    {
        a: ivy.array([1]),
        b: ivy.array([2])
    }

    With mix of :class:`ivy.Array` and :class:`ivy.Container` instances:, and backend
    set as `torch`:

    >>> x = ivy.Container(a=ivy.array([5, 6]), b=ivy.array([7, 8]))
    >>> y = ivy.array([1, 2])
    >>> ivy.inplace_update(x, y)
    >>> print(x)
    {
        a: ivy.array([1, 2]),
        b: ivy.array([1, 2])
    }
    """
    return current_backend(x).inplace_update(
        x,
        val,
        ensure_in_backend=ensure_in_backend,
        keep_input_dtype=keep_input_dtype,
    )


inplace_update.unsupported_dtypes = {"torch": ("bfloat16",)}


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def inplace_decrement(
    x: Union[ivy.Array, ivy.NativeArray],
    val: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """
    Perform in-place decrement for the input array.

    Parameters
    ----------
    x
        The input array to be decremented by the defined value.
    val
        The value of decrement.

    Returns
    -------
    ret
        The array following the in-place decrement.


    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[5.3, 7., 0.],[6.8, 8, 3.9],[0., 10., 6.3]])
    >>> y = ivy.inplace_decrement(x, 1.25)
    >>> print(y)
    ivy.array([[ 4.05,  5.75, -1.25],
       [ 5.55,  6.75,  2.65],
       [-1.25,  8.75,  5.05]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.5, -5., 30.]), b=ivy.array([0., -25., 50.]))
    >>> y = ivy.inplace_decrement(x, 1.5)
    >>> print(y)
    {
        a: ivy.array([-1., -6.5, 28.5]),
        b: ivy.array([-1.5, -26.5, 48.5])
    }

    >>> x = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
    >>> y = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
    >>> z = ivy.inplace_decrement(x, y)
    >>> print(z)
    {
        a: ivy.array([0., 0., 0.]),
        b: ivy.array([0., 0., 0.])
    }

    >>> x = ivy.Container(a=ivy.array([3., 7., 10.]), b=ivy.array([0., 75., 5.5]))
    >>> y = ivy.Container(a=ivy.array([2., 5.5, 7.]), b=ivy.array([0., 25., 2.]))
    >>> z = ivy.inplace_decrement(x, y)
    >>> print(z)
    {
        a: ivy.array([1., 1.5, 3.]),
        b: ivy.array([0., 50., 3.5])
    }
    """
    return current_backend(x).inplace_decrement(x, val)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def inplace_increment(
    x: Union[ivy.Array, ivy.NativeArray],
    val: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """
    Perform in-place increment for the input array.

    Parameters
    ----------
    x
        The input array to be incremented by the defined value.
    val
        The value of increment.

    Returns
    -------
    ret
        The array following the in-place increment.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[5.3, 7., 0.],[6.8, 8, 3.9],[0., 10., 6.3]])
    >>> y = ivy.inplace_increment(x, 3.)
    >>> print(y)
    ivy.array([[ 8.3, 10.,  3.],
       [ 9.8, 11.,  6.9],
       [ 3., 13.,  9.3]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
    >>> y = ivy.inplace_increment(x, 2.5)
    >>> print(y)
    {
        a: ivy.array([2.5, 17.5, 32.5]),
        b: ivy.array([2.5, 27.5, 52.5])
    }


    >>> x = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
    >>> y = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
    >>> z = ivy.inplace_increment(x, y)
    >>> print(z)
    {
        a: ivy.array([0., 30., 60.]),
        b: ivy.array([0., 50., 100.])
    }
    """
    return current_backend(x).inplace_increment(x, val)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def scatter_flat(
    indices: Union[ivy.Array, ivy.NativeArray],
    updates: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Scatter flat updates into a new flat array according to flat indices.

    Parameters
    ----------
    indices
        Indices for the new values to occupy.
    updates
        Values for the new array to hold.
    size
        The size of the result. Default is `None`, in which case tensor
        argument out must be provided.
    reduction
        The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> indices = ivy.array([0, 0, 1, 0, 2, 2, 3, 3])
    >>> updates = ivy.array([5, 1, 7, 2, 3, 2, 1, 3])
    >>> out = ivy.array([0, 0, 0, 0, 0, 0, 0, 0])
    >>> ivy.scatter_flat(indices, updates, out=out)
    >>> print(out)
    ivy.array([8, 7, 5, 4, 0, 0, 0, 0])


    With :class:`ivy.Array` input:
    >>> indices = ivy.array([1, 0, 1, 0, 2, 2, 3, 3])
    >>> updates = ivy.array([9, 2, 0, 2, 3, 2, 1, 8])
    >>> size = 8
    >>> print(ivy.scatter_flat(indices, updates, size=size))
    ivy.array([4, 9, 5, 9, 0, 0, 0, 0])


    With :class:`ivy.Container` and :class:`ivy.Array` input:
    >>> indices = ivy.array([1, 0, 1, 0, 2, 2, 3, 3])
    >>> updates = ivy.Container(a=ivy.array([9, 2, 0, 2, 3, 2, 1, 8]), \
                        b=ivy.array([5, 1, 7, 2, 3, 2, 1, 3]))
    >>> size = 8
    >>> print(ivy.scatter_flat(indices, updates, size=size))
    {
        a: ivy.array([4, 9, 5, 9, 0, 0, 0, 0]),
        b: ivy.array([3, 12, 5, 4, 0, 0, 0, 0])
    }


    With :class:`ivy.Container` input:
    >>> indices = ivy.Container(a=ivy.array([1, 0, 1, 0, 2, 2, 3, 3]), \
                        b=ivy.array([0, 0, 1, 0, 2, 2, 3, 3]))
    >>> updates = ivy.Container(a=ivy.array([9, 2, 0, 2, 3, 2, 1, 8]), \
                        b=ivy.array([5, 1, 7, 2, 3, 2, 1, 3]))
    >>> size = 8
    >>> print(ivy.scatter_flat(indices, updates, size=size))
    {
        a: ivy.array([4, 9, 5, 9, 0, 0, 0, 0]),
        b: ivy.array([8, 7, 5, 4, 0, 0, 0, 0])
    }
    """
    return current_backend(indices).scatter_flat(
        indices, updates, size=size, reduction=reduction, out=out
    )


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
@handle_array_function
@inputs_to_native_shapes
@handle_device_shifting
def scatter_nd(
    indices: Union[ivy.Array, ivy.NativeArray],
    updates: Union[ivy.Array, ivy.NativeArray],
    /,
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    *,
    reduction: str = "sum",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Scatter updates into a new array according to indices.

    Parameters
    ----------
    indices
        Indices for the new values to occupy.
    updates
        Values for the new array to hold.
    shape
        The shape of the result. Default is ``None``, in which case tensor
        argument must be provided.
    reduction
        The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    Examples
    --------
    scatter values into an empty array, With :class:`ivy.Array` input:

    >>> indices = ivy.array([[4], [3], [1], [7]])
    >>> updates = ivy.array([9, 10, 11, 12])
    >>> shape = ivy.array([8])
    >>> scatter = ivy.scatter_nd(indices, updates, shape)
    >>> print(scatter)
    ivy.array([ 0, 11,  0, 10,  9,  0,  0, 12])

    With scatter into an empty array, With :class:`ivy.Container` input:

    >>> indices = ivy.Container(a=ivy.array([[4],[3],[6]]),
    ...                         b=ivy.array([[5],[1],[2]]))
    >>> updates = ivy.Container(a=ivy.array([100, 200, 200]),
    ...                         b=ivy.array([20, 30, 40]))
    >>> shape = ivy.Container(a=ivy.array([10]),
    ...                       b = ivy.array([10]))
    >>> z = ivy.scatter_nd(indices, updates, shape=shape, reduction='replace')
    >>> print(z)
    {
        a: ivy.array([0, 0, 0, 200, 100, 0, 200, 0, 0, 0]),
        b: ivy.array([0, 30, 40, 0, 0, 20, 0, 0, 0, 0])
    }

    With :class:`ivy.Container` and :class:`ivy.Array` input:

    >>> indices = ivy.array([[4],[3],[1]])
    >>> updates = ivy.Container(a=ivy.array([10, 20, 30]),
    ...                         b=ivy.array([200, 300, 400]))
    >>> z = ivy.Container(a=ivy.array([1, 2, 3, 4, 5]),
    ...                   b=ivy.array([10, 20, 30, 40, 50]))
    >>> ivy.scatter_nd(indices, updates, reduction='replace', out=z)
    >>> print(z)
    {
        a: ivy.array([1, 30, 3, 20, 10]),
        b: ivy.array([10, 400, 30, 300, 200])
    }
    """
    return current_backend(indices).scatter_nd(
        indices, updates, shape=shape, reduction=reduction, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def gather(
    params: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gather slices from params at axis according to indices.

    Parameters
    ----------
    params
        The array from which to gather values.
    indices
        The array which indicates the indices that will be gathered along
        the specified axis.
    axis
        optional int, the axis from which to gather from.
        Default is ``-1``.
    batch_dims
        optional int, lets you gather different items from each element of a batch.
    out
        optional array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        New array with the values gathered at the specified indices along the
        specified axis.


    Both the description and the type hints above assumes an array input for
    simplicity, but this function is *nestable*, and therefore also accepts
    :class:`ivy.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.array([1, 2])
    >>> print(ivy.gather(x, y))
    ivy.array([1., 2.])

    >>> x = ivy.array([[0., 1., 2.],[3., 4., 5.]])
    >>> y = ivy.array([[0, 1],[1, 2]])
    >>> z = ivy.array([[0., 0.],[0., 0.]])
    >>> ivy.gather(x, y, out=z)
    >>> print(z)
    ivy.array([[[0., 1.],[1., 2.]],[[3., 4.],[4., 5.]]])

    >>> x = ivy.array([[[0., 1.], [2., 3.]],
    ...                [[8., 9.], [10., 11.]]])
    >>> y = ivy.array([[0, 1]])
    >>> ivy.gather(x, y, axis=0, out=x)
    >>> print(x)
    ivy.array(
        [[[[ 0.,  1.],
           [ 2.,  3.]],
          [[ 8.,  9.],
           [10., 11.]]]])

    >>> x = ivy.array([[0, 10, 20, 0, 0],
    ...                [0, 0, 0, 30, 40],
    ...                [0, 10, 0, 0, 40]])
    >>> y = ivy.array([[1, 2],[3, 4],[1, 4]])
    >>> z = ivy.gather(x, y, batch_dims=1)
    >>> print(z)
    ivy.array([[10, 20], [30, 40],[10, 40]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array([0., 1., 2.]),
    ...                   b = ivy.array([4., 5., 6.]))
    >>> y = ivy.Container(a = ivy.array([0, 1]),
    ...                   b = ivy.array([1, 2]))
    >>> print(ivy.gather(x, y))
    {
        a: ivy.array([0., 1.]),
        b: ivy.array([5., 6.])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.array([0., 1., 2.]),
    ...                   b = ivy.array([4., 5., 6.]))
    >>> y = ivy.array([0, 1])
    >>> print(ivy.gather(x, y))
    {
        a: ivy.array([0., 1.]),
        b: ivy.array([4., 5.])
    }
    """
    return current_backend(params, indices).gather(
        params, indices, axis=axis, batch_dims=batch_dims, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def gather_nd(
    params: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    batch_dims: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Gather slices from params into a array with shape specified by indices.

    Parameters
    ----------
    params
        The array from which to gather values.
    indices
        Index array.
    batch_dims
        optional int, lets you gather different items from each element of a batch.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values gathered at the indices.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6.])
    >>> y = ivy.array([1])
    >>> print(ivy.gather_nd(x, y))
    ivy.array(1.)

    >>> x = ivy.array([[0., 1.], [2., 3.], [4., 5.]])
    >>> y = ivy.array([[0],[1],[1]], dtype='int32')
    >>> z = ivy.gather_nd(x,y,batch_dims=1)
    ivy.array([0., 3., 5.])

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),b=ivy.array([4., 5., 6.]))
    >>> y = ivy.array([1])
    >>> print(ivy.gather_nd(x, y))
    {
        a: ivy.array(1.),
        b: ivy.array(5.)
    }

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[0., 10., 20.],[30.,40.,50.]]),
    ...                   b=ivy.array([[0., 100., 200.],[300.,400.,500.]]))
    >>> y = ivy.Container(a=ivy.array([1,0]),
    ...                   b=ivy.array([0]))
    >>> print(ivy.gather_nd(x, y))
    {
        a: ivy.array(30.),
        b: ivy.array([0., 100., 200.])
    }
    """
    res = current_backend(params, indices).gather_nd(
        params, indices, batch_dims=batch_dims
    )
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res


@handle_exceptions
@handle_nestable
@handle_array_function
def multiprocessing(context: Optional[str] = None):
    """
    Return backend-specific multiprocessing module.

    Parameters
    ----------
    context
        The context of the multiprocessing, either fork, forkserver or spawn.
        Default is ``None``.

    Returns
    -------
    ret
        Multiprocessing module
    """
    return current_backend().multiprocessing(context)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@outputs_to_ivy_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def shape(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    as_array: bool = False,
) -> Union[ivy.Shape, ivy.NativeShape]:
    """
    Return the shape of the array ``x``.

    Parameters
    ----------
    x
        Input array to infer the shape of.
    as_array
        Whether to return the shape as an array.
        Default is False.

    Returns
    -------
    ret
        Shape of the array ``x``.

    Examples
    --------
    >>> x = ivy.array([[-1, 0, 1], [1, 0, -1]])
    >>> y = ivy.shape(x)
    >>> z = ivy.shape(x, as_array = True)
    >>> print(y)
    (2, 3)

    >>> print(z)
    ivy.array([2, 3])

    """
    return current_backend(x).shape(x, as_array=as_array)


ivy.shape_array_mode = shape_array_mode_stack[-1] if shape_array_mode_stack else False


@handle_exceptions
def set_shape_array_mode(mode: bool) -> None:
    """
    Set the mode of returning shape as ivy.Array to the given mode instance.

    Parameter
    ---------
    mode
        boolean whether to return shape as ivy.Array

    Examples
    --------
    >>> ivy.set_shape_array_mode(False)
    >>> ivy.shape_array_mode
    False

    >>> ivy.set_shape_array_mode(True)
    >>> ivy.shape_array_mode
    True
    """
    global shape_array_mode_stack
    ivy.utils.assertions.check_isinstance(mode, bool)
    shape_array_mode_stack.append(mode)
    ivy.__setattr__("shape_array_mode", mode, True)


@handle_exceptions
def unset_shape_array_mode() -> None:
    """
    Reset the mode of returning shape as ivy.Array to the previous state.

    Examples
    --------
    >>> ivy.set_shape_array_mode(True)
    >>> ivy.shape_array_mode
    True

    >>> ivy.unset_shape_array_mode()
    >>> ivy.shape_array_mode
    False
    """
    global shape_array_mode_stack
    if shape_array_mode_stack:
        shape_array_mode_stack.pop(-1)
        mode = shape_array_mode_stack[-1] if shape_array_mode_stack else False
        ivy.__setattr__("shape_array_mode", mode, True)


@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def get_num_dims(
    x: Union[ivy.Array, ivy.NativeArray], /, *, as_array: bool = False
) -> int:
    """
    Return the number of dimensions of the array x.

    Parameters
    ----------
    x
        Input array to infer the number of dimensions for.
    as_array
        Whether to return the shape as a array, default False.

    Returns
    -------
    ret
        Shape of the array

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> a = ivy.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    >>> b = ivy.get_num_dims(a, as_array=False)
    >>> print(b)
    3

    With :class:`ivy.Container` input:

    >>> a = ivy.Container(b = ivy.asarray([[0.,1.,1.],[1.,0.,0.],[8.,2.,3.]]))
    >>> ivy.get_num_dims(a)
    2

    >>> b = ivy.get_num_dims(a, as_array=True)
    >>> print(b)
    ivy.array(3)
    """
    return current_backend(x).get_num_dims(x, as_array=as_array)


@handle_exceptions
def arg_info(fn: Callable, *, name: Optional[str] = None, idx: Optional[int] = None):
    """
    Return the index and `inspect.Parameter` representation of the specified argument.
    In the form of a dict with keys "idx" and "param".

    Parameters
    ----------
    fn
        The function to retrieve the argument information for
    name
        The name of the argument
    idx
        the index of the argument in the inputs

    Returns
    -------
    ret
        a `dict` containing the idx, and the `inspect.Parameter` for the argument,
        which itself contains the parameter name, type, and other helpful information.
    """
    ivy.utils.assertions.check_all_or_any_fn(
        name,
        idx,
        fn=ivy.exists,
        type="any",
        limit=[1],
        message="exactly one of the keyword arguments name or idx must be provided",
        as_array=False,
    )
    params = inspect.signature(fn).parameters
    if ivy.exists(name):
        return {"idx": list(params).index(name), "param": params[name]}
    return {"idx": idx, "param": list(params.values())[idx]}


def _valid_attrib_combinations(fn, backend, dnd_dict, first_attr_name, other_attr_name):
    attr_list = ()
    if hasattr(fn, other_attr_name):
        attr_list = getattr(fn, other_attr_name)
        if isinstance(attr_list, dict):
            attr_list = attr_list.get(backend, ())
    ivy.utils.assertions.check_false(
        dnd_dict and attr_list,
        (
            f"Cannot specify both {first_attr_name} and {other_attr_name} "
            "cannot both be defined for the same function"
        ),
    )


def _is_valid_device_and_dtypes_attributes(fn: Callable) -> bool:
    fn_unsupported_dnd = {}
    fn_supported_dnd = {}
    backend = ivy.current_backend_str()
    if hasattr(fn, "unsupported_device_and_dtype"):
        fn_unsupported_dnd = fn.unsupported_device_and_dtype
        # if it's a nested dict, unwrap for the current backend
        if isinstance(list(fn_unsupported_dnd.__get__().values())[0], dict):
            fn_unsupported_dnd = fn_unsupported_dnd.get(backend, {})
    if hasattr(fn, "supported_device_and_dtype"):
        fn_supported_dnd = fn.supported_device_and_dtype
        # if it's a nested dict, unwrap for the current backend
        if isinstance(list(fn_supported_dnd.__get__().values())[0], dict):
            fn_supported_dnd = fn_supported_dnd.get(backend, {})

    ivy.utils.assertions.check_false(
        fn_unsupported_dnd and fn_supported_dnd,
        (
            "unsupported_device_and_dtype and supported_device_and_dtype cannot"
            " both be defined for the same function"
        ),
    )

    us = "unsupported_device_and_dtype"
    _valid_attrib_combinations(fn, backend, fn_unsupported_dnd, us, "supported_devices")
    _valid_attrib_combinations(fn, backend, fn_unsupported_dnd, us, "supported_dtypes")

    ss = "supported_device_and_dtype"
    _valid_attrib_combinations(fn, backend, fn_supported_dnd, ss, "unsupported_device")
    _valid_attrib_combinations(fn, backend, fn_supported_dnd, ss, "unsupported_dtypes")

    return True


def _all_dnd_combinations():
    all_comb = {}
    for device in ivy.all_devices:
        all_comb[device] = ivy.all_dtypes
    return all_comb


def _dnd_dict_intersection(a, b):
    res = {}
    for device in a:
        if device in b:
            intersection = set.intersection(set(a[device]), set(b[device]))
            if intersection:
                res[device] = tuple(intersection)
    return res


def _dnd_dict_difference(a, b):
    res = a
    for device in list(a):
        if device in b:
            difference = set.difference(set(a[device]), set(b[device]))
            if difference:
                res[device] = tuple(difference)
            else:
                del res[device]
    return res


def _dnd_dict_union(a, b):
    res = {}
    for device in set(list(a) + list(b)):
        u1 = set(a.get(device, ()))
        u2 = set(b.get(device, ()))
        res[device] = tuple(set.union(u1, u2))

    return res


def _get_devices_and_dtypes(fn, recurse=False, complement=True):
    supported_devices = ivy.function_supported_devices(fn, recurse=recurse)
    supported_dtypes = ivy.function_supported_dtypes(fn, recurse=recurse)

    if hasattr(fn, "partial_mixed_handler"):
        supported_devices = supported_devices["primary"]
        supported_dtypes = supported_dtypes["primary"]

    supported = {}
    # Generate a base supported set from other attributes
    for device in supported_devices:
        supported[device] = supported_dtypes

    is_backend_fn = "backend" in fn.__module__
    is_frontend_fn = "frontend" in fn.__module__
    is_einops_fn = "einops" in fn.__name__
    if not is_backend_fn and not is_frontend_fn and not is_einops_fn:
        if complement:
            all_comb = _all_dnd_combinations()
            supported = _dnd_dict_difference(all_comb, supported)
        return supported

    backend = ivy.current_backend_str()

    # Their values are formatted like either
    # 1. fn.supported_device_and_dtype = {"cpu":("float16",)}
    if hasattr(fn, "supported_device_and_dtype"):
        fn_supported_dnd = fn.supported_device_and_dtype.__get__()

        if "einops" in fn.__name__ and isinstance(fn_supported_dnd, dict):
            fn_supported_dnd = fn_supported_dnd.get(backend, supported)

        ivy.utils.assertions.check_isinstance(list(fn_supported_dnd.values())[0], tuple)
        # dict intersection
        supported = _dnd_dict_intersection(supported, fn_supported_dnd)

    if hasattr(fn, "unsupported_device_and_dtype"):
        fn_unsupported_dnd = fn.unsupported_device_and_dtype.__get__()

        if "einops" in fn.__name__ and isinstance(fn_unsupported_dnd, dict):
            fn_unsupported_dnd = fn_unsupported_dnd.get(backend, supported)

        ivy.utils.assertions.check_isinstance(
            list(fn_unsupported_dnd.values())[0], tuple
        )
        # dict difference
        supported = _dnd_dict_difference(supported, fn_unsupported_dnd)

    if complement:
        # dict difference
        all_comb = _all_dnd_combinations()
        supported = _dnd_dict_difference(all_comb, supported)
    return supported


@handle_exceptions
@handle_nestable
def function_supported_devices_and_dtypes(fn: Callable, recurse: bool = True) -> Dict:
    """
    Return the supported combination of devices and dtypes of the current backend's
    function. The function returns a dict containing the supported combination of
    devices and dtypes of the primary and compositional implementations incase of
    partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the supported device and dtype attribute
    recurse
        Whether to recurse into used ivy functions.
        Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the supported devices and dtypes of the function
    """
    ivy.utils.assertions.check_true(
        _is_valid_device_and_dtypes_attributes(fn),
        (
            "supported_device_and_dtypes and unsupported_device_and_dtypes "
            "attributes cannot both exist in a particular backend"
        ),
    )

    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_supported_devices_and_dtypes(
                fn.compos, recurse=recurse
            ),
            "primary": _get_devices_and_dtypes(fn, complement=False),
        }
    else:
        supported_devices_dtypes = _get_devices_and_dtypes(fn, complement=False)
        if recurse:
            supported_devices_dtypes = ivy.functional.data_type._nested_get(
                fn,
                supported_devices_dtypes,
                _dnd_dict_intersection,
                function_supported_devices_and_dtypes,
                wrapper=lambda x: x,
            )

    return supported_devices_dtypes


@handle_exceptions
@handle_nestable
def function_unsupported_devices_and_dtypes(fn: Callable, recurse: bool = True) -> Dict:
    """
    Return the unsupported combination of devices and dtypes of the current backend's
    function. The function returns a dict containing the unsupported combination of
    devices and dtypes of the primary and compositional implementations incase of
    partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the unsupported device and dtype attribute
    recurse
        Whether to recurse into used ivy functions.
        Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the unsupported devices and dtypes of the function
    """
    ivy.utils.assertions.check_true(
        _is_valid_device_and_dtypes_attributes(fn),
        (
            "supported_device_and_dtypes and unsupported_device_and_dtypes "
            "attributes cannot both exist in a particular backend"
        ),
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_unsupported_devices_and_dtypes(
                fn.compos, recurse=recurse
            ),
            "primary": _get_devices_and_dtypes(fn, complement=True),
        }
    else:
        unsupported_devices_dtypes = _get_devices_and_dtypes(fn, complement=True)
        if recurse:
            unsupported_devices_dtypes = ivy.functional.data_type._nested_get(
                fn,
                unsupported_devices_dtypes,
                _dnd_dict_union,
                function_unsupported_devices_and_dtypes,
                wrapper=lambda x: x,
            )
    return unsupported_devices_dtypes


@handle_exceptions
def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    """
    Vectorizing map. Creates a function which maps func over argument axes.

    Parameters
    ----------
    func
        Function to be mapped over additional axes.
    in_axes
       An integer, None, or (nested) standard Python container
       (tuple/list) thereof specifying which input array
       axes to map over.If each positional argument to fun
       is an array, then in_axes can be an integer, a None,
       or a tuple of integers and Nones with length equal
       to the number of positional arguments to fun. An
       integer or None indicates which array axis to map
       over for all arguments (with None indicating not to map any axis),
       and a tuple indicates which axis to map for each
       corresponding positional argument. Axis integers must
       be in the range [-ndim, ndim) for each array,
       where ndim is the number of dimensions (axes) of the
       corresponding input array.
    out_axes
        An integer indicating where the mapped axis should appear in the output.

    Returns
    -------
    ret
        Batched/vectorized version of func with arguments
        that correspond to those of func, but with extra
        array axes at positions indicated by in_axes,
        and a return value that corresponds
        to that of fun, but with extra array axes
        at positions indicated by out_axes.


    This docstring is a summarised version of the `docstring
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax-vmap>`_
    for vmap from JAX documentation.

    Examples
    --------
    With :func:`ivy.matmul` and :class:`ivy.Array` input:

    >>> x = ivy.array(ivy.arange(60).reshape((3, 5, 4)))
    >>> y = ivy.array(ivy.arange(40).reshape((5, 4, 2)))
    >>> z = ivy.vmap(ivy.matmul, (1, 0), 1)(x, y)
    >>> print(z.shape)
    (3, 5, 2)
    """
    # TODO: optimize in the numpy and tensorflow backends and extend functionality
    return current_backend().vmap(func, in_axes, out_axes)


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
@handle_device_shifting
def isin(
    elements: Union[ivy.Array, ivy.NativeArray],
    test_elements: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> ivy.Array:
    """
    Test if each element of elements is in test_elements.

    Parameters
    ----------
    elements
        input array
    test_elements
        values against which to test for each input element
    assume_unique
        If True, assumes both elements and test_elements contain unique elements,
        which can speed up the calculation. Default value is False.
    invert
        If True, inverts the boolean return array, resulting in True values for
        elements not in test_elements. Default value is False.

    Returns
    -------
    ret
        output a boolean array of the same shape as elements that is True for elements
        in test_elements and False otherwise.

    Examples
    --------
    >>> x = ivy.array([[10, 7, 4], [3, 2, 1]])
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.isin(x, y)
    ivy.array([[False, False, False], [ True,  True,  True]])

    >>> x = ivy.array([3, 2, 1, 0])
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.isin(x, y, invert=True)
    ivy.array([False, False, False,  True])
    """
    return ivy.current_backend(elements, test_elements).isin(
        elements, test_elements, assume_unique=assume_unique, invert=invert
    )


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_device_shifting
def itemsize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> int:
    """
    Return the size of the input array's elements.

    Parameters
    ----------
    x
       The input array.

    Returns
    -------
    ret
        An integer specifying the element size in bytes.

    Examples
    --------
    >>> x = ivy.array([1,2,3], dtype=ivy.float64)
    >>> ivy.itemsize(x)
    8

    >>> x = ivy.array([1,2,3], dtype=ivy.complex128)
    >>> ivy.itemsize(x)
    16
    """
    return ivy.current_backend(x).itemsize(x)


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
@handle_device_shifting
def strides(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple[int]:
    """
    Return the input array's strides across each dimension.

    Parameters
    ----------
    x
       The input array.

    Returns
    -------
    ret
        A tuple containing the strides.

    Examples
    --------
    >>> x = ivy.array([[1, 5, 9], [2, 6, 10]])
    >>> ivy.strides(x)
    (4, 8)
    """
    return ivy.current_backend(x).strides(x)


def is_ivy_nested_array(x: Any, /) -> bool:
    """
    Determine whether the input x is an Ivy Nested Array.

    Parameters
    ----------
    x
        The input to check
    Returns
    -------
    ret
        Boolean, whether or not x is an ivy nested array.
    """
    return isinstance(x, ivy.NestedArray)
