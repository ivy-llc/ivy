"""Collection of general Ivy functions."""

# global
import gc
import math
import einops
import inspect
import builtins
import numpy as np
from numbers import Number
from typing import Callable, Any, Union, List, Tuple, Dict, Iterable, Optional

# local
import ivy
from ivy.functional.ivy.device import dev
from ivy.backend_handler import current_backend, backend_stack
from ivy.func_wrapper import (
    infer_device,
    inputs_to_native_arrays,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)

FN_CACHE = dict()
INF = float("inf")
TIMEOUT = 15.0
TMP_DIR = "/tmp"

array_mode_stack = list()
shape_array_mode_stack = list()
nestable_mode_stack = list()


def get_referrers_recursive(
    item, depth=0, max_depth=None, seen_set=None, local_set=None
):
    """Summary.

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


def is_native_array(
    x: Union[ivy.Array, ivy.NativeArray], exclusive: bool = False
) -> bool:
    """
    Determines whether the input x is a Native Array.

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
        Boolean, whether or not x is a native array.

    Examples
    --------
    >>> x = ivy.array([0, 1, 2])
    >>> ivy.is_native_array(x)
    False

    >>> x = ivy.native_array([1.5, 2.3, 4.9, 2.6])
    >>> ivy.is_native_array(x)
    True

    >>> x = ivy.native_array([-1, 2, 7, -3])
    >>> ivy.is_native_array(x, False)
    True

    >>> x = ivy.native_array([9.1, -8.3, 2.8, 3.0])
    >>> ivy.is_native_array(x, True)
    True

    >>> x = ivy.array([5, 2, 6, 9])
    >>> ivy.is_native_array(x, True)
    False

    """
    try:
        return current_backend(x).is_native_array(x, exclusive)
    except ValueError:
        return False


def is_ivy_array(x: Union[ivy.Array, ivy.NativeArray], exclusive: bool = False) -> bool:
    """
    Determines whether the input x is an Ivy Array.

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
    >>> ivy.is_ivy_array(x)
    True

    >>> x = ivy.native_array([1.5, 2.3, 4.9, 2.6])
    >>> ivy.is_ivy_array(x)
    False

    >>> x = ivy.native_array([-1, 2, 7, -3])
    >>> ivy.is_ivy_array(x, False)
    False

    >>> x = ivy.native_array([9.1, -8.3, 2.8, 3.0])
    >>> ivy.is_ivy_array(x, True)
    False

    >>> x = ivy.array([5, 2, 6, 9])
    >>> ivy.is_ivy_array(x, True)
    True

    """
    return isinstance(x, ivy.Array) and ivy.is_native_array(x.data, exclusive)


def is_array(x: Any, exclusive: bool = False) -> bool:
    """Determines whether the input x is either an Ivy Array or a Native Array.

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

    """
    return ivy.is_ivy_array(x, exclusive) or ivy.is_native_array(x, exclusive)


def is_ivy_container(x: Any) -> bool:
    """Determines whether the input x is an Ivy Container.

    Parameters
    ----------
    x
        The input to check

    Returns
    -------
    ret
        Boolean, whether or not x is an ivy container.

    """
    return isinstance(x, ivy.Container)


def set_array_mode(mode: bool) -> None:
    """Set the mode of whether to convert inputs to ivy.NativeArray, then convert
    outputs back to ivy.Array

    Parameter
    ---------
    mode
        boolean whether to perform ivy.Array conversions

    Examples
    --------
    >>> ivy.set_array_mode(False)
    >>> ivy.get_array_mode()
    False

    >>> ivy.set_array_mode(True)
    >>> ivy.get_array_mode()
    True
    """
    global array_mode_stack
    if not isinstance(mode, bool):
        raise Exception("set_array_mode only accepts type bool")
    array_mode_stack.append(mode)


def unset_array_mode() -> None:
    """Reset the mode of converting inputs to ivy.NativeArray, then converting
    outputs back to ivy.Array to the previous state

    Examples
    --------
    >>> ivy.set_array_mode(False)
    >>> ivy.get_array_mode()
    False

    >>> ivy.unset_shape_array_mode()
    >>> ivy.get_array_mode()
    True
    """
    global array_mode_stack
    if array_mode_stack:
        array_mode_stack.pop(-1)


def get_array_mode() -> bool:
    """Get the current state of array_mode

    Examples
    --------
    >>> ivy.get_array_mode()
    True

    >>> ivy.set_array_mode(False)
    >>> ivy.get_array_mode()
    False
    """
    global array_mode_stack
    if not array_mode_stack:
        return True
    return array_mode_stack[-1]


def set_nestable_mode(mode: bool) -> None:
    """Set the mode of whether to check if function inputs are ivy.Container

    Parameter
    ---------
    mode
        boolean whether to check if function inputs are ivy.Container

    Examples
    --------
    >>> ivy.set_nestable_mode(False)
    >>> ivy.get_nestable_mode()
    False

    >>> ivy.set_nestable_mode(True)
    >>> ivy.get_nestable_mode()
    True
    """
    global nestable_mode_stack
    if not isinstance(mode, bool):
        raise Exception("set_nestable_mode only accepts type bool")
    nestable_mode_stack.append(mode)


def unset_nestable_mode() -> None:
    """Reset the mode of whether to check if function inputs are ivy.Container
    to the previous state

    Examples
    --------
    >>> ivy.set_nestable_mode(False)
    >>> ivy.get_nestable_mode()
    False

    >>> ivy.unset_nestable_mode()
    >>> ivy.get_nestable_mode()
    True
    """
    global nestable_mode_stack
    if nestable_mode_stack:
        nestable_mode_stack.pop(-1)


def get_nestable_mode() -> bool:
    """Get the current mode of whether to check if function inputs are ivy.Container.
    Default is True.

    Examples
    --------
    >>> ivy.get_nestable_mode()
    True

    >>> ivy.set_nestable_mode(False)
    >>> ivy.get_nestable_mode()
    False
    """
    global nestable_mode_stack
    if not nestable_mode_stack:
        return True
    return nestable_mode_stack[-1]


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def copy_array(
    x: Union[ivy.Array, ivy.NativeArray], *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Copy an array.

    Parameters
    ----------
    x
        input array.

    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a copy of the input array ``x``.

    Examples
    --------
    With one :code:`ivy.Array` input:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([-1, 0, 1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([1, 0, 1, 1])

    >>> x = ivy.array([1, 0, 1, -1])
    >>> y = ivy.zeros((1, 4))
    >>> ivy.copy_array(x, out=y)
    >>> print(y)
    ivy.array([1, 0, 1, -1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> ivy.copy_array(x, out=x)
    >>> print(x)
    ivy.array([1, 0, 1, 1])

    With one :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]))
    >>> y = ivy.copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1])
    }

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),\
                          b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1, 1, 0])
    }

    With one :code:`ivy.Container` static method:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),\
                          b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.Container.static_copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1, 1, 0])
    }
    
    With one :code:`ivy.Array` instance method:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = x.copy_array()
    >>> print(y)
    ivy.array([-1, 0, 1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> y = x.copy_array()
    >>> print(y)
    ivy.array([1, 0, 1, 1])
    
    With :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([1, 0, 1]),\
                          b=ivy.array([-1, 0, 1, 1]))
    >>> y = x.copy_array()
    >>> print(y)
    {
        a: ivy.array([1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1])
    }

    """
    return current_backend(x).copy_array(x, out=out)


@inputs_to_native_arrays
@handle_nestable
def array_equal(
    x0: Union[ivy.Array, ivy.NativeArray], x1: Union[ivy.Array, ivy.NativeArray]
) -> bool:
    """Determines whether two input arrays are equal across all elements.

    Parameters
    ----------
    x0
        The first input array to compare.
    x1
        The second input array to compare.
    dtype
        array data type

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


@inputs_to_native_arrays
@handle_nestable
def arrays_equal(xs: List[Union[ivy.Array, ivy.NativeArray]]) -> bool:
    """Determines whether input arrays are equal across all elements.

    Parameters
    ----------
    xs
        Sequence of arrays to compare for equality
    dtype
        list data type

    Returns
    -------
    ret
        Boolean, whether or not all of the input arrays are equal across all elements.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> i = ivy.array([1, 2])
    >>> j = ivy.arrays_equal([i])
    >>> print(j)
    True

    >>> x = ivy.array([0, 1, 2])
    >>> y = ivy.array([1, 0, 2])
    >>> z = ivy.array([0, 1, 2])
    >>> w = ivy.arrays_equal([x, y, z])
    >>> print(w)
    False

    >>> a = ivy.array([-1, 0, 1])
    >>> b = ivy.array([-1, 0, 1])
    >>> c = ivy.array([-1, 0, 1])
    >>> d = ivy.arrays_equal([a, b, c])
    >>> print(d)
    True

    >>> x = ivy.array([0.1, 1.1])
    >>> y = ivy.array([0.1, 1.1, 2.1])
    >>> z = ivy.array([0.1, 1.1])
    >>> w = ivy.arrays_equal([x, y, z])
    >>> print(w)
    False


    With :code:`ivy.NativeArray` input:

    >>> m = ivy.native_array([1.1, 0.2, 1.3])
    >>> n = ivy.native_array([1.1, 0.2, 1.4])
    >>> o = ivy.arrays_equal([m, n])
    >>> print(o)
    False

    >>> a = ivy.native_array([1, 2, 3, 0, -1])
    >>> b = ivy.array([1, 2, 3, 0, -1])
    >>> c = ivy.arrays_equal([a,b])
    >>> print(c)
    True

    >>> a = ivy.native_array([1, 2, 3, 0, -1])
    >>> b = ivy.array([1, 2, 3, 0, -2])
    >>> c = ivy.arrays_equal([a,b])
    >>> print(c)
    False


    With :code:`ivy.Container` input:

    >>> r = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> s = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> t = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([6., 7., 8.]))
    >>> print(ivy.arrays_equal([r,s,t]))
    {
        a: true,
        b: false
    }

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([3, 4, 5]))
    >>> y = ivy.array([0,1,2])
    >>> z = ivy.arrays_equal([x,y])
    >>> print(z)
    {
        a: true,
        b: false
    }

    """
    x0 = xs[0]
    for x in xs[1:]:
        if not array_equal(x0, x):
            return False
    return True


@to_native_arrays_and_back
@handle_nestable
def all_equal(
    *xs: Iterable[Any], equality_matrix: bool = False
) -> Union[bool, ivy.Array, ivy.NativeArray]:
    """Determines whether the inputs are all equal.

    Parameters
    ----------
    xs
        inputs to compare.
    equality_matrix
        Whether to return a matrix of equalities comparing each input with every other.
        Default is False.

    Returns
    -------
    ret
        Boolean, whether or not the inputs are equal, or matrix array of booleans if
        equality_matrix=True is set.

    Examples
    --------
    With :code:`Number` inputs:

    >>> x1 = 1.2
    >>> x2 = 1.0
    >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    False

    With :code:`ivy.Array` inputs:

    >>> x1 = ivy.array([1, 1, 0, 0, 1, -1])
    >>> x2 = ivy.array([1, 1, 0, 0, 1, -1])
    >>> y = ivy.all_equal(x1, x2, equality_matrix=True)
    >>> print(y)
    ivy.array([[ True,  True], [ True,  True]])

    With :code:`ivy.NativeArray` inputs:

    >>> x1 = ivy.native_array([1, 1, 0, 0, 1, -1])
    >>> x2 = ivy.native_array([1, 1, 0, 0, 1, -1])
    >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    True

    With one :code:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.native_array([0, 0, -1, 1, 0]), \
                            b=ivy.array([0, 0, -1, 1, 0]))
    >>> x2 = ivy.array([0, 0, -1, 1, 0])
    >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    {
        a: true,
        b: true
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), \
                            b=ivy.native_array([1, 0, 0, 1]))
    >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1, 1]), \
                            b=ivy.array([1, 0, -1, -1]))
    >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    {
        a: true,
        b: false
    }

    """
    equality_fn = ivy.array_equal if ivy.is_native_array(xs[0]) else lambda a, b: a == b
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


@inputs_to_native_arrays
@handle_nestable
def to_numpy(x: Union[ivy.Array, ivy.NativeArray], copy: bool = True) -> np.ndarray:
    """Converts an array into a numpy array.

    Parameters
    ----------
    x
        input array
    copy
        whether to copy the array to a new address or not. Default is True.
    Returns
    -------
    ret
        a numpy array copying all the element of the array ``x``.

    Functional Method Examples
    --------------------------

    With :code:`ivy.Array` inputs:

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

    With :code:`ivy.NativeArray` inputs:

    >>> x = ivy.native_array([-1, 0, 1])
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    [-1 0 1]

    >>> x = ivy.native_array([[-1, 0, 1],[-1, 0, 1], [1,0,-1]])
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    [[-1  0  1]
    [-1  0  1]
    [ 1  0 -1]]

    With a mix of :code:`ivy.Container` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.Container(a=ivy.native_array([-1, 0, 1]))
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    {
        a: array([-1, 0, 1], dtype=int32)
    }

    >>> x = ivy.Container(a=ivy.native_array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]),\
                        b=ivy.native_array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    {
        a: array([[-1, 0, 1],
                  [-1, 0, 1],
                  [1, 0, -1]], dtype=int32),
        b: array([[-1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=int32)
    }

    With a mix of :code:`ivy.Container` and :code:`ivy.Array` inputs:

    >>> x = ivy.Container(x=ivy.array([-1, 0, 1]))
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    {x:array([-1,0,1],dtype=int32)}

    >>> x = ivy.Container(a=ivy.array([[-1.0, 0., 1.], [-1, 0, 1], [1, 0, -1]]),\
                      b=ivy.array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
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

    Instance Method Example
    -----------------------

    With :code:`ivy.Array` inputs:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = x.to_numpy()
    >>> print(y)
    [-1  0  1]

    >>> x = ivy.array([[-1, 0, 1],[-1, 0, 1], [1,0,-1]])
    >>> y = x.to_numpy()
    >>> print(y)
    [[-1  0  1]
    [-1  0  1]
    [ 1  0 -1]]

    With :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([[-1.0, 0., 1.], [-1, 0, 1], [1, 0, -1]]),\
                      b=ivy.native_array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
    >>> y = x.to_numpy()
    >>> print(y)
    {
        a: array([[-1., 0., 1.],
                  [-1., 0., 1.],
                  [1., 0., -1.]], dtype=float32),
        b: array([[-1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=int32)
    }

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]), b=ivy.array([1, 0, 1, 1]))
    >>> y = x.to_numpy()
    >>> print(y)
    {
        a: array([-1, 0, 1], dtype=int32),
        b: array([1, 0, 1, 1], dtype=int32)
    }

    """
    return current_backend(x).to_numpy(x, copy)


@inputs_to_native_arrays
@handle_nestable
def to_scalar(x: Union[ivy.Array, ivy.NativeArray]) -> Number:
    """Converts an array with a single element into a scalar.

    Parameters
    ----------
    x
        Input array with a single element.

    Returns
    -------
    ret
        a scalar copying the element of the array ``x``.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([-1])
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    -1

    >>> print(ivy.is_int_dtype(y))
    True

    >>> x = ivy.array([3])
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    3

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([-1])
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    -1

    >>> print(ivy.is_int_dtype(y))
    True

    >>> x = ivy.native_array([3])
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    3

    With a mix of :code:`ivy.Container` and :code:`ivy.Array` input:

    >>> x = ivy.Container(a=ivy.array([-1]), b=ivy.array([3]))
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    {
        a: -1,
        b: 3
    }

    >>> print(ivy.is_int_dtype(y))
    {
        a: true,
        b: true
    }

    >>> x = ivy.Container(a=ivy.array([1]), b=ivy.array([0]),\
                          c=ivy.array([-1]))
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    {
        a: 1,
        b: 0,
        c: -1
    }

    With a mix of :code:`ivy.Container` and :code:`ivy.NativeArray` input:

    >>> x = ivy.Container(a=ivy.native_array([-1]), b=ivy.native_array([3]))
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    {
        a: -1,
        b: 3
    }

    >>> print(ivy.is_int_dtype(y))
    {
        a: true,
        b: true
    }

    >>> x = ivy.Container(a=ivy.native_array([1]), b=ivy.native_array([0]),\
                          c=ivy.native_array([-1]))
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    {
        a: 1,
        b: 0,
        c: -1
    }

    Instance Method Examples
    ------------------------

    With :code:`ivy.Array` instance method:

    >>> x = ivy.array([-1])
    >>> y = x.to_scalar()
    >>> print(y)
    -1

    >>> print(ivy.is_int_dtype(y))
    True

    >>> x = ivy.array([3])
    >>> y = x.to_scalar()
    >>> print(y)
    3

    With a mix of :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([-1]), b=ivy.array([3]))
    >>> y = x.to_scalar()
    >>> print(y)
    {
        a: -1,
        b: 3
    }

    >>> print(ivy.is_int_dtype(y))
    {
        a: true,
        b: true
    }

    >>> x = ivy.Container(a=ivy.array([1]), b=ivy.array([0]),\
                          c=ivy.array([-1]))
    >>> y = x.to_scalar()
    >>> print(y)
    {
        a: 1,
        b: 0,
        c: -1
    }

    """
    return current_backend(x).to_scalar(x)


@inputs_to_native_arrays
@handle_nestable
def to_list(x: Union[ivy.Array, ivy.NativeArray]) -> List:
    """Creates a (possibly nested) list from input array.

    Parameters
    ----------
    x
        Input array.

    Returns
    -------
    ret
        A list representation of the input array ``x``.

    Functional Examples
    ------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [-1, 0, 1]

    >>> print(isinstance(y, list))
    True

    >>> x = ivy.array([[ 1.1,  2.2,  3.3], \
                       [-4.4, -5.5, -6.6]])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [[1.100000023841858,2.200000047683716,3.299999952316284],[-4.400000095367432,-5.5,-6.599999904632568]]

    >>> print(isinstance(y, list))
    True

    >>> x = ivy.array([[[-1,  0,  1],\
                        [ 1,  0, -1]], \
                       [[ 1, -1,  0], \
                        [ 1,  0, -1]]])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]

    >>> print(isinstance(y, list))
    True

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([-1, 0, 1])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [-1, 0, 1]

    >>> print(isinstance(y, list))
    True

    >>> x = ivy.native_array([[-1, 0, 1], \
                              [-1, 0, 1], \
                              [ 1, 0, -1]])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]

    >>> print(isinstance(y, list))
    True

    >>> x = ivy.native_array([[[-1, 0, 1], \
                               [1, 0, -1]], \
                              [[1, -1, 0], \
                               [1, 0, -1]]])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]

    >>> print(isinstance(y, list))
    True

    With a mix of :code:`ivy.Container` and :code:`ivy.Array` input:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [-1, 0, 1]
    }

    >>> x = ivy.Container(a=ivy.array([[-1, 0, 1], \
                                       [-1, 0, 1], \
                                       [1, 0, -1]]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [[-1, 0, 1], [-1, 0, 1], [1,0,-1]]
    }

    >>> x = \
    ivy.Container(a=ivy.array([[[-1, 0, 1],[1, 0, -1]],[[1, -1, 0],[1, 0, -1]]]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]
    }

    With a mix of :code:`ivy.Container` and :code:`ivy.NativeArray` input:

    >>> x = ivy.Container(a=ivy.native_array([-1, 0, 1]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [-1, 0, 1]
    }

    >>> x = ivy.Container(a=ivy.native_array([[-1, 0, 1],[-1, 0, 1],[1, 0, -1]]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]
    }

    >>> x =\
    ivy.Container(a=ivy.native_array([[[-1 ,0, 1],[1, 0 ,-1]],[[1, -1, 0],[1,0 ,-1]]]))
    >>> y = ivy.to_list(x)
    >>> print(y)
    {
        a: [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]
    }

    Instance Method Examples
    ------------------------

    With :code:`ivy.Array` instance method:

    >>> x = ivy.array([0, 1, 2])
    >>> y = x.to_list()
    >>> print(y)
    [0, 1, 2]

    With :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]))
    >>> y = x.to_list()
    >>> print(y)
    {a:[0,1,2]}

    """
    return current_backend(x).to_list(x)


@handle_nestable
def clip_vector_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    max_norm: float,
    p: float = 2.0,
    *,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Clips (limits) the vector p-norm of an array.

    Parameters
    ----------
    x
        array, input array containing elements to clip.
    max_norm
        float, the maximum value of the array norm.
    p
        optional float, the p-value for computing the p-norm. Default is 2.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the vector norm downscaled to the max norm if needed.

    Functional Examples
    ------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.clip_vector_norm(x, 2.0)
    >>> print(y)
    ivy.array([0.   , 0.894, 1.79 ])

    >>> x = ivy.array([0.5, -0.7, 2.4])
    >>> y = ivy.clip_vector_norm(x, 3.0, 1.0)
    >>> print(y)
    ivy.array([ 0.417, -0.583,  2.   ])

    >>> x = ivy.array([[[0., 0.], [1., 3.], [2., 6.]], \
                       [[3., 9.], [4., 12.], [5., 15.]]])
    >>> y = ivy.zeros(((2, 3, 2)))
    >>> ivy.clip_vector_norm(x, 4.0, 1.0, out=y)
    >>> print(y)
    ivy.array([[[0.    , 0.    ],
                [0.0667, 0.2   ],
                [0.133 , 0.4   ]],
               [[0.2   , 0.6   ],
                [0.267 , 0.8   ],
                [0.333 , 1.    ]]])

    >>> x = ivy.array([[1.1, 2.2, 3.3], \
                       [-4.4, -5.5, -6.6]])
    >>> ivy.clip_vector_norm(x, 1.0, 3.0, out=x)
    >>> print(x)
    ivy.array([[ 0.131,  0.263,  0.394],
               [-0.526, -0.657, -0.788]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.clip_vector_norm(x, 2.0)
    >>> print(y)
    ivy.array([0.   , 0.894, 1.79 ])

    >>> x = ivy.native_array([0.5, -0.7, 2.4])
    >>> y = ivy.clip_vector_norm(x, 3.0, 1.0)
    >>> print(y)
    ivy.array([ 0.417, -0.583,  2.   ])

    >>> x = ivy.native_array([[[0., 0.], [1., 3.], [2., 6.]], \
                              [[3., 9.], [4., 12.], [5., 15.]]])
    >>> y = ivy.zeros(((2, 3, 2)))
    >>> ivy.clip_vector_norm(x, 4.0, 1.0, out=y)
    >>> print(y)
    ivy.array([[[0.    , 0.    ],
                [0.0667, 0.2   ],
                [0.133 , 0.4   ]],
               [[0.2   , 0.6   ],
                [0.267 , 0.8   ],
                [0.333 , 1.    ]]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
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
        ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_native_arrays_and_back
@handle_nestable
def clip_matrix_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    max_norm: float,
    p: float = 2.0,
    *,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Clips (limits) the matrix norm of an array.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    max_norm
        The maximum value of the array norm.
    p
        The p-value for computing the p-norm. Default is 2.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the matrix norm downscaled to the max norm if needed.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([[0., 1., 2.]])
    >>> y = ivy.clip_matrix_norm(x, 2.0)
    >>> print(y)
    ivy.array([[0.   , 0.894, 1.79 ]])

    >>> x = ivy.array([[0.1, -1.2, 3.7], [0., 7.3, -0.5]])
    >>> y = ivy.clip_matrix_norm(x, 3.0, 1.0)
    >>> print(y)
    ivy.array([[ 0.0353, -0.424 ,  1.31  ],
               [ 0.    ,  2.58  , -0.176 ]])

    >>> x = ivy.array([[[5., 4.], [-2., 6.]], \
                       [[3., 7.], [0., -5.]]])
    >>> y = ivy.empty((2, 2, 2))
    >>> ivy.clip_matrix_norm(x, 0.5, 2.0, out=y)
    >>> print(y)
    ivy.array([[[ 0.339,  0.271],
                [-0.135,  0.406]],
               [[ 0.168,  0.391],
                [ 0.   , -0.279]]])

    >>> x = ivy.array([[0., 1.], \
                       [2., 3.]])
    >>> ivy.clip_matrix_norm(x, 5.0, 1.0, out=x)
    >>> print(x)
    ivy.array([[0., 1.],
               [2., 3.]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[0., 1., 2.]])
    >>> y = ivy.clip_matrix_norm(x, 2.0)
    >>> print(y)
    ivy.array([[0.   , 0.894, 1.79 ]])

    >>> x = ivy.native_array([[0.1, -1.2, 3.7], [0., 7.3, -0.5]])
    >>> y = ivy.clip_matrix_norm(x, 3.0, 1.0)
    >>> print(y)
    ivy.array([[ 0.0353, -0.424 ,  1.31  ],
               [ 0.    ,  2.58  , -0.176 ]])

    >>> x = ivy.native_array([[[5., 4.], [-2., 6.]], \
                       [[3., 7.], [0., -5.]]])
    >>> y = ivy.empty((2, 2, 2))
    >>> ivy.clip_matrix_norm(x, 0.5, 2.0, out=y)
    >>> print(y)
    ivy.array([[[ 0.339,  0.271],
                [-0.135,  0.406]],
               [[ 0.168,  0.391],
                [ 0.   , -0.279]]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]), \
                          b=ivy.array([[3., 4., 5.]]))
    >>> y = ivy.clip_matrix_norm(x, 2.0)
    >>> print(y)
    {
        a: ivy.array([[0., 0.894, 1.79]]),
        b: ivy.array([[0.849, 1.13, 1.41]])
    }
    """
    norms = ivy.matrix_norm(x, p, keepdims=True)
    ratios = ivy.minimum(ivy.stable_divide(max_norm, norms), 1.0)
    return ivy.multiply(ratios, x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def floormod(
    x: Union[ivy.Array, ivy.NativeArray],
    y: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns element-wise remainder of division.

    Parameters
    ----------
    x
        array, input to floormod
    y
        array, denominator input for floormod.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array of the same shape and type as x, with the elements floor modded.

    """
    return current_backend(x).floormod(x, y, out=out)


@to_native_arrays_and_back
@handle_nestable
def unstack(
    x: Union[ivy.Array, ivy.NativeArray], axis: int, keepdims: bool = False
) -> Union[ivy.Array, ivy.NativeArray]:
    """Unpacks the given dimension of a rank-R array into rank-(R-1) arrays.

    Parameters
    ----------
    x
        Input array to unstack.
    axis
        Axis for which to unpack the array.
    keepdims
        Whether to keep dimension 1 in the unstack dimensions. Default is False.

    Returns
    -------
    ret
        List of arrays, unpacked along specified dimensions.

    """
    return current_backend(x).unstack(x, axis, keepdims)


@to_native_arrays_and_back
@handle_nestable
def fourier_encode(
    x: Union[ivy.Array, ivy.NativeArray],
    max_freq: Union[float, ivy.Array, ivy.NativeArray],
    num_bands: int = 4,
    linear: bool = False,
    concat: bool = True,
    flatten: bool = False,
) -> Union[ivy.Array, ivy.NativeArray, Tuple]:
    """Pads an array with fourier encodings.

    Parameters
    ----------
    x
        Input array to encode.
    max_freq
        The maximum frequency of the encoding.
    num_bands
        The number of frequency bands for the encoding. Default is 4.
    linear
        Whether to space the frequency bands linearly as opposed to geometrically.
        Default is False.
    concat
        Whether to concatenate the position, sin and cos values, or return seperately.
        Default is True.
    flatten
        Whether to flatten the position dimension into the batch dimension. Default is
        False.

    Returns
    -------
    ret
        New array with the final dimension expanded, and the encodings stored in this
        channel.

    """
    x_in = x
    dim = x.shape[-1]
    x = ivy.expand_dims(x, -1)
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
        return ivy.concat([orig_x, sin_x, cos_x], -1)
    return sin_x, cos_x


@inputs_to_native_arrays
@handle_nestable
def value_is_nan(
    x: Union[ivy.Array, ivy.NativeArray, Number], include_infs: bool = True
) -> bool:
    """Determine whether the single valued array or scalar is of nan type.

    Parameters
    ----------
    x
        The input to check Input array.
    include_infs
        Whether to include infs and -infs in the check. Default is True.

    Returns
    -------
    ret
        Boolean as to whether the input value is a nan or not.

    """
    x_scalar = ivy.to_scalar(x) if ivy.is_native_array(x) else x
    if not x_scalar == x_scalar:
        return True
    if include_infs and x_scalar == INF or x_scalar == -INF:
        return True
    return False


@inputs_to_native_arrays
@handle_nestable
def has_nans(x: Union[ivy.Array, ivy.NativeArray], include_infs: bool = True) -> bool:
    """Determine whether the array contains any nans, as well as infs or -infs if
    specified.

    Parameters
    ----------
    x
        Input array.
    include_infs
        Whether to include ``+infinity`` and ``-infinity`` in the check. Default is True.

    Returns
    -------
    ret
        Boolean as to whether the array contains nans.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` input:

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

    With :code: `ivy.NativeArray` input:

    >>> x = ivy.native_array([1, 2, 3, float('nan')])
    >>> y = ivy.has_nans(x)
    >>> print(y)
    True

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.has_nans(x)
    >>> print(y)
    {
        a: false,
        b: false
    }

    With one :code:`ivy.Container` static method:
    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),\
                          b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.Container.static_has_nans(x)
    >>> print(y)
    {
        a: false,
        b: false
    }

     With one :code:`ivy.Array` instance method:
    >>> x = ivy.array([-1, 0, 1])
    >>> y = x.has_nans()
    >>> print(y)
    False

    With :code:`ivy.Container` instance method:
    >>> x = ivy.Container(a=ivy.array([1, 0, 1]),\
                          b=ivy.array([-1, 0, 1, 1]))
    >>> y = x.has_nans()
    >>> print(y)
    {
        a: false,
        b: false
    }

    """
    return ivy.value_is_nan(ivy.sum(x), include_infs)


def exists(x: Any) -> bool:
    """Simple check as to whether the input is None or not.

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

    >>> x = ivy.native_array([1, 2, 3, 1.2])
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    >>> x = ivy.array([1, 2, 3, 1.2])
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    With a mix of :code:`ivy.Container` and :code:`Any` input:

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

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.native_array([1, 0, 1.2]))
    >>> y = ivy.exists(x)
    >>> print(y)
    True

    """
    return x is not None


def default(
    x: Any,
    default_val: Any,
    catch_exceptions: bool = False,
    rev: bool = False,
    with_callable: bool = False,
) -> Any:
    """Returns x provided it exists (is not None), else returns default value.

    Parameters
    ----------
    x
        Input which may or may not exist (be None).
    default_val
        The default value.
    catch_exceptions
        Whether to catch exceptions from callable x. Default is False.
    rev
        Whether to reverse the input x and default_val. Default is False.
    with_callable
        Whether either of the arguments might be callable functions. Default is False.

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
    >>> y = ivy.default(x, lambda: ivy.array([1, 2, 3]), with_callable=True, catch_exceptions=True)
    >>> print(y)
    ivy.array([1, 2, 3])

    >>> x = lambda a, b: a + b
    >>> y = ivy.default(x, lambda: ivy.array([1, 2, 3]), with_callable=True, catch_exceptions=True, rev=True)
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


def to_ivy_shape(shape: Union[ivy.Shape, ivy.NativeShape]) -> ivy.Shape:
    """Returns the input shape in ivy.Shape form

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


def to_native_shape(shape: Union[ivy.Shape, ivy.NativeShape]) -> ivy.NativeShape:
    """Returns the input shape in its native backend framework form

    Parameters
    ----------
    shape
        The input to be converted

    Returns
    -------
     ret
        the input in its native framework form

    """
    if isinstance(shape, ivy.NativeShape):
        return shape
    assert isinstance(shape, (int, list, tuple))
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, list):
        shape = tuple(shape)
    assert builtins.all([isinstance(v, int) for v in shape])
    return ivy.NativeShape(shape)


@handle_nestable
def try_else_none(fn: Callable, *args: Any, **kwargs: Any) -> Union[Callable, None]:
    """Try and return the function, otherwise return None
        if an exception was raised during function execution.

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
    with: if the function is executed without any exception
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = ivy.try_else_none(ivy.add,x, y)
    >>> print(z.__name__)
    add

    with: if the function is executed with an exception
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


def arg_names(receiver):
    """
    Gets the expected keyword arguments for a function or class constructor.

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


def match_kwargs(kwargs, *receivers, allow_duplicates=False):
    """Match keyword arguments to either class or function receivers.

    Parameters
    ----------
    kwargs
        Keyword arguments to match.
    receivers
        Functions and/or classes to match the keyword arguments to.
    allow_duplicates
        Whether to allow one keyword argument to be used for multiple receivers.
        Default is False.

    Returns
    -------
    ret
        Sequence of keyword arguments split as best as possible.

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


def cache_fn(func: Callable) -> Callable:
    """Wrap a function, such that when cache=True is passed as an argument, a previously
    cached output is returned.

    Parameters
    ----------
    func
        The function to wrap, whose output should be cached for later.

    Returns
    -------
    ret
        The newly cache wrapped function.

    """
    global FN_CACHE
    if func not in FN_CACHE:
        FN_CACHE[func] = dict()

    def cached_fn(*args, **kwargs):
        """Summary.

        Parameters
        ----------
        *args

        **kwargs

        """
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


def current_backend_str() -> Union[str, None]:
    """Summary.

    Returns
    -------
    ret
        The framework string.

    """
    fw = current_backend()
    if not backend_stack:
        return ""
    return fw.current_backend_str()


@to_native_arrays_and_back
@handle_nestable
def einops_rearrange(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    *,
    out: Optional[ivy.Array] = None,
    **axes_lengths: Dict[str, int],
) -> ivy.Array:
    """Perform einops rearrange operation on input array x.

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

    """
    ret = einops.rearrange(x, pattern, **axes_lengths)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_native_arrays_and_back
@handle_nestable
def einops_reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    reduction: Union[str, Callable],
    *,
    out: Optional[ivy.Array] = None,
    **axes_lengths: Dict[str, int],
) -> ivy.Array:
    """Perform einops reduce operation on input array x.

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

    """
    ret = einops.reduce(x, pattern, reduction, **axes_lengths)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_native_arrays_and_back
@handle_nestable
def einops_repeat(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    *,
    out: Optional[ivy.Array] = None,
    **axes_lengths: Dict[str, int],
) -> Union[ivy.Array, ivy.NativeArray]:
    """Perform einops repeat operation on input array x.

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

    """
    ret = einops.repeat(x, pattern, **axes_lengths)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def get_min_denominator() -> float:
    """Get the global minimum denominator used by ivy for numerically stable division.

    Returns
    -------
    ret
        A float number of the global minimum denominator.

    Examples
    --------
    >>> x = ivy.get_min_denominator()
    >>> print(x)
    1e-12

    """
    return ivy._MIN_DENOMINATOR


def set_min_denominator(val: float) -> None:
    """Set the global minimum denominator used by ivy for numerically stable division.

    Parameters
    ----------
    val
        The new value to set the minimum denominator to.

    """
    ivy._MIN_DENOMINATOR = val


def get_min_base() -> float:
    """
    Gets the global minimum base used by ivy for numerically stable power raising.

    Returns
    -------
    ret
        Global minimum base number

    Examples
    --------
    >>> x = ivy.get_min_base()
    >>> print(x)
    1e-05

    """
    # noinspection PyProtectedMember
    return ivy._MIN_BASE


def set_min_base(val: float) -> None:
    """Set the global minimum base used by ivy for numerically stable power raising.

    Parameters
    ----------
    val
        The new value to set the minimum base to.

    """
    ivy._MIN_BASE = val


def stable_divide(
    numerator: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container],
    denominator: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container],
    min_denominator: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container] = None,
) -> Union[Number, ivy.Array, ivy.NativeArray, ivy.Container]:
    """Divide the numerator by the denominator, with min denominator added to the
    denominator for numerical stability.

    Parameters
    ----------
    numerator
        The numerator of the division.
    denominator
        The denominator of the division.
    min_denominator
        The minimum denominator to use, use global ivy._MIN_DENOMINATOR by default.

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

    With :code:`float` input:
    >>> x = ivy.stable_divide(5.0, 3.33)
    >>> print(x)
    1.5015015015010504

    With :code:`complex` input:
    >>> x = ivy.stable_divide(1+1j, 1-1j)
    >>> print(x)
    (5.000444502911705e-13+0.9999999999995j)

    With :code:`ivy.Array` input:
    >>> x = ivy.asarray([[10., 20., 30.],\
                        [40., 50., 60.]])
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

    With :code:`ivy.Container` input
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
    # noinspection PyProtectedMember
    return numerator / (denominator + default(min_denominator, ivy._MIN_DENOMINATOR))


def stable_pow(base: Any, exponent: Any, min_base: float = None) -> Any:
    """Raise the base by the power, with MIN_BASE added to the base when exponent > 1
    for numerical stability.

    Parameters
    ----------
    base
        The numerator of the division.
    exponent
        The denominator of the division.
    min_base
        The minimum base to use, use global ivy._MIN_BASE by default.

    Returns
    -------
    ret
        The new item following the numerically stable division.


    """
    # noinspection PyProtectedMember
    return (base + default(min_base, ivy._MIN_BASE)) ** exponent


def get_all_arrays_in_memory():
    """Gets all arrays which are currently alive."""
    all_arrays = list()
    for obj in gc.get_objects():
        # noinspection PyBroadException
        try:
            if ivy.is_native_array(obj):
                all_arrays.append(obj)
        except Exception:
            pass
    return all_arrays


def num_arrays_in_memory():
    """Returns the number of arrays which are currently alive."""
    return len(get_all_arrays_in_memory())


def print_all_arrays_in_memory():
    """Prints all arrays which are currently alive."""
    for arr in get_all_arrays_in_memory():
        print(type(arr), arr.shape)


def set_queue_timeout(timeout):
    """
    Set the global queue timeout value (in seconds)
    Default value without this function being called is 15 seconds.

    Parameters
    ----------
    timeout
        The timeout when waiting for containers to arrive from the queues.
        To be set in seconds.


    Examples
    --------
    >> x = ivy.queue_timeout()
    >> print(x)
    15.0

    To set the timeout for example 30 seconds

    >> ivy.set_queue_timeout(30)
    >> y = ivy.queue_timeout()
    >> print(y)
    30

    """
    global TIMEOUT
    TIMEOUT = timeout


def queue_timeout():
    """Get the global queue timeout values (in seconds).

    Default value without this function being called is 10 seconds.

    """
    global TIMEOUT
    return TIMEOUT


def tmp_dir():
    """Get the path for directory that saves temporary files.

    Returns
    -------
    ret
        The path of directory that saves temporary files.

    """
    return TMP_DIR


def set_tmp_dir(tmp_dr):
    """Set the directory for saving temporary files.

    Parameters
    ----------
    tmp_dr

    """
    global TMP_DIR
    TMP_DIR = tmp_dr


def container_types():
    """Summary.

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


def inplace_arrays_supported(f=None):
    """Determine whether inplace arrays are supported for the current backend framework.

    Parameters
    ----------
    f
         (Default value = None)

    Returns
    -------
    ret
        Boolean, whether or not inplace arrays are supported.

    """
    return current_backend().inplace_arrays_supported()


def inplace_variables_supported(f=None):
    """Determine whether inplace variables are supported for the current backend
    framework.

    Parameters
    ----------
    f
         (Default value = None)

    Returns
    -------
    ret
        Boolean, whether or not inplace variables are supported.

    """
    return current_backend().inplace_variables_supported()


@inputs_to_native_arrays
@handle_nestable
def supports_inplace(x):
    """Determine whether inplace operations are supported for the data type of x.

    Parameters
    ----------
    x
        Input variable or array to check for inplace support for.

    Returns
    -------
    ret
        Boolean, whether or not inplace operations are supported for x.

    """
    if ivy.is_variable(x):
        return ivy.inplace_variables_supported()
    elif ivy.is_native_array(x):
        return ivy.inplace_arrays_supported()
    raise Exception("Input x must be either a variable or an array.")


@inputs_to_native_arrays
@handle_nestable
def assert_supports_inplace(x):
    """Asserts that inplace operations are supported for x, else raises exception.

    Parameters
    ----------
    x
        Input variable or array to check for inplace support for.

    Returns
    -------
    ret
        True if support, raises exception otherwise

    """
    if not ivy.supports_inplace(x):
        raise Exception(
            "Inplace operations are not supported {} types with {} backend".format(
                type(x), ivy.current_backend_str()
            )
        )
    return True


@handle_nestable
def inplace_update(
    x: Union[ivy.Array, ivy.NativeArray],
    val: Union[ivy.Array, ivy.NativeArray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    """Perform in-place update for the input array. This will always be performed on
    ivy.Array instances pass in the input, and will also be performed on the native
    array classes in the backend when the backend supports this. If the backend does
    not natively support inplace updates, and x is an ivy.NativeArray instance,
    then an exception will be thrown.

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

    Returns
    -------
    ret
        The array following the in-place update.

    """
    return current_backend(x).inplace_update(x, val, ensure_in_backend)


@handle_nestable
def inplace_decrement(
    x: Union[ivy.Array, ivy.NativeArray],
    val: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Perform in-place decrement for the input array.

    Parameters
    ----------
    x
        The array to decrement.
    val
        The array to decrement the variable with.

    Returns
    -------
    ret
        The array following the in-place decrement.

    """
    return current_backend(x).inplace_decrement(x, val)


@handle_nestable
def inplace_increment(
    x: Union[ivy.Array, ivy.NativeArray],
    val: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Perform in-place increment for the input array.

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
    With :code:`ivy.Array` input:
    >>> x = ivy.array([[5.3, 7., 0.],\
                        [6.8, 8, 3.9],\
                        [0., 10., 6.3]])
    >>> y = ivy.inplace_increment(x, 3.)
    >>> print(y)
    ivy.array([[ 8.3, 10.,  3.],
       [ 9.8, 11.,  6.9],
       [ 3., 13.,  9.3]])

     With :code:`ivy.NativeArray` input:
     >>> x = ivy.native_array([10, 20, 30])
     >>> val = ivy.native_array([1, 2, 3])
     >>> y = ivy.inplace_increment(x, val)
     >>> print(y)
     ivy.array([11, 22, 33])

    With :code:`ivy.Container` input
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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def cumsum(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        int, Axis along which the cumulative sum is computed. By default 0.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Input array with cumulatively summed elements along axis

    """
    return current_backend(x).cumsum(x, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def cumprod(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    exclusive: Optional[bool] = False,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns the cumulative product of the elements along a given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        int , axis along which the cumulative product is computed. By default 0.
    exclusive
        optional bool, Whether to perform the cumprod exclusively. Defaults is False.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Input array with cumulatively multiplied elements along axis.

    Functional Examples
    --------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 4])
    >>> y = ivy.cumprod(x)
    >>> print(y)
    ivy.array([2, 6, 24])

    >>> x = ivy.array([2, 3, 4])
    >>> exclusive = True
    >>> y = ivy.cumprod(x, exclusive=exclusive)
    >>> print(y)
    ivy.array([1, 2, 6])

    Example specifying axes

    >>> x = ivy.array([[2, 3], \
                       [5, 7], \
                       [11, 13]])
    >>> exclusive = True
    >>> y = ivy.zeros((3, 2))
    >>> ivy.cumprod(x, axis=1, exclusive=exclusive, out=y)
    >>> print(y)
    ivy.array([[1.,2.],[1.,5.],[1.,11.]])

    >>> x = ivy.array([[2, 3],[5, 7],[11, 13]])
    >>> exclusive = True
    >>> ivy.cumprod(x, axis=0, exclusive=exclusive, out=x)
    >>> print(x)
    ivy.array([[1,  1],
               [2,  3],
               [10, 21]])


     With :code:`ivy.NativeArray` input:

     >>> x = ivy.native_array([2, 3, 4])
     >>> y = ivy.cumprod(x)
     >>> print(y)
     ivy.array([2, 6, 24])


     With :code:`ivy.Container` input:
     >>> x = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([3, 4, 5]))
     >>> y = ivy.cumprod(x)
     >>> print(y)
     {
         a: ivy.array([2, 6, 24]),
         b: ivy.array([3, 12, 60])
     }

    """
    return current_backend(x).cumprod(x, axis, exclusive, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def scatter_flat(
    indices: Union[ivy.Array, ivy.NativeArray],
    updates: Union[ivy.Array, ivy.NativeArray],
    size: Optional[int] = None,
    tensor: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    reduction: str = "sum",
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Scatter flat updates into a new flat array according to flat indices.

    Parameters
    ----------
    indices
        Indices for the new values to occupy.
    updates
        Values for the new array to hold.
    size
        The size of the result.
    tensor
        The tensor in which to scatter the results, default is None, in which case the
        size is used to
        scatter into a zeros array.
    reduction
        The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as
        updates if None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    """
    return current_backend(indices).scatter_flat(
        indices, updates, size, tensor, reduction, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def scatter_nd(
    indices: Union[ivy.Array, ivy.NativeArray],
    updates: Union[ivy.Array, ivy.NativeArray],
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    tensor: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    reduction: str = "sum",
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Scatter updates into a new array according to indices.

    Parameters
    ----------
    indices
        Indices for the new values to occupy.
    updates
        Values for the new array to hold.
    shape
        The shape of the result. Default is None, in which case tensor argument must be
        provided.
    tensor
        The tensor in which to scatter the results, default is None, in which case the
        shape arg is used to
        scatter into a zeros array.
    reduction
        The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as
        updates if None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    """
    return current_backend(indices).scatter_nd(
        indices, updates, shape, tensor, reduction, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def gather(
    params: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    axis: int = -1,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Gather slices from params at axis according to indices.

    Parameters
    ----------
    params
        array, the array from which to gather values.
    indices
        array, index array.
    axis
        optional int, the axis from which to gather from. Default is -1.
    device
        optional ivy.Device, device on which to create the array 'cuda:0', 'cuda:1',
        'cpu' etc. Same as x if None.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        New array with the values gathered at the specified indices along the specified
        axis.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.array([0, 1])
    >>> print(ivy.gather(x, y))
    ivy.array([0., 1.])

    >>> x = ivy.array([[0., 1., 2.], \
                        [3., 4., 5.]])
    >>> y = ivy.array([[0, 1], \
                        [1, 2]])
    >>> z = ivy.array([[0., 0.], \
                        [0., 0.]])
    >>> ivy.gather(x, y, out=z)
    >>> print(z)
    ivy.array([[0., 1.],
               [4., 5.]])

    >>> x = ivy.array([[[0., 1.], [2., 3.]], \
                        [[4., 5.], [6., 7.]], \
                        [[8., 9.], [10., 11.]]])
    >>> y = ivy.array([[[0, 1]], \
                        [[1, 2]], \
                        [[2, 0]]])
    >>> ivy.gather(x, y, axis=0, out=x)
    >>> print(x)
    ivy.array([[[0.,5.]],[[4.,9.]],[[8.,1.]]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.native_array([0, 1])
    >>> print(ivy.gather(x, y))
    ivy.array([0., 1.])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.array([0, 1])
    >>> print(ivy.gather(x, y))
    ivy.array([0., 1.])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.array([0., 1., 2.]), \
                          b = ivy.array([4., 5., 6.]))
    >>> y = ivy.array([0, 1])
    >>> print(ivy.gather(x, y))
    {
        a: ivy.array([0., 1.]),
        b: ivy.array([4., 5.])
    }

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array([0., 1., 2.]), \
                          b = ivy.array([4., 5., 6.]))
    >>> y = ivy.Container(a = ivy.array([0, 1]), \
                          b = ivy.array([1, 2]))
    >>> print(ivy.gather(x, y))
    {
        a: ivy.array([0., 1.]),
        b: ivy.array([5., 6.])
    }

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.array([0, 1])
    >>> print(x.gather(y))
    ivy.array([0., 1.])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a = ivy.array([0., 1., 2.]), \
                          b = ivy.array([4., 5., 6.]))
    >>> y = ivy.Container(a = ivy.array([0, 1]), \
                          b = ivy.array([1, 2]))
    >>> print(x.gather(y))
    {
        a: ivy.array([0., 1.]),
        b: ivy.array([5., 6.])
    }
    """
    return current_backend(params).gather(params, indices, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def gather_nd(
    params: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Gather slices from params into a array with shape specified by indices.

    Parameters
    ----------
    params
        The array from which to gather values.
    indices
        Index array.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values gathered at the indices.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6.])
    >>> y = ivy.array([1])
    >>> print(ivy.gather_nd(x, y))
    ivy.array(1.)

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.native_array([1])
    >>> print(ivy.gather_nd(x, y))
    ivy.array(1.)

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.array([1])
    >>> print(ivy.gather_nd(x, y))
    ivy.array(1.)

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([4., 5., 6.]))
    >>> y = ivy.array([1])
    >>> print(ivy.gather_nd(x, y))
    {
        a: ivy.array(1.),
        b: ivy.array(5.)
    }

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([4., 5., 6.]))
    >>> y = ivy.Container(a=ivy.array([0]), \
                          b=ivy.array([2]))
    >>> print(ivy.gather_nd(x, y))
    {
        a: ivy.array(0.),
        b: ivy.array(6.)
    }
    """
    res = current_backend(params, indices).gather_nd(params, indices)
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res


@handle_nestable
def multiprocessing(context: str = None):
    """Return backend-specific multiprocessing module.

    Parameters
    ----------
    context
        The context of the multiprocessing, either fork, forkserver or spawn.
        Default is None.

    Returns
    -------
    ret
        Multiprocessing module

    """
    return current_backend().multiprocessing(context)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def indices_where(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns indices or true elements in an input boolean array.

    Parameters
    ----------
    x
        Boolean array, for which indices are desired.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Indices for where the boolean array is True.

    """
    return current_backend(x).indices_where(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@handle_nestable
def one_hot(
    indices: Union[ivy.Array, ivy.NativeArray],
    depth: int,
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns a one-hot array.

    Parameters
    ----------
    indices
        Indices for where the ones should be scattered *[batch_shape, dim]*
    depth
        Scalar defining the depth of the one-hot dimension.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of zeros with the same shape and type as a, unless dtype provided which
        overrides.

    """
    return current_backend(indices).one_hot(indices, depth, device=device, out=out)


@to_native_arrays_and_back
@handle_nestable
def shape(
    x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False
) -> Union[ivy.Shape, ivy.NativeShape]:
    """Returns the shape of the array ``x``.

    Parameters
    ----------
    x
        Input array to infer the shape of.
    as_array
        Whether to return the shape as a array, default False.

    Returns
    -------
    ret
        Shape of the array ``x``.

    Examples
    --------
    >>> x = ivy.array([[-1, 0, 1],[1, 0, -1]])
    >>> y = ivy.shape(x)
    >>> z = ivy.shape(x, as_array = True)
    >>> print(y)
    (2, 3)

    >>> print(z)
    ivy.array([2, 3])

    """
    return current_backend(x).shape(x, as_array)


def set_shape_array_mode(mode: bool) -> None:
    """Set the mode of returning shape as ivy.Array to the given mode instance

    Parameter
    ---------
    mode
        boolean whether to return shape as ivy.Array

    Examples
    --------
    >>> ivy.set_shape_array_mode(False)
    >>> ivy.shape_array_mode()
    False

    >>> ivy.set_shape_array_mode(True)
    >>> ivy.shape_array_mode()
    True
    """
    global shape_array_mode_stack
    if not isinstance(mode, bool):
        raise Exception("set_shape_array_mode only accepts type bool")
    shape_array_mode_stack.append(mode)


def unset_shape_array_mode() -> None:
    """Reset the mode of returning shape as ivy.Array to the previous state

    Examples
    --------
    >>> ivy.set_shape_array_mode(True)
    >>> ivy.shape_array_mode()
    True

    >>> ivy.unset_shape_array_mode()
    >>> ivy.shape_array_mode()
    False
    """
    global shape_array_mode_stack
    if shape_array_mode_stack:
        shape_array_mode_stack.pop(-1)


def shape_array_mode() -> bool:
    """Get the current state of shape_array_mode

    Examples
    --------
    >>> ivy.shape_array_mode()
    False

    >>> ivy.set_shape_array_mode(True)
    >>> ivy.shape_array_mode()
    True
    """
    global shape_array_mode_stack
    if not shape_array_mode_stack:
        return False
    return shape_array_mode_stack[-1]


@to_native_arrays_and_back
@handle_nestable
def get_num_dims(x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False) -> int:
    """Returns the number of dimensions of the array x.

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

    """
    return current_backend(x).get_num_dims(x, as_array)


def arg_info(fn: Callable, *, name: str = None, idx: int = None):
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
    if (not ivy.exists(name) and not ivy.exists(idx)) or (
        ivy.exists(name) and ivy.exists(idx)
    ):
        raise Exception(
            "exactly one of the keyword arguments name or idx " "must be provided"
        )
    params = inspect.signature(fn).parameters
    if ivy.exists(name):
        return {"idx": list(params).index(name), "param": params[name]}
    return {"idx": idx, "param": list(params.values())[idx]}


def _is_valid_device_and_dtypes_attributes(fn: Callable) -> bool:
    if hasattr(fn, "unsupported_device_and_dtype") and hasattr(
        fn, "supported_device_and_dtype"
    ):
        fn_unsupported_device_and_dtype = fn.unsupported_device_and_dtype
        fn_supported_device_and_dtype = fn.supported_device_and_dtype
        if isinstance(fn_unsupported_device_and_dtype, dict):
            if isinstance(fn_supported_device_and_dtype, dict):
                backend_str = ivy.current_backend_str()
                if (
                    backend_str in fn_unsupported_device_and_dtype
                    and backend_str in fn_supported_device_and_dtype
                ):
                    return False
                elif (
                    "devices" in fn_unsupported_device_and_dtype
                    and "devices" in fn_supported_device_and_dtype
                ):
                    return False
    return True


@handle_nestable
def function_unsupported_devices_and_dtypes(fn: Callable) -> Dict:
    """Returns the unsupported combination of devices and dtypes
     of the current backend's function.

    Parameters
    ----------
    fn
        The function to check for the unsupported device and dtype attribute

    Returns
    -------
    ret
        The unsupported combination of devices and dtypes of the function
    """
    if not _is_valid_device_and_dtypes_attributes(fn):
        raise Exception(
            "supported_device_and_dtypes and unsupported_device_and_dtypes \
             attributes cannot both exist in a particular backend"
        )

    unsupported_devices_dtype = {"devices": (), "dtypes": ()}
    if hasattr(fn, "unsupported_device_and_dtype"):
        fn_unsupported_devices_dtypes = fn.unsupported_device_and_dtype
        if isinstance(fn_unsupported_devices_dtypes, dict):
            backend_str = ivy.current_backend_str()
            if backend_str in fn_unsupported_devices_dtypes:
                fn_unsupported_devices_dtypes = fn_unsupported_devices_dtypes[
                    backend_str
                ]

            elif "devices" not in fn_unsupported_devices_dtypes:
                return unsupported_devices_dtype

            keys = list(fn_unsupported_devices_dtypes.keys())
            if "dtypes" in keys and "devices" in keys:
                unsupported_devices_dtype["devices"] += fn_unsupported_devices_dtypes[
                    "devices"
                ]

                if isinstance(fn_unsupported_devices_dtypes["dtypes"][0], tuple):
                    for dtypes in fn_unsupported_devices_dtypes["dtypes"]:
                        unsupported_devices_dtype["dtypes"] += (dtypes,)
                else:
                    unsupported_devices_dtype["dtypes"] += (
                        fn_unsupported_devices_dtypes["dtypes"],
                    )
            else:
                raise Exception(
                    "'unsupported_device_and_dtype' attr must have keys \
                     'devices' and 'dtypes'"
                )
        else:
            raise Exception(
                "Have to provide a dictionary to 'unsupported_device_and_dtype' attr \
                 with keys 'devices' and 'dtypes'"
            )
    return unsupported_devices_dtype


@handle_nestable
def function_supported_devices_and_dtypes(fn: Callable) -> Dict:
    """Returns the supported combination of devices and dtypes
     of the current backend's function.

    Parameters
    ----------
    fn
        The function to check for the supported device and dtype attribute

    Returns
    -------
    ret
        The unsupported devices of the function
    """
    if not _is_valid_device_and_dtypes_attributes(fn):
        raise Exception(
            "supported_device_and_dtypes and unsupported_device_and_dtypes \
             attributes cannot both exist in a particular backend"
        )

    supported_devices_dtype = {"devices": (), "dtypes": ()}
    if hasattr(fn, "supported_device_and_dtype"):
        fn_supported_devices_dtypes = fn.supported_device_and_dtype
        if isinstance(fn_supported_devices_dtypes, dict):
            backend_str = ivy.current_backend_str()
            if backend_str in fn_supported_devices_dtypes:
                fn_supported_devices_dtypes = fn_supported_devices_dtypes[backend_str]
            elif "devices" not in fn_supported_devices_dtypes:
                return supported_devices_dtype
            keys = list(fn_supported_devices_dtypes.keys())
            if "dtypes" in keys and "devices" in keys:
                supported_devices_dtype["devices"] += fn_supported_devices_dtypes[
                    "devices"
                ]

                if isinstance(fn_supported_devices_dtypes["dtypes"][0], tuple):
                    for dtypes in fn_supported_devices_dtypes["dtypes"]:
                        supported_devices_dtype["dtypes"] += dtypes
                else:
                    supported_devices_dtype["dtypes"] += (
                        fn_supported_devices_dtypes["dtypes"],
                    )
            else:
                raise Exception(
                    "'supported_device_and_dtype' attr must have keys \
                     'devices' and 'dtypes'"
                )
        else:
            raise Exception(
                "Have to provide a dictionary to 'supported_device_and_dtype' attr \
                 with keys 'devices' and 'dtypes'"
            )
    return supported_devices_dtype
