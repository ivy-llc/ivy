"""
Collection of general Ivy functions.
"""

# global
import gc
import math
import einops
import inspect
import numpy as np
from numbers import Number
from typing import Callable, Any, Union, List, Tuple, Dict, Iterable, Optional

# local
import ivy
from ivy.functional.ivy.device import dev
from ivy.framework_handler import current_framework as _cur_framework

FN_CACHE = dict()
INF = float('inf')
TIMEOUT = 15.0
TMP_DIR = '/tmp'

def get_referrers_recursive(item, depth=0, max_depth=None, seen_set=None, local_set=None):
    seen_set = ivy.default(seen_set, set())
    local_set = ivy.default(local_set, set())
    ret_cont = ivy.Container(
        repr=str(item).replace(' ', ''), alphabetical_keys=False, keyword_color_dict={'repr': 'magenta'})
    referrers = [ref for ref in gc.get_referrers(item) if
                 not (isinstance(ref, dict) and
                      min([k in ref for k in ['depth', 'max_depth', 'seen_set', 'local_set']]))]
    local_set.add(str(id(referrers)))
    for ref in referrers:
        ref_id = str(id(ref))
        if ref_id in local_set or hasattr(ref, 'cell_contents'):
            continue
        seen = ref_id in seen_set
        seen_set.add(ref_id)
        refs_rec = lambda: get_referrers_recursive(ref, depth + 1, max_depth, seen_set, local_set)
        this_repr = 'tracked' if seen else str(ref).replace(' ', '')
        if not seen and (not max_depth or depth < max_depth):
            val = ivy.Container(
                repr=this_repr, alphabetical_keys=False, keyword_color_dict={'repr': 'magenta'})
            refs = refs_rec()
            for k, v in refs.items():
                val[k] = v
        else:
            val = this_repr
        ret_cont[str(ref_id)] = val
    return ret_cont


def is_array(x: Any, exclusive: bool = False)\
        -> bool:
    """
    Determines whether the input x is an Ivy Array.

    :param x: The input to check
    :type x: any
    :param exclusive: Whether to check if the data type is exclusively an array, rather than a variable or traced array.
    :type exclusive: bool, optional
    :return: Boolean, whether or not x is an array.
    """
    try:
        return _cur_framework(x).is_array(x, exclusive)
    except ValueError:
        return False


# noinspection PyShadowingNames
def copy_array(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Copy an array.

    :param x: The array to copy
    :type x: array
    :return: A copy of the input array.
    """
    return _cur_framework(x).copy_array(x)


def array_equal(x0: Union[ivy.Array, ivy.NativeArray], x1: Union[ivy.Array, ivy.NativeArray])\
        -> bool:
    """
    Determines whether two input arrays are equal across all elements.

    :param x0: The first input array to compare.
    :type x0: array
    :param x1: The second input array to compare.
    :type x1: array
    :return: Boolean, whether or not the input arrays are equal across all elements.
    """
    return _cur_framework(x0).array_equal(x0, x1)


def arrays_equal(xs: List[Union[ivy.Array, ivy.NativeArray]])\
        -> bool:
    """
    Determines whether input arrays are equal across all elements.

    :param xs: Sequence of arrays to compare for equality
    :type xs: sequence of arrays
    :return: Boolean, whether or not all of the input arrays are equal across all elements.
    """
    x0 = xs[0]
    for x in xs[1:]:
        if not array_equal(x0, x):
            return False
    return True


def all_equal(*xs: Iterable[Any], equality_matrix: bool = False)\
        -> Union[bool, Union[ivy.Array, ivy.NativeArray]]:
    """
    Determines whether the inputs are all equal.

    :param xs: inputs to compare.
    :type xs: any
    :param equality_matrix: Whether to return a matrix of equalities comparing each input with every other.
                            Default is False.
    :type equality_matrix: bool, optional
    :return: Boolean, whether or not the inputs are equal, or matrix array of booleans if equality_matrix=True is set.
    """
    equality_fn = ivy.array_equal if ivy.is_array(xs[0]) else lambda a, b: a == b
    if equality_matrix:
        num_arrays = len(xs)
        mat = [[None for _ in range(num_arrays)] for _ in range(num_arrays)]
        for i, xa in enumerate(xs):
            for j_, xb in enumerate(xs[i:]):
                j = j_ + i
                res = equality_fn(xa, xb)
                if ivy.is_array(res):
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


def to_numpy(x: Union[ivy.Array, ivy.NativeArray])\
        -> np.ndarray:
    """
    Converts array into a numpy array.

    :param x: Input array.
    :type x: array
    :return: A numpy array.
    """
    return _cur_framework(x).to_numpy(x)


def to_scalar(x: Union[ivy.Array, ivy.NativeArray])\
        -> Number:
    """
    Converts an array with a single element into a scalar.

    :param x: Input array with a single element.
    :type x: array
    :return: A scalar.
    """
    return _cur_framework(x).to_scalar(x)


def to_list(x: Union[ivy.Array, ivy.NativeArray])\
        -> List:
    """
    Creates a (possibly nested) list from input array.

    :param x: Input array.
    :type x: array
    :return: A list representation of the input array.
    """
    return _cur_framework(x).to_list(x)

def clip_vector_norm(x: Union[ivy.Array, ivy.NativeArray], max_norm: float, p: float = 2.0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Clips (limits) the vector p-norm of an array.

    :param x: Input array containing elements to clip.
    :type x: array
    :param max_norm: The maximum value of the array norm.
    :type max_norm: float
    :param p: The p-value for computing the p-norm. Default is 2.
    :type p: float, optional
    :return: An array with the vector norm downscaled to the max norm if needed.
    """
    norm = ivy.vector_norm(x, p, keepdims=True)
    ratio = ivy.stable_divide(max_norm, norm)
    if ratio < 1:
        return ratio * x
    return x


def clip_matrix_norm(x: Union[ivy.Array, ivy.NativeArray], max_norm: float, p: float = 2.0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Clips (limits) the matrix norm of an array.

    :param x: Input array containing elements to clip.
    :type x: array
    :param max_norm: The maximum value of the array norm.
    :type max_norm: float
    :param p: The p-value for computing the p-norm. Default is 2.
    :type p: float, optional
    :return: An array with the matrix norm downscaled to the max norm if needed.
    """
    norms = ivy.matrix_norm(x, p, keepdims=True)
    ratios = ivy.maximum(ivy.stable_divide(max_norm, norms), 1.)
    return ratios * x



def floormod(x: Union[ivy.Array, ivy.NativeArray], y: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns element-wise remainder of division.

    :param x: Input array to floormod.
    :type x: array
    :param y: Denominator input for floormod.
    :type y: array
    :return: An array of the same shape and type as x, with the elements floor modded.
    """
    return _cur_framework(x).floormod(x, y)




def unstack(x: Union[ivy.Array, ivy.NativeArray], axis: int, keepdims: bool = False)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Unpacks the given dimension of a rank-R array into rank-(R-1) arrays.

    :param x: Input array to unstack.
    :type x: array
    :param axis: Axis for which to unpack the array.
    :type axis: int
    :param keepdims: Whether to keep dimension 1 in the unstack dimensions. Default is False.
    :type keepdims: bool, optional
    :return: List of arrays, unpacked along specified dimensions.
    """
    return _cur_framework(x).unstack(x, axis, keepdims)


def fourier_encode(x: Union[ivy.Array, ivy.NativeArray], max_freq: Union[float, Union[ivy.Array, ivy.NativeArray]],
                   num_bands: int = 4, linear: bool = False, concat: bool = True, flatten: bool = False)\
        -> Union[ivy.Array, ivy.NativeArray, Tuple]:
    """
    Pads an array with fourier encodings.

    :param x: Input array to encode.
    :type x: array
    :param max_freq: The maximum frequency of the encoding.
    :type max_freq: float
    :param num_bands: The number of frequency bands for the encoding. Default is 4.
    :type num_bands: int, optional
    :param linear: Whether to space the frequency bands linearly as opposed to geometrically. Default is False.
    :type linear: bool, optional
    :param concat: Whether to concatenate the position, sin and cos values, or return seperately. Default is True.
    :type concat: bool, optional
    :param flatten: Whether to flatten the position dimension into the batch dimension. Default is False.
    :type flatten: bool, optional
    :return: New array with the final dimension expanded, and the encodings stored in this channel.
    """
    x_in = x
    dim = x.shape[-1]
    x = ivy.expand_dims(x, -1)
    orig_x = x
    if linear:
        scales = ivy.linspace(1., max_freq / 2, num_bands, dev=dev(x))
    else:
        scales = ivy.logspace(0., ivy.log(max_freq / 2) / math.log(10), num_bands, base=10, dev=dev(x))
    scales = ivy.cast(scales, ivy.dtype(x))
    scales = scales[(*((None,) * (len(x.shape) - len(scales.shape))), Ellipsis)]
    x = x * scales * math.pi
    sin_x = ivy.sin(x)
    cos_x = ivy.cos(x)
    if flatten:
        orig_x = x_in
        sin_x = ivy.reshape(sin_x, [-1, num_bands*dim])
        cos_x = ivy.reshape(cos_x, [-1, num_bands*dim])
    if concat:
        return ivy.concatenate([orig_x, sin_x, cos_x], -1)
    return sin_x, cos_x



def value_is_nan(x: Union[ivy.Array, ivy.NativeArray, Number], include_infs: bool = True)\
        -> bool:
    """
    Determine whether the single valued array or scalar is of nan type

    :param x: The input to check Input array.
    :type x: array
    :param include_infs: Whether to include infs and -infs in the check. Default is True.
    :type include_infs: bool, optional
    :return Boolean as to whether the input value is a nan or not.
    """
    x_scalar = ivy.to_scalar(x) if ivy.is_array(x) else x
    if not x_scalar == x_scalar:
        return True
    if include_infs and x_scalar == INF or x_scalar == -INF:
        return True
    return False


def has_nans(x: Union[ivy.Array, ivy.NativeArray], include_infs: bool = True)\
        -> bool:
    """
    Determine whether the array contains any nans, as well as infs or -infs if specified.

    :param x: Input array.
    :type x: array
    :param include_infs: Whether to include infs and -infs in the check. Default is True.
    :type include_infs: bool, optional
    :return: Boolean as to whether the array contains nans.
    """
    return value_is_nan(ivy.reduce_sum(x), include_infs)


def exists(x: Any)\
        -> bool:
    """
    Simple check as to whether the input is None or not.

    :param x: Input to check.
    :type x: any
    :return: True if x is not None, else False.
    """
    return x is not None


def default(x: Any, default_val: Any, catch_exceptions: bool = False, rev: bool = False, with_callable: bool = False)\
        -> Any:
    """
    Returns x provided it exists (is not None), else returns default value.
    :param x: Input which may or may not exist (be None).
    :type x: value if catch_exceptions=False else callable
    :param default_val: The default value.
    :type default_val: any
    :param catch_exceptions: Whether to catch exceptions from callable x. Default is False.
    :type catch_exceptions: bool, optional
    :param rev: Whether to reverse the input x and default_val. Default is False.
    :type rev: bool, optional
    :param with_callable: Whether either of the arguments might be callable functions. Default is False.
    :type with_callable: bool, optional
    :return: x if x exists (is not None), else default.
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

def shape_to_tuple(shape: Union[int, Tuple[int], List[int]]):
    """
    Returns a tuple representation of the input shape.

    :param shape: The shape input to convert to tuple representation.
    :retrn: The shape in tuple representation
    """
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)

def try_else_none(fn):
    """
    Try and return the function, otherwise return None if an exception was raised during function execution.

    :param fn: Function to try and call and return.
    :type fn: callable
    """
    return default(fn, None, True)


def arg_names(receiver):
    """
    Get the expected keyword arguments for a function or class constructor.
    """
    return list(inspect.signature(receiver).parameters.keys())


def match_kwargs(kwargs, *receivers, allow_duplicates=False):
    """
    Match keyword arguments to either class or function receivers.

    :param kwargs: Keyword arguments to match.
    :type kwargs: dict of any
    :param receivers: Functions and/or classes to match the keyword arguments to.
    :type receivers: callables and/or classes
    :param allow_duplicates: Whether to allow one keyword argument to be used for multiple receivers. Default is False.
    :type allow_duplicates: bool, optional
    :return: Sequence of keyword arguments split as best as possible.
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
