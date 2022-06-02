"""Collection of general Ivy functions."""

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
from ivy.backend_handler import current_backend as _cur_backend

FN_CACHE = dict()
INF = float("inf")
TIMEOUT = 15.0
TMP_DIR = "/tmp"


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
         (Default value = None)

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


def is_native_array(x: Any, exclusive: bool = False) -> bool:
    """Determines whether the input x is a Native Array.

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
    try:
        return _cur_backend(x).is_native_array(x, exclusive)
    except ValueError:
        return False


def is_ivy_array(x: Any, exclusive: bool = False) -> bool:
    """Determines whether the input x is an Ivy Array.

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


# noinspection PyShadowingNames
def copy_array(
    x: Union[ivy.Array, ivy.NativeArray]
) -> Union[ivy.Array, ivy.NativeArray]:
    """Copy an array.

    Returns
    -------
    ret
        a copy of the input array ``x``.

    Examples
    --------
    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([-1, 0, 1])

    """
    return _cur_backend(x).copy_array(x)


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
    return _cur_backend(x0).array_equal(x0, x1)


def arrays_equal(xs: List[Union[ivy.Array, ivy.NativeArray]]) -> bool:
    """Determines whether input arrays are equal across all elements.

    Parameters
    ----------
    xs
        Sequence of arrays to compare for equality

    Returns
    -------
    ret
        Boolean, whether or not all of the input arrays are equal across all elements.

    """
    x0 = xs[0]
    for x in xs[1:]:
        if not array_equal(x0, x):
            return False
    return True


def all_equal(
    *xs: Iterable[Any], equality_matrix: bool = False
) -> Union[bool, Union[ivy.Array, ivy.NativeArray]]:
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


def to_numpy(x: Union[ivy.Array, ivy.NativeArray]) -> np.ndarray:
    """Converts an array into a numpy array.

    Parameters
    ----------
    x
        input array

    Returns
    -------
    ret
        a numpy array copying all the element of the array ``x``.

    Examples
    --------
    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.to_numpy(x)
    >>> print(y)
    [-1 0 1]

    >>> print(type(y))
    <class 'numpy.ndarray'>

    """
    return _cur_backend(x).to_numpy(x)


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

    Examples
    --------
    >>> x = ivy.array([-1])
    >>> y = ivy.to_scalar(x)
    >>> print(y)
    -1


    >>> print(ivy.is_int_dtype(y))
    True

    """
    return _cur_backend(x).to_scalar(x)


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

    Examples
    --------
    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.to_list(x)
    >>> print(y)
    [-1, 0, 1]

    >>> print(isinstance(y, list))
    True

    """
    return _cur_backend(x).to_list(x)


def clip_vector_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    max_norm: float,
    p: float = 2.0,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
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

    """
    norm = ivy.vector_norm(x, keepdims=True, ord=p)
    ratio = ivy.stable_divide(max_norm, norm)
    if ratio < 1:
        return ratio * x
    return ivy.copy_array(x)._data


def clip_matrix_norm(
    x: Union[ivy.Array, ivy.NativeArray], max_norm: float, p: float = 2.0
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

    Returns
    -------
    ret
        An array with the matrix norm downscaled to the max norm if needed.

    """
    norms = ivy.matrix_norm(x, p, keepdims=True)
    ratios = ivy.maximum(ivy.stable_divide(max_norm, norms), 1.0)
    return ratios * x


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
    return _cur_backend(x).floormod(x, y, out=out)


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
    return _cur_backend(x).unstack(x, axis, keepdims)


def fourier_encode(
    x: Union[ivy.Array, ivy.NativeArray],
    max_freq: Union[float, Union[ivy.Array, ivy.NativeArray]],
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


def has_nans(x: Union[ivy.Array, ivy.NativeArray], include_infs: bool = True) -> bool:
    """Determine whether the array contains any nans, as well as infs or -infs if
    specified.

    Parameters
    ----------
    x
        Input array.
    include_infs
        Whether to include infs and -infs in the check. Default is True.

    Returns
    -------
    ret
        Boolean as to whether the array contains nans.

    """
    return value_is_nan(ivy.sum(x), include_infs)


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
    """Returns a tuple representation of the input shape.

    Parameters
    ----------
    shape
        The shape input to convert to tuple representation.

    Returns
    -------
        The shape in tuple representation

    """
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)


def try_else_none(fn):
    """Try and return the function, otherwise return None if an exception was raised
    during function execution.

    Parameters
    ----------
    fn
        Function to try and call and return.

    """
    return default(fn, None, True)


def arg_names(receiver):
    """Get the expected keyword arguments for a function or class constructor.

    Parameters
    ----------
    receiver

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
    fw = _cur_backend()
    if fw is None:
        return None
    return fw.current_backend_str()


def einops_rearrange(
    x: Union[ivy.Array, ivy.NativeArray], pattern: str, **axes_lengths: Dict[str, int]
) -> Union[ivy.Array, ivy.NativeArray]:
    """Perform einops rearrange operation on input array x.

    Parameters
    ----------
    x
        Input array to be re-arranged.
    pattern
        Rearrangement pattern.
    axes_lengths
        Any additional specifications for dimensions.

    Returns
    -------
    ret
        New array with einops.rearrange having been applied.

    """
    return einops.rearrange(x, pattern, **axes_lengths)


def einops_reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    pattern: str,
    reduction: Union[str, Callable],
    **axes_lengths: Dict[str, int],
) -> Union[ivy.Array, ivy.NativeArray]:
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

    Returns
    -------
    ret
        New array with einops.reduce having been applied.

    """
    return einops.reduce(x, pattern, reduction, **axes_lengths)


def einops_repeat(
    x: Union[ivy.Array, ivy.NativeArray], pattern: str, **axes_lengths: Dict[str, int]
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

    Returns
    -------
    ret
        New array with einops.repeat having been applied.

    """
    return einops.repeat(x, pattern, **axes_lengths)


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
    """Get the global minimum base used by ivy for numerically stable power raising."""
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
    numerator: Any, denominator: Any, min_denominator: float = None
) -> Any:
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
    """Set the global queue timeout values (in seconds). Default value without this
    function being called is 10 seconds.

    Parameters
    ----------
    timeout
        The timeout to set in seconds.

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
    """"""
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
        return _cur_backend().container_types()
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
    return _cur_backend().inplace_arrays_supported()


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
    return _cur_backend().inplace_variables_supported()


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


def inplace_update(
    x: Union[ivy.Array, ivy.NativeArray], val: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Perform in-place update for the input array. This will always be performed on
    ivy.Array instances pass in the input, and will also be performed on the native
    array classes in the backend, when the backend supports this.

    Parameters
    ----------
    x
        The variable to update.
    val
        The array to update the variable with.

    Returns
    -------
    ret
        The array following the in-place update.

    """
    return _cur_backend(x).inplace_update(x, val)


def inplace_decrement(x, val):
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
    return _cur_backend(x).inplace_decrement(x, val)


def inplace_increment(x, val):
    """Perform in-place increment for the input array.

    Parameters
    ----------
    x
        The array to increment.
    val
        The array to increment the variable with.

    Returns
    -------
    ret
        The array following the in-place increment.

    """
    return _cur_backend(x).inplace_increment(x, val)


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
    return _cur_backend(x).cumsum(x, axis, out=out)


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

    Returns
    -------
    ret
        Input array with cumulatively multiplied elements along axis.

    """
    return _cur_backend(x).cumprod(x, axis, exclusive, out=out)


# noinspection PyShadowingNames
def scatter_flat(
    indices: Union[ivy.Array, ivy.NativeArray],
    updates: Union[ivy.Array, ivy.NativeArray],
    size: Optional[int] = None,
    tensor: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    reduction: str = "sum",
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
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

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    """
    return _cur_backend(indices).scatter_flat(
        indices, updates, size, tensor, reduction, device=device
    )


# noinspection PyShadowingNames
def scatter_nd(
    indices: Union[ivy.Array, ivy.NativeArray],
    updates: Union[ivy.Array, ivy.NativeArray],
    shape: Optional[Iterable[int]] = None,
    tensor: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    reduction: str = "sum",
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
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

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    """
    return _cur_backend(indices).scatter_nd(
        indices, updates, shape, tensor, reduction, device=device
    )


# noinspection PyShadowingNames
def gather(
    params: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    axis: int = -1,
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
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

    """
    return _cur_backend(params).gather(params, indices, axis, device=device, out=out)


# noinspection PyShadowingNames
def gather_nd(
    params: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
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

    Returns
    -------
    ret
        New array of given shape, with the values gathered at the indices.

    """
    return _cur_backend(params).gather_nd(params, indices, device=device)


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
    return _cur_backend().multiprocessing(context)


def indices_where(
    x: Union[ivy.Array, ivy.NativeArray]
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns indices or true elements in an input boolean array.

    Parameters
    ----------
    x
        Boolean array, for which indices are desired.

    Returns
    -------
    ret
        Indices for where the boolean array is True.

    """
    return _cur_backend(x).indices_where(x)


# noinspection PyShadowingNames
def one_hot(
    indices: Union[ivy.Array, ivy.NativeArray],
    depth: int,
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
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

    Returns
    -------
    ret
        Tensor of zeros with the same shape and type as a, unless dtype provided which
        overrides.

    """
    return _cur_backend(indices).one_hot(indices, depth, device=device)


def shape(
    x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False
) -> Iterable[int]:
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
    >>> x = ivy.array([[-1, 0, 1],[1,0,-1]])
    >>> y_tuple = ivy.shape(x)
    >>> y_tensor = ivy.shape(x, as_tensor = True)
    >>> print(y_tuple)
    (2, 3)

    >>> print(y_tensor)
    ivy.array([2, 3])

    """
    return _cur_backend(x).shape(x, as_array)


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
    return _cur_backend(x).get_num_dims(x, as_array)
