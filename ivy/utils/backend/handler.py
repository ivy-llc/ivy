# global
import os
import copy
import types
import ivy
import importlib
import functools
import numpy as np
import gc
from ivy.utils import _importlib, verbosity

# local
from ivy.func_wrapper import _wrap_function
from ivy.utils.backend.sub_backend_handler import _clear_current_sub_backends

backend_stack = []
compiled_backends = {}
_compiled_backends_ids = {}
implicit_backend = "numpy"
ivy_original_dict = ivy.__dict__.copy()
ivy_original_fn_dict = dict()


class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        return set_backend(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        previous_backend()


_backends_subpackage_path = "ivy.functional.backends"
_backend_dict = dict()
_backend_reverse_dict = dict()

for backend in os.listdir(
    os.path.join(
        ivy.__path__[0].rpartition(os.path.sep)[0],  # type: ignore
        _backends_subpackage_path.replace(".", os.path.sep),
    )
):
    if backend.startswith("__"):
        continue
    backend_path = f"{_backends_subpackage_path}.{backend}"
    _backend_dict[backend] = backend_path
    _backend_reverse_dict[backend_path] = backend


# Backend Getting/Setting #
# ----------------------- #


def prevent_access_locally(fn):
    @functools.wraps(fn)
    def _prevent_access_locally(*args, **kwargs):
        if ivy.is_local():
            raise RuntimeError(f"Calling {fn.__name__} is not allowed on this object.")
        return fn(*args, **kwargs)

    return _prevent_access_locally


@functools.lru_cache
def _get_backend_for_arg(arg_module_name):
    for backend in _backend_dict:
        if backend in arg_module_name:
            module_name = _backend_dict[backend]
            return importlib.import_module(module_name)


def _determine_backend_from_args(args):
    """
    Return the appropriate Ivy backend, given some arguments.

    Parameters
    ----------
    args
        the arguments from which to figure out the corresponding Ivy backend.

    Returns
    -------
    ret
        the Ivy backend inferred from `args`.

    Examples
    --------
    If `args` is a jax.numpy array, then Ivy's jax backend will be returned:

    >>> from ivy.utils.backend.handler import _determine_backend_from_args
    >>> import jax.numpy as jnp
    >>> x = jnp.array([1])
    >>> print(_determine_backend_from_args(x))
    <module 'ivy.functional.backends.jax' from '/ivy/ivy/functional/backends/jax/__init__.py'>    # noqa
    """
    arg_type = type(args)
    if isinstance(args, ivy.Array):
        args = args.data

    if isinstance(args, dict):
        for key, value in args.items():
            # recursively call the function for each value in the dictionary
            lib = _determine_backend_from_args(value)
            if lib:
                return lib
        # check if args is a list or tuple
    elif arg_type in [list, tuple]:
        for arg in args:
            # recursively call the function for each element in the list/tuple
            lib = _determine_backend_from_args(arg)
            if lib:
                return lib
    else:
        # check and exclude if arg is a frontend array
        if not hasattr(args, "ivy_array"):
            # check if the class is of type ndarray
            if arg_type.__name__ == "ndarray":
                return _get_backend_for_arg(arg_type.__module__)
            # check if the class is of type Tensor
            elif arg_type.__name__ == "Tensor":
                return _get_backend_for_arg(arg_type.__module__)
    return None


def current_backend():
    if backend_stack:
        return backend_stack[-1]
    # if there is no backend on the stack, infer from the current default backend
    default_backend = _get_backend_for_arg(implicit_backend + ".")
    if default_backend:
        return default_backend
    raise RuntimeError("No current Ivy backend set.")


def _try_backend_attrs(backend):
    backend_vars = copy.copy(backend.__dict__)
    for var_name, var_value in backend_vars.items():
        if not hasattr(ivy, var_name):
            setattr(ivy, var_name, var_value)


def _delete_backend_attrs(backend):
    backend_vars = copy.copy(backend.__dict__)
    for var_name, var_value in backend_vars.items():
        if hasattr(ivy, var_name):
            delattr(ivy, var_name)


# Backend Switching #
# ----------------- #


def set_backend(backend, dynamic=True):
    """
    Set the global backend for Ivy.

    Parameters
    ----------
    backend : str
        The backend to be used. Should be one of the supported backends (e.g., 'numpy', 'jax').
    dynamic : bool, optional
        Whether to dynamically convert existing Ivy objects to the new backend. Default is True.

    Returns
    -------
    None

    Examples
    --------
    >>> from ivy.utils.backend.handler import set_backend
    >>> import jax
    >>> set_backend('jax')

    Notes
    -----
    The dynamic argument allows you to control whether Ivy objects will be converted to the new backend when you change
    the backend. This can be useful if you want to switch the backend globally but keep existing objects untouched.

    .. code-block:: python

        import ivy

        ivy.set_backend('jax', dynamic=False)
        x = ivy.array([1, 2, 3])  # this will still be a numpy array

        ivy.set_backend('jax', dynamic=True)
        x = ivy.array([1, 2, 3])  # this will be a jax array

    If dynamic is set to False, the backend change will only affect newly created Ivy objects.

    Ivy will automatically handle the necessary conversions between backends when performing operations between
    objects of different backends.

    """
    if backend in _backend_dict:
        backend = _backend_dict[backend]
    elif backend in _backend_reverse_dict:
        backend = backend
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Supported backends: {list(_backend_dict.keys())}"
        )

    current_backend_name = current_backend().__name__
    if current_backend_name == backend:
        # No need to switch if the requested backend is already active
        return

    if dynamic:
        ivy_module = importlib.import_module(backend)
        _try_backend_attrs(ivy_module)
    else:
        ivy_module = importlib.import_module(backend)
        ivy.__dict__.update(ivy_module.__dict__)
        _compiled_backends_ids.clear()
        _compiled_backends.clear()
        _clear_current_sub_backends()
        if "array_ops" in ivy.__dict__:
            ivy.array_ops.random_seed(ivy.random_seed())
        verbosity.set_ivy_version_specific_fn_names()

    backend_stack.append(ivy_module)


def previous_backend():
    """
    Return to the previous backend used in Ivy.

    Returns
    -------
    None

    Examples
    --------
    >>> from ivy.utils.backend.handler import previous_backend
    >>> import numpy as np
    >>> import jax
    >>> import ivy

    >>> with ivy.numpy.use:
    ...     print(ivy.current_backend().__name__)
    ...     with ivy.jax.use:
    ...         print(ivy.current_backend().__name__)
    ...         ivy.previous_backend()
    ...         print(ivy.current_backend().__name__)
    ...
    numpy
    jax
    numpy

    Notes
    -----
    The previous_backend function allows you to switch back to the previous Ivy backend after calling set_backend.

    Ivy will automatically handle the necessary conversions between backends when performing operations between
    objects of different backends.

    """
    if not backend_stack:
        raise RuntimeError("No previous Ivy backend to switch to.")
    backend = backend_stack.pop()

    _delete_backend_attrs(backend)

    if backend_stack:
        previous_backend = backend_stack[-1]
        _try_backend_attrs(previous_backend)
    else:
        default_backend = _get_backend_for_arg(implicit_backend + ".")
        if default_backend:
            _try_backend_attrs(default_backend)


def _forward_to_backend_attr(attr_name):
    """
    Helper function to forward the function call to the corresponding Ivy backend attribute.

    Parameters
    ----------
    attr_name : str
        The name of the attribute to forward the function call to.

    Returns
    -------
    Callable
        The function that forwards the call to the backend attribute.

    """

    @functools.wraps(attr_name)
    def forwarder(*args, **kwargs):
        backend = current_backend()
        return getattr(backend, attr_name)(*args, **kwargs)

    return forwarder


# Prevent local access to functions
set_backend = prevent_access_locally(set_backend)
previous_backend = prevent_access_locally(previous_backend)
_current_backend = prevent_access_locally(current_backend)
_forward_to_backend_attr = prevent_access_locally(_forward_to_backend_attr)


# Function Definitions #
# ------------------- #


def array(*args, **kwargs):
    return _forward_to_backend_attr("array")(*args, **kwargs)


def arange(*args, **kwargs):
    return _forward_to_backend_attr("arange")(*args, **kwargs)


def zeros(*args, **kwargs):
    return _forward_to_backend_attr("zeros")(*args, **kwargs)


def ones(*args, **kwargs):
    return _forward_to_backend_attr("ones")(*args, **kwargs)


def random_uniform(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform")(*args, **kwargs)


def random_normal(*args, **kwargs):
    return _forward_to_backend_attr("random_normal")(*args, **kwargs)


def reshape(*args, **kwargs):
    return _forward_to_backend_attr("reshape")(*args, **kwargs)


def concatenate(*args, **kwargs):
    return _forward_to_backend_attr("concatenate")(*args, **kwargs)


def stack(*args, **kwargs):
    return _forward_to_backend_attr("stack")(*args, **kwargs)


def split(*args, **kwargs):
    return _forward_to_backend_attr("split")(*args, **kwargs)


def tile(*args, **kwargs):
    return _forward_to_backend_attr("tile")(*args, **kwargs)


def expand_dims(*args, **kwargs):
    return _forward_to_backend_attr("expand_dims")(*args, **kwargs)


def squeeze(*args, **kwargs):
    return _forward_to_backend_attr("squeeze")(*args, **kwargs)


def transpose(*args, **kwargs):
    return _forward_to_backend_attr("transpose")(*args, **kwargs)


def matmul(*args, **kwargs):
    return _forward_to_backend_attr("matmul")(*args, **kwargs)


def abs(*args, **kwargs):
    return _forward_to_backend_attr("abs")(*args, **kwargs)


def sqrt(*args, **kwargs):
    return _forward_to_backend_attr("sqrt")(*args, **kwargs)


def exp(*args, **kwargs):
    return _forward_to_backend_attr("exp")(*args, **kwargs)


def log(*args, **kwargs):
    return _forward_to_backend_attr("log")(*args, **kwargs)


def mean(*args, **kwargs):
    return _forward_to_backend_attr("mean")(*args, **kwargs)


def sum(*args, **kwargs):
    return _forward_to_backend_attr("sum")(*args, **kwargs)


def max(*args, **kwargs):
    return _forward_to_backend_attr("max")(*args, **kwargs)


def min(*args, **kwargs):
    return _forward_to_backend_attr("min")(*args, **kwargs)


def all(*args, **kwargs):
    return _forward_to_backend_attr("all")(*args, **kwargs)


def any(*args, **kwargs):
    return _forward_to_backend_attr("any")(*args, **kwargs)


def greater(*args, **kwargs):
    return _forward_to_backend_attr("greater")(*args, **kwargs)


def greater_equal(*args, **kwargs):
    return _forward_to_backend_attr("greater_equal")(*args, **kwargs)


def less(*args, **kwargs):
    return _forward_to_backend_attr("less")(*args, **kwargs)


def less_equal(*args, **kwargs):
    return _forward_to_backend_attr("less_equal")(*args, **kwargs)


def equal(*args, **kwargs):
    return _forward_to_backend_attr("equal")(*args, **kwargs)


def logical_and(*args, **kwargs):
    return _forward_to_backend_attr("logical_and")(*args, **kwargs)


def logical_or(*args, **kwargs):
    return _forward_to_backend_attr("logical_or")(*args, **kwargs)


def logical_not(*args, **kwargs):
    return _forward_to_backend_attr("logical_not")(*args, **kwargs)


def clip(*args, **kwargs):
    return _forward_to_backend_attr("clip")(*args, **kwargs)


def sin(*args, **kwargs):
    return _forward_to_backend_attr("sin")(*args, **kwargs)


def cos(*args, **kwargs):
    return _forward_to_backend_attr("cos")(*args, **kwargs)


def tan(*args, **kwargs):
    return _forward_to_backend_attr("tan")(*args, **kwargs)


def arcsin(*args, **kwargs):
    return _forward_to_backend_attr("arcsin")(*args, **kwargs)


def arccos(*args, **kwargs):
    return _forward_to_backend_attr("arccos")(*args, **kwargs)


def arctan(*args, **kwargs):
    return _forward_to_backend_attr("arctan")(*args, **kwargs)


def sinh(*args, **kwargs):
    return _forward_to_backend_attr("sinh")(*args, **kwargs)


def cosh(*args, **kwargs):
    return _forward_to_backend_attr("cosh")(*args, **kwargs)


def tanh(*args, **kwargs):
    return _forward_to_backend_attr("tanh")(*args, **kwargs)


def arcsinh(*args, **kwargs):
    return _forward_to_backend_attr("arcsinh")(*args, **kwargs)


def arccosh(*args, **kwargs):
    return _forward_to_backend_attr("arccosh")(*args, **kwargs)


def arctanh(*args, **kwargs):
    return _forward_to_backend_attr("arctanh")(*args, **kwargs)


def einsum(*args, **kwargs):
    return _forward_to_backend_attr("einsum")(*args, **kwargs)


def cast(*args, **kwargs):
    return _forward_to_backend_attr("cast")(*args, **kwargs)


def where(*args, **kwargs):
    return _forward_to_backend_attr("where")(*args, **kwargs)


def random_seed(*args, **kwargs):
    return _forward_to_backend_attr("random_seed")(*args, **kwargs)


def shape(*args, **kwargs):
    return _forward_to_backend_attr("shape")(*args, **kwargs)


def dtype(*args, **kwargs):
    return _forward_to_backend_attr("dtype")(*args, **kwargs)


def device(*args, **kwargs):
    return _forward_to_backend_attr("device")(*args, **kwargs)


def to_numpy(*args, **kwargs):
    return _forward_to_backend_attr("to_numpy")(*args, **kwargs)


def to_scalar(*args, **kwargs):
    return _forward_to_backend_attr("to_scalar")(*args, **kwargs)


def concat(*args, **kwargs):
    return _forward_to_backend_attr("concat")(*args, **kwargs)


def tile_concat(*args, **kwargs):
    return _forward_to_backend_attr("tile_concat")(*args, **kwargs)


def tile_reshape(*args, **kwargs):
    return _forward_to_backend_attr("tile_reshape")(*args, **kwargs)


def repeat(*args, **kwargs):
    return _forward_to_backend_attr("repeat")(*args, **kwargs)


def strided_slice(*args, **kwargs):
    return _forward_to_backend_attr("strided_slice")(*args, **kwargs)


def gather(*args, **kwargs):
    return _forward_to_backend_attr("gather")(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return _forward_to_backend_attr("gather_nd")(*args, **kwargs)


def scatter(*args, **kwargs):
    return _forward_to_backend_attr("scatter")(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    return _forward_to_backend_attr("scatter_nd")(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return _forward_to_backend_attr("meshgrid")(*args, **kwargs)


def reduce_sum(*args, **kwargs):
    return _forward_to_backend_attr("reduce_sum")(*args, **kwargs)


def reduce_mean(*args, **kwargs):
    return _forward_to_backend_attr("reduce_mean")(*args, **kwargs)


def reduce_max(*args, **kwargs):
    return _forward_to_backend_attr("reduce_max")(*args, **kwargs)


def reduce_min(*args, **kwargs):
    return _forward_to_backend_attr("reduce_min")(*args, **kwargs)


def reduce_prod(*args, **kwargs):
    return _forward_to_backend_attr("reduce_prod")(*args, **kwargs)


def reduce_all(*args, **kwargs):
    return _forward_to_backend_attr("reduce_all")(*args, **kwargs)


def reduce_any(*args, **kwargs):
    return _forward_to_backend_attr("reduce_any")(*args, **kwargs)


def reduce_logsumexp(*args, **kwargs):
    return _forward_to_backend_attr("reduce_logsumexp")(*args, **kwargs)


def reduce_logsumexp2(*args, **kwargs):
    return _forward_to_backend_attr("reduce_logsumexp2")(*args, **kwargs)


def reduce_logaddexp(*args, **kwargs):
    return _forward_to_backend_attr("reduce_logaddexp")(*args, **kwargs)


def reduce_logaddexp2(*args, **kwargs):
    return _forward_to_backend_attr("reduce_logaddexp2")(*args, **kwargs)


def reduce_variance(*args, **kwargs):
    return _forward_to_backend_attr("reduce_variance")(*args, **kwargs)


def reduce_std(*args, **kwargs):
    return _forward_to_backend_attr("reduce_std")(*args, **kwargs)


def reduce_log_prob_normal(*args, **kwargs):
    return _forward_to_backend_attr("reduce_log_prob_normal")(*args, **kwargs)


def reduce_log_prob_categorical(*args, **kwargs):
    return _forward_to_backend_attr("reduce_log_prob_categorical")(*args, **kwargs)


def reduce_log_prob_bernoulli(*args, **kwargs):
    return _forward_to_backend_attr("reduce_log_prob_bernoulli")(*args, **kwargs)


def reduce_log_prob_multinomial(*args, **kwargs):
    return _forward_to_backend_attr("reduce_log_prob_multinomial")(*args, **kwargs)


def logical_and_reducer(*args, **kwargs):
    return _forward_to_backend_attr("logical_and_reducer")(*args, **kwargs)


def logical_or_reducer(*args, **kwargs):
    return _forward_to_backend_attr("logical_or_reducer")(*args, **kwargs)


def make_reduce_reducer(*args, **kwargs):
    return _forward_to_backend_attr("make_reduce_reducer")(*args, **kwargs)


def l2_normalize(*args, **kwargs):
    return _forward_to_backend_attr("l2_normalize")(*args, **kwargs)


def image_to_tensor(*args, **kwargs):
    return _forward_to_backend_attr("image_to_tensor")(*args, **kwargs)


def tensor_to_image(*args, **kwargs):
    return _forward_to_backend_attr("tensor_to_image")(*args, **kwargs)


def save(*args, **kwargs):
    return _forward_to_backend_attr("save")(*args, **kwargs)


def load(*args, **kwargs):
    return _forward_to_backend_attr("load")(*args, **kwargs)


def save_var(*args, **kwargs):
    return _forward_to_backend_attr("save_var")(*args, **kwargs)


def load_var(*args, **kwargs):
    return _forward_to_backend_attr("load_var")(*args, **kwargs)


def save_state_dict(*args, **kwargs):
    return _forward_to_backend_attr("save_state_dict")(*args, **kwargs)


def load_state_dict(*args, **kwargs):
    return _forward_to_backend_attr("load_state_dict")(*args, **kwargs)


def clone(*args, **kwargs):
    return _forward_to_backend_attr("clone")(*args, **kwargs)


def stop_gradient(*args, **kwargs):
    return _forward_to_backend_attr("stop_gradient")(*args, **kwargs)


def is_array(*args, **kwargs):
    return _forward_to_backend_attr("is_array")(*args, **kwargs)


def is_scalar(*args, **kwargs):
    return _forward_to_backend_attr("is_scalar")(*args, **kwargs)


def is_nan(*args, **kwargs):
    return _forward_to_backend_attr("is_nan")(*args, **kwargs)


def is_inf(*args, **kwargs):
    return _forward_to_backend_attr("is_inf")(*args, **kwargs)


def is_finite(*args, **kwargs):
    return _forward_to_backend_attr("is_finite")(*args, **kwargs)


def nan_to_num(*args, **kwargs):
    return _forward_to_backend_attr("nan_to_num")(*args, **kwargs)


def pad(*args, **kwargs):
    return _forward_to_backend_attr("pad")(*args, **kwargs)


def image_resize(*args, **kwargs):
    return _forward_to_backend_attr("image_resize")(*args, **kwargs)


def image_crop(*args, **kwargs):
    return _forward_to_backend_attr("image_crop")(*args, **kwargs)


def image_flip_left_right(*args, **kwargs):
    return _forward_to_backend_attr("image_flip_left_right")(*args, **kwargs)


def image_flip_up_down(*args, **kwargs):
    return _forward_to_backend_attr("image_flip_up_down")(*args, **kwargs)


def image_adjust_brightness(*args, **kwargs):
    return _forward_to_backend_attr("image_adjust_brightness")(*args, **kwargs)


def image_adjust_contrast(*args, **kwargs):
    return _forward_to_backend_attr("image_adjust_contrast")(*args, **kwargs)


def image_adjust_saturation(*args, **kwargs):
    return _forward_to_backend_attr("image_adjust_saturation")(*args, **kwargs)


def image_adjust_hue(*args, **kwargs):
    return _forward_to_backend_attr("image_adjust_hue")(*args, **kwargs)


def image_convert_color(*args, **kwargs):
    return _forward_to_backend_attr("image_convert_color")(*args, **kwargs)


def image_filter(*args, **kwargs):
    return _forward_to_backend_attr("image_filter")(*args, **kwargs)


def image_rotate(*args, **kwargs):
    return _forward_to_backend_attr("image_rotate")(*args, **kwargs)


def image_translate(*args, **kwargs):
    return _forward_to_backend_attr("image_translate")(*args, **kwargs)


def image_affine_transform(*args, **kwargs):
    return _forward_to_backend_attr("image_affine_transform")(*args, **kwargs)


def image_perspective_transform(*args, **kwargs):
    return _forward_to_backend_attr("image_perspective_transform")(*args, **kwargs)


def image_rescale_intensity(*args, **kwargs):
    return _forward_to_backend_attr("image_rescale_intensity")(*args, **kwargs)


def image_equalize_histogram(*args, **kwargs):
    return _forward_to_backend_attr("image_equalize_histogram")(*args, **kwargs)


def image_adjust_gamma(*args, **kwargs):
    return _forward_to_backend_attr("image_adjust_gamma")(*args, **kwargs)


def image_erosion(*args, **kwargs):
    return _forward_to_backend_attr("image_erosion")(*args, **kwargs)


def image_dilation(*args, **kwargs):
    return _forward_to_backend_attr("image_dilation")(*args, **kwargs)


def image_opening(*args, **kwargs):
    return _forward_to_backend_attr("image_opening")(*args, **kwargs)


def image_closing(*args, **kwargs):
    return _forward_to_backend_attr("image_closing")(*args, **kwargs)


def image_white_tophat(*args, **kwargs):
    return _forward_to_backend_attr("image_white_tophat")(*args, **kwargs)


def image_black_tophat(*args, **kwargs):
    return _forward_to_backend_attr("image_black_tophat")(*args, **kwargs)


def image_hit_or_miss(*args, **kwargs):
    return _forward_to_backend_attr("image_hit_or_miss")(*args, **kwargs)


def image_threshold(*args, **kwargs):
    return _forward_to_backend_attr("image_threshold")(*args, **kwargs)


def image_adaptive_threshold(*args, **kwargs):
    return _forward_to_backend_attr("image_adaptive_threshold")(*args, **kwargs)


def image_median_filter(*args, **kwargs):
    return _forward_to_backend_attr("image_median_filter")(*args, **kwargs)


def image_gaussian_filter(*args, **kwargs):
    return _forward_to_backend_attr("image_gaussian_filter")(*args, **kwargs)


def image_sobel_filter(*args, **kwargs):
    return _forward_to_backend_attr("image_sobel_filter")(*args, **kwargs)


def image_laplacian_filter(*args, **kwargs):
    return _forward_to_backend_attr("image_laplacian_filter")(*args, **kwargs)


def image_gabor_filter(*args, **kwargs):
    return _forward_to_backend_attr("image_gabor_filter")(*args, **kwargs)


def image_histogram(*args, **kwargs):
    return _forward_to_backend_attr("image_histogram")(*args, **kwargs)


def image_hog(*args, **kwargs):
    return _forward_to_backend_attr("image_hog")(*args, **kwargs)


def image_sift(*args, **kwargs):
    return _forward_to_backend_attr("image_sift")(*args, **kwargs)


def image_corner_harris(*args, **kwargs):
    return _forward_to_backend_attr("image_corner_harris")(*args, **kwargs)


def image_corner_shi_tomasi(*args, **kwargs):
    return _forward_to_backend_attr("image_corner_shi_tomasi")(*args, **kwargs)


def image_moments(*args, **kwargs):
    return _forward_to_backend_attr("image_moments")(*args, **kwargs)


def image_reconstruction(*args, **kwargs):
    return _forward_to_backend_attr("image_reconstruction")(*args, **kwargs)


def image_watershed(*args, **kwargs):
    return _forward_to_backend_attr("image_watershed")(*args, **kwargs)


def image_connected_components(*args, **kwargs):
    return _forward_to_backend_attr("image_connected_components")(*args, **kwargs)


def image_label(*args, **kwargs):
    return _forward_to_backend_attr("image_label")(*args, **kwargs)


def image_find_contours(*args, **kwargs):
    return _forward_to_backend_attr("image_find_contours")(*args, **kwargs)


def image_draw_contours(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_contours")(*args, **kwargs)


def image_draw_rectangle(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_rectangle")(*args, **kwargs)


def image_draw_circle(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_circle")(*args, **kwargs)


def image_draw_ellipse(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_ellipse")(*args, **kwargs)


def image_draw_line(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_line")(*args, **kwargs)


def image_draw_polygon(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_polygon")(*args, **kwargs)


def image_draw_text(*args, **kwargs):
    return _forward_to_backend_attr("image_draw_text")(*args, **kwargs)


def image_to_grayscale(*args, **kwargs):
    return _forward_to_backend_attr("image_to_grayscale")(*args, **kwargs)


def image_resize_nearest(*args, **kwargs):
    return _forward_to_backend_attr("image_resize_nearest")(*args, **kwargs)


def image_resize_bilinear(*args, **kwargs):
    return _forward_to_backend_attr("image_resize_bilinear")(*args, **kwargs)


def image_resize_bicubic(*args, **kwargs):
    return _forward_to_backend_attr("image_resize_bicubic")(*args, **kwargs)


def image_resize_area(*args, **kwargs):
    return _forward_to_backend_attr("image_resize_area")(*args, **kwargs)


def image_resize_lanczos3(*args, **kwargs):
    return _forward_to_backend_attr("image_resize_lanczos3")(*args, **kwargs)


def image_to_tensorflow_format(*args, **kwargs):
    return _forward_to_backend_attr("image_to_tensorflow_format")(*args, **kwargs)


def image_to_torch_format(*args, **kwargs):
    return _forward_to_backend_attr("image_to_torch_format")(*args, **kwargs)


def image_from_tensorflow_format(*args, **kwargs):
    return _forward_to_backend_attr("image_from_tensorflow_format")(*args, **kwargs)


def image_from_torch_format(*args, **kwargs):
    return _forward_to_backend_attr("image_from_torch_format")(*args, **kwargs)


def image_from_bytes(*args, **kwargs):
    return _forward_to_backend_attr("image_from_bytes")(*args, **kwargs)


def image_to_bytes(*args, **kwargs):
    return _forward_to_backend_attr("image_to_bytes")(*args, **kwargs)


def random_uniform(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform")(*args, **kwargs)


def random_normal(*args, **kwargs):
    return _forward_to_backend_attr("random_normal")(*args, **kwargs)


def random_truncated_normal(*args, **kwargs):
    return _forward_to_backend_attr("random_truncated_normal")(*args, **kwargs)


def random_binomial(*args, **kwargs):
    return _forward_to_backend_attr("random_binomial")(*args, **kwargs)


def random_bernoulli(*args, **kwargs):
    return _forward_to_backend_attr("random_bernoulli")(*args, **kwargs)


def random_categorical(*args, **kwargs):
    return _forward_to_backend_attr("random_categorical")(*args, **kwargs)


def random_shuffle(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle")(*args, **kwargs)


def random_choice(*args, **kwargs):
    return _forward_to_backend_attr("random_choice")(*args, **kwargs)


def random_gamma(*args, **kwargs):
    return _forward_to_backend_attr("random_gamma")(*args, **kwargs)


def random_poisson(*args, **kwargs):
    return _forward_to_backend_attr("random_poisson")(*args, **kwargs)


def random_exponential(*args, **kwargs):
    return _forward_to_backend_attr("random_exponential")(*args, **kwargs)


def random_weibull(*args, **kwargs):
    return _forward_to_backend_attr("random_weibull")(*args, **kwargs)


def random_power(*args, **kwargs):
    return _forward_to_backend_attr("random_power")(*args, **kwargs)


def random_negative_binomial(*args, **kwargs):
    return _forward_to_backend_attr("random_negative_binomial")(*args, **kwargs)


def random_geometric(*args, **kwargs):
    return _forward_to_backend_attr("random_geometric")(*args, **kwargs)


def random_logistic(*args, **kwargs):
    return _forward_to_backend_attr("random_logistic")(*args, **kwargs)


def random_multivariate_normal(*args, **kwargs):
    return _forward_to_backend_attr("random_multivariate_normal")(*args, **kwargs)


def random_seed(*args, **kwargs):
    return _forward_to_backend_attr("random_seed")(*args, **kwargs)


def random_get_state(*args, **kwargs):
    return _forward_to_backend_attr("random_get_state")(*args, **kwargs)


def random_set_state(*args, **kwargs):
    return _forward_to_backend_attr("random_set_state")(*args, **kwargs)


def random_permutation(*args, **kwargs):
    return _forward_to_backend_attr("random_permutation")(*args, **kwargs)


def random_shuffle_axis(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_axis")(*args, **kwargs)


def random_choice_axis(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_axis")(*args, **kwargs)


def random_cdf(*args, **kwargs):
    return _forward_to_backend_attr("random_cdf")(*args, **kwargs)


def random_pdf(*args, **kwargs):
    return _forward_to_backend_attr("random_pdf")(*args, **kwargs)


def random_log_pdf(*args, **kwargs):
    return _forward_to_backend_attr("random_log_pdf")(*args, **kwargs)


def random_sf(*args, **kwargs):
    return _forward_to_backend_attr("random_sf")(*args, **kwargs)


def random_isf(*args, **kwargs):
    return _forward_to_backend_attr("random_isf")(*args, **kwargs)


def random_randint(*args, **kwargs):
    return _forward_to_backend_attr("random_randint")(*args, **kwargs)


def random_uniform_like(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform_like")(*args, **kwargs)


def random_normal_like(*args, **kwargs):
    return _forward_to_backend_attr("random_normal_like")(*args, **kwargs)


def random_truncated_normal_like(*args, **kwargs):
    return _forward_to_backend_attr("random_truncated_normal_like")(*args, **kwargs)


def random_binomial_like(*args, **kwargs):
    return _forward_to_backend_attr("random_binomial_like")(*args, **kwargs)


def random_bernoulli_like(*args, **kwargs):
    return _forward_to_backend_attr("random_bernoulli_like")(*args, **kwargs)


def random_categorical_like(*args, **kwargs):
    return _forward_to_backend_attr("random_categorical_like")(*args, **kwargs)


def random_shuffle_like(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_like")(*args, **kwargs)


def random_choice_like(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_like")(*args, **kwargs)


def random_gamma_like(*args, **kwargs):
    return _forward_to_backend_attr("random_gamma_like")(*args, **kwargs)


def random_poisson_like(*args, **kwargs):
    return _forward_to_backend_attr("random_poisson_like")(*args, **kwargs)


def random_exponential_like(*args, **kwargs):
    return _forward_to_backend_attr("random_exponential_like")(*args, **kwargs)


def random_weibull_like(*args, **kwargs):
    return _forward_to_backend_attr("random_weibull_like")(*args, **kwargs)


def random_power_like(*args, **kwargs):
    return _forward_to_backend_attr("random_power_like")(*args, **kwargs)


def random_negative_binomial_like(*args, **kwargs):
    return _forward_to_backend_attr("random_negative_binomial_like")(*args, **kwargs)


def random_geometric_like(*args, **kwargs):
    return _forward_to_backend_attr("random_geometric_like")(*args, **kwargs)


def random_logistic_like(*args, **kwargs):
    return _forward_to_backend_attr("random_logistic_like")(*args, **kwargs)


def random_multivariate_normal_like(*args, **kwargs):
    return _forward_to_backend_attr("random_multivariate_normal_like")(*args, **kwargs)


def random_permutation_like(*args, **kwargs):
    return _forward_to_backend_attr("random_permutation_like")(*args, **kwargs)


def random_shuffle_axis_like(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_axis_like")(*args, **kwargs)


def random_choice_axis_like(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_axis_like")(*args, **kwargs)


def random_cdf_like(*args, **kwargs):
    return _forward_to_backend_attr("random_cdf_like")(*args, **kwargs)


def random_pdf_like(*args, **kwargs):
    return _forward_to_backend_attr("random_pdf_like")(*args, **kwargs)


def random_log_pdf_like(*args, **kwargs):
    return _forward_to_backend_attr("random_log_pdf_like")(*args, **kwargs)


def random_sf_like(*args, **kwargs):
    return _forward_to_backend_attr("random_sf_like")(*args, **kwargs)


def random_isf_like(*args, **kwargs):
    return _forward_to_backend_attr("random_isf_like")(*args, **kwargs)


def random_randint_like(*args, **kwargs):
    return _forward_to_backend_attr("random_randint_like")(*args, **kwargs)


def random_normal_from_dtype_scale(*args, **kwargs):
    return _forward_to_backend_attr("random_normal_from_dtype_scale")(*args, **kwargs)


def random_uniform_from_dtype_scale(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform_from_dtype_scale")(*args, **kwargs)


def random_seed_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_seed_tensorflow")(*args, **kwargs)


def random_set_seed(*args, **kwargs):
    return _forward_to_backend_attr("random_set_seed")(*args, **kwargs)


def random_get_state_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_get_state_tensorflow")(*args, **kwargs)


def random_set_state_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_set_state_tensorflow")(*args, **kwargs)


def random_normal_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_normal_tensorflow")(*args, **kwargs)


def random_uniform_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform_tensorflow")(*args, **kwargs)


def random_truncated_normal_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_truncated_normal_tensorflow")(*args, **kwargs)


def random_binomial_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_binomial_tensorflow")(*args, **kwargs)


def random_bernoulli_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_bernoulli_tensorflow")(*args, **kwargs)


def random_categorical_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_categorical_tensorflow")(*args, **kwargs)


def random_shuffle_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_tensorflow")(*args, **kwargs)


def random_choice_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_tensorflow")(*args, **kwargs)


def random_gamma_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_gamma_tensorflow")(*args, **kwargs)


def random_poisson_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_poisson_tensorflow")(*args, **kwargs)


def random_exponential_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_exponential_tensorflow")(*args, **kwargs)


def random_weibull_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_weibull_tensorflow")(*args, **kwargs)


def random_power_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_power_tensorflow")(*args, **kwargs)


def random_negative_binomial_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_negative_binomial_tensorflow")(*args, **kwargs)


def random_geometric_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_geometric_tensorflow")(*args, **kwargs)


def random_logistic_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_logistic_tensorflow")(*args, **kwargs)


def random_multivariate_normal_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_multivariate_normal_tensorflow")(*args, **kwargs)


def random_permutation_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_permutation_tensorflow")(*args, **kwargs)


def random_shuffle_axis_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_axis_tensorflow")(*args, **kwargs)


def random_choice_axis_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_axis_tensorflow")(*args, **kwargs)


def random_cdf_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_cdf_tensorflow")(*args, **kwargs)


def random_pdf_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_pdf_tensorflow")(*args, **kwargs)


def random_log_pdf_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_log_pdf_tensorflow")(*args, **kwargs)


def random_sf_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_sf_tensorflow")(*args, **kwargs)


def random_isf_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_isf_tensorflow")(*args, **kwargs)


def random_randint_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_randint_tensorflow")(*args, **kwargs)


def random_uniform_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform_like_tensorflow")(*args, **kwargs)


def random_normal_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_normal_like_tensorflow")(*args, **kwargs)


def random_truncated_normal_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_truncated_normal_like_tensorflow")(*args, **kwargs)


def random_binomial_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_binomial_like_tensorflow")(*args, **kwargs)


def random_bernoulli_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_bernoulli_like_tensorflow")(*args, **kwargs)


def random_categorical_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_categorical_like_tensorflow")(*args, **kwargs)


def random_shuffle_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_like_tensorflow")(*args, **kwargs)


def random_choice_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_like_tensorflow")(*args, **kwargs)


def random_gamma_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_gamma_like_tensorflow")(*args, **kwargs)


def random_poisson_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_poisson_like_tensorflow")(*args, **kwargs)


def random_exponential_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_exponential_like_tensorflow")(*args, **kwargs)


def random_weibull_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_weibull_like_tensorflow")(*args, **kwargs)


def random_power_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_power_like_tensorflow")(*args, **kwargs)


def random_negative_binomial_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_negative_binomial_like_tensorflow")(*args, **kwargs)


def random_geometric_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_geometric_like_tensorflow")(*args, **kwargs)


def random_logistic_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_logistic_like_tensorflow")(*args, **kwargs)


def random_multivariate_normal_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_multivariate_normal_like_tensorflow")(*args, **kwargs)


def random_permutation_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_permutation_like_tensorflow")(*args, **kwargs)


def random_shuffle_axis_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_shuffle_axis_like_tensorflow")(*args, **kwargs)


def random_choice_axis_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_choice_axis_like_tensorflow")(*args, **kwargs)


def random_cdf_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_cdf_like_tensorflow")(*args, **kwargs)


def random_pdf_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_pdf_like_tensorflow")(*args, **kwargs)


def random_log_pdf_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_log_pdf_like_tensorflow")(*args, **kwargs)


def random_sf_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_sf_like_tensorflow")(*args, **kwargs)


def random_isf_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_isf_like_tensorflow")(*args, **kwargs)


def random_randint_like_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_randint_like_tensorflow")(*args, **kwargs)


def random_normal_from_dtype_scale_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_normal_from_dtype_scale_tensorflow")(*args, **kwargs)


def random_uniform_from_dtype_scale_tensorflow(*args, **kwargs):
    return _forward_to_backend_attr("random_uniform_from_dtype_scale_tensorflow")(*args, **kwargs)
