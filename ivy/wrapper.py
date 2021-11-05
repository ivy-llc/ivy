import ivy
import inspect
import importlib
import numpy as np
from types import ModuleType


wrapped_modules_n_classes = []
NON_WRAPPED_METHODS = ['current_framework', 'current_framework_str', 'set_framework', 'get_framework',
                       'unset_framework', 'set_debug_mode', 'set_breakpoint_debug_mode', 'set_exception_debug_mode',
                       'unset_debug_mode', 'debug_mode', 'nested_map', 'to_ivy', 'args_to_ivy', 'to_native',
                       'args_to_native', 'default', 'exists', 'set_min_base', 'get_min_base', 'set_min_denominator',
                       'get_min_denominator', 'split_func_call_across_gpus', 'cache_fn', 'split_func_call',
                       'compile', 'compile_graph', 'dev', 'dev_str', 'dev_to_str', 'str_to_dev', 'memory_on_dev',
                       'gpu_is_available', 'num_gpus', 'tpu_is_available', 'dtype', 'dtype_to_str', 'cprint',
                       'to_ivy_module', 'tree_flatten', 'tree_unflatten', 'start_compiling', 'stop_compiling',
                       'get_compiled', 'set_wrapped_mode', 'unset_wrapped_mode', 'wrapped_mode', 'index_nest',
                       'set_nest_at_index', 'map_nest_at_index', 'multi_index_nest', 'set_nest_at_indices',
                       'map_nest_at_indices', 'nested_indices_where', 'map']

ARRAYLESS_RET_METHODS = ['to_numpy', 'to_list', 'to_scalar', 'shape', 'get_num_dims', 'is_array', 'is_variable']
NESTED_ARRAY_RET_METHODS = ['unstack', 'split']

FW_FN_KEYWORDS = {'numpy': [],
                  'jax': [],
                  'tensorflow': [],
                  'torch': [],
                  'mxnet': ['ndarray']}

NATIVE_KEYS_TO_SKIP = {'numpy': [],
                       'jax': [],
                       'tensorflow': [],
                       'torch': ['classes', 'torch', 'is_grad_enabled', 'get_default_dtype', 'numel', 'clone', 'cpu',
                                 'set_', 'type', 'requires_grad_'],
                       'mxnet': []}

wrapped_mode_val = False


# Methods #

def _wrap_method(fn):

    if hasattr(fn, '__name__') and (fn.__name__[0] == '_' or fn.__name__ in NON_WRAPPED_METHODS):
        return fn

    if hasattr(fn, 'wrapped') and fn.wrapped:
        return fn

    def _method_wrapped(*args, **kwargs):
        native_args, native_kwargs = ivy.args_to_native(*args, **kwargs)
        native_ret = fn(*native_args, **native_kwargs)
        if fn.__name__ in ARRAYLESS_RET_METHODS + NESTED_ARRAY_RET_METHODS:
            return native_ret
        return ivy.to_ivy(native_ret, nested=True)

    if hasattr(fn, '__name__'):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method(method_wrapped):

    if not hasattr(method_wrapped, 'wrapped') or not method_wrapped.wrapped:
        return method_wrapped
    return method_wrapped.inner_fn


def _invalid_fn(fn, fs=None):
    if fs is None:
        fs = ivy.current_framework_str()
    if isinstance(fn, np.ufunc):
        return False
    if not hasattr(fn, '__module__') or not fn.__module__:
        return True
    fw_fn_keywords = ['ivy', fs] + FW_FN_KEYWORDS[fs]
    for kw in fw_fn_keywords:
        if kw in fn.__module__:
            return False
    return True


def _wrap_or_unwrap_methods(wrap_or_unwrap_fn, val=None, fs=None, classes_to_wrap=None, native=False, depth=0):
    classes_to_wrap = [] if classes_to_wrap is None else classes_to_wrap
    if fs is None:
        fs = ivy.current_framework_str()
    if val is None:
        val = importlib.import_module(ivy.current_framework_str()) if native else ivy
    str_to_check = fs if native else 'ivy'
    is_class = inspect.isclass(val)
    if isinstance(val, ModuleType) or (val in classes_to_wrap):
        if val in wrapped_modules_n_classes or (('__file__' not in val.__dict__ or
                (str_to_check not in val.__file__) or 'framework_handler' in val.__file__) and not is_class):
            return val
        wrapped_modules_n_classes.append(val)
        if is_class:
            for k in dir(val):
                if native and (k in NATIVE_KEYS_TO_SKIP[fs]):
                    continue
                v = getattr(val, k)
                if v is not None:
                    # noinspection PyBroadException
                    try:
                        setattr(val, k, _wrap_or_unwrap_methods(
                            wrap_or_unwrap_fn, v, fs, classes_to_wrap, native, depth + 1))
                    except Exception:
                        pass
        else:
            for k, v in val.__dict__.items():
                if native and (k in NATIVE_KEYS_TO_SKIP[fs] or k[0] == '_'):
                    continue
                if v is None:
                    val.__dict__[k] = v
                else:
                    # noinspection PyBroadException
                    try:
                        val.__dict__[k] = _wrap_or_unwrap_methods(
                            wrap_or_unwrap_fn, v, fs, classes_to_wrap, native, depth + 1)
                    except Exception:
                        pass
        if depth == 0:
            wrapped_modules_n_classes.clear()
        return val
    elif callable(val) and not is_class:
        if depth == 0:
            wrapped_modules_n_classes.clear()
        if (hasattr(val, 'inner_fn') and (_invalid_fn(val.inner_fn) and not native))\
                or (_invalid_fn(val) and not native):
            return val
        return wrap_or_unwrap_fn(val)
    if depth == 0:
        wrapped_modules_n_classes.clear()
    return val


def _wrap_methods():
    return _wrap_or_unwrap_methods(_wrap_method)


def _unwrap_methods():
    return _wrap_or_unwrap_methods(_unwrap_method)


# Mode #

def set_wrapped_mode(val=True):
    global wrapped_mode_val
    wrapped_mode_val = val
    if wrapped_mode_val:
        _wrap_methods()
    else:
        _unwrap_methods()


def unset_wrapped_mode():
    global wrapped_mode_val
    wrapped_mode_val = False
    _unwrap_methods()


def wrapped_mode():
    return wrapped_mode_val
