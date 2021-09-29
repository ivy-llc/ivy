import ivy
import pdb
import inspect
import logging
import importlib
import collections
from ivy import verbosity
from types import ModuleType


framework_stack = []
wrap_methods_modules = []
NON_WRAPPED_METHODS = ['current_framework', 'current_framework_str', 'set_framework', 'get_framework',
                       'unset_framework', 'set_debug_mode', 'set_breakpoint_debug_mode', 'set_exception_debug_mode',
                       'unset_debug_mode', 'debug_mode', 'nested_map', 'to_ivy', 'args_to_ivy', 'to_native',
                       'args_to_native', 'default', 'exists', 'set_min_base', 'get_min_base', 'set_min_denominator',
                       'get_min_denominator', 'split_func_call_across_gpus', 'cache_fn', 'split_func_call',
                       'compile_fn', 'dev_to_str', 'str_to_dev', 'memory_on_dev', 'gpu_is_available', 'num_gpus',
                       'tpu_is_available', 'dtype_to_str', 'cprint', 'to_ivy_module']
NON_ARRAY_RET_METHODS = ['to_numpy', 'to_list', 'to_scalar', 'unstack', 'split', 'shape', 'get_num_dims', 'is_array',
                         'is_variable']
NON_ARRAY_METHODS = ['shape', 'dtype']

FW_FN_KEYWORDS = {'numpy': [],
                  'jax': [],
                  'tensorflow': [],
                  'torch': [],
                  'mxnd': ['ndarray']}

debug_mode_val = False
wrapped_mode_val = False
ivy_original_dict = ivy.__dict__.copy()
ivy_original_fn_dict = dict()


class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        set_framework(self.module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_framework()


_array_types = dict()
_array_types['numpy'] = 'ivy.numpy'
_array_types['jax.interpreters.xla'] = 'ivy.jax'
_array_types['tensorflow.python.framework.ops'] = 'ivy.tensorflow'
_array_types['torch'] = 'ivy.torch'
_array_types['mxnet.ndarray.ndarray'] = 'ivy.mxnd'

_framework_dict = dict()
_framework_dict['numpy'] = 'ivy.numpy'
_framework_dict['jax'] = 'ivy.jax'
_framework_dict['tensorflow'] = 'ivy.tensorflow'
_framework_dict['torch'] = 'ivy.torch'
_framework_dict['mxnd'] = 'ivy.mxnd'

_framework_reverse_dict = dict()
_framework_reverse_dict['ivy.numpy'] = 'numpy'
_framework_reverse_dict['ivy.jax'] = 'jax'
_framework_reverse_dict['ivy.tensorflow'] = 'tensorflow'
_framework_reverse_dict['ivy.torch'] = 'torch'
_framework_reverse_dict['ivy.mxnd'] = 'mxnd'


# Framework Getting/Setting #
# --------------------------#

def _determine_framework_from_args(args):
    for arg in args:
        arg_type = type(arg)
        if arg_type in [list, tuple]:
            lib = _determine_framework_from_args(arg)
            if lib:
                return lib
        elif arg_type is dict:
            lib = _determine_framework_from_args(list(arg.values()))
            if lib:
                return lib
        else:
            if arg.__class__.__module__ in _array_types:
                module_name = _array_types[arg.__class__.__module__]
                return importlib.import_module(module_name)


def current_framework(*args, f=None, **kwargs):
    """Priorities: framework > global_framework > input's framework."""

    if f:
        if verbosity.level > 0:
            verbosity.cprint('Using provided framework: {}'.format(f))
        return f

    if framework_stack:
        f = framework_stack[-1]
        if verbosity.level > 0:
            verbosity.cprint('Using framework from stack: {}'.format(f))
        return f

    f = _determine_framework_from_args(list(args) + list(kwargs.values()))
    if f is None:
        raise ValueError(
            'get_framework failed to find a valid library from the inputs: '
            '{} {}'.format(args, kwargs))
    if verbosity.level > 0:
        verbosity.cprint('Using framework from type: {}'.format(f))
    return f


def set_framework(f):
    global ivy_original_dict
    global ivy_original_fn_dict
    if not framework_stack:
        ivy_original_dict = ivy.__dict__.copy()
    if isinstance(f, str):
        f = importlib.import_module(_framework_dict[f])
    framework_stack.append(f)
    ivy_original_fn_dict.clear()
    for k, v in ivy_original_dict.items():
        if k not in f.__dict__:
            f.__dict__[k] = v
        specific_v = f.__dict__[k]
        ivy.__dict__[k] = specific_v
        if isinstance(specific_v, collections.Hashable):
            ivy_original_fn_dict[specific_v] = v
    # noinspection PyUnresolvedReferences
    if wrapped_mode_val and (not hasattr(ivy, 'wrapped') or not ivy.wrapped):
        _wrap_methods()
        ivy.wrapped = True
        f.wrapped = True
    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))


def get_framework(f=None):
    global ivy_original_dict
    if not framework_stack:
        ivy_original_dict = ivy.__dict__.copy()
    if f is None:
        f = ivy.current_framework()
    if isinstance(f, str):
        if framework_stack:
            for k, v in ivy_original_dict.items():
                ivy.__dict__[k] = v
        f = importlib.import_module(_framework_dict[f])
        if framework_stack:
            for k, v in framework_stack[-1].__dict__.items():
                ivy.__dict__[k] = v
    for k, v in ivy_original_dict.items():
        if k not in f.__dict__:
            f.__dict__[k] = v
    return f


def unset_framework():
    if framework_stack:
        framework_stack.pop(-1)
        f_dict = framework_stack[-1].__dict__ if framework_stack else ivy_original_dict
        wrapped = f_dict['wrapped'] if 'wrapped' in f_dict else False
        for k, v in f_dict.items():
            ivy.__dict__[k] = v
        ivy.wrapped = wrapped
    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))


# Wrapped Mode #
# -------------#

# Methods #

def _wrap_method(fn):

    if hasattr(fn, '__name__') and (fn.__name__[0] == '_' or fn.__name__ in NON_WRAPPED_METHODS):
        return fn

    if hasattr(fn, 'wrapped') and fn.wrapped:
        return fn

    def _method_wrapped(*args, **kwargs):
        native_args, native_kwargs = ivy.args_to_native(*args, **kwargs)
        native_ret = fn(*native_args, **native_kwargs)
        if fn.__name__ in NON_ARRAY_RET_METHODS:
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
    if not hasattr(fn, '__module__') or not fn.__module__:
        return True
    fw_fn_keywords = ['ivy', fs] + FW_FN_KEYWORDS[fs]
    for kw in fw_fn_keywords:
        if kw in fn.__module__:
            return False
    return True


def _wrap_or_unwrap_methods(wrap_or_unwrap_fn, val=None, fs=None, depth=0):
    if val is None:
        val = ivy
    if fs is None:
        fs = ivy.current_framework_str()
    if isinstance(val, ModuleType):
        if (val in wrap_methods_modules or '__file__' not in val.__dict__ or
                'ivy' not in val.__file__ or 'framework_handler' in val.__file__):
            return val
        wrap_methods_modules.append(val)
        for k, v in val.__dict__.items():
            if v is None:
                val.__dict__[k] = v
            else:
                val.__dict__[k] = _wrap_or_unwrap_methods(wrap_or_unwrap_fn, v, fs, depth+1)
        if depth == 0:
            wrap_methods_modules.clear()
        return val
    elif callable(val) and not inspect.isclass(val):
        if depth == 0:
            wrap_methods_modules.clear()
        if hasattr(val, 'inner_fn') and _invalid_fn(val.inner_fn):
            return val
        elif _invalid_fn(val):
            return val
        return wrap_or_unwrap_fn(val)
    if depth == 0:
        wrap_methods_modules.clear()
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


# Debug Mode #
# -----------#

# Methods #

def _wrap_method_for_debugging(fn):

    if hasattr(fn, '__name__') and (fn.__name__[0] == '_' or fn.__name__ in
                                    set(NON_WRAPPED_METHODS + ['has_nans', 'is_array', 'value_is_nan', 'reduce_sum',
                                                               'to_scalar'])):
        return fn

    if hasattr(fn, 'wrapped_for_debugging') and fn.wrapped_for_debugging:
        return fn

    def _method_wrapped(*args, **kwargs):

        def _check_nans(x):
            if ivy.is_array(x) and ivy.has_nans(x):
                if debug_mode_val == 'exception':
                    raise Exception('found nans in {}'.format(x))
                else:
                    logging.error('found nans in {}'.format(x))
                    pdb.set_trace()
            return x

        ivy.nested_map(args, _check_nans)
        ivy.nested_map(kwargs, _check_nans)
        ret = fn(*args, **kwargs)
        ivy.nested_map(ret, _check_nans)
        return ret

    if hasattr(fn, '__name__'):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped_for_debugging = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method_from_debugging(method_wrapped):

    if not hasattr(method_wrapped, 'wrapped_for_debugging') or not method_wrapped.wrapped_for_debugging:
        return method_wrapped
    return method_wrapped.inner_fn


def _wrap_methods_for_debugging():
    return _wrap_or_unwrap_methods(_wrap_method_for_debugging)


def _unwrap_methods_from_debugging():
    return _wrap_or_unwrap_methods(_unwrap_method_from_debugging)


# Mode #

def set_debug_mode(debug_mode_in='exception'):
    assert debug_mode_in in ['breakpoint', 'exception']
    global debug_mode_val
    debug_mode_val = debug_mode_in
    _wrap_methods_for_debugging()


def set_breakpoint_debug_mode():
    set_debug_mode('breakpoint')


def set_exception_debug_mode():
    set_debug_mode('exception')


def unset_debug_mode():
    global debug_mode_val
    debug_mode_val = False
    _unwrap_methods_from_debugging()


def debug_mode():
    return debug_mode_val
