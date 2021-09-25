import ivy
import inspect
import importlib
from ivy import verbosity
from types import ModuleType


framework_stack = []
wrap_methods_modules = []
# ToDo: add more non-wrapped methods to the list below, adding __name__ attribute to lambdas where needed
NON_WRAPPED_METHODS = ['current_framework', 'current_framework_str', 'set_framework', 'get_framework',
                       'unset_framework', 'set_debug_mode', 'set_breakpoint_debug_mode', 'set_exception_debug_mode',
                       'unset_debug_mode', 'debug_mode', 'to_ivy', 'to_native', 'args_to_native', 'default', 'exists',
                       'set_min_base', 'get_min_base', 'set_min_denominator', 'get_min_denominator',
                       'split_func_call_across_gpus', 'cache_fn', 'split_func_call', 'compile_fn', 'dev_to_str',
                       'str_to_dev', 'memory_on_dev', 'gpu_is_available', 'num_gpus', 'tpu_is_available',
                       'dtype_to_str']
NON_ARRAY_METHODS = ['to_numpy', 'to_list', 'to_scalar', 'unstack', 'split', 'shape', 'get_num_dims', 'is_array',
                     'is_variable']
debug_mode_val = False
wrapped_mode_val = True
ivy_original_dict = ivy.__dict__.copy()


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
    if not framework_stack:
        ivy_original_dict = ivy.__dict__.copy()
    if isinstance(f, str):
        f = importlib.import_module(_framework_dict[f])
    framework_stack.append(f)
    for k, v in ivy_original_dict.items():
        if k not in f.__dict__:
            f.__dict__[k] = v
        ivy.__dict__[k] = f.__dict__[k]

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

def _wrap_method(fn):

    if hasattr(fn, '__name__') and (fn.__name__[0] == '_' or fn.__name__ in NON_WRAPPED_METHODS):
        return fn

    if hasattr(fn, 'wrapped') and fn.wrapped:
        return fn

    def _method_wrapped(*args, **kwargs):
        # ToDo: try to modify ivy.Array built-ins so extracting the data is not needed here,
        #  and maybe even the wrapping in general
        native_args, native_kwargs = ivy.args_to_native(*args, **kwargs)
        native_ret = fn(*native_args, **native_kwargs)
        if fn.__name__ in NON_ARRAY_METHODS:
            return native_ret
        if isinstance(native_ret, (list, tuple)):
            ret = tuple([ivy.Array(r) if isinstance(r, ivy.NativeArray) else r for r in native_ret])
        else:
            ret = ivy.Array(native_ret) if isinstance(native_ret, ivy.NativeArray) else native_ret
        return ret

    if hasattr(fn, '__name__'):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method(method_wrapped):

    if not hasattr(method_wrapped, 'wrapped') or not method_wrapped.wrapped:
        return method_wrapped
    return method_wrapped.inner_fn


def _wrap_or_unwrap_methods(wrap_or_unwrap_fn, val=None, depth=0):
    if val is None:
        val = ivy
    if isinstance(val, ModuleType) and val not in wrap_methods_modules and '__file__' in val.__dict__ and \
            'ivy' in val.__file__ and 'framework_handler' not in val.__file__:
        wrap_methods_modules.append(val)
        for k, v in val.__dict__.items():
            if v is None:
                val.__dict__[k] = v
            else:
                val.__dict__[k] = _wrap_or_unwrap_methods(wrap_or_unwrap_fn, v, depth+1)
        if depth == 0:
            wrap_methods_modules.clear()
        return val
    elif callable(val) and not inspect.isclass(val):
        if depth == 0:
            wrap_methods_modules.clear()
        return wrap_or_unwrap_fn(val)
    if depth == 0:
        wrap_methods_modules.clear()
    return val


def _wrap_methods():
    return _wrap_or_unwrap_methods(_wrap_method)


def _unwrap_methods():
    return _wrap_or_unwrap_methods(_unwrap_method)


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

def set_debug_mode(debug_mode_in='exception'):
    assert debug_mode_in in ['breakpoint', 'exception']
    global debug_mode_val
    debug_mode_val = debug_mode_in


def set_breakpoint_debug_mode():
    set_debug_mode('breakpoint')


def set_exception_debug_mode():
    set_debug_mode('exception')


def unset_debug_mode():
    global debug_mode_val
    debug_mode_val = False


def debug_mode():
    return debug_mode_val
