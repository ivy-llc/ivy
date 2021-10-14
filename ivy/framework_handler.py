# global
import ivy
import importlib
import collections
from ivy import verbosity

# local
from ivy.wrapper import _wrap_methods, wrapped_mode_val


framework_stack = []
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
        temp_stack = list()
        while framework_stack:
            temp_stack.append(unset_framework())
        f = importlib.import_module(_framework_dict[f])
        for fw in reversed(temp_stack):
            framework_stack.append(fw)
    framework_stack.append(f)
    ivy_original_fn_dict.clear()
    for k, v in ivy_original_dict.items():
        if k not in f.__dict__:
            f.__dict__[k] = v
        specific_v = f.__dict__[k]
        ivy.__dict__[k] = specific_v
        if isinstance(specific_v, collections.Hashable):
            try:
                ivy_original_fn_dict[specific_v] = v
            except TypeError:
                pass
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
    fw = None
    if framework_stack:
        fw = framework_stack.pop(-1)
        f_dict = framework_stack[-1].__dict__ if framework_stack else ivy_original_dict
        wrapped = f_dict['wrapped'] if 'wrapped' in f_dict else False
        for k, v in f_dict.items():
            ivy.__dict__[k] = v
        ivy.wrapped = wrapped
    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))
    return fw
