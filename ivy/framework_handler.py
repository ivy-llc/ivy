import ivy
import importlib
from ivy import verbosity


framework_stack = []
ivy_original_dict = dict()


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


def _get_framework_from_args(args):
    for arg in args:
        arg_type = type(arg)
        if arg_type in [list, tuple]:
            lib = _get_framework_from_args(arg)
            if lib:
                return lib
        elif arg_type is dict:
            lib = _get_framework_from_args(list(arg.values()))
            if lib:
                return lib
        else:
            if arg.__class__.__module__ in _array_types:
                module_name = _array_types[arg.__class__.__module__]
                return importlib.import_module(module_name)


def get_framework(*args, f=None, **kwargs):
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

    f = _get_framework_from_args(list(args) + list(kwargs.values()))
    if f is None:
        raise ValueError(
            'get_framework failed to find a valid library from the inputs: '
            '{} {}'.format(args, kwargs)
        )

    if verbosity.level > 0:
        verbosity.cprint('Using framework from type: {}'.format(f))
    return f


def get_framework_str(*args, f=None, **kwargs):
    framework = get_framework(*args, f, **kwargs)
    return _framework_reverse_dict[
        framework.__repr__().split("<module '")[-1].split("' ")[0]
    ]


def set_framework(f):
    if not framework_stack:
        global ivy_original_dict
        ivy_original_dict = ivy.__dict__.copy()

    if isinstance(f, str):
        f = importlib.import_module(_framework_dict[f])
    framework_stack.append(f)

    for k, v in f.__dict__.items():
        ivy.__dict__[k] = v

    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))


def unset_framework():
    if framework_stack:
        framework_stack.pop(-1)
        f_dict = (
            framework_stack[-1].__dict__ if framework_stack else ivy_original_dict
        )

        for k, v in f_dict.items():
            ivy.__dict__[k] = v

    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))
