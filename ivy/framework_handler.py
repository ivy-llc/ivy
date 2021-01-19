import importlib

from ivy import verbosity


framework_stack = []


class ContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        framework_stack.append(self.module)
        if verbosity.level > 0:
            verbosity.cprint(
                'framework stack: {}'.format(framework_stack))

    def __exit__(self, exc_type, exc_val, exc_tb):
        framework_stack.pop(-1)
        if verbosity.level > 0:
            verbosity.cprint(
                'framework stack: {}'.format(framework_stack))


_array_types = dict()
_array_types['numpy'] = 'ivy.numpy'
_array_types['jax.interpreters.xla'] = 'ivy.jax'
_array_types['tensorflow.python.framework.ops'] = 'ivy.tensorflow'
_array_types['torch'] = 'ivy.torch'
_array_types['mxnet.ndarray.ndarray'] = 'ivy.mxnd'
_array_types['mxnet.symbol.symbol'] = 'ivy.mxsym'


def _get_framework_from_args(args):
    for arg in args:
        arg_type = type(arg)
        if arg_type in [list, tuple]:
            lib = _get_framework_from_args(*arg)
            if lib:
                return lib
        elif arg_type is dict:
            lib = _get_framework_from_args(*list(arg.values()))
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
            '{} {}'.format(args, kwargs))
    if verbosity.level > 0:
        verbosity.cprint('Using framework from type: {}'.format(f))
    return f


def set_framework(f):
    framework_stack.append(f)
    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))


def unset_framework():
    framework_stack.pop(-1)
    if verbosity.level > 0:
        verbosity.cprint(
            'framework stack: {}'.format(framework_stack))
