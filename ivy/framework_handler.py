# global
import ivy
import logging
import importlib
import collections
import numpy as np
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
_array_types['jaxlib.xla_extension'] = 'ivy.jax'
_array_types['tensorflow.python.framework.ops'] = 'ivy.tensorflow'
_array_types['torch'] = 'ivy.torch'
_array_types['mxnet.ndarray.ndarray'] = 'ivy.mxnet'

_framework_dict = dict()
_framework_dict['numpy'] = 'ivy.numpy'
_framework_dict['jax'] = 'ivy.jax'
_framework_dict['tensorflow'] = 'ivy.tensorflow'
_framework_dict['torch'] = 'ivy.torch'
_framework_dict['mxnet'] = 'ivy.mxnet'

_framework_reverse_dict = dict()
_framework_reverse_dict['ivy.numpy'] = 'numpy'
_framework_reverse_dict['ivy.jax'] = 'jax'
_framework_reverse_dict['ivy.tensorflow'] = 'tensorflow'
_framework_reverse_dict['ivy.torch'] = 'torch'
_framework_reverse_dict['ivy.mxnet'] = 'mxnet'


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


def clear_framework_stack():
    while framework_stack:
        unset_framework()

# Framework Getters #
# ------------------#

def try_import_ivy_jax(warn=False):
    try:
        import ivy.jax
        return ivy.jax
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\nEither jax or jaxlib appear to not be installed, '
                        'ivy.jax can therefore not be imported.\n'.format(e))


def try_import_ivy_tf(warn=False):
    try:
        import ivy.tensorflow
        return ivy.tensorflow
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\ntensorflow does not appear to be installed, '
                        'ivy.tensorflow can therefore not be imported.\n'.format(e))


def try_import_ivy_torch(warn=False):
    try:
        import ivy.torch
        return ivy.torch
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\ntorch does not appear to be installed, '
                        'ivy.torch can therefore not be imported.\n'.format(e))


def try_import_ivy_mxnet(warn=False):
    try:
        import ivy.mxnet
        return ivy.mxnet
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\nmxnet does not appear to be installed, '
                        'ivy.mxnet can therefore not be imported.\n'.format(e))


def try_import_ivy_numpy(warn=False):
    try:
        import ivy.numpy
        return ivy.numpy
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\nnumpy does not appear to be installed, '
                        'ivy.numpy can therefore not be imported.\n'.format(e))


FW_DICT = {'jax': try_import_ivy_jax,
           'tensorflow': try_import_ivy_tf,
           'torch': try_import_ivy_torch,
           'mxnet': try_import_ivy_mxnet,
           'numpy': try_import_ivy_numpy}


def choose_random_framework(excluded=None):
    excluded = list() if excluded is None else excluded
    while True:
        if len(excluded) == 5:
            raise Exception('Unable to select framework, all backends are either excluded or not installed.')
        f = np.random.choice([f_srt for f_srt in list(FW_DICT.keys()) if f_srt not in excluded])
        if f is None:
            excluded.append(f)
            continue
        else:
            print('\nselected framework: {}\n'.format(f))
            return f
