# global
import ivy
import sys
import types
import numbers
import inspect
import functools
import numpy as np

# local
from ivy.compiler import globals as glob


def _get_shape(x_in):
    glob.wrapping_paused = True
    if hasattr(x_in, 'shape') or hasattr(x_in, '__dict__') and 'shape' in x_in.__dict__:
        # noinspection PyBroadException
        try:
            glob.wrapping_paused = False
            return tuple(x_in.shape)
        except Exception:
            glob.wrapping_paused = False
            return None
    glob.wrapping_paused = False
    return None


def _get_id(x):
    glob.wrapping_paused = True
    if hasattr(x, 'param_id'):
        pid_raw = x.__dict__['param_id']
    else:
        pid_raw = id(x)
    if pid_raw in glob.params_removed_from_args and not ivy.exists(glob.params_removed_from_args[pid_raw]()):
        del glob.params_removed_from_args[pid_raw]
        glob.pid_to_unique_id_dict[pid_raw] = np.random.randint(0, sys.maxsize)
    glob.wrapping_paused = False
    return glob.pid_to_unique_id_dict[pid_raw] if pid_raw in glob.pid_to_unique_id_dict else pid_raw


def _terminal_pids_to_key(terminal_pids):
    return '_'.join([str(pid) for pid in terminal_pids])


def _args_str_from_fn(fn):
    if hasattr(fn, 'args') and hasattr(fn, 'kwargs'):
        keys = list(fn.signature.keys())
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in dict(**dict(
                [(keys[i] if i < len(keys) else str(i),
                  _to_label(a)) for i, a in enumerate(fn.args)]), **fn.kwargs).items()])
    else:
        return ''


def _to_label(x):
    return _format_label(
        type(x), ivy.default(lambda: tuple(x.shape), x if isinstance(x, (str, numbers.Number)) else None, True))


def _format_label(cls, shape_or_val):
    ptype_str = '{}'.format(cls).replace(
        "'", '').replace(' ', '').replace('<', '').replace('>', '').replace('class', '').split('.')[-1]
    if ivy.exists(shape_or_val):
        return ptype_str + ', {}'.format(shape_or_val)
    return ptype_str


def _output_str_from_fn(fn):
    if hasattr(fn, 'output'):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in dict([(str(i), _to_label(a)) for i, a in enumerate(fn.output)]).items()])
    else:
        return ''


def _param_to_label(param):
    return _format_label(param.ptype, param.shape)


def _copy_func(f):
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def _get_fn_signature(fn):
    try:
        return dict(inspect.signature(fn).parameters)
    except ValueError:
        return {}
