# global
import ivy
import sys
import copy
import types
import numbers
import inspect
import functools
import numpy as np

# local
from ivy.compiler import globals as glob


def _generate_pid():
    return np.random.randint(0, sys.maxsize)


# noinspection PyProtectedMember
def _clone_param(x, graph):
    glob.wrapping_paused = True
    orig_id = id(x)
    if hasattr(x, '__dict__'):
        if ivy.is_array(x):
            x_copy = ivy.copy_array(x)
        else:
            x_copy = copy.copy(x)
        new_id = _generate_pid()
    elif ivy.is_array(x):
        x_copy = ivy.copy_array(x)
        new_id = id(x_copy)
    else:
        x_copy = copy.copy(x)
        new_id = id(x_copy)
    if orig_id in graph._stateful_clone_pid_dict:
        graph._stateful_clone_pid_dict[new_id] = graph._stateful_clone_pid_dict[orig_id]
    if hasattr(x_copy, '__dict__'):
        x_copy.__dict__['param_id'] = new_id  # update the id of the new param
    if hasattr(x, '__dict__'):
        x.__dict__['param_id'] = new_id  # update the id of the original param (for preserved stateful objects)
    glob.wrapping_paused = False
    return x_copy


def _get_id(x):
    glob.wrapping_paused = True
    if hasattr(x, 'param_id'):
        pid_raw = x.__dict__['param_id']
    else:
        pid_raw = id(x)
    glob.wrapping_paused = False
    return pid_raw


def _get_unique_id(x):
    glob.wrapping_paused = True
    if hasattr(x, 'param_id'):
        unique_pid = x.__dict__['param_id']
        glob.wrapping_paused = False
        return unique_pid
    pid = id(x)
    if pid in glob.raw_pids_to_weakrefs and not ivy.exists(glob.raw_pids_to_weakrefs[pid]()):
        glob.raw_pids_to_weakrefs[pid] = lambda: False
        glob.raw_pids_to_unique_pids[pid] = np.random.randint(0, sys.maxsize)
    unique_pid = glob.raw_pids_to_unique_pids[pid] if pid in glob.raw_pids_to_unique_pids else pid
    glob.wrapping_paused = False
    return unique_pid


def _delete_dependent_param(x, graph):
    _pid = _get_unique_id(x)
    if _pid not in glob.dependent_pids and graph.with_array_caching:
        return x


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


def _terminal_pids_to_key(terminal_pids):
    return '_'.join([str(pid) for pid in terminal_pids])


def _args_str_from_fn(fn):
    if hasattr(fn, 'arg_n_kwarg_reprs'):
        return fn.arg_n_kwarg_reprs
    else:
        return ''


def _args_n_kwarg_reprs_from_keys_n_args_n_kwargs(keys, args, kwargs):
    return '\n'.join(
        ['{}: {}'.format(k, v) for k, v in dict(**dict(
            [(keys[i] if i < len(keys) else str(i),
              _to_label(a)) for i, a in enumerate(args)]), **kwargs).items()])


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
    if hasattr(fn, 'output_reprs'):
        return fn.output_reprs
    else:
        return ''


def _output_reprs_from_output(output):
    return '\n'.join(
        ['{}: {}'.format(k, v) for k, v in dict([(str(i), _to_label(a)) for i, a in enumerate(output)]).items()])


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
