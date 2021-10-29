# global
import ivy
import numbers

# local
import ivy.compiler.globals as glob


def _get_shape(x_in):
    # noinspection PyBroadException
    try:
        return tuple(x_in.shape)
    except Exception:
        return None


def _get_id(x):
    glob.wrapping_paused = True
    if hasattr(x, 'param_id'):
        glob.wrapping_paused = False
        return x.__dict__['param_id']
    glob.wrapping_paused = False
    return id(x)


def _terminal_pids_to_key(terminal_pids):
    return '_'.join([str(pid) for pid in terminal_pids])


def _args_str_from_fn(fn):
    if hasattr(fn, 'args') and hasattr(fn, 'kwargs'):
        return '\n'.join(
            ['{}: {}'.format(k, v) for k, v in dict(**dict(
                [(str(i), _to_label(a)) for i, a in enumerate(fn.args)]), **fn.kwargs).items()])
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
