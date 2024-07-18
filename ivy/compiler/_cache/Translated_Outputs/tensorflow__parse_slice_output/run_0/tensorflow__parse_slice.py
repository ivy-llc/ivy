from .tensorflow__helpers import tensorflow_arange
from .tensorflow__helpers import tensorflow_asarray


def tensorflow__parse_slice(idx, s):
    step = 1 if idx.step is None else idx.step
    if step > 0:
        start = 0 if idx.start is None else idx.start
        if start >= s:
            stop = start
        else:
            if start <= -s:
                start = 0
            elif start < 0:
                start = start + s
            stop = s if idx.stop is None else idx.stop
            if stop > s:
                stop = s
            elif start <= -s:
                stop = 0
            elif stop < 0:
                stop = stop + s
    else:
        start = s - 1 if idx.start is None else idx.start
        if start < -s:
            stop = start
        else:
            if start >= s:
                start = s - 1
            elif start < 0:
                start = start + s
            if idx.stop is None:
                stop = -1
            else:
                stop = idx.stop
                if stop > s:
                    stop = s
                elif stop < -s:
                    stop = -1
                elif stop == -s:
                    stop = 0
                elif stop < 0:
                    stop = stop + s
    q_i = tensorflow_arange(start, stop, step)
    ag__result_list_0 = []
    for q in q_i:
        if 0 <= q < s:
            res = q
            ag__result_list_0.append(res)
    q_i = ag__result_list_0
    q_i = (
        tensorflow_asarray(q_i)
        if len(q_i) or start == stop or idx.stop is not None
        else tensorflow_arange(0, s, 1)
    )
    return q_i
