# global
import numpy as np
from functools import reduce

#local
import ivy
from ivy.functional.backends.numpy import unstack

def np_map_fn(fn, elems, axis=0):
    return np.stack([fn(elem) for elem in unstack(elems, axis)])


def vmap(func, in_axis=0, out_axis=0):

    if isinstance(in_axis, list):
        in_axis = tuple(in_axis)

    @ivy.to_native_arrays_and_back
    def new_fn(*args):
        args = list(args)
        if isinstance(in_axis, (list, tuple)):
            try:
                assert (len(args)) == len(in_axis)
            except AssertionError:
                raise Exception("Length of in_axis and positional args incompatible")

        if isinstance(in_axis, (tuple, list)):
            for i in range(len(in_axis)):
                args[i] = np.moveaxis(args[i], in_axis[i], 0)
        elif isinstance(in_axis, int):
            args[0] = np.moveaxis(args[0], in_axis, 0)
        if len(args) == 1:
            ret = np_map_fn(func, args[0])
        else:
            ret = reduce(func, args)

        if out_axis:
            ret = np.moveaxis(ret, 0, out_axis)
        return ret
    return new_fn
