# global
import numpy as np
from functools import reduce
#local
import ivy
from ivy.functional.backends.numpy import unstack
import jax

def np_map_fn(fn, elems, axis=0):
    return np.stack([fn(elem) for elem in unstack(elems, axis)])


# def vmap(func, in_axes=0, out_axes=0):
#
#     @ivy.to_native_arrays_and_back
#     def new_fn(*args):
#         args = list(args)
#         if jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)(*args) is None:
#             return None
#
#         if isinstance(in_axes, (list, tuple)):
#             try:
#                 assert len(args) == len(in_axes)
#             except AssertionError:
#                 raise Exception("Length of in_axis and positional args incompatible")
#             for i in range(len(in_axis)):
#                 args[i] = np.moveaxis(args[i], in_axis[i], 0)
#         elif isinstance(in_axes, int):
#             args[0] = np.moveaxis(args[0], in_axes, 0)
#
#         ret = np_map_fn(func, *args)
#
#
#         if out_axes:
#             ret = np.moveaxis(ret, 0, out_axes)
#         return ret
#     return new_fn


def vmap(func, in_axes=0, out_axes=0):
    @ivy.to_native_arrays_and_back
    def _vmap(*args):

        # convert args tuple to list to allow mutability using moveaxis ahead.
        args = list(args)

        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            try:
                assert (len(args)) == len(in_axes)
            except AssertionError:
                raise Exception('''The in_axes should have length equivalent to the 
                number of positional arguments to the function being vectorized
                or it should be an integer.''')

        # checking axis_size consistency
        axis_size = set()

        if isinstance(in_axes, int):
            for arg in args:
                axis_size.add(arg.shape[in_axes])
        elif isinstance(in_axes, (list, tuple)):
            for (arg, axis) in zip(args, in_axes):
                if axis is not None:
                    axis_size.add(arg.shape[axis])

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            assert not all(ax is None for ax in in_axes), "All in_axes should be non-None"
        elif isinstance(in_axes, int):
            assert not (in_axes is None), "in_axes should be non-None if integer"


        # Handling None in in_axes

        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = list()
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = np.broadcast_to(args[none_mapped_axis],
                                                         (tuple(axis_size) + args[none_mapped_axis].shape))


        if len(axis_size) > 1:
            raise ValueError('''Inconsistent sizes. All axes should have the same size''')


        # set up the axis to be mapped
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                if in_axes[i] is not None:
                    args[i] = np.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = np.moveaxis(args[0], in_axes, 0)

        # vectorisation - applying map_fn if only one arg provided as reduce requires
        # two elements to begin with.


        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = np.stack(arr_results)

        if out_axes:
            res = np.moveaxis(res, 0, out_axes)

        return res

    return _vmap