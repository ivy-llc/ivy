# global
import numpy as np
from typing import Callable, Union, Sequence, Tuple, Optional

#local
import ivy


def vmap(func: Callable,
         in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
         out_axes: Optional[int] = 0) -> Callable:
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

        # checking uniqueness of axis_size
        axis_size = set()

        if isinstance(in_axes, int):
            for arg in args:
                axis_size.add(arg.shape[in_axes])
        elif isinstance(in_axes, (list, tuple)):
            for (arg, axis) in zip(args, in_axes):
                if axis is not None:
                    axis_size.add(arg.shape[axis])

        if len(axis_size) > 1:
            raise ValueError('''Inconsistent sizes. All mapped axes should have the same size''')

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            assert not all(ax is None for ax in in_axes), "At least one of the axes should be specified (not None)"
        else:
            assert not (in_axes is None), "in_axes should be non-None if integer"

        # Handling None in in_axes by broadcasting the axis_size
        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = list()
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = np.broadcast_to(args[none_mapped_axis],
                                                         (tuple(axis_size) + args[none_mapped_axis].shape))

        # set up the axis to be mapped to index zero.
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                if in_axes[i] is not None:
                    args[i] = np.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = np.moveaxis(args[0], in_axes, 0)

        # vectorisation. To be optimized.
        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = np.stack(arr_results)

        if out_axes:
            res = np.moveaxis(res, 0, out_axes)

        return res

    return _vmap