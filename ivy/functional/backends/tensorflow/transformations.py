# global
import functools
import tensorflow as tf

#local
import ivy


def vmap(fun, in_axes=0, out_axes=0):
    @ivy.to_native_arrays_and_back
    def _vmap(*args):

        # convert args tuple to list to allow mutability in the arg container.
        args = list(args)

        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            try:
                assert (len(args)) == len(in_axes)
            except AssertionError:
                raise Exception('''The in_axis should have length equivalent to the 
                number of positional arguments to the function being vectorized
                or it should be an integer.''')

        # set up the axis to be mapped
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                args[i] = tf.experimental.numpy.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = tf.experimental.numpy.moveaxis(args[0], in_axes, 0)

        # vecotrisation - applying map_fn if only one arg provided as reduce requires
        # two elements to begin with.
        if len(args) == 1:
            ret = tf.map_fn(fun, args[0])
        else:
            ret = functools.reduce(fun, args)

        if out_axes:
            ret = tf.experimental.numpy.moveaxis(ret, 0, out_axes)

        return ret

    return _vmap
