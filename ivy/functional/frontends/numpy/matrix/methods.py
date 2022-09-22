# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


class matrix:
    def __init__(self, data, dtype=None, copy=True):
        pass

    def _init_data(self, data):
        if isinstance(data, str):
            pass
        elif isinstance(data, list) or ivy.is_array(data):
            data = ivy.array(data)
            ivy.assertions.check_equal(len(ivy.shape(data)), 2)
            self._data = data
        else:
            raise ivy.exceptions.IvyException("data must be a 2D array, list, or str")

    # Properties #
    # ---------- #

    @property
    def A():
        pass

    @property
    def A1():
        pass

    @property
    def H():
        pass

    # @property
    # def I():
    #     pass

    @property
    def T():
        pass

    @property
    def data():
        pass

    @property
    def dtype():
        pass

    @property
    def flat():
        pass

    @property
    def itemsize():
        pass

    @property
    def nbytes():
        pass

    @property
    def ndim():
        pass

    @property
    def shape():
        pass

    @property
    def size():
        pass

    # Setters #
    # ------- #

    @data.setter
    def data():
        pass


@from_zero_dim_arrays_to_float
def argmax(a, axis=None, out=None, *, keepdims=None):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


def any(x, /, axis=None, out=None, keepdims=False, *, where=True):
    ret = ivy.where(ivy.array(where), ivy.array(x), ivy.zeros_like(x))
    ret = ivy.any(ret, axis=axis, keepdims=keepdims, out=out)
    return ret
