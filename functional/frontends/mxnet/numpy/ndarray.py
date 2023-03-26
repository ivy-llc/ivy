# global

# local
import ivy
import ivy.functional.frontends.mxnet as mxnet_frontend
from ivy.functional.frontends.numpy import dtype


class ndarray:
<<<<<<< HEAD
    # TODO Add dtype support
=======
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    def __repr__(self):
        return str(self._ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.mxnet.numpy.array"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def dtype(self):
        return dtype(self._ivy_array.dtype)

    @property
    def shape(self):
        return self._ivy_array.shape

    # Instance Methods #
    # ---------------- #

    def __add__(self, other):
        return mxnet_frontend.numpy.add(self, other)
