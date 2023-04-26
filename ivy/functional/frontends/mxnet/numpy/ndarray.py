# global

# local
import ivy
import ivy.functional.frontends.mxnet as mxnet_frontend


class ndarray:
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    def __repr__(self):
        return str(self.ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.mxnet.numpy.array"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def dtype(self):
        return self.ivy_array.dtype

    @property
    def shape(self):
        return self.ivy_array.shape

    # Instance Methods #
    # ---------------- #

    def __add__(self, other):
        return mxnet_frontend.numpy.add(self, other)
