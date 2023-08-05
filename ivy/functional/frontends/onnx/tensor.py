# global

# local
import ivy

# import ivy.functional.frontends.onnx as onnx_frontend


class Tensor:
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    def __len__(self):
        return len(self._ivy_array)

    def __repr__(self):
        return str(self.ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.onnx.Tensor"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def device(self):
        return self.ivy_array.device

    @property
    def dtype(self):
        return self.ivy_array.dtype

    @property
    def shape(self):
        return self.ivy_array.shape

    @property
    def ndim(self):
        return self.ivy_array.ndim

    # Setters #
    # --------#

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )
