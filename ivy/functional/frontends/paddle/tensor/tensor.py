# local
import ivy
import ivy.functional.frontends.paddle as paddle_frontend
from ivy.functional.frontends.paddle.func_wrapper import _to_ivy_array
from ivy.func_wrapper import with_unsupported_dtypes


class Tensor:
    def __init__(self, array, dtype=None, place="cpu", stop_gradient=True):
        self._ivy_array = (
            ivy.array(array, dtype=dtype, device=place)
            if not isinstance(array, ivy.Array)
            else array
        )
        self._dtype = dtype
        self._place = place
        self._stop_gradient = stop_gradient

    def __repr__(self):
        return (
            str(self._ivy_array.__repr__())
            .replace("ivy.array", "ivy.frontends.paddle.Tensor")
            .replace("dev", "place")
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def place(self):
        return self.ivy_array.device

    @property
    def dtype(self):
        return self._ivy_array.dtype

    @property
    def shape(self):
        return self._ivy_array.shape

    @property
    def ndim(self):
        return self.dim()

    # Setters #
    # --------#

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    # Special Methods #
    # -------------------#

    def __getitem__(self, item):
        ivy_args = ivy.nested_map([self, item], _to_ivy_array)
        ret = ivy.get_item(*ivy_args)
        return paddle_frontend.Tensor(ret)

    def __setitem__(self, item, value):
        item, value = ivy.nested_map([item, value], _to_ivy_array)
        self.ivy_array[item] = value

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor not supported")
        for i in range(self.shape[0]):
            yield self[i]

    # Instance Methods #
    # ---------------- #

    def reshape(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            return paddle_frontend.reshape(self._ivy_array, shape)
        if args:
            if isinstance(args[0], (tuple, list)):
                shape = args[0]
                return paddle_frontend.reshape(self._ivy_array, shape)
            else:
                return paddle_frontend.reshape(self._ivy_array, args)
        return paddle_frontend.reshape(self._ivy_array)

    def dim(self):
        return self.ivy_array.ndim

    @with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
    def abs(self):
        return paddle_frontend.abs(self)
