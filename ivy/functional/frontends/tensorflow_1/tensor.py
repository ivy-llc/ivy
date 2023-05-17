# global

# local
import ivy
import ivy.functional.frontends.tensorflow_1 as tf_1_frontend
from ivy.functional.frontends.tensorflow_1.func_wrapper import (
    to_ivy_dtype,
    _to_ivy_array,
)
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import array


class EagerTensor:
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    def __repr__(self):
        return (
            repr(self.ivy_array).replace(
                "ivy.array", "ivy.frontends.tensorflow_1.EagerTensor"
            )[:-1]
            + ", shape="
            + str(self.shape)
            + ", dtype="
            + str(self.ivy_array.dtype)
            + ")"
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
        return tf_1_frontend.DType(
            tf_1_frontend.tensorflow_type_to_enum[self.ivy_array.dtype]
        )

    @property
    def shape(self):
        return self.ivy_array.shape

    # Instance Methods #
    # ---------------- #

    def __array__(self, dtype=None, name="array"):
        dtype = to_ivy_dtype(dtype)
        return array(ivy.asarray(self.ivy_array, dtype=dtype))

    def __bool__(self, name="bool"):
        if isinstance(self.ivy_array, int):
            return self.ivy_array != 0

        temp = ivy.squeeze(ivy.asarray(self.ivy_array), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __getitem__(self, slice_spec, var=None, name="getitem"):
        ivy_args = ivy.nested_map([self, slice_spec], _to_ivy_array)
        ret = ivy.get_item(*ivy_args)
        return EagerTensor(ret)

    def __len__(self):
        return len(self.ivy_array)

    def __setitem__(self, key, value):
        raise ivy.utils.exceptions.IvyException(
            "ivy.functional.frontends.tensorflow.EagerTensor object "
            "doesn't support assignment"
        )

    def __iter__(self):
        ndim = len(self.shape)
        if ndim == 0:
            raise TypeError("iteration over a 0-d tensor not supported")
        for i in range(self.shape[0]):
            yield self[i]


# Dummy Tensor class to help with compilation, don't add methods here
class Tensor(EagerTensor):
    pass
