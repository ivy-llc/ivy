# global

# local
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_dtype
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import array


class EagerTensor:
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )
        self._dtype = tf_frontend.DType(
            tf_frontend.tensorflow_type_to_enum[self._ivy_array.dtype]
        )

    def __repr__(self):
        return (
            "ivy.frontends.tensorflow.EagerTensor("
            + str(ivy.to_list(self._ivy_array))
            + ",shape="
            + str(self._ivy_array.shape)
            + ","
            + "dtype="
            + str(self._ivy_array.dtype)
            + ")"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def device(self):
        return self._ivy_array.device

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return "TensorShape(" + str(list(self._ivy_array.shape)) + ")"

    # Instance Methods #
    # ---------------- #

    def get_shape(self):
        return tf_frontend.raw_ops.Shape(input=self._ivy_array)

    def set_shape(self, shape):
        if shape is None:
            return

        x_shape = self._ivy_array.shape
        if len(x_shape) != len(shape):
            raise ValueError(
                f"Tensor's shape {x_shape} is not compatible with supplied shape "
                f"{shape}."
            )
        for i, v in enumerate(x_shape):
            if v != shape[i] and (shape[i] is not None):
                raise ValueError(
                    f"Tensor's shape {x_shape} is not compatible with supplied shape "
                    f"{shape}."
                )

    def numpy(self):
        return ivy.to_numpy(self._ivy_array)

    def __add__(self, y, name="add"):
        return y.__radd__(self._ivy_array)

    def __div__(self, x, name="div"):
        return tf_frontend.math.divide(x, self._ivy_array, name=name)

    def __and__(self, y, name="and"):
        return y.__rand__(self._ivy_array)

    def __array__(self, dtype=None, name="array"):
        dtype = to_ivy_dtype(dtype)
        return array(ivy.asarray(self._ivy_array, dtype=dtype))

    def __bool__(self, name="bool"):
        if isinstance(self._ivy_array, int):
            return self._ivy_array != 0

        temp = ivy.squeeze(ivy.asarray(self._ivy_array), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __eq__(self, other):
        return tf_frontend.raw_ops.Equal(
            x=self._ivy_array, y=other, incompatible_shape_error=False
        )

    def __floordiv__(self, y, name="floordiv"):
        return y.__rfloordiv__(self._ivy_array)

    def __ge__(self, y, name="ge"):
        return tf_frontend.raw_ops.GreaterEqual(
            x=self._ivy_array, y=y._ivy_array, name=name
        )

    def __getitem__(self, slice_spec, var=None, name="getitem"):
        ret = ivy.get_item(self._ivy_array, slice_spec)
        return EagerTensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))

    def __gt__(self, y, name="gt"):
        return tf_frontend.raw_ops.Greater(x=self._ivy_array, y=y._ivy_array, name=name)

    def __invert__(self, name="invert"):
        return tf_frontend.raw_ops.Invert(x=self._ivy_array, name=name)

    def __le__(self, y, name="le"):
        return tf_frontend.raw_ops.LessEqual(
            x=self._ivy_array, y=y._ivy_array, name=name
        )

    def __lt__(self, y, name="lt"):
        return tf_frontend.raw_ops.Less(x=self._ivy_array, y=y._ivy_array, name=name)

    def __matmul__(self, y, name="matmul"):
        return y.__rmatmul__(self._ivy_array)

    def __mul__(self, x, name="mul"):
        return tf_frontend.math.multiply(x, self._ivy_array, name=name)

    def __mod__(self, x, name="mod"):
        return ivy.remainder(x, self._ivy_array, name=name)

    def __ne__(self, other):
        return tf_frontend.raw_ops.NotEqual(
            x=self._ivy_array, y=other._ivy_array, incompatible_shape_error=False
        )

    def __neg__(self, name="neg"):
        return tf_frontend.raw_ops.Neg(x=self._ivy_array, name=name)

    __nonzero__ = __bool__

    def __or__(self, y, name="or"):
        return y.__ror__(self._ivy_array)

    def __pow__(self, y, name="pow"):
        return tf_frontend.math.pow(x=self, y=y, name=name)

    def __radd__(self, x, name="radd"):
        return tf_frontend.math.add(x, self._ivy_array, name=name)

    def __rand__(self, x, name="rand"):
        return tf_frontend.math.logical_and(x, self._ivy_array, name=name)

    def __rfloordiv__(self, x, name="rfloordiv"):
        return tf_frontend.raw_ops.FloorDiv(x=x, y=self._ivy_array, name=name)

    def __rmatmul__(self, x, name="rmatmul"):
        return tf_frontend.raw_ops.MatMul(a=x, b=self._ivy_array, name=name)

    def __rmul__(self, x, name="rmul"):
        return tf_frontend.raw_ops.Mul(x=x, y=self._ivy_array, name=name)

    def __ror__(self, x, name="ror"):
        return tf_frontend.raw_ops.LogicalOr(x=x, y=self._ivy_array, name=name)

    def __rpow__(self, x, name="rpow"):
        return tf_frontend.raw_ops.Pow(x=x, y=self._ivy_array, name=name)

    def __rsub__(self, x, name="rsub"):
        return tf_frontend.math.subtract(x, self._ivy_array, name=name)

    def __rtruediv__(self, x, name="rtruediv"):
        return tf_frontend.math.truediv(x, self._ivy_array, name=name)

    def __rxor__(self, x, name="rxor"):
        return tf_frontend.math.logical_xor(x, self._ivy_array, name=name)

    def __sub__(self, y, name="sub"):
        return y.__rsub__(self._ivy_array)

    def __truediv__(self, y, name="truediv"):
        dtype = ivy.dtype(self._ivy_array)
        if dtype in [ivy.uint8, ivy.int8, ivy.uint16, ivy.int16]:
            return ivy.astype(y, ivy.float32).__rtruediv__(
                ivy.astype(self._ivy_array, ivy.float32)
            )
        if dtype in [ivy.uint32, ivy.int32, ivy.uint64, ivy.int64]:
            return ivy.astype(y, ivy.float64).__rtruediv__(
                ivy.astype(self._ivy_array, ivy.float64)
            )
        return y.__rtruediv__(self._ivy_array)

    def __len__(self):
        return len(self._ivy_array)

    def __xor__(self, y, name="xor"):
        return y.__rxor__(self._ivy_array)

    def __setitem__(self, key, value):
        raise ivy.exceptions.IvyException(
            "ivy.functional.frontends.tensorflow.EagerTensor object "
            "doesn't support assignment"
        )
