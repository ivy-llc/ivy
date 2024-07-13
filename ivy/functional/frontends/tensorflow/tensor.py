# global

# local
import ivy
from ivy import with_unsupported_dtypes
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow.func_wrapper import _to_ivy_array
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import array


class EagerTensor:
    def __init__(self, array):
        self._ivy_array = array if isinstance(array, ivy.Array) else ivy.array(array)

    def __repr__(self):
        return (
            repr(self.ivy_array).replace(
                "ivy.array", "ivy.frontends.tensorflow.EagerTensor"
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
        return tf_frontend.DType(
            tf_frontend.tensorflow_type_to_enum[self.ivy_array.dtype]
        )

    @property
    def shape(self):
        return TensorShape(self.ivy_array.shape.shape)

    # Instance Methods #
    # ---------------- #

    def get_shape(self):
        return tf_frontend.raw_ops.Shape(input=self)

    def set_shape(self, shape):
        if shape is None:
            return

        x_shape = self.shape
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
        return array(self.ivy_array)

    def __add__(self, y, name="add"):
        return self.__radd__(y)

    def __div__(self, y, name="div"):
        if "int" in self._ivy_array.dtype:
            return tf_frontend.raw_ops.FloorDiv(x=self, y=y, name=name)
        ret = tf_frontend.math.divide(self, y, name=name)
        return tf_frontend.cast(ret, self.dtype)

    def __and__(self, y, name="and"):
        return self.__rand__(y)

    def __array__(self, dtype=None, name="array"):
        if not dtype:
            return ivy.to_numpy(self.ivy_array)
        return ivy.to_numpy(self.ivy_array).astype(dtype)

    def __bool__(self, name="bool"):
        temp = ivy.squeeze(self.ivy_array, axis=None)
        if temp.shape != ():
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )
        return temp != 0

    def __eq__(self, other):
        return tf_frontend.raw_ops.Equal(
            x=self, y=other, incompatible_shape_error=False
        )

    def __floordiv__(self, y, name="floordiv"):
        return tf_frontend.raw_ops.FloorDiv(x=self, y=y, name=name)

    @with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )
    def __ge__(self, y, name="ge"):
        return tf_frontend.raw_ops.GreaterEqual(x=self, y=y, name=name)

    def __getitem__(self, slice_spec, var=None, name="getitem"):
        ivy_args = ivy.nested_map(_to_ivy_array, [self, slice_spec])
        ret = ivy.get_item(*ivy_args)
        return EagerTensor(ret)

    @with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )
    def __gt__(self, y, name="gt"):
        return tf_frontend.raw_ops.Greater(x=self, y=y, name=name)

    def __invert__(self, name="invert"):
        return tf_frontend.raw_ops.Invert(x=self, name=name)

    @with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )
    def __le__(self, y, name="le"):
        return tf_frontend.raw_ops.LessEqual(x=self, y=y, name=name)

    @with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )
    def __lt__(self, y, name="lt"):
        return tf_frontend.raw_ops.Less(x=self, y=y, name=name)

    def __matmul__(self, y, name="matmul"):
        return tf_frontend.linalg.matmul(a=self, b=y, name=name)

    def __mul__(self, y, name="mul"):
        return tf_frontend.math.multiply(self, y, name=name)

    @with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )
    def __mod__(self, y, name="mod"):
        return tf_frontend.floormod(self, y, name=name)

    def __ne__(self, other):
        return tf_frontend.raw_ops.NotEqual(
            x=self, y=other, incompatible_shape_error=False
        )

    def __neg__(self, name="neg"):
        return tf_frontend.raw_ops.Neg(x=self, name=name)

    __nonzero__ = __bool__

    def __or__(self, y, name="or"):
        return self.__ror__(y)

    def __pow__(self, y, name="pow"):
        return tf_frontend.math.pow(x=self, y=y, name=name)

    def __radd__(self, x, name="radd"):
        return tf_frontend.math.add(self, x, name=name)

    def __rand__(self, x, name="rand"):
        return tf_frontend.raw_ops.BitwiseAnd(y=self, x=x, name=name)

    def __rfloordiv__(self, x, name="rfloordiv"):
        return tf_frontend.raw_ops.FloorDiv(x=x, y=self, name=name)

    def __rmatmul__(self, x, name="rmatmul"):
        return tf_frontend.linalg.matmul(a=x, b=self, name=name)

    def __rmul__(self, x, name="rmul"):
        return tf_frontend.raw_ops.Mul(x=self, y=x, name=name)

    def __ror__(self, x, name="ror"):
        return tf_frontend.raw_ops.BitwiseOr(x=self, y=x, name=name)

    def __rpow__(self, x, name="rpow"):
        return tf_frontend.math.pow(x=x, y=self, name=name)

    def __rsub__(self, x, name="rsub"):
        return tf_frontend.math.subtract(x, self, name=name)

    def __rtruediv__(self, x, name="rtruediv"):
        return tf_frontend.math.truediv(x, self, name=name)

    def __rxor__(self, x, name="rxor"):
        return tf_frontend.raw_ops.BitwiseXor(x=self, y=x, name=name)

    def __sub__(self, y, name="sub"):
        return tf_frontend.math.subtract(self, y, name=name)

    def __truediv__(self, y, name="truediv"):
        return tf_frontend.math.truediv(self, y, name=name)

    def __len__(self):
        return len(self.ivy_array)

    def __xor__(self, y, name="xor"):
        return self.__rxor__(y)

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


class TensorShape:
    # TODO: there are still some methods that may need implementing

    def __init__(self, dims):
        self._dims = tuple(dims)

    def __repr__(self):
        if self._dims is not None:
            return f"TensorShape({list(self._dims)})"
        else:
            return "TensorShape(None)"

    def __str__(self):
        if self.rank is None:
            return "<unknown>"
        elif self.rank == 1:
            return f"({self._dims[0]},)"
        else:
            return f'({", ".join(str(d) for d in self._dims)})'

    # Properties #
    # ---------- #

    @property
    def dims(self):
        return self._dims

    @property
    def ivy_shape(self):
        return ivy.Shape(self._dims)

    @property
    def ndims(self):
        return self.__len__()

    @property
    def rank(self):
        return self.__len__()

    # Instance Methods #
    # ---------------- #

    def __add__(self, other):
        return self.concatenate(other)

    def __bool__(self):
        return self._dims is not None

    def __concat__(self, other):
        return self.concatenate(other)

    def __eq__(self, other):
        return self._dims == other.dims

    def __getitem__(self, key):
        if isinstance(key, slice):
            return TensorShape(self._dims[key])
        else:
            return self._dims[key]

    def __iter__(self):
        return iter(d for d in self._dims)

    def __len__(self):
        return len(self._dims)

    def __nonzero__(self):
        return self.__bool__()

    def __radd__(self, other):
        return other.concatenate(self)

    def as_list(self):
        return list(self._dims)

    def concatenate(self, other):
        other = as_shape(other)
        if self.dims is None or other.dims is None:
            return unknown_shape()
        else:
            return TensorShape(self.dims + other.dims)

    def num_elements(self):
        return ivy.to_scalar(ivy.prod(self._dims))


# Dummy Tensor class to help with compilation, don't add methods here
class Tensor(EagerTensor):
    pass


# Helpers


def as_shape(shape):
    """Converts the given object to a TensorShape."""
    if isinstance(shape, TensorShape):
        return shape
    else:
        return TensorShape(shape)


def unknown_shape(rank=None, **kwargs):
    if rank is None and "ndims" in kwargs:
        rank = kwargs.pop("ndims")
    if kwargs:
        raise TypeError(f"Unknown argument: {kwargs}")
    if rank is None:
        return TensorShape(None)
    else:
        return TensorShape([None] * rank)
