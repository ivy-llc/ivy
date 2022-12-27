# local
import ivy
import ivy.functional.frontends.tensorflow as tensorflow_frontend
import ivy.functional.frontends.numpy as np_frontend
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back


class DType:
    def __init__(self, dtype_int):
        self._ivy_dtype = tensorflow_frontend.tensorflow_enum_to_type[dtype_int]

    def __repr__(self):
        return "ivy.frontends.tensorflow." + self._ivy_dtype

    @property
    def as_datatype_enum(self):
        return tensorflow_frontend.tensorflow_type_to_enum[self._ivy_dtype]

    @property
    def as_numpy_dtype(self):
        return np_frontend.dtype(self._ivy_dtype)

    @property
    def base_dtype(self):
        return self

    @property
    def is_bool(self):
        return self._ivy_dtype.is_bool_dtype

    @property
    def is_complex(self):
        return "complex" in self._ivy_dtype

    @property
    def is_floating(self):
        return self._ivy_dtype.is_float_dtype

    @property
    def is_integer(self):
        return self._ivy_dtype.is_int_dtype

    @property
    def is_numpy_compatible(self):
        return self._ivy_dtype in np_frontend.numpy_type_to_str_and_num_table

    @property
    def is_unsigned(self):
        return self._ivy_dtype.is_uint_dtype

    @property
    def limits(self):
        if self._ivy_dtype is ivy.bool:
            return False, True
        if self._ivy_dtype.is_int_dtype:
            return 0, self._ivy_dtype.info.max
        if self._ivy_dtype.is_float_dtype:
            return 0, 1
        else:
            raise ivy.exceptions.IvyException(
                f"{self._ivy_dtype} does not have defined limits"
            )

    @property
    def max(self):
        if self._ivy_dtype in (ivy.bool, ivy.complex128, ivy.complex64):
            raise ivy.exceptions.IvyException(
                f"Cannot find maximum value of {self._ivy_dtype}"
            )
        if self._ivy_dtype is ivy.bfloat16:
            return float.fromhex("0x1.FEp127")
        return self._ivy_dtype.info.max

    @property
    def min(self):
        if self._ivy_dtype in (ivy.bool, ivy.complex128, ivy.complex64):
            raise ivy.exceptions.IvyException(
                f"Cannot find maximum value of {self._ivy_dtype}"
            )
        if self._ivy_dtype is ivy.bfloat16:
            return float.fromhex("-0x1.FEp127")
        return self._ivy_dtype.info.min

    @property
    def real_dtype(self):
        if self._ivy_dtype is ivy.complex64:
            return DType(1)
        if self._ivy_dtype is ivy.complex128:
            return DType(2)
        else:
            return self

    def __eq__(self, other):
        if other is None:
            return False

        if type(other) != DType:  # pylint: disable=unidiomatic-typecheck
            try:
                other = as_dtype(other)
            except ivy.exceptions.IvyException:
                return False

        return self._ivy_dtype == other._ivy_dtype

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))


def as_dtype(type_value):
    if isinstance(type_value, DType):
        return type_value
    if type_value in tensorflow_frontend.tensorflow_type_to_enum:
        return DType(tensorflow_frontend.tensorflow_type_to_enum[type_value])
    if isinstance(type_value, np_frontend.dtype):
        return DType(tensorflow_frontend.tensorflow_type_to_enum[type_value._ivy_dtype])
    if type_value in tensorflow_frontend.tensorflow_enum_to_type:
        return DType(type_value)
    raise ivy.exceptions.IvyException(
        f"Cannot convert the argument 'type_value': {type_value!r} "
        "to a TensorFlow Dtype"
    )


@to_ivy_arrays_and_back
def cast(x, dtype, name=None):
    assert ivy.can_cast(x.dtype, dtype), "Cannot cast from {} to {}".format(
        x.dtype, dtype
    )
    return ivy.astype(x, dtype, copy=False)
