# global

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class dtype:
    def __init__(self, dtype_in, align=False, copy=False):
        self._ivy_dtype = (
            to_ivy_dtype(dtype_in)
            if not isinstance(dtype_in, dtype)
            else dtype_in._ivy_dtype
        )

    def __repr__(self):
        return "ivy.frontends.numpy.dtype('" + self._ivy_dtype + "')"

    def __ge__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self == np_frontend.promote_numpy_dtypes(
            self._ivy_dtype, other._ivy_dtype
        )

    def __gt__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self >= other and self != other

    def __lt__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self != np_frontend.promote_numpy_dtypes(
            self._ivy_dtype, other._ivy_dtype
        )

    def __le__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self < other or self == other

    @property
    def type(self):
        return np_frontend.numpy_dtype_to_scalar[self._ivy_dtype]

    @property
    def alignment(self):
        if self._ivy_dtype.is_bool_dtype:
            return 1
        return self._ivy_dtype.dtype_bits // 8

    @property
    def base(self):
        return self

    @property
    def char(self):
        return np_frontend.numpy_type_to_str_and_num_table[self._ivy_dtype][0]

    @property
    def byteorder(self):
        if self._ivy_dtype[-1] == 8:
            return "|"
        else:
            return "="

    @property
    def itemsize(self):
        return self._ivy_dtype.dtype_bits // 8

    @property
    def kind(self):
        if self._ivy_dtype.is_bool_dtype:
            return "b"
        elif self._ivy_dtype.is_int_dtype:
            return "i"
        elif self._ivy_dtype.is_uint_dtype:
            return "u"
        elif self._ivy_dtype.is_float_dtype:
            return "f"
        else:
            return "V"

    @property
    def num(self):
        return np_frontend.numpy_type_to_str_and_num_table[self._ivy_dtype][1]

    @property
    def shape(self):
        return ()

    @property
    def str(self):
        if self._ivy_dtype.is_bool_dtype:
            return "|b1"
        elif self._ivy_dtype.is_uint_dtype:
            if self._ivy_dtype[4::] == "8":
                return "|u1"
            return "<u" + str(self.alignment)
        elif self._ivy_dtype.is_int_dtype:
            if self._ivy_dtype[3::] == "8":
                return "|i1"
            return "<i" + str(self.alignment)
        elif self._ivy_dtype.is_float_dtype:
            return "<f" + str(self.alignment)

    @property
    def subtype(self):
        return None

    @property
    def ivy_dtype(self):
        return self._ivy_dtype

    @property
    def name(self):
        return self._ivy_dtype.__repr__()


def to_ivy_dtype(dtype_in):
    if dtype_in is None:
        return
    if isinstance(dtype_in, ivy.Dtype):
        return dtype_in
    if isinstance(dtype_in, str):
        if dtype_in.strip("><=") in np_frontend.numpy_str_to_type_table:
            return ivy.Dtype(np_frontend.numpy_str_to_type_table[dtype_in.strip("><=")])
        return ivy.Dtype(dtype_in)
    if ivy.is_native_dtype(dtype_in):
        return ivy.as_ivy_dtype(dtype_in)
    if dtype_in in (int, float, bool):
        return {int: ivy.int64, float: ivy.float64, bool: ivy.bool}[dtype_in]
    if isinstance(dtype_in, np_frontend.dtype):
        return dtype_in.ivy_dtype
    if isinstance(dtype_in, type):
        if issubclass(dtype_in, np_frontend.generic):
            return np_frontend.numpy_scalar_to_dtype[dtype_in]
        if hasattr(dtype_in, "dtype"):
            return dtype_in.dtype.ivy_dtype
    else:
        return ivy.as_ivy_dtype(dtype_in)
