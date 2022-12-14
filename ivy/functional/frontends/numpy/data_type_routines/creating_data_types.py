# global

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class dtype:
    def __init__(self, dtype, align=False, copy=False):
        self._ivy_dtype = (
            ivy.Dtype(dtype) if not isinstance(dtype, ivy.Dtype) else dtype
        )

    def __repr__(self):
        return "ivy.frontends.numpy.dtype('" + self._ivy_dtype + "')"

    def __ge__(self, other):
        if isinstance(other, str):
            other = dtype(other)

        if not isinstance(other, dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self == np_frontend.promote_numpy_dtypes(self, other)

    def __gt__(self, other):
        if isinstance(other, str):
            other = dtype(other)

        if not isinstance(other, dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self >= other and self != other

    def __lt__(self, other):
        if isinstance(other, str):
            other = dtype(other)

        if not isinstance(other, dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self != np_frontend.promote_numpy_dtypes(self, other)

    def __le__(self, other):
        if isinstance(other, str):
            other = dtype(other)

        if not isinstance(other, dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self < other or self == other

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
