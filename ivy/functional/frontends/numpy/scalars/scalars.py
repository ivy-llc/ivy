# global

# local
import ivy
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import array
from ivy.functional.frontends.numpy.ndarray.ndarray import ndarray


class generic(ndarray):
    _name = "generic"

    def __init__(self):
        raise ivy.utils.exceptions.IvyException(
            f"cannot create 'numpy.{self._name}' instances"
        )


class bool_(generic):
    def __new__(cls, value=0):
        ret = array(value, dtype="bool")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "bool")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="bool")
        )

    def __repr__(self):
        return "True" if self.ivy_array else "False"


class number(generic):
    _name = "number"


class bfloat16(generic):
    def __new__(cls, value=0):
        ret = array(value, dtype="bfloat16")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "bfloat16")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="bfloat16")
        )


class integer(number):
    _name = "integer"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


class inexact(number):
    _name = "inexact"


class signedinteger(integer):
    _name = "signedinteger"


class unsignedinteger(integer):
    _name = "unsignedinteger"


class floating(inexact):
    _name = "floating"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


class complexfloating(inexact):
    _name = "complexfloating"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


class byte(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="int8")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "int8")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="int8")
        )


class short(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="int16")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "int16")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="int16")
        )


class intc(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="int32")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "int32")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="int32")
        )


class int_(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="int64")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "int64")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="int64")
        )


class longlong(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="int64")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "int64")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="int64")
        )


class uint(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="uint64")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "uint64")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="uint64")
        )


class ulonglong(signedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="uint64")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "uint64")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="uint64")
        )


class ubyte(unsignedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="uint8")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "uint8")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="uint8")
        )


class ushort(unsignedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="uint16")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "uint16")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="uint16")
        )


class uintc(unsignedinteger):
    def __new__(cls, value=0):
        ret = array(value, dtype="uint32")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "uint32")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="uint32")
        )


class half(floating):
    def __new__(cls, value=0):
        ret = array(value, dtype="float16")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "float16")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="float16")
        )


class single(floating):
    def __new__(cls, value=0):
        ret = array(value, dtype="float32")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "float32")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="float32")
        )


class double(floating, float):
    def __new__(cls, value=0):
        ret = array(value, dtype="float64")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "float64")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="float64")
        )


class csingle(complexfloating):
    def __new__(cls, value=0):
        ret = array(value, dtype="complex64")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "complex64")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="complex64")
        )


class cdouble(complexfloating, complex):
    def __new__(cls, value=0):
        ret = array(value, dtype="complex128")
        if ret.shape != ():
            return ret
        obj = super().__new__(cls)
        return obj

    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = (
            ivy.astype(value.ivy_array, "complex128")
            if hasattr(value, "ivy_array")
            else ivy.array(value, dtype="complex128")
        )


bool8 = bool_
complex128 = cfloat = complex_ = cdouble
complex64 = singlecomplex = csingle
float16 = half
float32 = single
float64 = float_ = double
int16 = short
int32 = intc
int64 = intp = int_
int8 = byte
uint16 = ushort
uint32 = uintc
uint64 = uintp = uint
uint8 = ubyte
