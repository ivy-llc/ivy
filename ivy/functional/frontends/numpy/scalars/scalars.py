# global

# local
import ivy
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import array
from ivy.functional.frontends.numpy.ndarray.ndarray import ndarray


class generic(ndarray):
    _name = "generic"

    def __init__(self):
        raise ivy.exceptions.IvyException(
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


bool8 = bool_


class number(generic):
    _name = "number"
    pass


class integer(number):
    _name = "integer"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


class signedinteger(integer):
    _name = "signedinteger"
    pass


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


int8 = byte


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


int16 = short


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


int32 = intc


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


int64 = intp = int_


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


class unsignedinteger(integer):
    _name = "unsignedinteger"
    pass


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


uint8 = ubyte


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


uint16 = ushort


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


uint32 = uintc


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


uint64 = uintp = uint


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


class inexact(number):
    _name = "inexact"
    pass


class floating(inexact):
    _name = "floating"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


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


float16 = half


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


float32 = single


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


float64 = float_ = double


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


class complexfloating(inexact):
    _name = "complexfloating"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


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


complex64 = singlecomplex = csingle


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


complex128 = cfloat = complex_ = cdouble
