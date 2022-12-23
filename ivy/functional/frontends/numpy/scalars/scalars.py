# global

# local
import ivy
from ivy.functional.frontends.numpy.ndarray.ndarray import ndarray


class generic(ndarray):
    _name = "generic"

    def __init__(self):
        raise ivy.exceptions.IvyException(
            f"cannot create 'numpy.{self._name}' instances"
        )


class bool_(generic):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="bool")
        self.dtype = "bool"

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
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="int8")
        self.dtype = "int8"


int8 = byte


class short(signedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="int16")
        self.dtype = "int16"


int16 = short


class intc(signedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="int32")
        self.dtype = "int32"


int32 = intc


class int_(signedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="int64")
        self.dtype = "int64"


int64 = intp = int_


class longlong(signedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="int64")
        self.dtype = "int64"


class unsignedinteger(integer):
    _name = "unsignedinteger"
    pass


class ubyte(unsignedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="uint8")
        self.dtype = "uint8"


uint8 = ubyte


class ushort(unsignedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="uint16")
        self.dtype = "uint16"


uint16 = ushort


class uintc(unsignedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="uint32")
        self.dtype = "uint32"


uint32 = uintc


class uint(signedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="uint64")
        self.dtype = "uint64"


uint64 = uintp = uint


class ulonglong(signedinteger):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="uint64")
        self.dtype = "uint64"


class inexact(number):
    _name = "inexact"
    pass


class floating(inexact):
    _name = "floating"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


class half(floating):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="float16")
        self.dtype = "float16"


float16 = half


class single(floating):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="float32")
        self.dtype = "float32"


float32 = single


class double(floating, float):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="float64")
        self.dtype = "float64"


float64 = float_ = double


class complexfloating(inexact):
    _name = "complexfloating"

    def __repr__(self):
        return self.ivy_array.__repr__()[10:-1]


class csingle(complexfloating):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="complex64")
        self.dtype = "complex64"


complex64 = singlecomplex = csingle


class cdouble(complexfloating, complex):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="complex128")
        self.dtype = "complex128"


complex128 = cfloat = complex_ = cdouble


class clongdouble(complexfloating):
    def __init__(self, value=0):
        ndarray.__init__(self, 0)
        self.ivy_array = ivy.array(value, dtype="complex256")
        self.dtype = "complex256"


complex256 = clongfloat = longcomplex = clongdouble
