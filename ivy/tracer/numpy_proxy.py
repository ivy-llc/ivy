import numpy as np
import platform


IS_MAC_ARM = platform.system() == "Darwin" and platform.machine() == "arm64"
IS_WINDOWS = platform.system() == "Windows"


class NewNDArray(np.ndarray):
    def __new__(cls, data):
        obj = np.asarray(data) if not isinstance(data, np.ndarray) else data
        new_obj = obj.view(cls)
        new_obj._data = obj
        return new_obj

    def __init__(self, data):
        self._data = np.asarray(data) if not isinstance(data, np.ndarray) else data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._data = getattr(obj, "_data", None)

    def resize(self, new_shape, refcheck=True):
        try:
            self._data.resize(new_shape, refcheck=refcheck)
        except ValueError:
            self._data = np.resize(self._data, new_shape)

    @property
    def data(self):
        return self._data


if not IS_MAC_ARM and not IS_WINDOWS:

    class NewFloat128(np.float128):
        def __init__(self, data):
            self._data = data

        @property
        def data(self):
            return self._data

    class NewComplex256(np.complex256):
        def __init__(self, data):
            self._data = data

        @property
        def data(self):
            return self._data


class NewFloat64(np.float64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewFloat32(np.float32):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewFloat16(np.float16):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewComplex128(np.complex128):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewComplex64(np.complex64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt64(np.int64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt32(np.int32):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt16(np.int16):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt8(np.int8):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint64(np.uint64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint32(np.uint32):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint16(np.uint16):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint8(np.uint8):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewBool(np.bool_):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


NUMPY_TO_CUSTOM = {
    np.ndarray: NewNDArray,
    np.float64: NewFloat64,
    np.float32: NewFloat32,
    np.float16: NewFloat16,
    np.complex128: NewComplex128,
    np.complex64: NewComplex64,
    np.int64: NewInt64,
    np.int32: NewInt32,
    np.int16: NewInt16,
    np.int8: NewInt8,
    np.uint64: NewUint64,
    np.uint32: NewUint32,
    np.uint16: NewUint16,
    np.uint8: NewUint8,
    np.bool_: NewBool,
}

CUSTOM_TO_NUMPY = {value: key for key, value in NUMPY_TO_CUSTOM.items()}

custom_np_classes = [
    NewNDArray,
    NewBool,
    NewFloat64,
    NewFloat32,
    NewFloat16,
    NewComplex128,
    NewComplex64,
    NewInt64,
    NewInt32,
    NewInt16,
    NewInt8,
    NewUint64,
    NewUint32,
    NewUint16,
    NewUint8,
]

custom_np_class_names = [
    "NewNDArray",
    "NewBool",
    "NewFloat64",
    "NewFloat32",
    "NewFloat16",
    "NewComplex128",
    "NewComplex64",
    "NewInt64",
    "NewInt32",
    "NewInt16",
    "NewInt8",
    "NewUint64",
    "NewUint32",
    "NewUint16",
    "NewUint8",
]

if not IS_MAC_ARM and not IS_WINDOWS:
    custom_np_classes.extend([NewFloat128, NewComplex256])
    custom_np_class_names.extend(["NewFloat128", "NewComplex256"])
    NUMPY_TO_CUSTOM.update({np.float128: NewFloat128, np.complex256: NewComplex256})
