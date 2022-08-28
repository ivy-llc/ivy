# global
import ivy


def tile(A, reps):
    return ivy.tile(A, reps)


tile.unsupported_dtypes = {
    "tensorflow": ("uint8", "uint16", "uint32", "int8", "int16"),
    "torch": ("uint16", "uint32", "uint64"),
}


def repeat(a, repeats, axis=None):
    return ivy.repeat(a, repeats, axis=axis)


repeat.supported_dtypes = {"tensorflow": ("int32", "int64")}
