import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
import ivy.functional.frontends.numpy as np_frontend


@to_ivy_arrays_and_back
def can_cast(from_, to, casting="safe"):
    ivy.utils.assertions.check_elem_in_list(
        casting,
        ["no", "equiv", "safe", "same_kind", "unsafe"],
        message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
    )

    if ivy.is_array(from_):
        from_ = ivy.as_ivy_dtype(ivy.dtype(from_))
    elif isinstance(from_, (str, type)):
        from_ = ivy.as_ivy_dtype(from_)
    elif isinstance(from_, np_frontend.dtype):
        from_ = from_._ivy_dtype
    else:
        raise ivy.utils.exceptions.IvyException(
            "from_ must be one of dtype, dtype specifier, scalar, or array"
        )

    if isinstance(to, (str, type)):
        to = ivy.as_ivy_dtype(to)
    elif isinstance(to, np_frontend.dtype):
        to = to._ivy_dtype
    else:
        raise ivy.utils.exceptions.IvyException("to must be dtype or dtype specifier")

    if casting == "no" or casting == "equiv":
        return from_ == to

    if casting == "safe" and to in np_frontend.numpy_casting_rules[from_]:
        return True

    if casting == "same_kind":
        if from_ == to or "bool" in from_:
            return True
        if "int" in from_ and ("float" in to or "complex" in to):
            return True
        elif "float" in from_ and ("float" in to or "complex" in to):
            return True
        elif "uint" in from_ and ("int" in to or "float" in to or "complex" in to):
            return True
        elif "int" in from_ and "int" in to and "uint" not in to:
            return True
        else:
            return to in np_frontend.numpy_casting_rules[from_]
    if casting == "unsafe":
        return True
    return False


def promote_types(type1, type2, /):
    if isinstance(type1, np_frontend.dtype):
        type1 = type1._ivy_dtype
    if isinstance(type2, np_frontend.dtype):
        type2 = type2._ivy_dtype
    return np_frontend.dtype(np_frontend.promote_numpy_dtypes(type1, type2))


# dtypes as string
all_int_dtypes = ["int8", "int16", "int32", "int64"]
all_uint_dtypes = ["uint8", "uint16", "uint32", "uint64"]
all_float_dtypes = [
    "float16",
    "float32",
    "float64",
]
all_complex_dtypes = ["complex64", "complex128"]


def min_scalar_type(a, /):
    if ivy.is_array(a) and a.shape == ():
        a = a.item()
    if np_frontend.isscalar(a):
        validation_dtype = type(a)
        if "int" in validation_dtype.__name__:
            for dtype in all_uint_dtypes:
                if np_frontend.iinfo(dtype).min <= a <= np_frontend.iinfo(dtype).max:
                    return np_frontend.dtype(dtype)
            for dtype in all_int_dtypes:
                if np_frontend.iinfo(dtype).min <= a <= np_frontend.iinfo(dtype).max:
                    return np_frontend.dtype(dtype)

        elif "float" in validation_dtype.__name__:
            for dtype in all_float_dtypes:
                if np_frontend.finfo(dtype).min <= a <= np_frontend.finfo(dtype).max:
                    return np_frontend.dtype(dtype)
        elif "complex" in validation_dtype.__name__:
            for dtype in all_complex_dtypes:
                if np_frontend.finfo(dtype).min <= a <= np_frontend.finfo(dtype).max:
                    return np_frontend.dtype(dtype)
        else:
            return np_frontend.dtype(validation_dtype)
    else:
        return np_frontend.dtype(a.dtype)


@to_ivy_arrays_and_back
def result_type(*arrays_and_dtypes):
    if len(arrays_and_dtypes) == 0:
        raise ivy.utils.exceptions.IvyException(
            "At least one array or dtype must be provided"
        )
    if len(arrays_and_dtypes) == 1:
        if isinstance(arrays_and_dtypes[0], np_frontend.dtype):
            return arrays_and_dtypes[0]
        else:
            return np_frontend.dtype(arrays_and_dtypes[0].dtype)
    else:
        res = (
            arrays_and_dtypes[0]
            if not ivy.is_array(arrays_and_dtypes[0])
            else np_frontend.dtype(arrays_and_dtypes[0].dtype)
        )
        for elem in arrays_and_dtypes:
            if ivy.is_array(elem):
                elem = np_frontend.dtype(elem.dtype)
            res = promote_types(res, elem)
        return res
