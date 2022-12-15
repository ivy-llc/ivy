import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
import ivy.functional.frontends.numpy as np_frontend


@to_ivy_arrays_and_back
def can_cast(from_, to, casting="safe"):
    ivy.assertions.check_elem_in_list(
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
        raise ivy.exceptions.IvyException(
            "from_ must be one of dtype, dtype specifier, scalar, or array"
        )

    if isinstance(to, (str, type)):
        to = ivy.as_ivy_dtype(to)
    elif isinstance(to, np_frontend.dtype):
        to = to._ivy_dtype
    else:
        raise ivy.exceptions.IvyException("to must be dtype or dtype specifier")

    if casting == "no" or casting == "equiv":
        return from_ == to

    if casting == "safe" and to in np_frontend.numpy_casting_rules[from_]:
        return True

    if casting == "same_kind":
        if from_ == to or "bool" in from_:
            return True
        if "int" in from_ and "float" in to:
            return True
        elif "float" in from_ and "float" in to:
            return True
        elif "uint" in from_ and ("int" in to or "float" in to):
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
