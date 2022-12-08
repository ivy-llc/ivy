import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.numpy import numpy_casting_rules


@to_ivy_arrays_and_back
def can_cast(from_, to, casting="safe"):
    ivy.assertions.check_elem_in_list(
        casting,
        ["no", "equiv", "safe", "same_kind", "unsafe"],
        message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
    )
    if casting == "no" or casting == "equiv":
        return from_ == to

    if ivy.is_array(from_):
        from_ = ivy.as_ivy_dtype(ivy.dtype(from_))
    elif isinstance(from_, str) or isinstance(from_, type):
        from_ = ivy.as_ivy_dtype(from_)
    elif isinstance(from_, (int, float, bool)):
        from_ = ivy.as_ivy_dtype(type(from_))
    else:
        raise ivy.exceptions.IvyException(
            "from_ must be one of dtype, dtype specifier, scalar, or array"
        )

    if isinstance(to, (str, type)):
        to = ivy.as_ivy_dtype(to)
    else:
        raise ivy.exceptions.IvyException("to must be dtype or dtype specifier")

    if casting == "safe" and to in numpy_casting_rules[from_]:
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
            return to in numpy_casting_rules[from_]
    if casting == "unsafe":
        return True
    return False
