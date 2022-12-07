import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def can_cast(from_, to, casting="safe"):
    ivy.assertions.check_elem_in_list(
        casting,
        ["no", "equiv", "safe", "same_kind", "unsafe"],
        message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
    )
    if casting == "no":
        return False

    if ivy.is_array(from_):
        from_ = ivy.as_ivy_dtype(ivy.dtype(from_))
    else:
        from_ = ivy.as_ivy_dtype(from_)

    to = ivy.as_ivy_dtype(to)
    if casting == "equiv":
        return from_ == to

    if "bool" in from_ and (("int" in to) or ("float" in to)):
        return False
    if "int" in from_ and "float" in to:
        return False

    if casting == "safe":
        pass
    elif casting == "same_kind":
        pass
    return True
