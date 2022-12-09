# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.jax.numpy import jax_numpy_casting_table


@to_ivy_arrays_and_back
def can_cast(from_, to, casting="safe"):
    """
    Returns True if casting betweer two dtypes is possible according to casting rules,
    False otherwise.
    """
    ivy.assertions.check_elem_in_list(
        casting,
        ["no", "equiv", "safe", "same_kind", "unsafe"],
        message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
    )

    if ivy.is_array(from_):
        from_ = ivy.as_ivy_dtype(from_.dtype)
    elif isinstance(from_, str) or isinstance(from_, type):
        from_ = ivy.as_ivy_dtype(from_)
    elif isinstance(from_, (bool, int, float, complex)):
        from_ = ivy.as_ivy_dtype(type(from_))
    else:
        raise ivy.exceptions.IvyException(
            "from_ must be one of dtype, dtype specifier, scalar type, or array, "
        )

    if isinstance(to, str) or isinstance(to, type):
        to = ivy.as_ivy_dtype(to)
    elif isinstance(to, (bool, int, float, complex)):
        to = ivy.as_ivy_dtype(type(to))
    else:
        raise ivy.exceptions.IvyException("to must be one of dtype, or dtype specifier")

    if casting == "no" or casting == "equiv":
        return from_ == to

    if casting == "safe":
        return to in jax_numpy_casting_table[from_]

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
            return to in jax_numpy_casting_table[from_]
    if casting == "unsafe":
        return True
    return False
