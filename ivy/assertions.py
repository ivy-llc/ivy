import ivy
import builtins


# General #
# ------- #


def check_elem_in_list(elem, list):
    if elem not in list:
        raise ivy.exceptions.IvyException("{} is not one of {}".format(elem, list))


def check_less(x1, x2, allow_equal=False):
    if allow_equal and ivy.any(ivy.greater(x1, x2)):
        raise ivy.exceptions.IvyException(
            "{} is not lesser than or equal to {}".format(x1, x2)
        )
    elif ivy.any(ivy.greater_equal(x1, x2)):
        raise ivy.exceptions.IvyException("{} is not lesser than {}".format(x1, x2))


def check_greater(x1, x2, allow_equal=False):
    if allow_equal and ivy.any(ivy.less(x1, x2)):
        raise ivy.exceptions.IvyException(
            "{} is not greater than or equal to {}".format(x1, x2)
        )
    elif ivy.any(ivy.less_equal(x1, x2)):
        raise ivy.exceptions.IvyException("{} is not greater than {}".format(x1, x2))


def check_all(results, message="one of the args is None"):
    if not builtins.all(results):
        raise ivy.exceptions.IvyException(message)


def check_isinstance(x, allowed_types):
    if not isinstance(x, allowed_types):
        raise ivy.exceptions.IvyException(
            "type of x: {} is not one of the allowed types: {}".format(
                type(x), allowed_types
            )
        )


# Creation #
# -------- #


def check_fill_value_and_dtype_are_compatible(fill_value, dtype):
    if not (
        (ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype))
        and isinstance(fill_value, int)
    ) and not (
        ivy.is_float_dtype(dtype)
        and isinstance(fill_value, float)
        or isinstance(fill_value, bool)
    ):
        raise ivy.exceptions.IvyException(
            "the fill_value: {} and data type: {} are not compatible".format(
                fill_value, dtype
            )
        )
