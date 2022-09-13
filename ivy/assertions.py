import ivy
import builtins


# General #
# ------- #


def check_elem_in_list(elem, list):
    if elem not in list:
        raise ivy.exceptions.IvyException("{} must be one of {}".format(elem, list))


def check_less(x1, x2, allow_equal=False):
    if allow_equal and ivy.any(x1 > x2):
        raise ivy.exceptions.IvyException(
            "{} must be lesser than or equal to {}".format(x1, x2)
        )
    elif ivy.any(x1 >= x2):
        raise ivy.exceptions.IvyException("{} must be lesser than {}".format(x1, x2))


def check_greater(x1, x2, allow_equal=False):
    if allow_equal and ivy.any(x1 < x2):
        raise ivy.exceptions.IvyException(
            "{} must be greater than or equal to {}".format(x1, x2)
        )
    elif ivy.any(x1 <= x2):
        raise ivy.exceptions.IvyException("{} must be greater than {}".format(x1, x2))


def check_equal(x1, x2, inverse=False):
    if inverse and ivy.any(x1 == x2):
        raise ivy.exceptions.IvyException("{} must be equal to {}".format(x1, x2))
    elif not ivy.all(x1 == x2):
        raise ivy.exceptions.IvyException("{} must not be equal to {}".format(x1, x2))


def check_isinstance(x, allowed_types):
    if not isinstance(x, allowed_types):
        raise ivy.exceptions.IvyException(
            "type of x: {} must be one of the allowed types: {}".format(
                type(x), allowed_types
            )
        )


# General with Custom Message #
# --------------------------- #


def check_true(expression, message="expression must be True"):
    if not expression:
        raise ivy.exceptions.IvyException(message)


def check_false(expression, message="expression must be False"):
    if expression:
        raise ivy.exceptions.IvyException(message)


def check_all(results, message="one of the args is None"):
    if not builtins.all(results):
        raise ivy.exceptions.IvyException(message)


def check_any(results, message="all of the args are None"):
    if not builtins.any(results):
        raise ivy.exceptions.IvyException(message)


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
