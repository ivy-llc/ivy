import ivy


# General with Custom Message #
# --------------------------- #


def check_less(x1, x2, allow_equal=False, message=""):
    # less_equal
    if allow_equal and ivy.any(x1 > x2):
        raise ivy.exceptions.IvyException(
            "{} must be lesser than or equal to {}".format(x1, x2)
            if message == ""
            else message
        )
    # less
    elif not allow_equal and ivy.any(x1 >= x2):
        raise ivy.exceptions.IvyException(
            "{} must be lesser than {}".format(x1, x2) if message == "" else message
        )


def check_greater(x1, x2, allow_equal=False, message=""):
    # greater_equal
    if allow_equal and ivy.any(x1 < x2):
        raise ivy.exceptions.IvyException(
            "{} must be greater than or equal to {}".format(x1, x2)
            if message == ""
            else message
        )
    # greater
    elif not allow_equal and ivy.any(x1 <= x2):
        raise ivy.exceptions.IvyException(
            "{} must be greater than {}".format(x1, x2) if message == "" else message
        )


def check_equal(x1, x2, inverse=False, message=""):
    # not_equal
    if inverse and ivy.any(x1 == x2):
        raise ivy.exceptions.IvyException(
            "{} must not be equal to {}".format(x1, x2) if message == "" else message
        )
    # equal
    elif not inverse and ivy.any(x1 != x2):
        raise ivy.exceptions.IvyException(
            "{} must be equal to {}".format(x1, x2) if message == "" else message
        )


def check_isinstance(x, allowed_types, message=""):
    if not isinstance(x, allowed_types):
        raise ivy.exceptions.IvyException(
            "type of x: {} must be one of the allowed types: {}".format(
                type(x), allowed_types
            )
            if message == ""
            else message
        )


def check_exists(x, inverse=False, message=""):
    # not_exists
    if inverse and ivy.exists(x):
        raise ivy.exceptions.IvyException(
            "arg must be None" if message == "" else message
        )
    # exists
    elif not inverse and not ivy.exists(x):
        raise ivy.exceptions.IvyException(
            "arg must not be None" if message == "" else message
        )


def check_elem_in_list(elem, list, message=""):
    message = message if message != "" else "{} must be one of {}".format(elem, list)
    if elem not in list:
        raise ivy.exceptions.IvyException(message)


def check_true(expression, message="expression must be True"):
    if not expression:
        raise ivy.exceptions.IvyException(message)


def check_false(expression, message="expression must be False"):
    if expression:
        raise ivy.exceptions.IvyException(message)


def check_all(results, message="one of the args is False"):
    if not ivy.all(results):
        raise ivy.exceptions.IvyException(message)


def check_any(results, message="all of the args are False"):
    if not ivy.any(results):
        raise ivy.exceptions.IvyException(message)


def check_all_or_any_fn(
    *args,
    fn,
    type="all",
    limit=[0],
    message="args must exist according to type and limit given"
):
    if type == "all":
        check_all([fn(arg) for arg in args], message)
    elif type == "any":
        count = 0
        for arg in args:
            count = count + 1 if fn(arg) else count
        if count not in limit:
            raise ivy.exceptions.IvyException(message)
    else:
        raise ivy.exceptions.IvyException("type must be all or any")


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
