import ivy


# General #
# ------- #


def check_elem_in_list(elem, list):
    if elem not in list:
        raise ivy.exceptions.IvyException("{} is not one of {}".format(elem, list))


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
