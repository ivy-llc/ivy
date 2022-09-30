from hypothesis import strategies as st
from hypothesis.internal.floats import float_of

# local
import ivy
import ivy.functional.backends.numpy as ivy_np  # ToDo should be removed.
from . import array_helpers as ah


@st.composite
def floats(
    draw,
    *,
    min_value=None,
    max_value=None,
    allow_nan=False,
    allow_inf=False,
    allow_subnormal=False,
    width=None,
    exclude_min=True,
    exclude_max=True,
    safety_factor=0.99,
    small_value_safety_factor=1.1,
):
    """Draws an arbitrarily sized list of floats with a safety factor applied
        to avoid values being generated at the edge of a dtype limit.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_value
        minimum value of floats generated.
    max_value
        maximum value of floats generated.
    allow_nan
        if True, allow Nans in the list.
    allow_inf
        if True, allow inf in the list.
    allow_subnormal
        if True, allow subnormals in the list.
    width
        The width argument specifies the maximum number of bits of precision
        required to represent the generated float. Valid values are 16, 32, or 64.
    exclude_min
        if True, exclude the minimum limit.
    exclude_max
        if True, exclude the maximum limit.
    safety_factor
        default = 0.99. Only values which are 99% or less than the edge of
        the limit for a given dtype are generated.
    small_value_safety_factor
        default = 1.1.

    Returns
    -------
    ret
        list of floats.
    """
    lim_float16 = 65504
    lim_float32 = 3.4028235e38
    lim_float64 = 1.7976931348623157e308

    if min_value is not None and max_value is not None:
        if (
            min_value > -lim_float16 * safety_factor
            and max_value < lim_float16 * safety_factor
            and (width == 16 or not ivy.exists(width))
        ):
            # dtype float16
            width = 16
        elif (
            min_value > -lim_float32 * safety_factor
            and max_value < lim_float32 * safety_factor
            and (width == 32 or not ivy.exists(width))
        ):
            # dtype float32
            width = 32
        else:
            # dtype float64
            width = 64

        min_value = float_of(min_value, width)
        max_value = float_of(max_value, width)

        values = draw(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=allow_nan,
                allow_subnormal=allow_subnormal,
                allow_infinity=allow_inf,
                width=width,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            )
        )

    else:
        if min_value is not None:
            if min_value > -lim_float16 * safety_factor and (
                width == 16 or not ivy.exists(width)
            ):
                dtype_min = "float16"
            elif min_value > -lim_float32 * safety_factor and (
                width == 32 or not ivy.exists(width)
            ):
                dtype_min = "float32"
            else:
                dtype_min = "float64"
        else:
            dtype_min = draw(st.sampled_from(ivy_np.valid_float_dtypes))

        if max_value is not None:
            if max_value < lim_float16 * safety_factor and (
                width == 16 or not ivy.exists(width)
            ):
                dtype_max = "float16"
            elif max_value < lim_float32 * safety_factor and (
                width == 32 or not ivy.exists(width)
            ):
                dtype_max = "float32"
            else:
                dtype_max = "float64"
        else:
            dtype_max = draw(st.sampled_from(ivy_np.valid_float_dtypes))

        dtype = ivy.promote_types(dtype_min, dtype_max)

        if dtype == "float16" or 16 == ivy.default(width, 0):
            width = 16
            min_value = float_of(-lim_float16 * safety_factor, width)
            max_value = float_of(lim_float16 * safety_factor, width)
        elif dtype in ["float32", "bfloat16"] or 32 == ivy.default(width, 0):
            width = 32
            min_value = float_of(-lim_float32 * safety_factor, width)
            max_value = float_of(lim_float32 * safety_factor, width)
        else:
            width = 64
            min_value = float_of(-lim_float64 * safety_factor, width)
            max_value = float_of(lim_float64 * safety_factor, width)

        values = draw(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=allow_nan,
                allow_subnormal=allow_subnormal,
                allow_infinity=allow_inf,
                width=width,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            )
        )
    return values


@st.composite
def ints(draw, *, min_value=None, max_value=None, safety_factor=0.95):
    """Draws an arbitrarily sized list of integers with a safety factor
    applied to values.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_value
        minimum value of integers generated.
    max_value
        maximum value of integers generated.
    safety_factor
        default = 0.95. Only values which are 95% or less than the edge of
        the limit for a given dtype are generated.

    Returns
    -------
    ret
        list of integers.
    """
    dtype = draw(st.sampled_from(ivy_np.valid_int_dtypes))

    if dtype == "int8":
        min_value = ivy.default(min_value, round(-128 * safety_factor))
        max_value = ivy.default(max_value, round(127 * safety_factor))
    elif dtype == "int16":
        min_value = ivy.default(min_value, round(-32768 * safety_factor))
        max_value = ivy.default(max_value, round(32767 * safety_factor))
    elif dtype == "int32":
        min_value = ivy.default(min_value, round(-2147483648 * safety_factor))
        max_value = ivy.default(max_value, round(2147483647 * safety_factor))
    elif dtype == "int64":
        min_value = ivy.default(min_value, round(-9223372036854775808 * safety_factor))
        max_value = ivy.default(max_value, round(9223372036854775807 * safety_factor))
    elif dtype == "uint8":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(255 * safety_factor))
    elif dtype == "uint16":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(65535 * safety_factor))
    elif dtype == "uint32":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(4294967295 * safety_factor))
    elif dtype == "uint64":
        min_value = ivy.default(min_value, round(0 * safety_factor))
        max_value = ivy.default(max_value, round(18446744073709551615 * safety_factor))

    return draw(st.integers(min_value, max_value))


@st.composite
def ints_or_floats(draw, *, min_value=None, max_value=None, safety_factor=0.95):
    """Draws integers or floats with a safety factor
    applied to values.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_value
        minimum value of integers generated.
    max_value
        maximum value of integers generated.
    safety_factor
        default = 0.95. Only values which are 95% or less than the edge of
        the limit for a given dtype are generated.

    Returns
    -------
    ret
        integer or float.
    """
    return draw(
        ints(
            min_value=int(min_value),
            max_value=int(max_value),
            safety_factor=safety_factor,
        )
        | floats(min_value=min_value, max_value=max_value, safety_factor=safety_factor)
    )


def none_or_list_of_floats(
    *,
    dtype,
    size,
    min_value=None,
    max_value=None,
    exclude_min=False,
    exclude_max=False,
    no_none=False,
):
    """Draws a List containing Nones or Floats.

    Parameters
    ----------
    dtype
        float data type ('float16', 'float32', or 'float64').
    size
        size of the list required.
    min_value
        lower bound for values in the list
    max_value
        upper bound for values in the list
    exclude_min
        if True, exclude the min_value
    exclude_max
        if True, exclude the max_value
    no_none
        if True, List does not contains None

    Returns
    -------
    A strategy that draws a List containing Nones or Floats.
    """
    if no_none:
        if dtype == "float16":
            values = ah.list_of_length(
                x=floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float32":
            values = ah.list_of_length(
                x=st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=32,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float64":
            values = ah.list_of_length(
                x=st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=64,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
    else:
        if dtype == "float16":
            values = ah.list_of_length(
                x=st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float32":
            values = ah.list_of_length(
                x=st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=32,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
        elif dtype == "float64":
            values = ah.list_of_length(
                x=st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=64,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                length=size,
            )
    return values
