import math
from hypothesis import strategies as st
from hypothesis.internal.floats import float_of

# local
from . import general_helpers as gh, dtype_helpers


floats_info = {
    "float16": {"cast_type": "float16", "width": 16},
    "bfloat16": {"cast_type": "float32", "width": 32},
    "float32": {"cast_type": "float32", "width": 32},
    "float64": {"cast_type": "float64", "width": 64},
}


@st.composite
def floats(
    draw,
    *,
    min_value=None,
    max_value=None,
    abs_smallest_val=None,
    allow_nan=False,
    allow_inf=False,
    allow_subnormal=False,
    exclude_min=True,
    exclude_max=True,
    large_abs_safety_factor=1.1,
    small_abs_safety_factor=1.1,
    safety_factor_scale="linear",
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
    # ToDo assert that if min or max can be represented
    dtype = draw(dtype_helpers.get_dtypes("float", full=False))
    dtype = dtype[0]
    # ToDo add support for not applying safety factor
    min_value, max_value, abs_smallest_val = gh.apply_safety_factor(
        dtype,
        min_value=min_value,
        max_value=max_value,
        abs_smallest_val=abs_smallest_val,
        small_abs_safety_factor=small_abs_safety_factor,
        large_abs_safety_factor=large_abs_safety_factor,
        safety_factor_scale=safety_factor_scale,
    )
    # The smallest possible value is determined by one of the arguments
    if min_value > -abs_smallest_val or max_value < abs_smallest_val:
        float_strategy = st.floats(
            min_value=float_of(min_value, floats_info[dtype]["width"]),
            max_value=float_of(max_value, floats_info[dtype]["width"]),
            allow_nan=allow_nan,
            allow_subnormal=allow_subnormal,
            allow_infinity=allow_inf,
            width=floats_info[dtype]["width"],
            exclude_min=exclude_min,
            exclude_max=exclude_max,
        )
    else:
        float_strategy = st.one_of(
            st.floats(
                min_value=float_of(min_value, floats_info[dtype]["width"]),
                max_value=float_of(-abs_smallest_val, floats_info[dtype]["width"]),
                allow_nan=allow_nan,
                allow_subnormal=allow_subnormal,
                allow_infinity=allow_inf,
                width=floats_info[dtype]["width"],
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            ),
            st.floats(
                min_value=float_of(abs_smallest_val, floats_info[dtype]["width"]),
                max_value=float_of(max_value, floats_info[dtype]["width"]),
                allow_nan=allow_nan,
                allow_subnormal=allow_subnormal,
                allow_infinity=allow_inf,
                width=floats_info[dtype]["width"],
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            ),
        )
    values = draw(float_strategy)
    return values


@st.composite
def ints(
    draw,
    *,
    min_value=None,
    max_value=None,
    safety_factor=1.1,
    safety_factor_scale=None,
):
    """Draws an arbitrarily sized list of integers with a safety factor
    applied to values if a safety scale is specified.

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
    safety_factor_scale
    Returns
    -------
    ret
        list of integers.
    """
    dtype = draw(dtype_helpers.get_dtypes("integer", full=False))
    if min_value is None and max_value is None:
        safety_factor_scale = "linear"
    if safety_factor_scale is not None:
        min_value, max_value, _ = gh.apply_safety_factor(
            dtype[0],
            min_value=min_value,
            max_value=max_value,
            large_abs_safety_factor=safety_factor,
            safety_factor_scale=safety_factor_scale,
        )
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
            min_value=int(math.ceil(min_value)),
            max_value=int(math.ceil(max_value)),
            safety_factor=safety_factor,
        )
        | floats(
            min_value=min_value,
            max_value=max_value,
            safety_factor=safety_factor,
        )
    )
