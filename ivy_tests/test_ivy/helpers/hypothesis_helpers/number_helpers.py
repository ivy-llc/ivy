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
def min_max_bound(draw, min_value=None, max_value=None):
    min_val = draw(st.just(min_value))
    max_val = draw(st.just(max_value))
    return min_val, max_val


@st.composite
def exclude_min_max(draw, exclude_min=True, exclude_max=True):
    min_ex = draw(st.just(exclude_min))
    max_ex = draw(st.just(exclude_max))
    return min_ex, max_ex


@st.composite
def min_max_bound_exclusion(draw,
                            min_value=None,
                            max_value=None,
                            exclude_min=True,
                            exclude_max=True):
    min_val, max_val = draw(min_max_bound(min_value=min_value, max_value=max_value))
    min_ex, max_ex = draw(exclude_min_max(exclude_min=exclude_min, exclude_max=exclude_max))
    return min_val, max_val, min_ex, max_ex

@st.composite
def floats(
    draw,
    *,
    min_max_bound_exclusion=min_max_bound_exclusion,
    abs_smallest_val=None,
    allow_nan=False,
    allow_inf=False,
    allow_subnormal=False,
    large_abs_safety_factor=1.1,
    small_abs_safety_factor=1.1,
    safety_factor_scale="linear",
):
    """
    Draws an arbitrarily sized list of floats with a safety factor applied to avoid
    values being generated at the edge of a dtype limit.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_max_bound_exclusion
        strategy that generates the min and maximum values and whether to exclude them.
    allow_nan
        if True, allow Nans in the list.
    allow_inf
        if True, allow inf in the list.
    allow_subnormal
        if True, allow subnormals in the list.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used,  a safety factor of 2 means
        that only 50% of the range is included, a safety factor of 3 means that
        only 33% of the range is included etc.

        when a "log" safety factor scaler is used, a data type with maximum
        value of 2^32 and a safety factor of 2 transforms the maximum to 2^16.
    small_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used, a data type with minimum
        representable number of 0.0001 and a safety factor of 2 transforms the
        minimum to 0.0002, a safety factor of 3 transforms the minimum to 0.0003 etc.

        when a "log" safety factor scaler is used, a data type with minimum
        representable number of 0.5 * 2^-16 and a safety factor of 2 transforms the
        minimum to 0.5 * 2^-8, a safety factor of 3 transforms the minimum to 0.5 * 2^-4
    safety_factor_scale
        The operation to use for the safety factor scaling. Can be "linear" or "log".
        Default value = "linear".

    Returns
    -------
    ret
        Float.
    """

    min_value, max_value, exclude_min, exclude_max = draw(min_max_bound_exclusion)
    # ToDo assert that if min or max can be represented
    dtype = draw(dtype_helpers.get_dtypes("float", full=False, prune_function=False))
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
    min_max_values=min_max_bound,
    safety_factor=1.1,
    safety_factor_scale=None,
):
    """
    Draws an integer with a safety factor if specified.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_max_values
        strategy that draws a tuple of min and max values.
    safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used,  a safety factor of 2 means
        that only 50% of the range is included, a safety factor of 3 means that
        only 33% of the range is included etc.

        when a "log" safety factor scaler is used, a data type with maximum
        value of 2^32 and a safety factor of 2 transforms the maximum to 2^16.
    safety_factor_scale
        The operation to use for the safety factor scaling. Can be "linear" or "log".
        Default value = "linear".

    Returns
    -------
    ret
        Integer.
    """
    dtype = draw(dtype_helpers.get_dtypes("integer", full=False, prune_function=False))
    min_value, max_value = min_max_values()
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
def number(
    draw,
    *,
    min_max_values=min_max_bound,
    large_abs_safety_factor=1.1,
    small_abs_safety_factor=1.1,
    safety_factor_scale="linear",
):
    """
    Draws integers or floats with a safety factor applied to values.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    min_max_values
        strategy that draws a tuple of min and max values.
    large_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,

        when a "linear" safety factor scaler is used,  a safety factor of 2 means
        that only 50% of the range is included, a safety factor of 3 means that
        only 33% of the range is included etc.

        when a "log" safety factor scaler is used, a data type with maximum
        value of 2^32 and a safety factor of 2 transforms the maximum to 2^16.
    small_abs_safety_factor
        A safety factor of 1 means that all values are included without limitation,
        this has no effect on integer data types.

        when a "linear" safety factor scaler is used, a data type with minimum
        representable number of 0.0001 and a safety factor of 2 transforms the
        minimum to 0.0002, a safety factor of 3 transforms the minimum to 0.0003 etc.

        when a "log" safety factor scaler is used, a data type with minimum
        representable number of 0.5 * 2^-16 and a safety factor of 2 transforms the
        minimum to 0.5 * 2^-8, a safety factor of 3 transforms the minimum to 0.5 * 2^-4
    safety_factor_scale
        The operation to use for the safety factor scaling. Can be "linear" or "log".
        Default value = "linear".

    Returns
    -------
    ret
        An integer or float.
    """
    min_value, max_value = min_max_values()
    return draw(
        ints(
            min_value=min_value,
            max_value=max_value,
            safety_factor=large_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
        )
        | floats(
            min_value=min_value,
            max_value=max_value,
            small_abs_safety_factor=small_abs_safety_factor,
            large_abs_safety_factor=large_abs_safety_factor,
            safety_factor_scale=safety_factor_scale,
        )
    )
