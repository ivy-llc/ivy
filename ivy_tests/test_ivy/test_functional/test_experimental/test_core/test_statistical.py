# global
from hypothesis import strategies as st

# local
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
    _get_castable_dtype,
)


# --- Helpers --- #
# --------------- #


@st.composite
def _get_castable_float_dtype_nan(draw, min_value=None, max_value=None):
    available_dtypes = helpers.get_dtypes("float")
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=4, max_dim_size=6))
    dtype3, where = draw(
        helpers.dtype_and_values(available_dtypes=["bool"], shape=shape)
    )
    dtype, values = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            num_arrays=1,
            large_abs_safety_factor=6,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            allow_nan=True,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    dtype1, values, dtype2 = draw(
        helpers.get_castable_dtype(draw(available_dtypes), dtype[0], values[0])
    )
    return dtype1, [values], axis, dtype2, dtype3, where


@st.composite
def _get_dtype_value1_value2_cov(
    draw,
    available_dtypes,
    min_num_dims,
    max_num_dims,
    min_dim_size,
    max_dim_size,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )

    value1, value2 = values[0], values[1]

    # modifiers: rowVar, bias, ddof
    rowVar = draw(st.booleans())
    bias = draw(st.booleans())
    ddof = draw(helpers.ints(min_value=0, max_value=1))

    numVals = None
    if rowVar is False:
        numVals = -1 if numVals == 0 else 0
    else:
        numVals = 0 if len(shape) == 1 else -1

    fweights = draw(
        helpers.array_values(
            dtype="int64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
        )
    )

    aweights = draw(
        helpers.array_values(
            dtype="float64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
            small_abs_safety_factor=1,
        )
    )

    return [dtype], value1, value2, rowVar, bias, ddof, fweights, aweights


@st.composite
def _histogram_helper(draw):
    dtype_input = draw(st.sampled_from(draw(helpers.get_dtypes("float"))))
    bins = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=(draw(helpers.ints(min_value=1, max_value=10)),),
            abs_smallest_val=-10,
            min_value=-10,
            max_value=10,
        )
    )
    bins = np.asarray(sorted(set(bins)), dtype=dtype_input)
    if len(bins) == 1:
        bins = int(abs(bins[0]))
        if bins == 0:
            bins = 1
        if dtype_input in draw(helpers.get_dtypes("unsigned")):
            range = (
                draw(
                    helpers.floats(
                        min_value=0, max_value=10, exclude_min=False, exclude_max=False
                    )
                ),
                draw(
                    helpers.floats(
                        min_value=11, max_value=20, exclude_min=False, exclude_max=False
                    )
                ),
            )
        else:
            range = (
                draw(helpers.floats(min_value=-10, max_value=0)),
                draw(helpers.floats(min_value=1, max_value=10)),
            )
        range = draw(st.sampled_from([range, None]))
    else:
        range = None
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=5
        )
    )
    a = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=shape,
            min_value=-20,
            max_value=20,
        )
    )
    weights = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=shape,
            min_value=-20,
            max_value=20,
        )
    )
    # weights = draw(st.sampled_from([weights, None]))
    axes = draw(
        helpers.get_axis(
            shape=shape,
            # TODO: negative axes
            allow_neg=False,
            min_size=1,
            max_size=10,
        )
    )
    dtype_out = draw(
        st.sampled_from(
            draw(
                helpers.get_castable_dtype(
                    draw(helpers.get_dtypes("float")), str(dtype_input)
                )
            )
        )
    )
    if range:
        if np.min(a) < range[0]:
            extend_lower_interval = True
        else:
            extend_lower_interval = draw(st.booleans())
        if np.max(a) > range[1]:
            extend_upper_interval = True
        else:
            extend_upper_interval = draw(st.booleans())
    else:
        if isinstance(bins, int):
            extend_lower_interval = draw(st.booleans())
            extend_upper_interval = draw(st.booleans())
        else:
            if np.min(a) < bins[0]:
                extend_lower_interval = True
            else:
                extend_lower_interval = draw(st.booleans())
            if np.max(a) > bins[-1]:
                extend_upper_interval = True
            else:
                extend_upper_interval = draw(st.booleans())
    density = draw(st.booleans())
    return (
        a,
        bins,
        axes,
        extend_lower_interval,
        extend_upper_interval,
        dtype_out,
        range,
        weights,
        density,
        dtype_input,
    )


@st.composite
def _quantile_helper(draw):
    large_abs_safety_factor = 2
    small_abs_safety_factor = 2
    dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            valid_axis=True,
            allow_neg_axes=False,
            min_axes_size=1,
            force_int_axis=True,
        )
    )
    q = draw(
        st.one_of(
            helpers.array_values(
                dtype=helpers.get_dtypes("float"),
                shape=helpers.get_shape(min_dim_size=1, max_num_dims=1, min_num_dims=1),
                min_value=0.0,
                max_value=1.0,
                exclude_max=False,
                exclude_min=False,
            ),
            st.floats(min_value=0.0, max_value=1.0),
        )
    )

    interpolation_names = [
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
        "nearest_jax",
    ]
    interpolation = draw(
        helpers.list_of_size(
            x=st.sampled_from(interpolation_names),
            size=1,
        )
    )
    return dtype, values, axis, interpolation, q


# bincount
@st.composite
def bincount_dtype_and_values(draw):
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            num_arrays=2,
            shared_dtype=True,
            min_num_dims=1,
            max_num_dims=1,
            min_dim_size=1,
            max_dim_size=10,
            min_value=0,
            max_value=10,
            allow_nan=False,
        )
    )
    dtype_and_x[1][1] = dtype_and_x[1][0]
    if draw(st.booleans()):
        dtype_and_x[1][1] = None

    min_length = draw(st.integers(min_value=0, max_value=10))
    return dtype_and_x, min_length


# --- Main --- #
# ------------ #


@handle_test(
    fn_tree="functional.ivy.experimental.bincount",
    dtype_and_x=bincount_dtype_and_values(),
    test_gradients=st.just(False),
)
def test_bincount(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype_and_x, min_length = dtype_and_x
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        weights=x[1],
        minlength=min_length,
    )


# corrcoef
@handle_test(
    fn_tree="functional.ivy.experimental.corrcoef",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        abs_smallest_val=1e-5,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    rowvar=st.booleans(),
    test_gradients=st.just(False),
)
def test_corrcoef(*, dtype_and_x, rowvar, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        y=x[1],
        rowvar=rowvar,
    )


# cov
@handle_test(
    fn_tree="functional.ivy.experimental.cov",
    dtype_x1_x2_cov=_get_dtype_value1_value2_cov(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_cov(
    *,
    dtype_x1_x2_cov,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, x1, x2, rowVar, bias, ddof, fweights, aweights = dtype_x1_x2_cov
    helpers.test_function(
        input_dtypes=[dtype[0], dtype[0], "int64", "float64"],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x1,
        x2=x2,
        rowVar=rowVar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
        return_flat_np_arrays=True,
        rtol_=1e-2,
        atol_=1e-2,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.cummax",
    dtype_x_axis_castable=_get_castable_dtype(),
    exclusive=st.booleans(),
    reverse=st.booleans(),
)
def test_cummax(
    *,
    dtype_x_axis_castable,
    exclusive,
    reverse,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis, castable_dtype = dtype_x_axis_castable
    helpers.test_function(
        input_dtypes=[input_dtype],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
        dtype=castable_dtype,
        rtol_=1e-1,
        atol_=1e-1,
    )


# cummin
@handle_test(
    fn_tree="functional.ivy.experimental.cummin",
    dtype_x_axis_castable=_get_castable_dtype(),
    exclusive=st.booleans(),
    reverse=st.booleans(),
    test_gradients=st.just(False),
)
def test_cummin(
    *,
    dtype_x_axis_castable,
    exclusive,
    reverse,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis, castable_dtype = dtype_x_axis_castable
    helpers.test_function(
        input_dtypes=[input_dtype],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
        dtype=castable_dtype,
        rtol_=1e-1,
        atol_=1e-1,
    )


# TODO: - Error message from Tensorflow: 'Number of dimensions of `x` and `weights`
#       must coincide. Found: x has <nd1>, weights has <nd2>'
#       - Error description: typo that throws unintended exceptions when using both
#       weights and multiple axis.
#       - fixed in TFP 0.20 release.
#       - Test helper needs to be modified to handle this case in older versions.
@handle_test(
    fn_tree="functional.ivy.experimental.histogram",
    values=_histogram_helper(),
    test_gradients=st.just(False),
)
def test_histogram(
    *,
    values,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    (
        a,
        bins,
        axis,
        extend_lower_interval,
        extend_upper_interval,
        dtype,
        range,
        weights,
        density,
        dtype_input,
    ) = values
    helpers.test_function(
        a=a,
        bins=bins,
        axis=axis,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype,
        range=range,
        weights=weights,
        density=density,
        input_dtypes=[dtype_input],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
    )


# igamma
@handle_test(
    fn_tree="functional.ivy.experimental.igamma",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        min_value=2,
        max_value=100,
    ),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_igamma(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-04,
        a=x[0],
        x=x[1],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.median",
    dtype_x_axis=_statistical_dtype_values(function="median"),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_median(*, dtype_x_axis, keep_dims, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        input=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# nanmean
@handle_test(
    fn_tree="functional.ivy.experimental.nanmean",
    dtype_x_axis=_statistical_dtype_values(function="nanmean"),
    keep_dims=st.booleans(),
    dtype=helpers.get_dtypes("valid", full=False),
    test_gradients=st.just(False),
)
def test_nanmean(
    *, dtype_x_axis, keep_dims, dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x, axis, *_ = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        atol_=1e-02,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        dtype=dtype[0],
    )


# nanmedian
@handle_test(
    fn_tree="functional.ivy.experimental.nanmedian",
    dtype_x_axis=_statistical_dtype_values(function="nanmedian"),
    keep_dims=st.booleans(),
    dtype=helpers.get_dtypes("valid", full=False),
    overwriteinput=st.booleans(),
    test_gradients=st.just(False),
)
def test_nanmedian(
    *,
    dtype_x_axis,
    keep_dims,
    overwriteinput,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        atol_=1e-02,
        fn_name=fn_name,
        on_device=on_device,
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        overwrite_input=overwriteinput,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.nanmin",
    dtype_x_axis_castable=_get_castable_float_dtype_nan(),
    test_gradients=st.just(False),
    initial=st.integers(min_value=-5, max_value=5),
    keep_dims=st.booleans(),
)
def test_nanmin(
    *,
    dtype_x_axis_castable,
    initial,
    keep_dims,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis, castable_dtype, dtype3, where = dtype_x_axis_castable
    x = x[0]
    helpers.test_function(
        input_dtypes=[input_dtype, dtype3[0]],
        test_flags=test_flags,
        rtol_=1e-1,
        atol_=1e-1,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        a=x,
        axis=axis,
        keepdims=keep_dims,
        initial=initial,
        where=where[0],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.nanprod",
    dtype_x_axis_castable=_get_castable_float_dtype_nan(),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
    initial=st.integers(min_value=-5, max_value=5),
)
def test_nanprod(
    *,
    dtype_x_axis_castable,
    keep_dims,
    test_flags,
    initial,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis, castable_dtype = dtype_x_axis_castable
    x = x[0]
    helpers.test_function(
        input_dtypes=[input_dtype],
        test_flags=test_flags,
        rtol_=1e-1,
        atol_=1e-1,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        a=x,
        axis=axis,
        keepdims=keep_dims,
        dtype=castable_dtype,
        initial=initial,
    )


# quantile
@handle_test(
    fn_tree="functional.ivy.experimental.quantile",
    dtype_and_x=_quantile_helper(),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_quantile(
    *, dtype_and_x, keep_dims, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x, axis, interpolation, q = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        a=x[0],
        q=q,
        axis=axis,
        interpolation=interpolation[0],
        keepdims=keep_dims,
        atol_=1e-3,
        rtol_=1e-3,
    )
