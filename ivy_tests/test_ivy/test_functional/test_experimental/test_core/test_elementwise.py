# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# Helpers #
# ------- #


# sinc
@handle_test(
    fn_tree="functional.ivy.experimental.sinc",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    test_gradients=st.just(False),
)
def test_sinc(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        atol_=1e-02,
        ground_truth_backend=ground_truth_backend,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# fmax
@handle_test(
    fn_tree="functional.ivy.experimental.fmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_nan=True,
    ),
    test_gradients=st.just(False),
)
def test_fmax(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend=ground_truth_backend,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )


# fmin
@handle_test(
    fn_tree="functional.ivy.experimental.fmin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_nan=True,
    ),
    test_gradients=st.just(False),
)
def test_fmin(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend=ground_truth_backend,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )


# trapz
@st.composite
def _either_x_dx(draw):
    rand = (draw(st.integers(min_value=0, max_value=1)),)
    if rand == 0:
        either_x_dx = draw(
            helpers.dtype_and_values(
                avaliable_dtypes=st.shared(
                    helpers.get_dtypes("float"), key="trapz_dtype"
                ),
                min_value=-100,
                max_value=100,
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            )
        )
        return rand, either_x_dx
    else:
        either_x_dx = draw(
            st.floats(min_value=-10, max_value=10),
        )
        return rand, either_x_dx


@handle_test(
    fn_tree="functional.ivy.experimental.trapz",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=st.shared(helpers.get_dtypes("float"), key="trapz_dtype"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_neg_axes=True,
        valid_axis=True,
        force_int_axis=True,
    ),
    rand_either=_either_x_dx(),
    test_gradients=st.just(False),
)
def test_trapz(
    dtype_values_axis,
    rand_either,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, y, axis = dtype_values_axis
    rand, either_x_dx = rand_either
    if rand == 0:
        dtype_x, x = either_x_dx
        x = np.asarray(x, dtype=dtype_x)
        dx = None
    else:
        x = None
        dx = either_x_dx
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend=ground_truth_backend,
        fn_name=fn_name,
        rtol_=1e-1,
        atol_=1e-1,
        y=np.asarray(y[0], dtype=input_dtype[0]),
        x=x,
        dx=dx,
        axis=axis,
    )


# float_power_helper
@st.composite
def _float_power_helper(draw, *, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            small_abs_safety_factor=16,
            large_abs_safety_factor=16,
            safety_factor_scale="log",
        )
    )
    dtype2 = draw(helpers.get_dtypes("numeric"))
    if ivy.is_int_dtype(dtype2[0]):
        min_value = 0
    else:
        min_value = -10
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            min_value=min_value,
            max_value=10,
            dtype=dtype2,
        )
    )
    return (dtype1[0], dtype2[0]), (x1[0], x2[0])


# float_power
@handle_test(
    fn_tree="functional.ivy.experimental.float_power",
    dtype_and_x=_float_power_helper(),
    test_gradients=st.just(False),
)
def test_float_power(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
        rtol_=1e-1,
        atol_=1e-1,
    )


# exp2
@handle_test(
    fn_tree="functional.ivy.experimental.exp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_gradients=st.just(False),
)
def test_exp2(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        x=np.asarray(x[0], dtype=input_dtype[0]),
    )


# copysign
@handle_test(
    fn_tree="functional.ivy.experimental.copysign",
    dtype_x1_x2=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=0,
        allow_nan=False,
        shared_dtype=False,
    ),
    test_gradients=st.just(False),
)
def test_copysign(
    dtype_x1_x2,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    (x1_dtype, x2_dtype), (x1, x2) = dtype_x1_x2
    helpers.test_function(
        input_dtypes=[x1_dtype, x2_dtype],
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        x1=x1,
        x2=x2,
    )


@st.composite
def _get_dtype_values_axis_for_count_nonzero(
    draw,
    in_available_dtypes,
    out_available_dtypes,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
):
    input_dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes(in_available_dtypes),
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            valid_axis=True,
        )
    )
    axis = draw(st.one_of(st.just(axis), st.none()))
    output_dtype = draw(helpers.get_dtypes(out_available_dtypes))
    return [input_dtype, output_dtype], values, axis


# count_nonzero
@handle_test(
    fn_tree="functional.ivy.experimental.count_nonzero",
    dtype_values_axis=_get_dtype_values_axis_for_count_nonzero(
        in_available_dtypes="integer",
        out_available_dtypes="integer",
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
    ),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_count_nonzero(
    *,
    dtype_values_axis,
    keepdims,
    test_flags,
    on_device,
    fn_name,
    backend_fw,
    ground_truth_backend,
):
    i_o_dtype, a, axis = dtype_values_axis
    helpers.test_function(
        input_dtypes=i_o_dtype[0],
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend=ground_truth_backend,
        fn_name=fn_name,
        a=a[0],
        axis=axis,
        keepdims=keepdims,
        dtype=i_o_dtype[1][0],
    )


# nansum
@st.composite
def _get_castable_dtypes_values(draw, *, allow_nan=False):
    available_dtypes = helpers.get_dtypes("numeric")
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=4, max_dim_size=6))
    dtype, values = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            num_arrays=1,
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            allow_nan=allow_nan,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    dtype1, values, dtype2 = draw(
        helpers.get_castable_dtype(
            draw(helpers.get_dtypes("float")), dtype[0], values[0]
        )
    )
    return [dtype1], [values], axis, dtype2


# nansum
@handle_test(
    fn_tree="functional.ivy.experimental.nansum",
    dtype_x_axis_dtype=_get_castable_dtypes_values(allow_nan=True),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
)
def test_nansum(
    *,
    dtype_x_axis_dtype,
    keep_dims,
    test_flags,
    on_device,
    fn_name,
    backend_fw,
    ground_truth_backend,
):
    input_dtype, x, axis, dtype = dtype_x_axis_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        axis=axis,
        keepdims=keep_dims,
        dtype=dtype,
    )


# gcd
@handle_test(
    fn_tree="functional.ivy.experimental.gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=False,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    test_gradients=st.just(False),
)
def test_gcd(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# isclose
@handle_test(
    fn_tree="functional.ivy.experimental.isclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        allow_nan=True,
        shared_dtype=True,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    rtol=st.floats(min_value=0.0, max_value=0.1, exclude_min=True, exclude_max=True),
    atol=st.floats(min_value=0.0, max_value=0.1, exclude_min=True, exclude_max=True),
    equal_nan=st.booleans(),
    test_gradients=st.just(False),
)
def test_isclose(
    *,
    dtype_and_x,
    rtol,
    atol,
    equal_nan,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        a=x[0],
        b=x[1],
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


# angle
@handle_test(
    fn_tree="functional.ivy.experimental.angle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float64"],
        min_value=-5,
        max_value=5,
        max_dim_size=5,
        max_num_dims=5,
        min_dim_size=1,
        min_num_dims=1,
        allow_inf=False,
        allow_nan=False,
    ),
    deg=st.booleans(),
    test_gradients=st.just(False),
)
def test_angle(
    *,
    dtype_and_x,
    deg,
    test_flags,
    ground_truth_backend,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, z = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        z=z[0],
        deg=deg,
    )


# imag
@handle_test(
    fn_tree="functional.ivy.experimental.imag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-5,
        max_value=5,
        max_dim_size=5,
        max_num_dims=5,
        min_dim_size=1,
        min_num_dims=1,
        allow_inf=False,
        allow_nan=False,
    ),
    test_gradients=st.just(False),
)
def test_imag(
    *,
    dtype_and_x,
    test_flags,
    ground_truth_backend,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        val=x[0],
    )


# nan_to_num
@handle_test(
    fn_tree="functional.ivy.experimental.nan_to_num",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=True,
        allow_inf=True,
    ),
    copy=st.booleans(),
    nan=st.floats(min_value=0.0, max_value=100),
    posinf=st.floats(min_value=5e100, max_value=5e100),
    neginf=st.floats(min_value=-5e100, max_value=-5e100),
    test_gradients=st.just(False),
)
def test_nan_to_num(
    *,
    dtype_and_x,
    copy,
    nan,
    posinf,
    neginf,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )


# logaddexp2
@handle_test(
    fn_tree="functional.ivy.experimental.logaddexp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    test_gradients=st.just(False),
)
def test_logaddexp2(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x1=x[0],
        x2=x[1],
    )


# allclose
@handle_test(
    fn_tree="functional.ivy.experimental.allclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
    ),
    rtol=st.floats(min_value=1e-5, max_value=1e-5),
    atol=st.floats(min_value=1e-5, max_value=1e-5),
    equal_nan=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_allclose(
    dtype_and_x,
    rtol,
    atol,
    equal_nan,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


# fix
@handle_test(
    fn_tree="functional.ivy.experimental.fix",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_gradients=st.just(False),
)
def test_fix(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# nextafter
@handle_test(
    fn_tree="functional.ivy.experimental.nextafter",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_nextafter(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# diff
@handle_test(
    fn_tree="functional.ivy.experimental.diff",
    dtype_n_x_n_axis=helpers.dtype_values_axis(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype_prepend=helpers.dtype_and_values(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    dtype_append=helpers.dtype_and_values(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    n=st.integers(min_value=0, max_value=5),
    test_gradients=st.just(False),
)
def test_diff(
    *,
    dtype_n_x_n_axis,
    n,
    dtype_prepend,
    dtype_append,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis = dtype_n_x_n_axis
    _, prepend = dtype_prepend
    _, append = dtype_append
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        n=n,
        axis=axis,
        prepend=prepend[0],
        append=append[0],
    )


# zeta
@handle_test(
    fn_tree="functional.ivy.experimental.zeta",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_zeta(
    dtype_and_x,
    test_flags,
    fn_name,
    on_device,
    ground_truth_backend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        q=x[1],
    )


# gradient
@handle_test(
    fn_tree="functional.ivy.experimental.gradient",
    dtype_n_x_n_axis=helpers.dtype_values_axis(
        available_dtypes=("float32", "float16", "float64"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=4,
        valid_axis=True,
        force_int_axis=True,
    ),
    spacing=helpers.ints(
        min_value=-3,
        max_value=3,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_gradient(
    *,
    dtype_n_x_n_axis,
    spacing,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, axis = dtype_n_x_n_axis
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        spacing=spacing,
        axis=axis,
    )


# xlogy
@handle_test(
    fn_tree="functional.ivy.experimental.xlogy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        num_arrays=2,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_xlogy(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# real
@handle_test(
    fn_tree="functional.ivy.experimental.real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex")
    ),
)
def test_real(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# hypot
@handle_test(
    fn_tree="functional.ivy.experimental.hypot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_hypot(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-2,
        ground_truth_backend=ground_truth_backend,
        x1=x[0],
        x2=x[1],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.binarizer",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    threshold=helpers.floats(),
    container_flags=st.just([False]),
)
def test_binarizer(
    *,
    dtype_and_x,
    threshold,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        threshold=threshold,
    )


# conj
@handle_test(
    fn_tree="conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex")
    ),
    test_with_out=st.just(False),
)
def test_conj(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# ldexp
@st.composite
def ldexp_args(draw):
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            num_arrays=1,
            shared_dtype=True,
            min_value=-100,
            max_value=100,
            min_num_dims=1,
            max_num_dims=3,
        )
    )
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            num_arrays=1,
            shared_dtype=True,
            min_value=-100,
            max_value=100,
            min_num_dims=1,
            max_num_dims=3,
        )
    )
    return (dtype1[0], dtype2[0]), (x1[0], x2[0])


@handle_test(
    fn_tree="functional.ivy.experimental.ldexp",
    dtype_and_x=ldexp_args(),
    test_gradients=st.just(False),
)
def test_ldexp(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# lerp
@handle_test(
    fn_tree="functional.ivy.experimental.lerp",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=3,
        shared_dtype=True,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        allow_nan=False,
        allow_inf=False,
    ),
    test_gradients=st.just(False),
)
def test_lerp(
    *,
    dtype_and_input,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, inputs = dtype_and_input
    start, end, weight = inputs
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        atol_=1e-01,
        ground_truth_backend=ground_truth_backend,
        on_device=on_device,
        input=start,
        end=end,
        weight=weight,
    )


# frexp
@handle_test(
    fn_tree="functional.ivy.experimental.frexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=1,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_frexp(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )
