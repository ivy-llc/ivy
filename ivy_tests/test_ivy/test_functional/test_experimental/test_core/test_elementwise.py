# global
import numpy as np
from hypothesis import strategies as st, assume

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
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        atol_=1e-02,
        ground_truth_backend="jax",
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# lcm
@handle_test(
    fn_tree="functional.ivy.experimental.lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["int16", "int32", "int64"],
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
def test_lcm(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend="numpy",
        fw=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )


# fmod
@handle_test(
    fn_tree="functional.ivy.experimental.fmod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=False,
        large_abs_safety_factor=6,
        small_abs_safety_factor=6,
        safety_factor_scale="log",
    ),
    test_gradients=st.just(False),
)
def test_fmod(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    # Make sure values is not too close to zero
    assume(not np.any(np.isclose(x[0], 0)))
    assume(not np.any(np.isclose(x[1], 0)))
    # jax raises inconsistent gradients for negative numbers in x1
    if (np.any(x[0] < 0) or np.any(x[1] < 0)) and ivy.current_backend_str() == "jax":
        test_flags.test_gradients = False
    test_flags.as_variable = [test_flags.as_variable, False]
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend="numpy",
        fw=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )


# fmax
@handle_test(
    fn_tree="functional.ivy.experimental.fmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
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
    fn_tree="functional.ivy.experimental.fmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
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
        y=np.asarray(y[0], dtype=input_dtype[0]),
        x=x,
        dx=dx,
        axis=axis,
    )


# float_power
@handle_test(
    fn_tree="functional.ivy.experimental.float_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
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
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        fn_name=fn_name,
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
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
):
    (x1_dtype, x2_dtype), (x1, x2) = dtype_x1_x2
    helpers.test_function(
        input_dtypes=[x1_dtype, x2_dtype],
        test_flags=test_flags,
        on_device=on_device,
        ground_truth_backend="torch",
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
@handle_test(
    fn_tree="functional.ivy.experimental.nansum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        min_axes_size=1,
        force_tuple_axis=True,
        allow_nan=True,
    ),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
)
def test_nansum(
    *,
    dtype_x_axis,
    keep_dims,
    test_flags,
    on_device,
    fn_name,
    backend_fw,
    ground_truth_backend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        fn_name=fn_name,
        x=x[0],
        axis=axis,
        keepdims=keep_dims,
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
        available_dtypes=["float32"],
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
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name="allclose",
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
        available_dtypes=helpers.get_dtypes("float", index=2),
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
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
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
    n=st.integers(min_value=0, max_value=5),
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
    ground_truth_backend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name="zeta",
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        q=x[1],
    )


# gradient
@handle_test(
    fn_tree="functional.ivy.experimental.gradient",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=("float32", "float16", "float64"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=4,
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
    dtype_and_x,
    spacing,
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
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        spacing=spacing,
    )


# xlogy
@handle_test(
    fn_tree="functional.ivy.experimental.xlogy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float16", "float32", "float64"],
        num_arrays=2,
        shared_dtype=False,
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
