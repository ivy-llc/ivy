# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# Helpers #
# ------- #


# sinc
@handle_test(
    fn_tree="functional.experimental.sinc",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
)
def test_sinc(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        atol_=1e-02,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="jax",
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# lcm
@handle_test(
    fn_tree="functional.experimental.lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_lcm(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        ground_truth_backend="jax",
        fw=backend_fw,
        fn_name=fn_name,
        test_gradients=False,
        x1=x[0],
        x2=x[1],
    )


# fmod
@handle_test(
    fn_tree="functional.experimental.fmod",
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
)
def test_fmod(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        ground_truth_backend="numpy",
        fw=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )


# fmax
@handle_test(
    fn_tree="functional.experimental.fmax",
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
        allow_nan=True,
    ),
)
def test_fmax(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend="tensorflow",
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
    fn_tree="functional.experimental.trapz",
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
)
def test_trapz(
    dtype_values_axis,
    rand_either,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
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
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend="tensorflow",
        fn_name=fn_name,
        y=np.asarray(y[0], dtype=input_dtype[0]),
        x=x,
        dx=dx,
        axis=axis,
    )


# float_power
@handle_test(
    fn_tree="functional.experimental.float_power",
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
)
def test_float_power(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# exp2
@handle_test(
    fn_tree="functional.experimental.exp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_exp2(
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        x=np.asarray(x[0], dtype=input_dtype[0]),
    )


# copysign
@handle_test(
    fn_tree="functional.experimental.copysign",
    dtype_x1_x2=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=0,
        allow_nan=False,
        shared_dtype=True,
    ),
)
def test_copysign(
    dtype_x1_x2,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    (x1_dtype, x2_dtype), (x1, x2) = dtype_x1_x2
    helpers.test_function(
        input_dtypes=[x1_dtype, x2_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        ground_truth_backend="torch",
        fw=backend_fw,
        fn_name=fn_name,
        test_values=True,
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
    fn_tree="functional.experimental.count_nonzero",
    dtype_values_axis=_get_dtype_values_axis_for_count_nonzero(
        in_available_dtypes="integer",
        out_available_dtypes="integer",
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
    ),
    keepdims=st.booleans(),
)
def test_count_nonzero(
    dtype_values_axis,
    keepdims,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    on_device,
    backend_fw,
):
    i_o_dtype, a, axis = dtype_values_axis
    helpers.test_function(
        input_dtypes=i_o_dtype[0],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        ground_truth_backend="tensorflow",
        fn_name="count_nonzero",
        a=a[0],
        axis=axis,
        keepdims=keepdims,
        dtype=i_o_dtype[1][0],
    )


# nansum
@handle_test(
    fn_tree="functional.experimental.nansum",
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
        allow_nan=True,
    ),
    keep_dims=st.booleans(),
)
def test_nansum(
    *,
    dtype_x_axis,
    keep_dims,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    on_device,
    fn_name,
    instance_method,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        on_device=on_device,
        ground_truth_backend="tensorflow",
        fn_name=fn_name,
        x=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# gcd
@handle_test(
    fn_tree="functional.experimental.gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_gcd(
    *,
    dtype_and_x,
    with_out,
    num_positional_args,
    as_variable,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# isclose
@handle_test(
    fn_tree="functional.experimental.isclose",
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
)
def test_isclose(
    *,
    dtype_and_x,
    rtol,
    atol,
    equal_nan,
    with_out,
    num_positional_args,
    as_variable,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        a=x[0],
        b=x[1],
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


# isposinf
@handle_test(
    fn_tree="functional.experimental.isposinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_isposinf(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# isneginf
@handle_test(
    fn_tree="functional.experimental.isneginf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_isneginf(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# nan_to_num
@handle_test(
    fn_tree="functional.experimental.nan_to_num",
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
)
def test_nan_to_num(
    *,
    dtype_and_x,
    copy,
    nan,
    posinf,
    neginf,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        ground_truth_backend="tensorflow",
        x=x[0],
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )


# logaddexp2
@handle_test(
    fn_tree="functional.experimental.logaddexp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_logaddexp2(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-03,
        atol_=1e-03,
        x1=x[0],
        x2=x[1],
    )


# allclose
@handle_test(
    fn_tree="functional.experimental.allclose",
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
    num_positional_args=helpers.num_positional_args(fn_name="allclose"),
)
def test_allclose(
    dtype_and_x,
    rtol,
    atol,
    equal_nan,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
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
    fn_tree="functional.experimental.fix",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=2),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_fix(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name="fix",
        x=np.asarray(x[0], dtype=input_dtype[0]),
    )


# nextafter
@handle_test(
    fn_tree="functional.experimental.nextafter",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=2),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_nextafter(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name="nextafter",
        x1=x[0],
        x2=x[1],
    )


# diff
@handle_test(
    fn_tree="functional.experimental.diff",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_diff(
    *,
    dtype_and_x,
    with_out,
    num_positional_args,
    as_variable,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        ground_truth_backend="tensorflow",
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# zeta
@handle_test(
    fn_tree="functional.experimental.zeta",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=2),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_zeta(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="zeta",
        rtol_=1e-03,
        atol_=1e-03,
        x=x[0],
        q=x[1],
    )


# gradient
@handle_test(
    fn_tree="functional.experimental.gradient",
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
)
def test_gradient(
    *,
    dtype_and_x,
    spacing,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=False,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        spacing=spacing,
    )
