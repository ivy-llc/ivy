# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import numpy as np

# Helpers #
# ------- #


# sinc
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sinc"),
)
def test_sinc(
    *,
    dtype_and_x,
    as_variable,
    with_out,
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
        rtol_=1e-03,
        atol_=1e-03,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sinc",
        x=x[0],
    )


# lcm
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="lcm"),
)
def test_lcm(
    dtype_and_x,
    as_variable,
    with_out,
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
        fn_name="lcm",
        test_gradients=True,
        x1=x[0],
        x2=x[1],
    )


# fmod
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="fmod"),
)
def test_fmod(
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
        fn_name="fmod",
        x1=x[0],
        x2=x[0],
    )


# fmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
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
    num_positional_args=helpers.num_positional_args(fn_name="fmax"),
)
def test_fmax(
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
        fn_name="fmax",
        x1=x[0],
        x2=x[0],
    )


# trapz
@st.composite
def _either_x_dx(draw):
    rand = (draw(st.integers(min_value=0, max_value=1)),)
    if rand == 0:
        either_x_dx = draw(
            helpers.dtype_and_x(
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


@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="trapz"),
)
def test_trapz(
    dtype_values_axis,
    rand_either,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
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
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        rtol_=1e-2,
        atol_=1e-2,
        fw=fw,
        fn_name="trapz",
        y=np.asarray(y[0], dtype=input_dtype[0]),
        x=x,
        dx=dx,
        axis=axis,
    )


# float_power
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="float_power"),
)
def test_float_power(
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
        fn_name="float_power",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# exp2
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="exp2"),
)
def test_exp2(
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
        fn_name="exp2",
        x=np.asarray(x[0], dtype=input_dtype[0]),
    )


# nansum
@handle_cmd_line_args
@given(
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
    ),
    num_positional_args=helpers.num_positional_args(fn_name="nansum"),
    keep_dims=st.booleans(),
)
def test_nansum(
    dtype_x_axis,
    keep_dims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="nansum",
        input=x[0],
        axis=axis,
        keepdims=keep_dims,
    )


# gcd
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="gcd"),
)
def test_gcd(
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
        fn_name="gcd",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
    )


# isclose
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="isclose"),
)
def test_isclose(
    dtype_and_x,
    rtol,
    atol,
    equal_nan,
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
        fn_name="isclose",
        a=np.asarray(x[0], dtype=input_dtype[0]),
        b=np.asarray(x[1], dtype=input_dtype[1]),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


# isposinf
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="isposinf"),
)
def test_isposinf(
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
        fn_name="isposinf",
        x=np.asarray(x[0], dtype=input_dtype[0]),
    )


# isneginf
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="isneginf"),
)
def test_isneginf(
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
        fn_name="isneginf",
        x=np.asarray(x[0], dtype=input_dtype[0]),
    )

    
# nan_to_num
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=True,
        allow_inf=True
    ),
    copy=st.booleans(),
    nan=st.floats(min_value=0.0, max_value=100),
    posinf=st.floats(min_value=5e+100, max_value=5e+100),
    neginf=st.floats(min_value=-5e+100, max_value=-5e+100),
    num_positional_args=helpers.num_positional_args(fn_name="nan_to_num"),
)
def test_nan_to_num(
    dtype_and_x,
    copy,
    nan,
    posinf,
    neginf,
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
        fn_name="nan_to_num",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf
    )

    
# logaddexp2
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(fn_name="logaddexp2"),
)
def test_logaddexp2(
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
        fn_name="logaddexp2",
        rtol_=1e-03,
        atol_=1e-03,
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1])
    )
