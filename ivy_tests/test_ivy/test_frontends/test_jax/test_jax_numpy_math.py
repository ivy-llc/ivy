# global
from hypothesis import strategies as st, assume
import numpy as np
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
    _get_dtype_value1_value2_axis_for_tensordot,
)


# absolute
@handle_frontend_test(
    fn_tree="jax.numpy.absolute",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
)
def test_jax_numpy_absolute(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        all_aliases=["numpy.abs"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# add
@handle_frontend_test(
    fn_tree="jax.numpy.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_jax_numpy_add(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# arctan
@handle_frontend_test(
    fn_tree="jax.numpy.arctan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arctan(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arctan2
@handle_frontend_test(
    fn_tree="jax.numpy.arctan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_jax_numpy_arctan2(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.cos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_cos(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# cosh
@handle_frontend_test(
    fn_tree="jax.numpy.cosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_cosh(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# tanh
@handle_frontend_test(
    fn_tree="jax.numpy.tanh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_tanh(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sinh
@handle_frontend_test(
    fn_tree="jax.numpy.sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
)
def test_jax_numpy_sinh(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.sin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_sin(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# floor
@handle_frontend_test(
    fn_tree="jax.numpy.floor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_floor(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# tensordot
@handle_frontend_test(
    fn_tree="jax.numpy.tensordot",
    dtype_values_and_axes=_get_dtype_value1_value2_axis_for_tensordot(
        helpers.get_dtypes(kind="numeric")
    ),
)
def test_jax_numpy_tensordot(
    dtype_values_and_axes,
    as_variable,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
):
    dtype, a, b, axes = dtype_values_and_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        a=a,
        b=b,
        axes=axes,
    )


# divide
@handle_frontend_test(
    fn_tree="jax.numpy.divide",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=4,
        safety_factor_scale="linear",
        shared_dtype=True,
    ),
)
def test_jax_numpy_divide(
    *,
    dtype_values,
    as_variable,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_values
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        all_aliases=["numpy.true_divide"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        a=x[0],
        b=x[1],
    )


# exp
@handle_frontend_test(
    fn_tree="jax.numpy.exp",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_exp(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# dot
@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    if dim_size == 1:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
    else:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
    return dtype, vec1, vec2


@handle_frontend_test(
    fn_tree="jax.numpy.dot",
    dtype_x_y=_get_dtype_input_and_vectors(),
)
def test_jax_numpy_dot(
    *,
    dtype_x_y,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, y = dtype_x_y
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        b=y,
        precision=None,
    )


# mod
@handle_frontend_test(
    fn_tree="jax.numpy.mod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_jax_numpy_mod(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)) and "bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# tan
@handle_frontend_test(
    fn_tree="jax.numpy.tan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_tan(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arccos
@handle_frontend_test(
    fn_tree="jax.numpy.arccos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arccos(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arccosh
@handle_frontend_test(
    fn_tree="jax.numpy.arccosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arccosh(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arcsin
@handle_frontend_test(
    fn_tree="jax.numpy.arcsin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_arcsin(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arcsinh
@handle_frontend_test(
    fn_tree="jax.numpy.arcsinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
)
def test_jax_numpy_arcsinh(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# power
@handle_frontend_test(
    fn_tree="jax.numpy.power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_jax_numpy_power(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# trunc
@handle_frontend_test(
    fn_tree="jax.numpy.trunc",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_trunc(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# ceil
@handle_frontend_test(
    fn_tree="jax.numpy.ceil",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_ceil(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# float_power
@handle_frontend_test(
    fn_tree="jax.numpy.float_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_jax_numpy_float_power(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# deg2rad
@handle_frontend_test(
    fn_tree="jax.numpy.deg2rad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_numpy_deg2rad(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# exp2
@handle_frontend_test(
    fn_tree="jax.numpy.exp2",
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
def test_jax_numpy_exp2(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        rtol=1e-01,
        atol=1e-02,
    )


# gcd
@handle_frontend_test(
    fn_tree="jax.numpy.gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
    ).filter(lambda x: all([dtype != "uint64" for dtype in x[0]])),
)
def test_jax_numpy_gcd(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# i0
@handle_frontend_test(
    fn_tree="jax.numpy.i0",
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
def test_jax_numpy_i0(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# kron
@handle_frontend_test(
    fn_tree="jax.numpy.kron",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
    ),
)
def test_jax_numpy_kron(
    *,
    dtype_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
    )


# lcm
@handle_frontend_test(
    fn_tree="jax.numpy.lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_jax_numpy_lcm(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    value_test = True
    # Skip Tensorflow backend value test for lcm
    # https://github.com/tensorflow/tensorflow/issues/58955
    if ivy.current_backend_str() == "tensorflow":
        value_test = False
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        test_values=value_test,
    )


# logaddexp2
@handle_frontend_test(
    fn_tree="jax.numpy.logaddexp2",
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
def test_jax_numpy_logaddexp2(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x1=x[0],
        x2=x[1],
    )


# matmul
@st.composite
def _get_safe_casting_dtype(draw, *, dtypes):
    target_dtype = dtypes[0]
    for dtype in dtypes[1:]:
        if ivy.can_cast(target_dtype, dtype):
            target_dtype = dtype
    if ivy.is_float_dtype(target_dtype):
        dtype = draw(st.sampled_from(["float64", None]))
    elif ivy.is_uint_dtype(target_dtype):
        dtype = draw(st.sampled_from(["uint64", None]))
    elif ivy.is_int_dtype(target_dtype):
        dtype = draw(st.sampled_from(["int64", None]))
    else:
        dtype = draw(st.sampled_from(["bool", None]))
    return dtype


@st.composite
def dtypes_values_casting_dtype(
    draw,
    *,
    arr_func,
    get_dtypes_kind="valid",
    get_dtypes_index=0,
    get_dtypes_none=True,
    get_dtypes_key=None,
    special=False,
):
    dtypes, values = [], []
    casting = draw(st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"]))
    for func in arr_func:
        typ, val = draw(func())
        dtypes += typ if isinstance(typ, list) else [typ]
        values += val if isinstance(val, list) else [val]

    if casting in ["no", "equiv"] and len(dtypes) > 0:
        dtypes = [dtypes[0]] * len(dtypes)

    if special:
        dtype = draw(st.sampled_from(["bool", None]))
    elif casting in ["no", "equiv"]:
        dtype = draw(st.just(None))
    elif casting in ["safe", "same_kind"]:
        dtype = draw(_get_safe_casting_dtype(dtypes=dtypes))
    else:
        dtype = draw(
            helpers.get_dtypes(
                get_dtypes_kind,
                index=get_dtypes_index,
                full=False,
                none=get_dtypes_none,
                key=get_dtypes_key,
            )
        )[0]
    return dtypes, values, casting, dtype


# matmul
@handle_frontend_test(
    fn_tree="jax.numpy.matmul",
    dtypes_values_casting=dtypes_values_casting_dtype(
        arr_func=[_get_first_matrix_and_dtype, _get_second_matrix_and_dtype],
        get_dtypes_kind="numeric",
    ),
)
def test_jax_numpy_matmul(
    dtypes_values_casting,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
        precision=None,
    )


# trapz
@st.composite
def _either_x_dx(draw):
    dtype_values_axis = draw(
        helpers.dtype_values_axis(
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
    )
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
        return dtype_values_axis, rand, either_x_dx
    else:
        either_x_dx = draw(
            st.floats(min_value=-10, max_value=10),
        )
        return dtype_values_axis, rand, either_x_dx


@handle_frontend_test(
    fn_tree="jax.numpy.trapz",
    dtype_x_axis_rand_either=_either_x_dx(),
)
def test_jax_numpy_trapz(
    *,
    dtype_x_axis_rand_either,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype_values_axis, rand, either_x_dx = dtype_x_axis_rand_either
    input_dtype, y, axis = dtype_values_axis
    if rand == 0:
        dtype_x, x = either_x_dx
        x = np.asarray(x, dtype=dtype_x)
        dx = None
    else:
        x = None
        dx = either_x_dx
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        y=y[0],
        x=x,
        dx=dx,
        axis=axis,
    )


# sqrt
@handle_frontend_test(
    fn_tree="jax.numpy.sqrt",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_sqrt(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# square
@handle_frontend_test(
    fn_tree="jax.numpy.square",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_numpy_square(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arctanh
@handle_frontend_test(
    fn_tree="jax.numpy.arctanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=0,
    ),
)
def test_jax_numpy_arctanh(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x=x[0],
    )


# multiply
@handle_frontend_test(
    fn_tree="jax.numpy.multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_jax_numpy_multiply(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# log10
@handle_frontend_test(
    fn_tree="jax.numpy.log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
    ),
)
def test_jax_numpy_log10(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x=x[0],
    )


# logaddexp
@handle_frontend_test(
    fn_tree="jax.numpy.logaddexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_jax_numpy_logaddexp(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x1=x[0],
        x2=x[1],
    )


# degrees
@handle_frontend_test(
    fn_tree="jax.numpy.degrees",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_numpy_degrees(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# negative
@handle_frontend_test(
    fn_tree="jax.numpy.negative",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
)
def test_jax_numpy_negative(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# rad2deg
@handle_frontend_test(
    fn_tree="jax.numpy.rad2deg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1
    ),
)
def test_jax_numpy_rad2deg(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# fmax
@handle_frontend_test(
    fn_tree="jax.numpy.fmax",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_jax_numpy_fmax(
    *,
    dtype_and_inputs,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, inputs = dtype_and_inputs
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=inputs[0],
        x2=inputs[1],
    )


# maximum
@handle_frontend_test(
    fn_tree="jax.numpy.maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_jax_numpy_maximum(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# minimum
@handle_frontend_test(
    fn_tree="jax.numpy.minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.minimum"
    ),
)
def test_jax_numpy_minimum(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# heaviside
@handle_frontend_test(
    fn_tree="jax.numpy.heaviside",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
    ),
)
def test_jax_numpy_heaviside(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# log
@handle_frontend_test(
    fn_tree="jax.numpy.log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
    ),
)
def test_jax_numpy_log(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x=x[0],
    )


# copysign
@handle_frontend_test(
    fn_tree="jax.numpy.copysign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True
    ),
)
def test_jax_numpy_copysign(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )
