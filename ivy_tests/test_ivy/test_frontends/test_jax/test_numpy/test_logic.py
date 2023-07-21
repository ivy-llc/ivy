# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_helpers


# allclose
@handle_frontend_test(
    fn_tree="jax.numpy.allclose",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_allclose(
    *,
    dtype_and_input,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=input[0],
        b=input[1],
        equal_nan=equal_nan,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.array_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_array_equal(
    *,
    dtype_and_x,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a1=x[0],
        a2=x[1],
        equal_nan=equal_nan,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.array_equiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_array_equiv(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a1=x[0],
        a2=x[1],
    )


# isneginf
@handle_frontend_test(
    fn_tree="jax.numpy.isneginf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_isneginf(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# isposinf
@handle_frontend_test(
    fn_tree="jax.numpy.isposinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_jax_isposinf(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# less
@handle_frontend_test(
    fn_tree="jax.numpy.less",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_less(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# less_equal
@handle_frontend_test(
    fn_tree="jax.numpy.less_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_less_equal(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# greater
@handle_frontend_test(
    fn_tree="jax.numpy.greater",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_greater(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# greater_equal
@handle_frontend_test(
    fn_tree="jax.numpy.greater_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_greater_equal(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# isnan
@handle_frontend_test(
    fn_tree="jax.numpy.isnan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_jax_isnan(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# equal
@handle_frontend_test(
    fn_tree="jax.numpy.equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_equal(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# not_equal
@handle_frontend_test(
    fn_tree="jax.numpy.not_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_not_equal(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# all
@handle_frontend_test(
    fn_tree="jax.numpy.all",
    # aliases=["jax.numpy.alltrue"], deprecated since 0.4.12.
    #  uncomment with multi-version testing pipeline
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
    test_with_out=st.just(False),
)
def test_jax_all(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# bitwise_and
# TODO: add testing for other dtypes
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"), num_arrays=2
    ),
    test_with_out=st.just(False),
)
def test_jax_bitwise_and(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# bitwise_not
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_not",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("bool")),
    test_with_out=st.just(False),
)
def test_jax_bitwise_not(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# bitwise_or
# TODO: add testing for other dtypes
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"), num_arrays=2
    ),
    test_with_out=st.just(False),
)
def test_jax_bitwise_or(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# bitwise_xor
# TODO: add testing for other dtypes
@handle_frontend_test(
    fn_tree="jax.numpy.bitwise_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"), num_arrays=2
    ),
    test_with_out=st.just(False),
)
def test_jax_bitwise_xor(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# any
@handle_frontend_test(
    fn_tree="jax.numpy.any",
    # aliases=["jax.numpy.sometrue"], deprecated since 0.4.12.
    # uncomment with multi-version testing pipeline
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        max_axes_size=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    where=np_helpers.where(),
    test_with_out=st.just(False),
)
def test_jax_any(
    *,
    dtype_x_axis,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    where, input_dtypes, test_flags = np_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
    )


# logical_and
@handle_frontend_test(
    fn_tree="jax.numpy.logical_and",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_jax_logical_and(
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtypes, x = dtypes_values
    np_helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# invert
@handle_frontend_test(
    fn_tree="jax.numpy.invert",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
)
def test_jax_invert(
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtypes, x = dtypes_values
    np_helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# isfinite
@handle_frontend_test(
    fn_tree="jax.numpy.isfinite",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), allow_nan=True
    ),
    test_with_out=st.just(False),
)
def test_jax_isfinite(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# isinf
@handle_frontend_test(
    fn_tree="jax.numpy.isinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), allow_inf=True
    ),
    test_with_out=st.just(False),
)
def test_jax_isinf(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# isclose
@handle_frontend_test(
    fn_tree="jax.numpy.isclose",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
    equal_nan=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_isclose(
    *,
    dtype_and_input,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=input[0],
        b=input[1],
        equal_nan=equal_nan,
    )


# logical_not
@handle_frontend_test(
    fn_tree="jax.numpy.logical_not",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=1,
    ),
)
def test_jax_logical_not(
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtypes, x = dtypes_values
    np_helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# logical_or
@handle_frontend_test(
    fn_tree="jax.numpy.logical_or",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_jax_logical_or(
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtypes, x = dtypes_values
    np_helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# isscalar
@handle_frontend_test(
    fn_tree="jax.numpy.isscalar",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_jax_isscalar(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtypes, x = dtype_and_x
    np_helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# left_shift
@handle_frontend_test(
    fn_tree="jax.numpy.left_shift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_jax_left_shift(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# isreal
@handle_frontend_test(
    fn_tree="jax.numpy.isreal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_inf=True,
    ),
)
def test_jax_isreal(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# logical_xor
@handle_frontend_test(
    fn_tree="jax.numpy.logical_xor",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_jax_logical_xor(
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtypes, x = dtypes_values
    np_helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.right_shift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_right_shift(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, xs = dtype_and_x

    xs[1] = np.asarray(np.clip(xs[1], 0, np.iinfo(dtype[1]).bits - 1), dtype=dtype[1])

    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
    )


# iscomplex
@handle_frontend_test(
    fn_tree="jax.numpy.iscomplex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_iscomplex(
    dtype_and_x,
    frontend,
    on_device,
    *,
    fn_tree,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.isrealobj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_isrealobj(
    dtype_and_x,
    frontend,
    on_device,
    *,
    fn_tree,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# iscomplexobj
@handle_frontend_test(
    fn_tree="jax.numpy.iscomplexobj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(False),
)
def test_jax_iscomplexobj(
    dtype_and_x,
    frontend,
    on_device,
    *,
    fn_tree,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# setxor1d
@handle_frontend_test(
    fn_tree="jax.numpy.setxor1d",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
    assume_unique=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_numpy_setxor1d(
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    assume_unique,
):
    x_dtypes, x = dtypes_values
    helpers.test_frontend_function(
        input_dtypes=x_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ar1=x[0],
        ar2=x[1],
        assume_unique=assume_unique,
    )
