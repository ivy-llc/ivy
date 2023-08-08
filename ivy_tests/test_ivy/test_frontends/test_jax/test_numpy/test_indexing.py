# global
from hypothesis import strategies as st, assume
import numpy as np
from jax.numpy import tril, triu


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# diagonal
@st.composite
def dims_and_offset(draw, shape):
    shape_actual = draw(shape)
    dim1 = draw(helpers.get_axis(shape=shape, force_int=True))
    dim2 = draw(helpers.get_axis(shape=shape, force_int=True))
    offset = draw(
        st.integers(min_value=-shape_actual[dim1], max_value=shape_actual[dim1])
    )
    return dim1, dim2, offset


@handle_frontend_test(
    fn_tree="jax.numpy.diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_jax_diagonal(
    *,
    dtype_and_values,
    dims_and_offset,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, value = dtype_and_values
    axis1, axis2, offset = dims_and_offset
    a = value[0]
    num_of_dims = len(np.shape(a))
    assume(axis1 != axis2)
    if axis1 < 0:
        assume(axis1 + num_of_dims != axis2)
    if axis2 < 0:
        assume(axis1 != axis2 + num_of_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


# diag
@st.composite
def _diag_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            small_abs_safety_factor=2,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=2,
            min_dim_size=1,
            max_dim_size=50,
        )
    )
    shape = x[0].shape
    if len(shape) == 2:
        k = draw(helpers.ints(min_value=-shape[0] + 1, max_value=shape[1] - 1))
    else:
        k = draw(helpers.ints(min_value=0, max_value=shape[0]))
    return dtype, x, k


@handle_frontend_test(
    fn_tree="jax.numpy.diag",
    dtype_x_k=_diag_helper(),
    test_with_out=st.just(False),
)
def test_jax_diag(
    *,
    dtype_x_k,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    dtype, x, k = dtype_x_k
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.diag_indices",
    n=helpers.ints(min_value=1, max_value=10),
    ndim=helpers.ints(min_value=2, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_diag_indices(
    n,
    ndim,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        ndim=ndim,
    )


# take_along_axis
@handle_frontend_test(
    fn_tree="jax.numpy.take_along_axis",
    dtype_x_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int32", "int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
        valid_bounds=False,
    ),
    mode=st.sampled_from(["clip", "fill", "drop"]),
    test_with_out=st.just(False),
)
def test_jax_take_along_axis(
    *,
    dtype_x_indices_axis,
    mode,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtypes, x, indices, axis, _ = dtype_x_indices_axis
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=x,
        indices=indices,
        axis=axis,
        mode=mode,
    )


# Tril_indices
@handle_frontend_test(
    fn_tree="jax.numpy.tril_indices",
    n_rows=helpers.ints(min_value=1, max_value=10),
    k=helpers.ints(min_value=2, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_tril_indices(
    n_rows,
    k,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n_rows,
        k=k,
    )


# triu_indices
@handle_frontend_test(
    fn_tree="jax.numpy.triu_indices",
    n=helpers.ints(min_value=2, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    input_dtypes=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_triu_indices(
    n,
    k,
    input_dtypes,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        n=n,
        k=k,
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
    )


# triu_indices_from
@handle_frontend_test(
    fn_tree="jax.numpy.triu_indices_from",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
    ),
    k=helpers.ints(min_value=-5, max_value=5),
    test_with_out=st.just(False),
)
def test_jax_triu_indices_from(
    dtype_and_x,
    k,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=x[0],
        k=k,
    )


# tril_indices_from
@handle_frontend_test(
    fn_tree="jax.numpy.tril_indices_from",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
    ),
    k=helpers.ints(min_value=-5, max_value=5),
    test_with_out=st.just(False),
)
def test_jax_tril_indices_from(
    dtype_and_x,
    k,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=x[0],
        k=k,
    )


# unravel_index
@st.composite
def max_value_as_shape_prod(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        )
    )
    dtype_and_x = draw(
        helpers.dtype_values_axis(
            available_dtypes=["int32", "int64"],
            min_value=0,
            max_value=np.prod(shape) - 1,
        )
    )
    return dtype_and_x, shape


@handle_frontend_test(
    fn_tree="jax.numpy.unravel_index",
    dtype_x_shape=max_value_as_shape_prod(),
    test_with_out=st.just(False),
)
def test_jax_unravel_index(
    *,
    dtype_x_shape,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype_and_x, shape = dtype_x_shape
    input_dtype, x = dtype_and_x[0], dtype_and_x[1]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        indices=x[0],
        shape=shape,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.mask_indices",
    n=helpers.ints(min_value=3, max_value=10),
    mask_func=st.sampled_from([triu, tril]),
    k=helpers.ints(min_value=-5, max_value=5),
    input_dtype=helpers.get_dtypes("numeric"),
    test_with_out=st.just(False),
    number_positional_args=st.just(2),
)
def test_jax_mask_indices(
    n,
    mask_func,
    k,
    input_dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        mask_func=mask_func,
        k=k,
    )


@st.composite
def _get_dtype_square_x(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"), shape=(dim_size, dim_size)
        )
    )
    return dtype_x


@handle_frontend_test(
    dtype_x=_get_dtype_square_x(),
    fn_tree="jax.numpy.diag_indices_from",
    test_with_out=st.just(False),
)
def test_jax_diag_indices_from(
    dtype_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.indices",
    dimensions=helpers.get_shape(min_num_dims=1),
    dtype=helpers.get_dtypes("numeric"),
    sparse=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_numpy_indices(
    *,
    dimensions,
    dtype,
    sparse,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        dimensions=dimensions,
        dtype=dtype[0],
        sparse=sparse,
    )
