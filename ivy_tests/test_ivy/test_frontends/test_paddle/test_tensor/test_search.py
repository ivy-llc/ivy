# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.argmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_paddle_argmax(
    dtype_x_and_axis,
    keepdim,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    # Skipped dtype test due to paddle functions only accepting str and np.ndarray,
    # but test_frontend_function changes dtype kwargs to native dtype
    input_dtypes, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        axis=axis,
        keepdim=keepdim,
    )


@handle_frontend_test(
    fn_tree="paddle.argmin",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_paddle_argmin(
    dtype_x_and_axis,
    keepdim,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtypes, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        axis=axis,
        keepdim=keepdim,
    )


# argsort
@handle_frontend_test(
    fn_tree="paddle.argsort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    descending=st.booleans(),
)
def test_paddle_argsort(
    dtype_input_axis,
    descending,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        descending=descending,
    )


# sort
@handle_frontend_test(
    fn_tree="paddle.tensor.search.sort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    descending=st.booleans(),
)
def test_paddle_sort(
    *,
    dtype_input_axis,
    descending,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        x=x[0],
        axis=axis,
        descending=descending,
    )


# nonzero
@handle_frontend_test(
    fn_tree="paddle.nonzero",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    as_tuple=st.booleans(),
)
def test_paddle_nonzero(
    *,
    dtype_and_values,
    as_tuple,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, input = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        as_tuple=as_tuple,
    )


# searchsorted
@handle_frontend_test(
    fn_tree="paddle.searchsorted",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        min_num_dims=1,
        num_arrays=2,
    ),
    out_int32=st.booleans(),
    right=st.booleans(),
)
def test_paddle_searchsorted(
    *,
    dtype_and_values,
    out_int32,
    right,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, input = dtype_and_values
    input[0] = np.sort(input[0])
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        sorted_sequence=input[0],
        values=input[1],
        out_int32=out_int32,
        right=right,
    )


# masked_select
@st.composite
def _dtypes_input_mask(draw):
    _shape = draw(helpers.get_shape(min_num_dims=1, min_dim_size=1))
    _mask = draw(helpers.array_values(dtype="bool", shape=_shape))
    _dtype, _x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=1,
            shape=_shape,
        )
    )

    return _dtype, _x, _mask


@handle_frontend_test(
    fn_tree="paddle.masked_select",
    dtype_input_mask=_dtypes_input_mask(),
)
def test_paddle_masked_select(
    *,
    dtype_input_mask,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    (
        input_dtype,
        x,
        mask,
    ) = dtype_input_mask

    helpers.test_frontend_function(
        input_dtypes=input_dtype + ["bool"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        mask=mask,
    )


@handle_frontend_test(
    fn_tree="paddle.topk",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    k=st.data(),
    sorted=st.booleans(),
    largest=st.booleans(),
)
def test_paddle_topk(
    *,
    dtype_x_and_axis,
    k,
    sorted,
    largest,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtypes, x, axis = dtype_x_and_axis
    k = k.draw(st.integers(min_value=1, max_value=x[0].shape[axis]))
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        k=k,
        axis=axis,
        largest=largest,
        sorted=sorted,
        test_values=False,
    )


@handle_frontend_test(
    fn_tree="paddle.where",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_paddle_where(
    *,
    dtype_x_and_axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtypes, x, axis = dtype_x_and_axis
    condition = x > 0.5
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        condition=condition,
        x=x[0],
        y=axis,
    )
    
