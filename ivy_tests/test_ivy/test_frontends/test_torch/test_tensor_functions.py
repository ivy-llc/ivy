# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="torch.is_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_is_tensor(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        obj=x[0],
    )


@handle_frontend_test(
    fn_tree="torch.numel",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
)
def test_torch_numel(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@handle_frontend_test(
    fn_tree="torch.is_floating_point",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
)
def test_torch_is_floating_point(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    ivy.set_backend(backend_fw)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=ivy.asarray(x[0]),
    )
    ivy.previous_backend()


@handle_frontend_test(
    fn_tree="torch.is_nonzero",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_torch_is_nonzero(
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@handle_frontend_test(
    fn_tree="torch.is_complex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_torch_is_complex(
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@st.composite
def put_along_axis_helper(draw):
    _shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=2, min_dim_size=3, max_dim_size=5
        )
    )
    _idx = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=len(_shape),
            max_num_dims=len(_shape),
            min_dim_size=1,
            max_dim_size=1,
            min_value=0,
            max_value=len(_shape),
        ),
    )
    dtype_x_axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=_shape,
            min_axis=-len(_shape),
            max_axis=len(_shape) - 1,
            min_value=0,
            max_value=len(_shape) - 1,
        ),
    )
    dtype, x, axis = dtype_x_axis
    _, idx = _idx

    return dtype, x, axis, idx


# scatter
@handle_frontend_test(
    fn_tree="torch.scatter",
    dtype_x_ax_idx=put_along_axis_helper(),
    value=st.integers(min_value=0, max_value=100),
    mode=st.sampled_from(["add", "multiply"]),
    test_with_out=st.just(False),
)
def test_torch_scatter(
    *,
    dtype_x_ax_idx,
    value,
    mode,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis, indices = dtype_x_ax_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        index=indices,
        src=value,
        dim=axis,
        reduce=mode,
    )


# scatter_add
@handle_frontend_test(
    fn_tree="torch.scatter_add",
    dtype_x_ax_idx=put_along_axis_helper(),
    value=st.integers(min_value=0, max_value=100),
    test_with_out=st.just(False),
)
def test_torch_scatter_add(
    *,
    dtype_x_ax_idx,
    value,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis, indices = dtype_x_ax_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        index=indices,
        src=value,
        dim=axis,
    )


# scatter_reduce
@handle_frontend_test(
    fn_tree="torch.scatter",
    dtype_x_ax_idx=put_along_axis_helper(),
    value=st.integers(min_value=0, max_value=100),
    mode=st.sampled_from(["add", "multiply"]),
    test_with_out=st.just(False),
)
def test_torch_scatter_reduce(
    *,
    dtype_x_ax_idx,
    value,
    mode,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis, indices = dtype_x_ax_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        index=indices,
        src=value,
        dim=axis,
        reduce=mode,
    )
