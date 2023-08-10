# global
from hypothesis import strategies as st
import hypothesis.extra.numpy as nph
import numpy as np

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
    input_dtype, x, axis, shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=["int64"],
            min_num_dims=2,
            min_dim_size=2,
            valid_axis=True,
            force_int_axis=True,
            ret_shape=True,
            min_axis=0,
            max_axis=1,
        )
    )

    if axis < 0:
        axis = 0
    idx_shape = list(shape)
    idx_shape[axis] = 1
    idx_shape = tuple(idx_shape)

    idx_strategy = nph.arrays(
        dtype=np.int64, shape=idx_shape, elements=st.integers(0, len(idx_shape) - 1)
    )
    indices = draw(idx_strategy)

    values_strategy = nph.arrays(
        dtype=input_dtype[0], shape=idx_shape, elements=st.integers(0, 1e3)
    )
    values = draw(values_strategy)

    return input_dtype, x[0], indices, values, axis


# scatter
@handle_frontend_test(
    fn_tree="torch.scatter",
    args=put_along_axis_helper(),
    test_with_out=st.just(False),
)
def test_torch_scatter(
    *,
    args,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, indices, value, axis = args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x[0],
        index=indices,
        src=value,
        dim=axis,
    )


# scatter_add
@handle_frontend_test(
    fn_tree="torch.scatter_add",
    args=put_along_axis_helper(),
    test_with_out=st.just(False),
)
def test_torch_scatter_add(
    *,
    args,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, indices, value, axis = args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x[0],
        index=indices,
        src=value,
        dim=axis,
    )


# scatter_reduce
@handle_frontend_test(
    fn_tree="torch.scatter",
    args=put_along_axis_helper(),
    mode=st.sampled_from(["add", "mul", "mean", "amin", "amax"]),
    test_with_out=st.just(False),
)
def test_torch_scatter_reduce(
    *,
    args,
    mode,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, indices, value, axis = args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x[0],
        index=indices,
        src=value,
        dim=axis,
        reduce=mode,
    )
