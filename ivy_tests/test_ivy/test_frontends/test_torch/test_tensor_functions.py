# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa: E501
    put_along_axis_helper,
)


# broadcast_tensors
@handle_frontend_test(
    fn_tree="torch.broadcast_tensors",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        num_arrays=helpers.ints(min_value=2, max_value=5),
    ),
)
def test_torch_broadcast_tensors(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    kw = {}
    for i, array in enumerate(x):
        kw[f"x{i}"] = array
    test_flags.num_positional_args = len(kw)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        **kw,
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
    backend_fw,
    test_flags,
):
    dtypes, x, indices, value, axis = args
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x,
        dim=axis,
        index=indices,
        src=value,
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
    backend_fw,
    test_flags,
):
    dtypes, x, indices, value, axis = args
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x,
        dim=axis,
        index=indices,
        src=value,
    )


# scatter_reduce
@handle_frontend_test(
    fn_tree="torch.scatter_reduce",
    args=put_along_axis_helper(),
    # ToDo: test for "mean" as soon as ivy.put_along_axis supports it
    mode=st.sampled_from(["sum", "prod", "amin", "amax"]),
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
    dtypes, x, indices, value, axis = args
    test_flags.ground_truth_backend = "torch"
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x,
        dim=axis,
        index=indices,
        src=value,
        reduce=mode,
    )
