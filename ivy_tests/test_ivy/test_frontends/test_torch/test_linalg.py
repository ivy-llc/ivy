# global
import sys
import numpy as np
from hypothesis import strategies as st


# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test


# helpers
@st.composite
def _get_dtype_and_square_matrix(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


@st.composite
def _get_dtype_and_matrix(draw):
    arbitrary_dims = draw(helpers.get_shape(max_dim_size=5))
    random_size = draw(st.integers(min_value=1, max_value=4))
    shape = (*arbitrary_dims, random_size, random_size)
    return draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )


# inv
@handle_frontend_test(
    fn_tree="torch.linalg.inv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_value=0,
        max_value=25,
        shape=helpers.ints(min_value=2, max_value=10).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon),
)
def test_torch_inv(
    *,
    dtype_and_x,
    as_variable,
    with_out,
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
        with_out=with_out,
        all_aliases=["inverse"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        input=x[0],
    )


# pinv
@handle_frontend_test(
    fn_tree="torch.linalg.pinv",
    dtype_and_input=_get_dtype_and_matrix(),
)
def test_torch_pinv(
    *,
    dtype_and_input,
    num_positional_args,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        native_array_flags=native_array,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        atol=1e-15,
        rtol=1e-15,
    )


# det
@handle_frontend_test(
    fn_tree="torch.linalg.det",
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_det(
    *,
    dtype_and_x,
    as_variable,
    with_out,
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
        with_out=with_out,
        all_aliases=["det"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
    )


# qr
@handle_frontend_test(
    fn_tree="torch.linalg.qr",
    dtype_and_input=_get_dtype_and_matrix(),
)
def test_torch_qr(
    *,
    dtype_and_input,
    num_positional_args,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        native_array_flags=native_array,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-05,
        input=x[0],
    )


# slogdet
@handle_frontend_test(
    fn_tree="torch.linalg.slogdet",
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_slogdet(
    *,
    dtype_and_x,
    as_variable,
    with_out,
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
        with_out=with_out,
        all_aliases=["slogdet"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
    )


# eigvalsh
@handle_frontend_test(
    fn_tree="torch.linalg.eigvalsh",
    dtype_and_input=_get_dtype_and_matrix(),
)
def test_torch_eigvalsh(
    *,
    dtype_and_input,
    num_positional_args,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        native_array_flags=native_array,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# matrix_power
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_power",
    dtype_and_x=_get_dtype_and_square_matrix(),
    n=helpers.ints(min_value=2, max_value=5),
)
def test_torch_matrix_power(
    *,
    dtype_and_x,
    n,
    as_variable,
    with_out,
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
        with_out=with_out,
        all_aliases=["matrix_power"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=x,
        n=n,
    )


@st.composite
def _matrix_rank_helper(draw):
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            min_value=-1e05,
            max_value=1e05,
        )
    )
    return dtype_x


# matrix_rank
@handle_frontend_test(
    fn_tree="torch.linalg.matrix_rank",
    dtype_and_x=_matrix_rank_helper(),
    atol=st.floats(min_value=1e-5, max_value=0.1, exclude_min=True, exclude_max=True),
    rtol=st.floats(min_value=1e-5, max_value=0.1, exclude_min=True, exclude_max=True),
)
def test_matrix_rank(
    dtype_and_x,
    rtol,
    atol,
    as_variable,
    with_out,
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
        with_out=with_out,
        all_aliases=["matrix_rank"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        rtol=rtol,
        atol=atol,
    )


# svd
@handle_frontend_test(
    fn_tree="torch.linalg.svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
    full_matrices=st.booleans(),
)
def test_torch_svd(
    *,
    dtype_and_x,
    full_matrices,
    with_out,
    num_positional_args,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        atol=1e-03,
        rtol=1e-05,
        input=x,
        full_matrices=full_matrices,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    u, s, vh = ret
    frontend_u, frontend_s, frontend_vh = frontend_ret

    assert_all_close(
        ret_np=u @ np.diag(s) @ vh,
        ret_from_gt_np=frontend_u @ np.diag(frontend_s) @ frontend_vh,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )
    
    
#svdvals
@handle_frontend_test(
    fn_tree="torch.linalg.svdvals",
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_torch_svdvals(
    *,
    dtype_and_x,
    as_variable,
    with_out,
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
        with_out=with_out,
        all_aliases=["svdvals"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
    )
