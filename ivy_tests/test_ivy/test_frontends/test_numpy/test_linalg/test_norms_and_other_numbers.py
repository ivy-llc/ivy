# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
    _matrix_rank_helper,
)
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import (
    matrix_is_stable,
)


# norm
@st.composite
def _norm_helper(draw):
    def _matrix_norm_example():
        x_dtype, x = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                shape=helpers.get_shape(min_num_dims=2, max_num_dims=2),
                min_num_dims=2,
                max_num_dims=2,
                min_dim_size=1,
                max_dim_size=10,
                min_value=-1e4,
                max_value=1e4,
                large_abs_safety_factor=10,
                small_abs_safety_factor=10,
                safety_factor_scale="log",
            ),
        )
        ord = draw(st.sampled_from(["fro", "nuc"]))
        axis = (-2, -1)
        check_stable = True
        return x_dtype, x, axis, ord, check_stable

    def _vector_norm_example():
        x_dtype, x, axis = draw(
            helpers.dtype_values_axis(
                available_dtypes=helpers.get_dtypes("float"),
                min_num_dims=2,
                max_num_dims=5,
                min_dim_size=2,
                max_dim_size=10,
                valid_axis=True,
                force_int_axis=True,
                min_value=-1e04,
                max_value=1e04,
                large_abs_safety_factor=10,
                small_abs_safety_factor=10,
                safety_factor_scale="log",
            )
        )
        ints = draw(helpers.ints(min_value=1, max_value=2))
        floats = draw(helpers.floats(min_value=1, max_value=2))
        ord = draw(st.sampled_from([ints, floats, float("inf"), float("-inf")]))
        check_stable = False
        return x_dtype, x, axis, ord, check_stable

    is_vec_norm = draw(st.booleans())
    if is_vec_norm:
        return _vector_norm_example()
    return _matrix_norm_example()


@handle_frontend_test(
    fn_tree="numpy.linalg.norm",
    norm_values=_norm_helper(),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_norm(
    norm_values,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axis, ord, check_stable = norm_values
    if check_stable:
        assume(matrix_is_stable(x[0], cond_limit=10))
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        ord=ord,
        axis=axis,
        keepdims=keepdims,
    )


# matrix_rank
@handle_frontend_test(
    fn_tree="numpy.linalg.matrix_rank",
    dtype_x_hermitian_atol_rtol=_matrix_rank_helper(),
    test_with_out=st.just(False),
)
def test_numpy_matrix_rank(
    dtype_x_hermitian_atol_rtol,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, hermitian, atol, rtol = dtype_x_hermitian_atol_rtol
    assume(matrix_is_stable(x, cond_limit=10))
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        A=x,
        tol=atol,
        hermitian=hermitian,
    )


# det
@handle_frontend_test(
    fn_tree="numpy.linalg.det",
    dtype_and_x=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_numpy_det(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        a=x[0],
    )


# slogdet
@handle_frontend_test(
    fn_tree="numpy.linalg.slogdet",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_value=5,
        min_value=2,
        shape=st.tuples(
            st.shared(st.integers(1, 5), key="sq"),
            st.shared(st.integers(1, 5), key="sq"),
        ),
        num_arrays=1,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_numpy_slogdet(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    assume(matrix_is_stable(x[0]))
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        test_values=False,
    )
    for ret_f, ret_gtt in zip(ret, ret_gt):
        frontend_ret = ret_f
        frontend_ret_gt = ret_gtt
        ret_flattened = helpers.flatten_and_to_np(ret=frontend_ret)
        ret_gt_flattened = helpers.flatten_and_to_np(ret=frontend_ret_gt)
        helpers.value_test(
            ret_np_flat=ret_flattened,
            ret_np_from_gt_flat=ret_gt_flattened,
            rtol=1e-1,
            atol=1e-1,
            ground_truth_backend="numpy",
        )


@handle_frontend_test(
    fn_tree="numpy.trace",
    dtype_and_x_axes=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        min_axes_size=2,
        max_axes_size=2,
        min_num_dims=2,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
    offset=st.integers(min_value=-4, max_value=4),
)
def test_numpy_trace(
    dtype_and_x_axes,
    offset,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, axes = dtype_and_x_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        a=x[0],
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],
    )
