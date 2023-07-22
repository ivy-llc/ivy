# global
import numpy as np
from hypothesis import assume, strategies as st
import sys
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test, assert_all_close
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_value1_value2_axis_for_tensordot,
)
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import (
    matrix_is_stable,
)
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _matrix_rank_helper


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


@handle_frontend_test(
    fn_tree="tensorflow.linalg.det",
    dtype_and_input=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_det(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.eigh",
    dtype_and_input=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_eigh(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    assume(matrix_is_stable(x[0]))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.eigvals",
    dtype_and_input=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_eigvals(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    assume(matrix_is_stable(x[0]))
    if x[0].dtype == ivy.float32:
        x[0] = x[0].astype("float64")
        input_dtype = [ivy.float64]
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
        test_values=False,
    )

    ret = ivy.to_numpy(ret)
    ret = ret.round(6)
    ret = np.sort(ret)
    frontend_ret = frontend_ret[0].numpy()
    frontend_ret = frontend_ret.round(6)
    frontend_ret = np.sort(frontend_ret)

    assert_all_close(
        ret_np=ret,
        ret_from_gt_np=frontend_ret,
        rtol=1e-06,
        atol=1e-06,
        ground_truth_backend=frontend,
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.eigvalsh",
    dtype_and_input=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_eigvalsh(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    assume(matrix_is_stable(x[0]))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.matrix_rank",
    dtype_x_hermitian_atol_rtol=_matrix_rank_helper(),
    test_with_out=st.just(False),
)
def test_matrix_rank(
    *,
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
        a=x,
        tol=atol,
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.matmul",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=(3, 3),
        num_arrays=2,
        shared_dtype=True,
        min_value=-1,
        max_value=100,
    ),
    transpose_a=st.booleans(),
    transpose_b=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_matmul(
    *,
    dtype_x,
    transpose_a,
    transpose_b,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


# solve
@st.composite
def _get_first_matrix(draw):
    input_dtype_strategy = st.shared(
        st.sampled_from(draw(helpers.get_dtypes("float"))),
        key="shared_dtype",
    )
    input_dtype = draw(input_dtype_strategy)
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([shared_size, shared_size]),
            min_value=2,
            max_value=5,
        ).filter(lambda x: np.linalg.cond(x) < 1 / sys.float_info.epsilon)
    )
    return input_dtype, matrix


# solve
@st.composite
def _get_second_matrix(draw):
    input_dtype_strategy = st.shared(
        st.sampled_from(draw(helpers.get_dtypes("float"))),
        key="shared_dtype",
    )
    input_dtype = draw(input_dtype_strategy)

    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype, shape=tuple([shared_size, 1]), min_value=2, max_value=5
        )
    )


# solve
@handle_frontend_test(
    fn_tree="tensorflow.linalg.solve",
    x=_get_first_matrix(),
    y=_get_second_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_solve(
    *,
    x,
    y,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype1, x1 = x
    input_dtype2, x2 = y
    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        matrix=x1,
        rhs=x2,
    )


# logdet
@st.composite
def _get_hermitian_pos_def_matrix(draw):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("float"))),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    gen = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([shared_size, shared_size]),
            min_value=2,
            max_value=5,
        ).filter(lambda x: np.linalg.cond(x.tolist()) < 1 / sys.float_info.epsilon)
    )
    hpd = np.matmul(np.matrix(gen).getH(), np.matrix(gen)) + np.identity(gen.shape[0])
    return [input_dtype], hpd


@handle_frontend_test(
    fn_tree="tensorflow.linalg.logdet",
    dtype_and_x=_get_hermitian_pos_def_matrix(),
)
def test_tensorflow_logdet(
    *,
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
        matrix=x,
    )


# slogdet
@handle_frontend_test(
    fn_tree="tensorflow.linalg.slogdet",
    dtype_and_x=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_slogdet(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# cholesky_solve
@st.composite
def _get_cholesky_matrix(draw):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("float"))),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    gen = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([shared_size, shared_size]),
            min_value=2,
            max_value=5,
        ).filter(lambda x: np.linalg.cond(x.tolist()) < 1 / sys.float_info.epsilon)
    )
    spd = np.matmul(gen.T, gen) + np.identity(gen.shape[0])
    spd_chol = np.linalg.cholesky(spd)
    return input_dtype, spd_chol


@st.composite
def _get_second_matrix(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("float"))),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype, shape=tuple([shared_size, 1]), min_value=2, max_value=5
        )
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.cholesky_solve",
    x=_get_cholesky_matrix(),
    y=_get_second_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_cholesky_solve(
    *,
    x,
    y,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype1, x1 = x
    input_dtype2, x2 = y
    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        chol=x1,
        rhs=x2,
    )


# pinv
@handle_frontend_test(
    fn_tree="tensorflow.linalg.pinv",
    dtype_and_input=_get_dtype_and_matrix(),
)
def test_tensorflow_pinv(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        a=x[0],
        rcond=1e-15,
    )


# tensordot
@handle_frontend_test(
    fn_tree="tensorflow.linalg.tensordot",
    dtype_x_y_axes=_get_dtype_value1_value2_axis_for_tensordot(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_tensorflow_tensordot(
    *,
    dtype_x_y_axes,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    (
        dtype,
        x,
        y,
        axes,
    ) = dtype_x_y_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        b=y,
        axes=axes,
    )


# norm
@handle_frontend_test(
    fn_tree="tensorflow.linalg.norm",
    aliases=["tensorflow.norm"],
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=4,
        min_axis=-3,
        max_axis=2,
    ),
    ord=st.sampled_from([1, 2, np.inf]),
    keepdims=st.booleans(),
)
def test_tensorflow_norm(
    *,
    dtype_values_axis,
    ord,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
        ord=ord,
        axis=axis,
        keepdims=keepdims,
    )


# normalize
@handle_frontend_test(
    fn_tree="tensorflow.linalg.normalize",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=4,
        min_axis=-3,
        max_axis=2,
    ),
    ord=st.sampled_from([1, 2, np.inf]),
    test_with_out=st.just(False),
)
def test_tensorflow_normalize(
    *,
    dtype_values_axis,
    ord,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
        ord=ord,
        axis=axis,
        atol=1e-08,
    )


# l2_normalize
@handle_frontend_test(
    fn_tree="tensorflow.linalg.l2_normalize",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=4,
        min_axis=-3,
        max_axis=2,
    ),
)
def test_tensorflow_l2_normalize(
    *,
    dtype_values_axis,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )


# trace
@handle_frontend_test(
    fn_tree="tensorflow.linalg.trace",
    dtype_and_input=_get_dtype_and_matrix(),
    test_with_out=st.just(False),
)
def test_tensorflow_trace(
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
    )


# matrix_transpose
@handle_frontend_test(
    fn_tree="tensorflow.linalg.matrix_transpose",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
    ),
    conjugate=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_matrix_transpose(
    dtype_and_input,
    conjugate,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=x[0],
        conjugate=conjugate,
    )


@st.composite
def _get_dtype_and_sequence_of_arrays(draw):
    array_dtype = draw(helpers.get_dtypes("float", full=False))
    arbitrary_size = draw(st.integers(min_value=2, max_value=10))
    values = []
    for i in range(arbitrary_size):
        values.append(
            draw(
                helpers.array_values(
                    dtype=array_dtype[0], shape=helpers.get_shape(), allow_nan=True
                )
            )
        )
    return array_dtype, values


@handle_frontend_test(
    fn_tree="tensorflow.linalg.global_norm",
    dtype_and_input=_get_dtype_and_sequence_of_arrays(),
    test_with_out=st.just(False),
)
def test_tensorflow_global_norm(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        t_list=x,
    )


# cholesky
@handle_frontend_test(
    fn_tree="tensorflow.linalg.cholesky",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_linalg_cholesky(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive-definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        input=x,
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.cross",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=3,
        max_dim_size=3,
        shared_dtype=True,
    ),
)
def test_tensorflow_linalg_cross(
    frontend, on_device, dtype_and_x, *, fn_tree, test_flags
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.linalg.svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
    full_matrices=st.booleans(),
    compute_uv=st.just(True),
)
def test_tensorflow_svd(
    *,
    dtype_and_x,
    full_matrices,
    compute_uv,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    # make symmetric positive definite beforehand
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        atol=1e-03,
        rtol=1e-05,
        tensor=x,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    u, s, vh = ret
    frontend_s, frontend_u, frontend_vh = frontend_ret

    assert_all_close(
        ret_np=u @ np.diag(s) @ vh,
        ret_from_gt_np=frontend_u @ np.diag(frontend_s) @ frontend_vh.T,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )


# einsum
@handle_frontend_test(
    fn_tree="tensorflow.linalg.einsum",
    eq_n_op_n_shp=st.sampled_from(
        [
            ("ii", (np.arange(25).reshape(5, 5),), ()),
            ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
            ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_tensorflow_linalg_einsum(
    *,
    eq_n_op_n_shp,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    eq, operands, _ = eq_n_op_n_shp
    kw = {}
    i = 0
    for x_ in operands:
        kw["x{}".format(i)] = x_
        i += 1
    # len(operands) + 1 because of the equation
    test_flags.num_positional_args = len(operands) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        equation=eq,
        **kw,
    )


# adjoint
@handle_frontend_test(
    fn_tree="tensorflow.linalg.adjoint",
    dtype_and_x=_get_dtype_and_matrix().filter(
        lambda x: "float16" not in x[0] and "bfloat16" not in x[0]
    ),  # TODO : remove this filter when paddle.conj supports float16
    test_with_out=st.just(False),
)
def test_tensorflow_adjoint(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        matrix=x[0],
    )


# diag
@handle_frontend_test(
    fn_tree="tensorflow.linalg.diag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["int64", "int32"],
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=5,
        max_dim_size=10,
        min_value=0,
        max_value=10,
    ),
    k=st.just(0),
)
def test_tensorflow_diag(
    dtype_and_x,
    k,
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
        v=x[0],
        k=k,
    )


@st.composite
def _get_dtype_and_matrix_and_num(draw):
    arbitrary_dims = draw(helpers.get_shape(max_dim_size=5))
    random_size = draw(st.integers(min_value=1, max_value=4))
    shape = (*arbitrary_dims, random_size, random_size)
    dtype_and_values = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            min_value=-10,
            max_value=10,
        )
    )
    num_lower = draw(st.integers(min_value=-1, max_value=random_size - 1))
    num_upper = draw(st.integers(min_value=-1, max_value=random_size - 1))
    return (*dtype_and_values, num_lower, num_upper)


# band_part
@handle_frontend_test(
    fn_tree="tensorflow.linalg.band_part",
    dtype_and_input=_get_dtype_and_matrix_and_num(),
    test_with_out=st.just(False),
)
def test_tensorflow_band_part(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, num_lower, num_upper = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        num_lower=num_lower,
        num_upper=num_upper,
    )


# inv
@handle_frontend_test(
    fn_tree="tensorflow.linalg.inv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-100,
        max_value=100,
        shape=helpers.ints(min_value=1, max_value=20).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    adjoint=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_inv(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    adjoint,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        adjoint=adjoint,
    )


# qr
@handle_frontend_test(
    fn_tree="tensorflow.linalg.qr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
)
def test_qr(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])
    x = np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        atol=1e-03,
        rtol=1e-05,
        input=x,
    )
    ret = [ivy.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    assert_all_close(
        ret_np=ret[0],
        ret_from_gt_np=frontend_ret[0],
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend=frontend,
    )


# tensor_diag
@handle_frontend_test(
    fn_tree="tensorflow.linalg.tensor_diag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=5,
        max_dim_size=10,
        min_value=1,
        max_value=10,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_tensor_diag(
    *,
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
        diagonal=x[0],
    )
