# global
import math
from hypothesis import strategies as st
from hypothesis import assume
import numpy as np
import pytest
import itertools
import sys

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, BackendHandler
import ivy


# --- Helpers --- #
# --------------- #


# batched_outer
@st.composite
def _batched_outer_data(draw):
    shape = draw(helpers.get_shape(min_num_dims=2, max_num_dims=3))
    tensors_num = draw(helpers.ints(min_value=1, max_value=5))
    dtype, tensors = draw(
        helpers.dtype_and_values(
            num_arrays=tensors_num,
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
        )
    )
    return dtype, tensors


@st.composite
def _generate_diag_args(draw):
    x_shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=2, min_dim_size=1, max_dim_size=5
        )
    )

    flat_x_shape = math.prod(x_shape)

    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=x_shape,
            min_value=-1e2,
            max_value=1e2,
        )
    )

    offset = draw(helpers.ints(min_value=-5, max_value=5))

    dtype = dtype_x[0]

    dtype_padding_value = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            max_dim_size=1,
            min_dim_size=1,
            min_num_dims=1,
            max_num_dims=1,
            min_value=-1e2,
            max_value=1e2,
        )
    )

    align = draw(
        st.sampled_from(["RIGHT_LEFT", "RIGHT_RIGHT", "LEFT_LEFT", "LEFT_RIGHT"])
    )

    if offset < 0:
        num_rows_is_negative = draw(st.booleans())
        if num_rows_is_negative:
            num_rows = -1
            num_cols = draw(
                st.one_of(
                    st.integers(min_value=-1, max_value=-1),
                    st.integers(min_value=flat_x_shape, max_value=50),
                )
            )
        else:
            num_rows_is_as_expected = draw(st.booleans())
            if num_rows_is_as_expected:
                num_rows = flat_x_shape + abs(offset)
                num_cols = draw(
                    st.one_of(
                        st.integers(min_value=-1, max_value=-1),
                        st.integers(min_value=flat_x_shape, max_value=50),
                    )
                )
            else:
                num_rows = draw(
                    st.integers(min_value=flat_x_shape + abs(offset) + 1, max_value=50)
                )
                num_cols = draw(st.sampled_from([-1, flat_x_shape]))
    if offset > 0:
        num_cols_is_negative = draw(st.booleans())
        if num_cols_is_negative:
            num_cols = -1
            num_rows = draw(
                st.one_of(
                    st.integers(min_value=-1, max_value=-1),
                    st.integers(min_value=flat_x_shape, max_value=50),
                )
            )
        else:
            num_cols_is_as_expected = draw(st.booleans())
            if num_cols_is_as_expected:
                num_cols = flat_x_shape + abs(offset)
                num_rows = draw(
                    st.one_of(
                        st.integers(min_value=-1, max_value=-1),
                        st.integers(min_value=flat_x_shape, max_value=50),
                    )
                )
            else:
                num_cols = draw(
                    st.integers(min_value=flat_x_shape + abs(offset) + 1, max_value=50)
                )
                num_rows = draw(st.sampled_from([-1, flat_x_shape]))

    if offset == 0:
        num_rows_is_negative = draw(st.booleans())
        num_cols_is_negative = draw(st.booleans())

        if num_rows_is_negative and num_cols_is_negative:
            num_rows = -1
            num_cols = -1

        if num_rows_is_negative:
            num_rows = -1
            num_cols = draw(
                st.integers(min_value=flat_x_shape + abs(offset), max_value=50)
            )

        if num_cols_is_negative:
            num_cols = -1
            num_rows = draw(
                st.integers(min_value=flat_x_shape + abs(offset), max_value=50)
            )

        else:
            num_rows_is_as_expected = draw(st.booleans())
            if num_rows_is_as_expected:
                num_rows = flat_x_shape
                num_cols = draw(
                    st.integers(min_value=flat_x_shape + abs(offset), max_value=50)
                )
            else:
                num_cols = flat_x_shape
                num_rows = draw(
                    st.integers(min_value=flat_x_shape + abs(offset), max_value=50)
                )

    return dtype_x, offset, dtype_padding_value, align, num_rows, num_cols


# dot
@st.composite
def _generate_dot_dtype_and_arrays(draw, min_num_dims=0):
    shape_a = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=5, min_num_dims=min_num_dims, max_num_dims=5
        )
    )
    shape_b = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=5, min_num_dims=min_num_dims, max_num_dims=5
        )
    )

    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if len(shape_a) == 1 and len(shape_b) == 1:
        shape_b[0] = shape_a[0]
    elif len(shape_a) == 2 and len(shape_b) == 2:
        shape_b[0] = shape_a[1]
    elif len(shape_a) >= 2 and len(shape_b) == 1:
        shape_b[0] = shape_a[-1]
    elif len(shape_a) >= 1 and len(shape_b) >= 2:
        shape_a[-1] = shape_b[-2]

    dtype_1, a = draw(
        helpers.dtype_and_values(
            shape=shape_a,
            available_dtypes=helpers.get_dtypes("float"),
            min_value=-10,
            max_value=10,
        )
    )
    dtype_2, b = draw(
        helpers.dtype_and_values(
            shape=shape_b,
            dtype=dtype_1,
            min_value=-10,
            max_value=10,
        )
    )

    return [dtype_1[0], dtype_2[0]], [a[0], b[0]]


@st.composite
def _generate_eigh_tridiagonal_args(draw):
    dtype, alpha = draw(
        helpers.dtype_and_values(
            min_dim_size=2,
            min_num_dims=1,
            max_num_dims=1,
            min_value=2.0,
            max_value=5,
            available_dtypes=helpers.get_dtypes("float"),
        )
    )
    beta_shape = len(alpha[0]) - 1
    dtype, beta = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype,
            shape=(beta_shape,),
            min_value=2.0,
            max_value=5,
        )
    )

    select = draw(st.sampled_from(("a", "i", "v")))
    if select == "a":
        select_range = None
    elif select == "i":
        range_slice = draw(
            st.slices(beta_shape).filter(
                lambda x: x.start
                and x.stop
                and x.step
                and x.start >= 0
                and x.stop >= 0
                and x.step >= 0
                and x.start < x.stop
            )
        )

        select_range = [range_slice.start, range_slice.stop]
    else:
        select_range = [-100, 100]

    eigvals_only = draw(st.booleans())
    tol = draw(st.floats(1e-5, 1e-3) | st.just(None))
    return dtype, alpha, beta, eigvals_only, select, select_range, tol


@st.composite
def _generate_general_inner_product_args(draw):
    dim = draw(st.integers(min_value=1, max_value=3))
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=(dim, dim),
            min_value=1,
            max_value=10.0,
            num_arrays=2,
            shared_dtype=True,
            allow_nan=False,
        )
    )
    max_value = dim - 1 if dim > 1 else dim
    n_modes = draw(st.integers(min_value=1, max_value=max_value) | st.just(None))
    return x_dtype, x, n_modes


# multi_dot
@st.composite
def _generate_multi_dot_dtype_and_arrays(draw):
    input_dtype = [draw(st.sampled_from(draw(helpers.get_dtypes("numeric"))))]
    matrices_dims = draw(
        st.lists(st.integers(min_value=2, max_value=10), min_size=4, max_size=4)
    )
    shape_1 = (matrices_dims[0], matrices_dims[1])
    shape_2 = (matrices_dims[1], matrices_dims[2])
    shape_3 = (matrices_dims[2], matrices_dims[3])

    matrix_1 = draw(
        helpers.dtype_and_values(
            shape=shape_1,
            dtype=input_dtype,
            min_value=-10,
            max_value=10,
        )
    )
    matrix_2 = draw(
        helpers.dtype_and_values(
            shape=shape_2,
            dtype=input_dtype,
            min_value=-10,
            max_value=10,
        )
    )
    matrix_3 = draw(
        helpers.dtype_and_values(
            shape=shape_3,
            dtype=input_dtype,
            min_value=-10,
            max_value=10,
        )
    )

    return input_dtype, [matrix_1[1][0], matrix_2[1][0], matrix_3[1][0]]


# solve_triangular
@st.composite
def _generate_solve_triangular_args(draw):
    shape = draw(
        st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=5)
    )
    shape_b = list(shape)
    shape_a = list(shape)
    shape_a[-1] = shape_a[-2]  # Make square

    dtype_a, a = draw(
        helpers.dtype_and_values(
            shape=shape_a,
            available_dtypes=helpers.get_dtypes("float"),
            min_value=-10,
            max_value=10,
        )
    )

    dtype_b, b = draw(
        helpers.dtype_and_values(
            shape=shape_b,
            available_dtypes=helpers.get_dtypes("float"),
            min_value=-10,
            max_value=10,
        )
    )

    dtype_a = dtype_a[0]
    dtype_b = dtype_b[0]
    a = a[0]
    b = b[0]
    upper = draw(st.booleans())
    adjoint = draw(st.booleans())
    unit_diagonal = draw(st.booleans())

    for i in range(shape_a[-2]):
        a[ivy.abs(a[..., i, i]) < 0.01, i, i] = 0.01  # Make diagonals non-zero

    return upper, adjoint, unit_diagonal, [dtype_a, dtype_b], [a, b]


@st.composite
def _get_dtype_value1_value2_cov(
    draw,
    available_dtypes,
    min_num_dims,
    max_num_dims,
    min_dim_size,
    max_dim_size,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )

    value1, value2 = values[0], values[1]

    # modifiers: rowVar, bias, ddof
    rowVar = draw(st.booleans())
    bias = draw(st.booleans())
    ddof = draw(helpers.ints(min_value=0, max_value=1))

    numVals = None
    if rowVar is False:
        numVals = -1 if numVals == 0 else 0
    else:
        numVals = 0 if len(shape) == 1 else -1

    fweights = draw(
        helpers.array_values(
            dtype="int64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
        )
    )

    aweights = draw(
        helpers.array_values(
            dtype="float64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
            small_abs_safety_factor=1,
        )
    )

    return [dtype], value1, value2, rowVar, bias, ddof, fweights, aweights


# higher_order_moment
@st.composite
def _higher_order_moment_data(draw):
    shape = draw(helpers.get_shape(min_num_dims=2, max_num_dims=4))
    order = draw(helpers.ints(min_value=0, max_value=5))
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
        )
    )
    return dtype, x[0], order


# initialize tucker
@st.composite
def _initialize_tucker_data(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=5,
            min_value=0.1,
            max_value=10.0,
            ret_shape=True,
        )
    )
    dims = len(shape)
    rank = []
    for i in range(dims):
        rank.append(draw(helpers.ints(min_value=1, max_value=shape[i])))
    n_modes = draw(helpers.ints(min_value=2, max_value=dims))
    modes = [*range(dims)][:n_modes]
    mask_dtype, mask = draw(
        helpers.dtype_and_values(
            dtype=["int32"],
            shape=shape,
            min_value=0,
            max_value=1,
        )
    )
    svd_mask_repeats = draw(helpers.ints(min_value=0, max_value=3))
    non_negative = draw(st.booleans())
    return (
        x_dtype + mask_dtype,
        x[0],
        rank,
        modes,
        non_negative,
        mask[0],
        svd_mask_repeats,
    )


@st.composite
def _khatri_rao_data(draw):
    num_matrices = draw(helpers.ints(min_value=2, max_value=4))
    m = draw(helpers.ints(min_value=1, max_value=5))
    input_dtypes, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=num_matrices,
            min_dim_size=m,
            max_dim_size=m,
            min_num_dims=2,
            max_num_dims=2,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
        )
    )
    skip_matrix = draw(helpers.ints(min_value=0, max_value=len(input) - 1))
    weight_dtype, weights = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"), shape=(m,)
        )
    )
    mask_dtype, mask = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=1,
            shape=(m,),
        )
    )
    return (
        input_dtypes + weight_dtype + mask_dtype,
        input,
        skip_matrix,
        weights[0],
        mask[0],
    )


@st.composite
def _kronecker_data(draw):
    num_arrays = draw(helpers.ints(min_value=2, max_value=5))
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=num_arrays,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
            shared_dtype=True,
            min_num_dims=2,
            max_num_dims=2,
        )
    )
    skip_matrix = draw(
        st.lists(st.integers(min_value=0, max_value=num_arrays - 1), unique=True)
    )
    reverse = draw(st.booleans())
    return x_dtype, x, skip_matrix, reverse


# truncated svd
@st.composite
def _make_svd_nn_data(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=2,
            min_dim_size=2,
            max_dim_size=5,
            min_value=1.0,
            max_value=10.0,
            ret_shape=True,
        )
    )
    n, m = shape
    _, U = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            available_dtypes=helpers.get_dtypes("float"),
            shape=(n, m),
            min_value=1.0,
            max_value=10.0,
        )
    )
    _, S = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(m,),
            min_value=1.0,
            max_value=10.0,
        )
    )
    _, V = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=(m, m),
            min_value=1.0,
            max_value=10.0,
        )
    )
    nntype = draw(st.sampled_from(["nndsvd", "nndsvda"]))
    return x_dtype, x[0], U[0], S[0], V[0], nntype


@st.composite
def _mode_dot_data(draw):
    shape_t1 = draw(helpers.get_shape(min_num_dims=2, max_num_dims=5))
    mode = draw(helpers.ints(min_value=0, max_value=len(shape_t1) - 1))
    mode_dimsize = shape_t1[mode]
    t1_dtype, t1 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape_t1,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
        )
    )
    t2_rows = draw(helpers.ints(min_value=1, max_value=4))
    shape_t2 = draw(st.sampled_from([(mode_dimsize,), (t2_rows, mode_dimsize)]))
    t2_dtype, t2 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape_t2,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
        )
    )
    return t1_dtype + t2_dtype, t1[0], t2[0], mode


@st.composite
def _multi_mode_dot_data(draw):
    t1_dtype, t1, shape_t1 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            ret_shape=True,
            min_num_dims=2,
            large_abs_safety_factor=20,
            small_abs_safety_factor=20,
            safety_factor_scale="log",
        )
    )
    modes = [*range(len(shape_t1))]
    skip = draw(st.lists(helpers.ints(min_value=0, max_value=len(shape_t1) - 1)))
    t2 = []
    t2_dtype = []
    for i in modes:
        mode_dimsize = shape_t1[i]
        rows = draw(helpers.ints(min_value=1, max_value=4))
        shape = draw(st.sampled_from([(mode_dimsize,), (rows, mode_dimsize)]))
        mat_or_vec_dtype, mat_or_vec = draw(
            helpers.dtype_and_values(
                dtype=t1_dtype,
                shape=shape,
                large_abs_safety_factor=20,
                small_abs_safety_factor=20,
                safety_factor_scale="log",
            )
        )
        t2.append(mat_or_vec[0])
        t2_dtype.append(mat_or_vec_dtype[0])

    return t1_dtype + t2_dtype, t1[0], t2, modes, skip


# partial tucker
@st.composite
def _partial_tucker_data(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=5,
            min_value=0.1,
            max_value=10.0,
            ret_shape=True,
        )
    )
    dims = len(shape)
    rank = []
    for i in range(dims):
        rank.append(draw(helpers.ints(min_value=1, max_value=shape[i])))
    n_modes = draw(helpers.ints(min_value=2, max_value=dims))
    modes = [*range(dims)][:n_modes]
    mask_dtype, mask = draw(
        helpers.dtype_and_values(
            dtype=["int32"],
            shape=shape,
            min_value=0,
            max_value=1,
        )
    )
    svd_mask_repeats = draw(helpers.ints(min_value=0, max_value=3))
    n_iter_max = draw(helpers.ints(min_value=1, max_value=7))
    tol = draw(helpers.floats(min_value=1e-5, max_value=1e-1))
    return (
        x_dtype + mask_dtype,
        x[0],
        rank,
        modes,
        n_iter_max,
        mask[0],
        svd_mask_repeats,
        tol,
    )


# tensor train
@st.composite
def _tensor_train_data(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_value=0.1,
            max_value=10,
            min_num_dims=2,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=5,
            ret_shape=True,
        ).filter(lambda x: "float16" not in x[0] and "bfloat16" not in x[0])
    )
    dims = len(shape)
    rank = [1]
    for i in range(dims - 1):
        rank.append(draw(helpers.ints(min_value=1, max_value=shape[i])))
    rank.append(1)

    return x_dtype, x[0], rank


# truncated svd
@st.composite
def _truncated_svd_data(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=2,
            min_dim_size=2,
            max_dim_size=5,
            min_value=0.1,
            max_value=10.0,
            ret_shape=True,
        )
    )
    uv = draw(st.booleans())
    n_eigen = draw(helpers.ints(min_value=1, max_value=max(shape[-2:])))
    return x_dtype, x[0], uv, n_eigen


@st.composite
def _tt_matrix_to_tensor_data(draw):
    rank = 1
    num_factors = draw(st.integers(min_value=1, max_value=3))
    factor_dims = draw(
        st.tuples(
            st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)
        )
    )
    shape = (num_factors, rank, *factor_dims, rank)
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=1,
            shape=shape,
            shared_dtype=True,
        )
    )
    return x_dtype, x


# tucker
@st.composite
def _tucker_data(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=2,
            max_num_dims=4,
            min_dim_size=2,
            max_dim_size=3,
            min_value=0.1,
            max_value=10.0,
            ret_shape=True,
        )
    )
    dims = len(shape)
    rank = []
    for i in range(dims):
        rank.append(draw(helpers.ints(min_value=1, max_value=shape[i])))
    mask_dtype, mask = draw(
        helpers.dtype_and_values(
            dtype=["int32"],
            shape=shape,
            min_value=0,
            max_value=1,
        )
    )
    svd_mask_repeats = draw(helpers.ints(min_value=0, max_value=1))
    n_iter_max = draw(helpers.ints(min_value=0, max_value=2))
    tol = draw(helpers.floats(min_value=1e-5, max_value=1e-1))
    init = draw(st.sampled_from(["svd", "random"]))
    fixed_factors = draw(st.booleans())
    if fixed_factors:
        _, core = draw(
            helpers.dtype_and_values(
                dtype=x_dtype,
                min_value=0.1,
                max_value=10.0,
                shape=rank,
            )
        )
        factors = []
        for i in range(dims):
            _, factor = draw(
                helpers.dtype_and_values(
                    dtype=x_dtype,
                    min_value=0.1,
                    max_value=10.0,
                    shape=(shape[i], rank[i]),
                )
            )
            factors.append(factor[0])
        fixed_factors = draw(
            st.lists(
                helpers.ints(min_value=0, max_value=dims - 1), unique=True, min_size=1
            )
        )
        rank = [rank[i] for i in range(dims) if i not in fixed_factors]
        init = ivy.TuckerTensor((core[0], factors))
    return (
        x_dtype + mask_dtype,
        x[0],
        rank,
        fixed_factors,
        init,
        n_iter_max,
        mask[0],
        svd_mask_repeats,
        tol,
    )


# --- Main --- #
# ------------ #


@handle_test(
    fn_tree="functional.ivy.experimental.adjoint",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=(
            ivy.float32,
            ivy.float64,
            ivy.complex64,
            ivy.complex128,
        ),
        min_num_dims=2,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
        min_value=-1.0e5,
        max_value=1.0e5,
        allow_nan=False,
        shared_dtype=True,
    ),
)
def test_adjoint(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        on_device=on_device,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.batched_outer",
    data=_batched_outer_data(),
)
def test_batched_outer(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, tensors = data
    if backend_fw == "paddle":
        # to avoid large dimension results since paddle don't support them
        tensors = tensors[:2]
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-1,
        rtol_=1e-1,
        input_dtypes=input_dtypes,
        tensors=tensors,
    )


# test adapted from tensorly
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/tests/test_outer_product.py#L22
@pytest.mark.skip(
    reason=(
        "ivy.tensordot does not support batched_modes argument for the moment. "
        "TODO please remove this when the functionality is added. "
        "see https://github.com/unifyai/ivy/issues/21914"
    )
)
def test_batched_outer_product():
    batch_size = 3
    X = ivy.random_uniform(shape=(batch_size, 4, 5, 6))
    Y = ivy.random_uniform(shape=(batch_size, 3))
    Z = ivy.random_uniform(shape=(batch_size, 2))
    res = ivy.batched_outer([X, Y, Z])
    true_res = ivy.tensordot(X, Y, (), batched_modes=0)
    true_res = ivy.tensordot(true_res, Z, (), batched_modes=0)
    np.testing.assert_array_almost_equal(res, true_res)


@handle_test(
    fn_tree="functional.ivy.experimental.cond",
    dtype_x=helpers.cond_data_gen_helper(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_cond(dtype_x, test_flags, backend_fw, on_device, fn_name):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-3,
        atol_=1e-3,
        x=x[0],
        p=x[1],
    )


# cov
@handle_test(
    fn_tree="functional.ivy.experimental.cov",
    dtype_x1_x2_cov=_get_dtype_value1_value2_cov(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_cov(*, dtype_x1_x2_cov, test_flags, backend_fw, fn_name, on_device):
    dtype, x1, x2, rowVar, bias, ddof, fweights, aweights = dtype_x1_x2_cov
    helpers.test_function(
        input_dtypes=[dtype[0], dtype[0], "int64", "float64"],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x1,
        x2=x2,
        rowVar=rowVar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
        return_flat_np_arrays=True,
        rtol_=1e-2,
        atol_=1e-2,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.diagflat",
    args_packet=_generate_diag_args(),
    test_gradients=st.just(False),
)
def test_diagflat(*, test_flags, backend_fw, fn_name, args_packet, on_device):
    dtype_x, offset, dtype_padding_value, align, num_rows, num_cols = args_packet

    x_dtype, x = dtype_x
    padding_value_dtype, padding_value = dtype_padding_value
    padding_value = padding_value[0][0]

    helpers.test_function(
        input_dtypes=x_dtype + ["int64"] + padding_value_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        offset=offset,
        padding_value=padding_value,
        align=align,
        num_rows=num_rows,
        num_cols=num_cols,
        on_device=on_device,
        atol_=1e-01,
        rtol_=1 / 64,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.dot",
    data=_generate_dot_dtype_and_arrays(),
)
def test_dot(*, data, test_flags, backend_fw, fn_name, on_device):
    (input_dtypes, x) = data
    return helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        xs_grad_idxs=[[0, 0]],
        input_dtypes=input_dtypes,
        test_values=True,
        rtol_=0.5,
        atol_=0.5,
        a=x[0],
        b=x[1],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.eig",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=(
            ivy.float32,
            ivy.float64,
            ivy.int32,
            ivy.int64,
            ivy.complex64,
            ivy.complex128,
        ),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=10,
        max_dim_size=10,
        min_value=1.0,
        max_value=1.0e5,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_eig(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        test_values=False,
        x=x[0],
    )


# eigh_tridiagonal
@handle_test(
    fn_tree="eigh_tridiagonal",
    args_packet=_generate_eigh_tridiagonal_args(),
    ground_truth_backend="numpy",
    test_gradients=st.just(False),
)
def test_eigh_tridiagonal(
    *,
    args_packet,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, alpha, beta, eigvals_only, select, select_range, tol = args_packet
    test_flags.with_out = False
    results = helpers.test_function(
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        input_dtypes=dtype,
        alpha=alpha[0],
        beta=beta[0],
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
        test_values=eigvals_only,
        return_flat_np_arrays=True,
    )
    if results is None:
        return
    ret_np_flat, ret_np_from_gt_flat = results
    reconstructed_np = None
    for i in range(len(ret_np_flat) // 2):
        eigenvalue = ret_np_flat[i]
        eigenvector = ret_np_flat[len(ret_np_flat) // 2 + i]
        if reconstructed_np is not None:
            reconstructed_np += eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
        else:
            reconstructed_np = eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )

    reconstructed_from_np = None
    for i in range(len(ret_np_from_gt_flat) // 2):
        eigenvalue = ret_np_from_gt_flat[i]
        eigenvector = ret_np_from_gt_flat[len(ret_np_flat) // 2 + i]
        if reconstructed_from_np is not None:
            reconstructed_from_np += eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
        else:
            reconstructed_from_np = eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
    # value test
    helpers.assert_all_close(
        reconstructed_np,
        reconstructed_from_np,
        rtol=1e-1,
        atol=1e-2,
        backend=backend_fw,
        ground_truth_backend=test_flags.ground_truth_backend,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.eigvals",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=(
            ivy.float32,
            ivy.float64,
            ivy.int32,
            ivy.int64,
            ivy.complex64,
            ivy.complex128,
        ),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=10,
        max_dim_size=10,
        min_value=1.0,
        max_value=1.0e5,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_eigvals(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        test_values=False,
        x=x[0],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.general_inner_product",
    data=_generate_general_inner_product_args(),
)
def test_general_inner_product(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, x, n_modes = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtypes,
        a=x[0],
        b=x[1],
        n_modes=n_modes,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.higher_order_moment",
    data=_higher_order_moment_data(),
)
def test_higher_order_moment(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, x, order = data
    if backend_fw == "paddle":
        # to avoid large dimension results since paddle don't support them
        order = min(order, 2)
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-1,
        rtol_=1e-1,
        input_dtypes=input_dtypes,
        x=x,
        order=order,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.initialize_tucker",
    data=_initialize_tucker_data(),
    test_with_out=st.just(False),
)
def test_initialize_tucker(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, x, rank, modes, non_negative, mask, svd_mask_repeats = data
    results = helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=input_dtypes,
        x=x,
        rank=rank,
        modes=modes,
        non_negative=non_negative,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    core = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
    factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
    core_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
    )
    factors_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
    )

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        n_elem = int(ivy_backend.prod(rank[: len(modes)])) * int(
            ivy_backend.prod(x.shape[len(modes) :])
        )
    for c, c_gt in zip(core, core_gt):
        assert np.prod(c.shape) == n_elem
        assert np.prod(c_gt.shape) == n_elem

    for f, f_gt in zip(factors, factors_gt):
        assert np.prod(f.shape) == np.prod(f_gt.shape)


@handle_test(
    fn_tree="functional.ivy.experimental.khatri_rao",
    data=_khatri_rao_data(),
    test_instance_method=st.just(False),
)
def test_khatri_rao(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, input, skip_matrix, weights, mask = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        test_values=False,
        input_dtypes=input_dtypes,
        x=input,
        weights=weights,
        skip_matrix=skip_matrix,
        mask=mask,
    )


# The following two tests have been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/tests/test_khatri_rao.py
@pytest.mark.parametrize(("columns", "rows"), [(4, [3, 4, 2])])
def test_khatri_rao_tensorly_1(columns, rows):
    columns = columns
    rows = rows
    matrices = [ivy.arange(k * columns).reshape((k, columns)) for k in rows]
    res = ivy.khatri_rao(matrices)
    # resulting matrix must be of shape (prod(n_rows), n_columns)
    n_rows = 3 * 4 * 2
    n_columns = 4
    assert res.shape[0] == n_rows
    assert res.shape[1] == n_columns


@pytest.mark.parametrize(
    ("t1", "t2", "true_res"),
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
            [
                [1.0, 8.0, 21.0],
                [2.0, 10.0, 24.0],
                [3.0, 12.0, 27.0],
                [4.0, 20.0, 42.0],
                [8.0, 25.0, 48.0],
                [12.0, 30.0, 54.0],
                [7.0, 32.0, 63.0],
                [14.0, 40.0, 72.0],
                [21.0, 48.0, 81.0],
            ],
        )
    ],
)
def test_khatri_rao_tensorly_2(t1, t2, true_res):
    t1 = ivy.array(t1)
    t2 = ivy.array(t2)
    true_res = ivy.array(true_res)
    res = ivy.khatri_rao([t1, t2])
    assert np.allclose(res, true_res)


@handle_test(
    fn_tree="functional.ivy.experimental.kron",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_gradients=st.just(False),
)
def test_kron(*, dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        a=x[0],
        b=x[1],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.kronecker",
    data=_kronecker_data(),
    test_instance_method=st.just(False),
)
def test_kronecker(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, input, skip_matrix, reverse = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtypes,
        x=input,
        skip_matrix=skip_matrix,
        reverse=reverse,
    )


# lu_factor
@handle_test(
    fn_tree="functional.ivy.experimental.lu_factor",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
    ).filter(
        lambda x: np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    test_gradients=st.just(False),
)
def test_lu_factor(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    ret = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        test_values=False,
    )
    # check decomp is correct manually by getting the values from test_function above
    # this is because the decomposition is not unique and test_values will not work
    ret_f, ret_gt = ret

    # check that the decomposition is correct for current fw at least
    LU, p = ret_f.LU, ret_f.p
    L = np.tril(LU, -1) + np.eye(LU.shape[0])
    U = np.triu(LU)
    P = np.eye(LU.shape[0])[p]
    assert np.allclose(L @ U, P @ x[0])


@handle_test(
    fn_tree="functional.ivy.experimental.lu_solve",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.get_shape(
            min_num_dims=2, max_num_dims=2, min_dim_size=2, max_dim_size=2
        ),
        num_arrays=2,
        shared_dtype=True,
    ).filter(
        lambda x: "float16" not in x[0]
        and "bfloat16" not in x[0]
        and np.linalg.cond(x[1][0]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1][0])) != 0
    ),
    test_gradients=st.just(False),
)
def test_lu_solve(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, arr = dtype_x
    A, B = arr[0], arr[1]
    ivy.set_backend(backend_fw)
    lu_ = ivy.lu_factor(A)
    lu, p = lu_.LU, lu_.p
    X, X_gt = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        lu=lu,
        p=p,
        b=B,
        test_values=False,
    )

    assert np.allclose(A @ X, B)


@handle_test(
    fn_tree="functional.ivy.experimental.make_svd_non_negative",
    data=_make_svd_nn_data(),
    test_with_out=st.just(False),
)
def test_make_svd_non_negative(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, U, S, V, nntype = data
    results = helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=input_dtype,
        x=x,
        U=U,
        S=S,
        V=V,
        nntype=nntype,
        test_values=False,
        return_flat_np_arrays=True,
    )
    if results is None:
        return

    # returned values should be non negative
    ret_flat_np, ret_from_gt_flat_np = results
    W_flat_np, H_flat_np = ret_flat_np[0], ret_flat_np[1]
    W_flat_np_gt, H_flat_np_gt = ret_from_gt_flat_np[0], ret_from_gt_flat_np[1]
    assert np.all(W_flat_np >= 0)
    assert np.all(H_flat_np >= 0)
    assert np.all(W_flat_np_gt >= 0)
    assert np.all(H_flat_np_gt >= 0)
    helpers.assert_all_close(
        W_flat_np,
        W_flat_np_gt,
        atol=1e-02,
        backend=backend_fw,
        ground_truth_backend=test_flags.ground_truth_backend,
    )
    helpers.assert_all_close(
        H_flat_np,
        H_flat_np_gt,
        atol=1e-02,
        backend=backend_fw,
        ground_truth_backend=test_flags.ground_truth_backend,
    )


# matrix_exp
@handle_test(
    fn_tree="functional.ivy.experimental.matrix_exp",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=2,
        min_value=-100,
        max_value=100,
        allow_nan=False,
        shared_dtype=True,
    ),
    test_gradients=st.just(False),
)
def test_matrix_exp(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.mode_dot",
    data=_mode_dot_data(),
)
def test_mode_dot(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, t1, t2, mode = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtypes,
        x=t1,
        matrix_or_vector=t2,
        mode=mode,
    )


@pytest.mark.parametrize(
    ("X", "U", "true_res"),
    [
        (
            [
                [[1, 13], [4, 16], [7, 19], [10, 22]],
                [[2, 14], [5, 17], [8, 20], [11, 23]],
                [[3, 15], [6, 18], [9, 21], [12, 24]],
            ],
            [[1, 3, 5], [2, 4, 6]],
            [
                [[22, 130], [49, 157], [76, 184], [103, 211]],
                [[28, 172], [64, 208], [100, 244], [136, 280]],
            ],
        )
    ],
)
def test_mode_dot_tensorly(X, U, true_res):
    X = ivy.array(X)
    U = ivy.array(U)
    true_res = ivy.array(true_res)
    res = ivy.mode_dot(X, U, 0)
    assert np.allclose(true_res, res, atol=1e-1, rtol=1e-1)


@handle_test(
    fn_tree="functional.ivy.experimental.multi_dot",
    dtype_x=_generate_multi_dot_dtype_and_arrays(),
    test_gradients=st.just(False),
)
def test_multi_dot(dtype_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        test_values=True,
        x=x,
        rtol_=1e-1,
        atol_=6e-1,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.multi_mode_dot",
    data=_multi_mode_dot_data(),
)
def test_multi_mode_dot(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, t1, t2, modes, skip = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtypes,
        x=t1,
        mat_or_vec_list=t2,
        modes=modes,
        skip=skip,
    )


# The following 2 tests have been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/tests/test_n_mode_product.py#L81
@pytest.mark.parametrize(
    ("X", "U", "true_res"),
    [
        ([[1, 2], [0, -1]], [[2, 1], [-1, 1]], [1]),
    ],
)
def test_multi_mode_dot_tensorly_1(X, U, true_res):
    X, U, true_res = ivy.array(X), ivy.array(U), ivy.array(true_res)
    res = ivy.multi_mode_dot(X, U, [0, 1])
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 5, 4, 2),
    ],
)
def test_multi_mode_dot_tensorly_2(shape):
    print(shape)
    X = ivy.ones(shape)
    vecs = [ivy.ones(s) for s in shape]
    res = ivy.multi_mode_dot(X, vecs)
    # result should be a scalar
    assert ivy.shape(res) == ()
    assert np.allclose(res, np.prod(shape))

    # Average pooling each mode
    # Order should not matter
    vecs = [vecs[i] / s for i, s in enumerate(shape)]
    for modes in itertools.permutations(range(len(shape))):
        res = ivy.multi_mode_dot(X, [vecs[i] for i in modes], modes=modes)
        assert ivy.shape(res) == ()
        assert np.allclose(res, 1)


@handle_test(
    fn_tree="functional.ivy.experimental.partial_tucker",
    data=_partial_tucker_data(),
    test_with_out=st.just(False),
)
def test_partial_tucker(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, x, rank, modes, n_iter_max, mask, svd_mask_repeats, tol = data
    results = helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=input_dtypes,
        x=x,
        rank=rank,
        modes=modes,
        n_iter_max=n_iter_max,
        tol=tol,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    core = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
    factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
    core_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
    )
    factors_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
    )

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        n_elem = int(ivy_backend.prod(rank[: len(modes)])) * int(
            ivy_backend.prod(x.shape[len(modes) :])
        )
    for c, c_gt in zip(core, core_gt):
        assert np.prod(c.shape) == n_elem
        assert np.prod(c_gt.shape) == n_elem

    for f, f_gt in zip(factors, factors_gt):
        assert np.prod(f.shape) == np.prod(f_gt.shape)


# test adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/tests/test_tucker.py#L24
@pytest.mark.parametrize(
    ("tol_norm_2", "tol_max_abs", "modes", "shape"),
    [
        (
            10e-3,
            10e-1,
            [1, 2],
            (3, 4, 3),
        )
    ],
)
def test_partial_tucker_tensorly(tol_norm_2, tol_max_abs, modes, shape):
    tensor = ivy.random_uniform(shape=shape)
    (core, factors) = ivy.partial_tucker(
        tensor, None, modes, n_iter_max=200, verbose=True
    )
    reconstructed_tensor = ivy.multi_mode_dot(core, factors, modes=modes)
    norm_rec = ivy.sqrt(ivy.sum(reconstructed_tensor**2))
    norm_tensor = ivy.sqrt(ivy.sum(tensor**2))
    assert (norm_rec - norm_tensor) / norm_rec < tol_norm_2

    # Test the max abs difference between the reconstruction and the tensor
    assert ivy.max(ivy.abs(norm_rec - norm_tensor)) < tol_max_abs

    # Test the shape of the core and factors
    ranks = [3, 1]
    (core, factors) = ivy.partial_tucker(
        tensor, ranks, modes, n_iter_max=100, verbose=True
    )
    for i, rank in enumerate(ranks):
        np.testing.assert_equal(
            factors[i].shape,
            (tensor.shape[i + 1], rank),
            err_msg=(
                f"factors[i].shape = {factors[i].shape}, expected"
                f" {(tensor.shape[i + 1], rank)}"
            ),
        )
    np.testing.assert_equal(
        core.shape,
        [tensor.shape[0]] + ranks,
        err_msg=f"core.shape = {core.shape}, expected {[tensor.shape[0]] + ranks}",
    )

    # Test random_state fixes the core and the factor matrices
    (core1, factors1) = ivy.partial_tucker(
        tensor,
        ranks,
        modes,
        seed=0,
        init="random",
    )
    (core2, factors2) = ivy.partial_tucker(
        tensor,
        ranks,
        modes,
        seed=0,
        init="random",
    )
    np.allclose(core1, core2)
    for factor1, factor2 in zip(factors1, factors2):
        np.allclose(factor1, factor2)


@handle_test(
    fn_tree="functional.ivy.experimental.solve_triangular",
    data=_generate_solve_triangular_args(),
    test_instance_method=st.just(False),
)
def test_solve_triangular(*, data, test_flags, backend_fw, fn_name, on_device):
    # Temporarily ignore gradients on paddlepaddle backend
    # See: https://github.com/unifyai/ivy/pull/25917
    assume(not (backend_fw == "paddle" and test_flags.test_gradients))
    upper, adjoint, unit_diagonal, input_dtypes, x = data
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-3,
        atol_=1e-3,
        input_dtypes=input_dtypes,
        x1=x[0],
        x2=x[1],
        upper=upper,
        adjoint=adjoint,
        unit_diagonal=unit_diagonal,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.svd_flip",
    uv=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_num_dims=2,
        max_num_dims=2,
    ),
    u_based_decision=st.booleans(),
    test_with_out=st.just(False),
)
def test_svd_flip(*, uv, u_based_decision, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, input = uv
    helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtypes,
        U=input[0],
        V=input[1],
        u_based_decision=u_based_decision,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.tensor_train",
    data=_tensor_train_data(),
    # TODO: add support for more modes
    svd=st.just("truncated_svd"),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_tensor_train(*, data, svd, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, rank = data
    results = helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=input_dtype,
        input_tensor=x,
        rank=rank,
        svd=svd,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    factors = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    factors_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
    )

    for f, f_gt in zip(factors, factors_gt):
        assert np.prod(f.shape) == np.prod(f_gt.shape)


# The following 3 tests have been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/tests/test_tt_decomposition.py
@pytest.mark.parametrize(
    ("shape", "rank"), [((3, 4, 5, 6, 2, 10), (1, 3, 3, 4, 2, 2, 1))]
)
def test_tensor_train_tensorly_1(shape, rank):
    tensor = ivy.random_uniform(shape=shape)
    tensor_shape = tensor.shape
    factors = ivy.tensor_train(tensor, rank)

    assert len(factors) == 6, "Number of factors should be 6, currently has " + str(
        len(factors)
    )

    r_prev_iteration = 1
    for k in range(6):
        (r_prev_k, n_k, r_k) = factors[k].shape
        assert tensor_shape[k] == n_k, (
            "Mode 1 of factor "
            + str(k)
            + "needs "
            + str(tensor_shape[k])
            + " dimensions, currently has "
            + str(n_k)
        )
        assert r_prev_k == r_prev_iteration, " Incorrect ranks of factors "
        r_prev_iteration = r_k


@pytest.mark.parametrize(
    ("shape", "rank"), [((3, 4, 5, 6, 2, 10), (1, 5, 4, 3, 8, 10, 1))]
)
def test_tensor_train_tensorly_2(shape, rank):
    tensor = ivy.random_uniform(shape=shape)
    factors = ivy.tensor_train(tensor, rank)

    for k in range(6):
        (r_prev, n_k, r_k) = factors[k].shape

        first_error_message = (
            "TT rank " + str(k) + " is greater than the maximum allowed "
        )
        first_error_message += str(r_prev) + " > " + str(rank[k])
        assert r_prev <= rank[k], first_error_message

        first_error_message = (
            "TT rank " + str(k + 1) + " is greater than the maximum allowed "
        )
        first_error_message += str(r_k) + " > " + str(rank[k + 1])
        assert r_k <= rank[k + 1], first_error_message


@pytest.mark.parametrize(("shape", "rank", "tol"), [((3, 3, 3), (1, 3, 3, 1), (10e-5))])
def test_tensor_train_tensorly_3(shape, rank, tol):
    tensor = ivy.random_uniform(shape=shape)
    factors = ivy.tensor_train(tensor, rank)
    reconstructed_tensor = ivy.TTTensor.tt_to_tensor(factors)
    error = ivy.vector_norm(ivy.matrix_norm(tensor - reconstructed_tensor, ord=2))
    error /= ivy.vector_norm(ivy.matrix_norm(tensor, ord=2))
    np.testing.assert_(error < tol, "norm 2 of reconstruction higher than tol")


@handle_test(
    fn_tree="functional.ivy.experimental.truncated_svd",
    data=_truncated_svd_data(),
    test_with_out=st.just(False),
)
def test_truncated_svd(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, uv, n_eigenvecs = data
    results = helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=input_dtype,
        x=x,
        compute_uv=uv,
        n_eigenvecs=n_eigenvecs,
        test_values=False,
        return_flat_np_arrays=True,
    )

    if results is None:
        return

    # value test based on recreating the original matrix and testing the consistency
    ret_flat_np, ret_from_gt_flat_np = results

    if uv:
        for i in range(len(ret_flat_np) // 3):
            U = ret_flat_np[i]
            S = ret_flat_np[len(ret_flat_np) // 3 + i]
            Vh = ret_flat_np[2 * len(ret_flat_np) // 3 + i]
        m = U.shape[-1]
        n = Vh.shape[-1]
        S = np.expand_dims(S, -2) if m > n else np.expand_dims(S, -1)

        for i in range(len(ret_from_gt_flat_np) // 3):
            U_gt = ret_from_gt_flat_np[i]
            S_gt = ret_from_gt_flat_np[len(ret_from_gt_flat_np) // 3 + i]
            Vh_gt = ret_from_gt_flat_np[2 * len(ret_from_gt_flat_np) // 3 + i]
        S_gt = np.expand_dims(S_gt, -2) if m > n else np.expand_dims(S_gt, -1)

        with BackendHandler.update_backend("numpy") as ivy_backend:
            S_mat = (
                S
                * ivy_backend.eye(
                    U.shape[-1], Vh.shape[-2], batch_shape=U.shape[:-2]
                ).data
            )
            S_mat_gt = (
                S_gt
                * ivy_backend.eye(
                    U_gt.shape[-1], Vh_gt.shape[-2], batch_shape=U_gt.shape[:-2]
                ).data
            )
        reconstructed = np.matmul(np.matmul(U, S_mat), Vh)
        reconstructed_gt = np.matmul(np.matmul(U_gt, S_mat_gt), Vh_gt)

        # value test
        helpers.assert_all_close(
            reconstructed,
            reconstructed_gt,
            atol=1e-04,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
        )
    else:
        S = ret_flat_np
        S_gt = ret_from_gt_flat_np
        helpers.assert_all_close(
            S[0],
            S_gt[0],
            atol=1e-04,
            backend=backend_fw,
            ground_truth_backend=test_flags.ground_truth_backend,
        )


@handle_test(
    fn_tree="functional.ivy.experimental.tt_matrix_to_tensor",
    data=_tt_matrix_to_tensor_data(),
    test_gradients=st.just(False),
)
def test_tt_matrix_to_tensor(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = data
    helpers.test_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e8,
        atol_=1e8,
        tt_matrix=x[0],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.tucker",
    data=_tucker_data(),
    test_with_out=st.just(False),
)
def test_tucker(*, data, test_flags, backend_fw, fn_name, on_device):
    (
        input_dtypes,
        x,
        rank,
        fixed_factors,
        init,
        n_iter_max,
        mask,
        svd_mask_repeats,
        tol,
    ) = data
    results = helpers.test_function(
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        input_dtypes=input_dtypes,
        x=x,
        rank=rank,
        fixed_factors=fixed_factors,
        n_iter_max=n_iter_max,
        init=init,
        tol=tol,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    core = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
    factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
    core_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
    )
    factors_gt = helpers.flatten_and_to_np(
        ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
    )

    n_elem = 1
    if isinstance(init, ivy.TuckerTensor):
        for index in fixed_factors:
            n_elem *= init[0].shape[index]
    n_elem *= np.prod(rank)

    for c, c_gt in zip(core, core_gt):
        assert np.prod(c.shape) == n_elem
        assert np.prod(c_gt.shape) == n_elem

    for f, f_gt in zip(factors, factors_gt):
        assert np.prod(f.shape) == np.prod(f_gt.shape)


# test adapted from tensorly
# https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/tests/test_tucker.py#L71
@pytest.mark.parametrize(
    ("tol_norm_2", "tol_max_abs", "shape", "ranks"),
    [(10e-3, 10e-1, (3, 4, 3), [2, 3, 1])],
)
def test_tucker_tensorly(tol_norm_2, tol_max_abs, shape, ranks):
    tensor = ivy.random_uniform(shape=shape)
    tucker = ivy.tucker(tensor, None, n_iter_max=200, verbose=True)
    reconstructed_tensor = tucker.to_tensor()
    norm_rec = ivy.sqrt(ivy.sum(reconstructed_tensor**2))
    norm_tensor = ivy.sqrt(ivy.sum(tensor**2))
    assert (norm_rec - norm_tensor) / norm_rec < tol_norm_2

    # Test the max abs difference between the reconstruction and the tensor
    assert ivy.max(ivy.abs(reconstructed_tensor - tensor)) < tol_max_abs

    # Test the shape of the core and factors
    core, factors = ivy.tucker(tensor, ranks, n_iter_max=100)
    for i, rank in enumerate(ranks):
        np.testing.assert_equal(
            factors[i].shape,
            (tensor.shape[i], ranks[i]),
            err_msg=(
                f"factors[i].shape = {factors[i].shape}, expected"
                f" {(tensor.shape[i], ranks[i])}"
            ),
        )
        np.testing.assert_equal(
            core.shape[i],
            rank,
            err_msg=f"core.shape[i] = {core.shape[i]}, expected {rank}",
        )

    # try fixing the core
    factors_init = [ivy.copy_array(f) for f in factors]
    _, factors = ivy.tucker(
        tensor,
        ranks,
        init=(core, factors),
        fixed_factors=[1],
        n_iter_max=100,
        verbose=1,
    )
    assert np.allclose(factors[1], factors_init[1])

    # Random and SVD init should converge to a similar solution
    rank = shape
    tucker_svd = ivy.tucker(tensor, rank, n_iter_max=200, init="svd")
    tucker_random = ivy.tucker(tensor, rank, n_iter_max=200, init="random", seed=1234)
    rec_svd = tucker_svd.to_tensor()
    rec_random = tucker_random.to_tensor()
    error = ivy.sqrt(ivy.sum((rec_svd - rec_random) ** 2))
    error /= ivy.sqrt(ivy.sum(rec_svd**2))

    tol_norm_2 = 1e-1
    np.testing.assert_(
        error < tol_norm_2, "norm 2 of difference between svd and random init too high"
    )
    np.testing.assert_(
        ivy.max(ivy.abs(rec_svd - rec_random)) < tol_max_abs,
        "abs norm of difference between svd and random init too high",
    )
