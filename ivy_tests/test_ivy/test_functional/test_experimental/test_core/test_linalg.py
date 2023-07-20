# global
import math
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
import ivy


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
        fw=backend_fw,
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
        reconstructed_np, reconstructed_from_np, rtol=1e-1, atol=1e-2
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
        fw=backend_fw,
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
        fw=backend_fw,
        fn_name=fn_name,
        a=x[0],
        b=x[1],
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
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
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
def test_eig(dtype_x, test_flags, backend_fw, fn_name):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        test_values=False,
        x=x[0],
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
def test_eigvals(dtype_x, test_flags, backend_fw, fn_name):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        test_values=False,
        x=x[0],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.adjoint",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=(
            ivy.float16,
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
def test_adjoint(dtype_x, test_flags, backend_fw, fn_name):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


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


@handle_test(
    fn_tree="functional.ivy.experimental.multi_dot",
    dtype_x=_generate_multi_dot_dtype_and_arrays(),
    test_gradients=st.just(False),
)
def test_multi_dot(dtype_x, test_flags, backend_fw, fn_name):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        test_values=True,
        x=x,
        rtol_=1e-1,
        atol_=6e-1,
    )


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
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-3,
        atol_=1e-3,
        x=x[0],
        p=x[1],
    )


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
        fw=backend_fw,
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


@st.composite
def _khatri_rao_data(draw):
    num_matrices = draw(helpers.ints(min_value=2, max_value=4))
    m = draw(helpers.ints(min_value=1, max_value=5))
    input_dtypes, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
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
    _, weights = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"), shape=(m,)
        )
    )
    _, mask = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=1,
            shape=(m,),
        )
    )
    return input_dtypes, input, skip_matrix, weights[0], mask[0]


# TODO fix instance method
# TODO fix out argument
@handle_test(
    fn_tree="functional.ivy.experimental.khatri_rao",
    data=_khatri_rao_data(),
)
def test_khatri_rao(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, input, skip_matrix, weights, mask = data
    test_flags.instance_method = False
    test_flags.with_out = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtypes,
        input=input,
        weights=weights,
        skip_matrix=skip_matrix,
        mask=mask,
    )
