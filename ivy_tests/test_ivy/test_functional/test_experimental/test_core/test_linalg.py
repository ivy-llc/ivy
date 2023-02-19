# global
import math
from hypothesis import strategies as st

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
            min_value=1.0,
            max_value=1.0e5,
            available_dtypes=(
                ivy.float32,
                ivy.float64,
                ivy.complex64,
                ivy.complex128,
            ),
        )
    )
    beta_shape = len(alpha[0]) - 1
    dtype, beta = draw(
        helpers.dtype_and_values(
            available_dtypes=st.just(dtype),
            shape=(beta_shape,),
            min_value=1.0,
            max_value=1.0e5,
        )
    )

    select = draw(st.sampled_from(("a", "i", "v")))
    if select == "a":
        select_range = None
    elif select == "i":
        range_slice = draw(
            st.slices(beta_shape + 1).filter(
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
        range_slice = draw(
            st.slices(beta_shape + 1).filter(
                lambda x: x.start
                and x.stop
                and x.step
                and x.step >= 0
                and x.start < x.stop
            )
        )

        select_range = [range_slice.start, range_slice.stop]

    eigvals_only = draw(st.booleans())
    tol = draw(st.floats() | st.just(None))
    return dtype, alpha[0], beta[0], eigvals_only, select, select_range, tol


# eigh_tridiagonal
@handle_test(
    fn_tree="eigh_tridiagonal",
    args_packet=_generate_eigh_tridiagonal_args(),
    test_gradients=st.just(False),
)
def test_eigh_tridiagonal(
    *,
    args_packet,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, alpha, beta, eigvals_only, select, select_range, tol = args_packet
    test_flags.with_out = False
    helpers.test_function(
        ground_truth_backend="numpy",
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        input_dtypes=dtype,
        alpha=alpha,
        beta=beta,
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.diagflat",
    args_packet=_generate_diag_args(),
    test_gradients=st.just(False),
)
def test_diagflat(
    *,
    test_flags,
    backend_fw,
    fn_name,
    args_packet,
    ground_truth_backend,
):
    dtype_x, offset, dtype_padding_value, align, num_rows, num_cols = args_packet

    x_dtype, x = dtype_x
    padding_value_dtype, padding_value = dtype_padding_value
    padding_value = padding_value[0][0]

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
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
        atol_=1e-01,
        rtol_=1 / 64,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.kron",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_gradients=st.just(False),
)
def test_kron(
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        a=x[0],
        b=x[1],
    )


# matrix_exp
@handle_test(
    fn_tree="functional.ivy.experimental.matrix_exp",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=(ivy.double, ivy.complex64, ivy.complex128),
        min_num_dims=2,
        max_num_dims=10,
        min_dim_size=2,
        max_dim_size=50,
        min_value=-100,
        max_value=100,
        allow_nan=False,
        shared_dtype=True,
    ),
    test_gradients=st.just(False),
)
def test_matrix_exp(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
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
def test_eig(
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
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
def test_eigvals(
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
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
def test_adjoint(
    dtype_x,
    test_flags,
    backend_fw,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )
