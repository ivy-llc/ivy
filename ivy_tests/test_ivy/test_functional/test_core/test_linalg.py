"""Collection of tests for unified linear algebra functions."""

# global
import sys
import numpy as np
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, BackendHandler
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import (
    matrix_is_stable,
)


@st.composite
def dtype_value1_value2_axis(
    draw,
    available_dtypes,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
    specific_dim_size=3,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    # For cross product, a dim with size 3 is required
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    axis = draw(helpers.ints(min_value=0, max_value=len(shape)))
    # make sure there is a dim with specific dim size
    shape = list(shape)
    shape = shape[:axis] + [specific_dim_size] + shape[axis:]
    shape = tuple(shape)

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
    return [dtype], value1, value2, axis


@st.composite
def _get_dtype_value1_value2_axis_for_tensordot(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
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
    axis = draw(helpers.ints(min_value=1, max_value=len(shape)))
    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=72,
                    small_abs_safety_factor=72,
                    safety_factor_scale="log",
                )
            )
        )

    value1, value2 = values[0], values[1]
    if not isinstance(axis, list):
        value2 = value2.transpose(
            [k for k in range(len(shape) - axis, len(shape))]
            + [k for k in range(0, len(shape) - axis)]
        )
    return [dtype], value1, value2, axis


@st.composite
def _get_dtype_and_matrix(draw, *, symmetric=False):
    # batch_shape, shared, random_size
    input_dtype = draw(st.shared(st.sampled_from(draw(helpers.get_dtypes("float")))))
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=3))
    if symmetric:
        num_independnt_vals = int((random_size**2) / 2 + random_size / 2)
        array_vals_flat = np.array(
            draw(
                helpers.array_values(
                    dtype=input_dtype,
                    shape=tuple(list(batch_shape) + [num_independnt_vals]),
                    min_value=2,
                    max_value=5,
                )
            )
        )
        array_vals = np.zeros(batch_shape + (random_size, random_size))
        c = 0
        for i in range(random_size):
            for j in range(random_size):
                if j < i:
                    continue
                array_vals[..., i, j] = array_vals_flat[..., c]
                array_vals[..., j, i] = array_vals_flat[..., c]
                c += 1
        return [input_dtype], array_vals
    return [input_dtype], draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [random_size, random_size]),
            min_value=2,
            max_value=5,
        )
    )


@st.composite
def _get_first_matrix_and_dtype(draw, *, transpose=False, conjugate=False):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("numeric"))),
            key="shared_dtype",
        ).filter(lambda x: "float16" not in x)
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([random_size, shared_size]),
            min_value=2,
            max_value=5,
        )
    )
    if conjugate:
        conjugate = draw(st.booleans())
        return [input_dtype], matrix, conjugate
    if transpose:
        transpose = draw(st.booleans())
        adjoint = draw(st.booleans())
        if adjoint and transpose:
            adjoint = draw(st.just("False"))
        if transpose and not adjoint:
            matrix = np.transpose(matrix)
        if adjoint and not transpose:
            matrix = np.transpose(np.conjugate(matrix))
        return [input_dtype], matrix, transpose, adjoint
    return [input_dtype], matrix


@st.composite
def _get_second_matrix_and_dtype(draw, *, transpose=False):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("numeric"))),
            key="shared_dtype",
        ).filter(lambda x: "float16" not in x)
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    matrix = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([random_size, shared_size]),
            min_value=2,
            max_value=5,
        )
    )
    if transpose:
        transpose = draw(st.booleans())
        adjoint = draw(st.booleans())
        if adjoint and transpose:
            adjoint = draw(st.just("False"))
        if transpose and not adjoint:
            matrix = np.transpose(matrix)
        if adjoint and not transpose:
            matrix = np.transpose(np.conjugate(matrix))
        return [input_dtype], matrix, transpose, adjoint
    return [input_dtype], matrix


# vector_to_skew_symmetric_matrix
@st.composite
def _get_dtype_and_vector(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("numeric"))),
            key="shared_dtype",
        )
    )
    batch_shape = draw(helpers.get_shape(min_num_dims=2, max_num_dims=4))
    return [input_dtype], draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [3]),
            min_value=2,
            max_value=5,
        )
    )


@handle_test(
    fn_tree="functional.ivy.vector_to_skew_symmetric_matrix",
    dtype_x=_get_dtype_and_vector(),
)
def test_vector_to_skew_symmetric_matrix(
    *, dtype_x, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        vector=x,
    )


# matrix_power
@handle_test(
    fn_tree="functional.ivy.matrix_power",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-3,
        max_value=20,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])),
    ),
    n=helpers.ints(min_value=-6, max_value=6),
)
def test_matrix_power(*, dtype_x, n, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    assume(matrix_is_stable(x[0]))
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        n=n,
    )


# matmul
@handle_test(
    fn_tree="functional.ivy.matmul",
    x=_get_first_matrix_and_dtype(transpose=True),
    y=_get_second_matrix_and_dtype(transpose=True),
)
def test_matmul(*, x, y, test_flags, backend_fw, fn_name, on_device):
    input_dtype1, x_1, transpose_a, adjoint_a = x
    input_dtype2, y_1, transpose_b, adjoint_b = y
    helpers.test_function(
        input_dtypes=input_dtype1 + input_dtype2,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x1=x_1,
        x2=y_1,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b,
    )


@st.composite
def _det_helper(draw):
    square = draw(helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])))
    shape_prefix = draw(helpers.get_shape())
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_value=2,
            max_value=5,
            shape=shape_prefix + square,
        )
    )
    return dtype_x


# det
@handle_test(
    fn_tree="functional.ivy.det",
    dtype_x=_det_helper(),
)
def test_det(*, dtype_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_x
    assume(matrix_is_stable(x[0]))
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
    )


# eigh
@handle_test(
    fn_tree="functional.ivy.eigh",
    dtype_x=_get_dtype_and_matrix(symmetric=True),
    UPLO=st.sampled_from(("L", "U")),
    test_gradients=st.just(False),
)
def test_eigh(*, dtype_x, UPLO, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_x
    results = helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        UPLO=UPLO,
        test_values=False,
        return_flat_np_arrays=True,
    )
    if results is None:
        return
    ret_np_flat, ret_from_np_flat = results
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
    for i in range(len(ret_from_np_flat) // 2):
        eigenvalue = ret_from_np_flat[i]
        eigenvector = ret_from_np_flat[len(ret_np_flat) // 2 + i]
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
    )


# eigvalsh
@handle_test(
    fn_tree="functional.ivy.eigvalsh",
    dtype_x=_get_dtype_and_matrix(symmetric=True),
    UPLO=st.sampled_from(("L", "U")),
    test_gradients=st.just(False),
)
def test_eigvalsh(*, dtype_x, UPLO, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-3,
        test_values=False,
        x=x,
        UPLO=UPLO,
    )


# inner
@handle_test(
    fn_tree="functional.ivy.inner",
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_inner(*, dtype_xy, test_flags, backend_fw, fn_name, on_device):
    types, arrays = dtype_xy
    helpers.test_function(
        input_dtypes=types,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-2,
        x1=arrays[0],
        x2=arrays[1],
    )


# inv
@handle_test(
    fn_tree="functional.ivy.inv",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=24,
        large_abs_safety_factor=24,
        safety_factor_scale="log",
        shape=helpers.ints(min_value=2, max_value=20).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1][0].tolist()) < 1 / sys.float_info.epsilon),
    adjoint=st.booleans(),
)
def test_inv(*, dtype_x, adjoint, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        adjoint=adjoint,
    )


# matrix_transpose
@handle_test(
    fn_tree="functional.ivy.matrix_transpose",
    dtype_x=_get_first_matrix_and_dtype(conjugate=True),
)
def test_matrix_transpose(*, dtype_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x, conjugate = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        conjugate=conjugate,
    )


# outer
@handle_test(
    fn_tree="functional.ivy.outer",
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=1,
        max_value=50,
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_outer(*, dtype_xy, test_flags, backend_fw, fn_name, on_device):
    types, arrays = dtype_xy
    helpers.test_function(
        input_dtypes=types,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=arrays[0],
        x2=arrays[1],
    )


# slogdet
# TODO: add with_out testing when testing with tuples is supported
# execute with grads error
@handle_test(
    fn_tree="functional.ivy.slogdet",
    dtype_x=_det_helper(),
    test_with_out=st.just(False),
)
def test_slogdet(*, dtype_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_x
    assume(matrix_is_stable(x[0]))
    ret_grad_idxs = (
        [[1, "a"], [1, "b", "c"], [1, "b", "d"]] if test_flags.container[0] else [[1]]
    )
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        rtol_=1e-1,
        atol_=1e-2,
        fn_name=fn_name,
        on_device=on_device,
        ret_grad_idxs=ret_grad_idxs,
        x=x[0],
    )


@handle_test(
    fn_tree="functional.ivy.solve",
    x=helpers.get_first_solve_matrix(adjoint=True),
    y=helpers.get_second_solve_matrix(),
)
def test_solve(*, x, y, test_flags, backend_fw, fn_name, on_device):
    input_dtype1, x1, adjoint = x
    input_dtype2, x2 = y
    helpers.test_function(
        input_dtypes=[input_dtype1, input_dtype2],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x1=x1,
        x2=x2,
        adjoint=adjoint,
    )


# svdvals
@handle_test(
    fn_tree="functional.ivy.svdvals",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        min_num_dims=2,
    ),
    test_gradients=st.just(False),
)
def test_svdvals(*, dtype_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# tensordot
@handle_test(
    fn_tree="functional.ivy.tensordot",
    dtype_x1_x2_axis=_get_dtype_value1_value2_axis_for_tensordot(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_tensordot(*, dtype_x1_x2_axis, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x1,
        x2,
        axis,
    ) = dtype_x1_x2_axis

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=0.8,
        atol_=0.8,
        x1=x1,
        x2=x2,
        axes=axis,
    )


# trace
@handle_test(
    fn_tree="functional.ivy.trace",
    dtype_x_axes=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        min_axes_size=2,
        max_axes_size=2,
        min_num_dims=2,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    # TODO: test for more offsets
    offset=st.integers(min_value=-3, max_value=3),
)
def test_trace(*, dtype_x_axes, offset, test_flags, backend_fw, fn_name, on_device):
    dtype, x, axes = dtype_x_axes
    axis1, axis2 = axes
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


# vecdot
@handle_test(
    fn_tree="functional.ivy.vecdot",
    dtype_x1_x2_axis=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=100,
        small_abs_safety_factor=100,
        safety_factor_scale="log",
        min_num_dims=1,
        max_num_dims=4,
        min_dim_size=1,
        max_dim_size=4,
    ),
)
def test_vecdot(*, dtype_x1_x2_axis, test_flags, backend_fw, fn_name, on_device):
    dtype, x1, x2, axis = dtype_x1_x2_axis
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=5e-1,
        atol_=5e-1,
        x1=x1,
        x2=x2,
        axis=axis,
    )


# vector_norm
@handle_test(
    fn_tree="functional.ivy.vector_norm",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        min_value=-1e04,
        max_value=1e04,
        abs_smallest_val=1e-04,
        max_axes_size=2,
        allow_neg_axes=True,
    ),
    kd=st.booleans(),
    ord=st.one_of(
        helpers.ints(min_value=-5, max_value=5),
        helpers.floats(min_value=-5, max_value=5.0),
        st.sampled_from((float("inf"), -float("inf"))),
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
)
def test_vector_norm(
    *, dtype_values_axis, kd, ord, dtype, test_flags, backend_fw, fn_name, on_device
):
    x_dtype, x, axis = dtype_values_axis
    # to avoid tuple axis with only one axis as force_int_axis can't generate
    # axis with two axes
    if isinstance(axis, tuple) and len(axis) == 1:
        axis = axis[0]
    helpers.test_function(
        input_dtypes=x_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        axis=axis,
        keepdims=kd,
        ord=ord,
        dtype=dtype[0],
        atol_=1e-08,
    )

    # Specific value test to handle cases when ord is one of {inf, -inf}

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        arr = ivy_backend.array([[1.0, 2.0, 3.0], [-1.0, 2.0, 4.0]])
        arr_normed_inf = ivy_backend.vector_norm(arr, axis=0, ord=float("inf"))
        arr_normed_min_inf = ivy_backend.vector_norm(arr, axis=0, ord=float("-inf"))

    with BackendHandler.update_backend(test_flags.ground_truth_backend) as gt_backend:
        gt_arr_normed_inf = gt_backend.array([1.0, 2.0, 4.0])
        gt_arr_normed_min_inf = gt_backend.array([1.0, 2.0, 3.0])

    helpers.assert_all_close(
        arr_normed_inf,
        gt_arr_normed_inf,
        backend=backend_fw,
        ground_truth_backend=test_flags.ground_truth_backend,
    )
    helpers.assert_all_close(
        arr_normed_min_inf,
        gt_arr_normed_min_inf,
        backend=backend_fw,
        ground_truth_backend=test_flags.ground_truth_backend,
    )


# pinv
@handle_test(
    fn_tree="functional.ivy.pinv",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        large_abs_safety_factor=32,
        small_abs_safety_factor=32,
        safety_factor_scale="log",
    ),
    rtol=st.floats(1e-5, 1e-3),
)
def test_pinv(*, dtype_x, rtol, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        rtol=rtol,
    )


# qr
@handle_test(
    fn_tree="functional.ivy.qr",
    dtype_x=_get_dtype_and_matrix(),
    mode=st.sampled_from(("reduced", "complete")),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_qr(*, dtype_x, mode, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    results = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        mode=mode,
        test_values=False,
        return_flat_np_arrays=True,
    )
    if results is None:
        return

    ret_np_flat, ret_from_np_flat = results
    for i in range(len(ret_np_flat) // 2):
        q_np_flat = ret_np_flat[i]
        r_np_flat = ret_np_flat[len(ret_np_flat) // 2 + i]
    reconstructed_np_flat = np.matmul(q_np_flat, r_np_flat)
    for i in range(len(ret_from_np_flat) // 2):
        q_from_np_flat = ret_from_np_flat[i]
        r_from_np_flat = ret_from_np_flat[len(ret_np_flat) // 2 + i]
    reconstructed_from_np_flat = np.matmul(q_from_np_flat, r_from_np_flat)

    # value test
    helpers.assert_all_close(
        reconstructed_np_flat,
        reconstructed_from_np_flat,
        rtol=1e-1,
        atol=1e-1,
        backend=backend_fw,
        ground_truth_backend=test_flags.ground_truth_backend,
    )


# svd
@handle_test(
    fn_tree="functional.ivy.svd",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=0.1,
        max_value=10.0,
    ),
    fm=st.booleans(),
    uv=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_svd(*, dtype_x, uv, fm, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x

    results = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        compute_uv=uv,
        full_matrices=fm,
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
        helpers.assert_all_close(
            reconstructed,
            x[0],
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


# matrix_norm
@handle_test(
    fn_tree="functional.ivy.matrix_norm",
    # ground_truth_backend="numpy",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        valid_axis=True,
        min_axes_size=2,
        max_axes_size=2,
        force_tuple_axis=True,
        allow_neg_axes=False,
    ),
    kd=st.booleans(),
    ord=st.sampled_from((-2, -1, 1, 2, -float("inf"), float("inf"), "fro", "nuc")),
)
def test_matrix_norm(
    *, dtype_value_axis, kd, ord, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, axis = dtype_value_axis
    assume(matrix_is_stable(x[0], cond_limit=10))
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-2,
        x=x[0],
        axis=axis,
        keepdims=kd,
        ord=ord,
    )


@st.composite
def _matrix_rank_helper(draw):
    _batch_shape = draw(
        helpers.get_shape(min_num_dims=1, max_num_dims=3, min_dim_size=1)
    )
    _batch_dim = draw(st.sampled_from([(), _batch_shape]))
    _matrix_dim = draw(helpers.ints(min_value=2, max_value=20))
    shape = _batch_dim + (_matrix_dim, _matrix_dim)
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
            min_value=-1e05,
            max_value=1e05,
            abs_smallest_val=1e-05,
            safety_factor_scale="log",
        )
    )
    if np.all(np.swapaxes(x[0], -1, -2) == x[0]):
        hermitian = True
    else:
        hermitian = False

    tol_strategy = st.one_of(
        st.none(),
        st.floats(allow_nan=False, allow_infinity=False),
        helpers.array_values(
            dtype=helpers.get_dtypes("float", prune_function=False),
            shape=_batch_shape,
            min_value=-1e05,
            max_value=1e05,
            abs_smallest_val=1e-05,
            safety_factor_scale="log",
        ),
    )
    atol = draw(tol_strategy)
    rtol = draw(tol_strategy)
    return dtype, x[0], hermitian, atol, rtol


# matrix_rank
@handle_test(
    fn_tree="functional.ivy.matrix_rank",
    dtype_x_hermitian_atol_rtol=_matrix_rank_helper(),
    ground_truth_backend="numpy",
)
def test_matrix_rank(
    *, dtype_x_hermitian_atol_rtol, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, hermitian, atol, rtol = dtype_x_hermitian_atol_rtol
    assume(matrix_is_stable(x, cond_limit=10))
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        atol=atol,
        rtol=rtol,
        hermitian=hermitian,
    )


# cholesky
# execute with grads error
@handle_test(
    fn_tree="functional.ivy.cholesky",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
    upper=st.booleans(),
)
def test_cholesky(*, dtype_x, upper, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    x = x[0]
    x = np.matmul(x.T, x) + np.identity(x.shape[0])  # make symmetric positive-definite

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x,
        upper=upper,
        rtol_=1e-3,
        atol_=1e-3,
    )


# cross
@handle_test(
    fn_tree="functional.ivy.cross",
    dtype_x1_x2_axis=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e5,
        max_value=1e5,
        abs_smallest_val=0.01,
        safety_factor_scale="log",
    ),
)
def test_cross(*, dtype_x1_x2_axis, test_flags, backend_fw, fn_name, on_device):
    dtype, x1, x2, axis = dtype_x1_x2_axis
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x1=x1,
        x2=x2,
        axis=axis,
    )


# diagonal
@handle_test(
    fn_tree="functional.ivy.diagonal",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=50,
    ),
    offset=helpers.ints(min_value=-10, max_value=50),
    axes=st.lists(
        helpers.ints(min_value=-2, max_value=1), min_size=2, max_size=2, unique=True
    ).filter(lambda axes: axes[0] % 2 != axes[1] % 2),
)
def test_diagonal(*, dtype_x, offset, axes, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],
    )


@st.composite
def _diag_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            small_abs_safety_factor=2,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=2,
            min_dim_size=1,
            max_dim_size=50,
        )
    )
    shape = x[0].shape
    if len(shape) == 2:
        k = draw(helpers.ints(min_value=-shape[0] + 1, max_value=shape[1] - 1))
    else:
        k = draw(helpers.ints(min_value=0, max_value=shape[0]))
    return dtype, x, k


# diag
@handle_test(
    fn_tree="functional.ivy.diag",
    dtype_x_k=_diag_helper(),
)
def test_diag(*, dtype_x_k, test_flags, backend_fw, fn_name, on_device):
    dtype, x, k = dtype_x_k
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        k=k,
    )


# vander
@handle_test(
    fn_tree="functional.ivy.vander",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(
            helpers.ints(min_value=1, max_value=10),
        ),
        large_abs_safety_factor=15,
        small_abs_safety_factor=15,
        safety_factor_scale="log",
    ),
    N=st.integers(min_value=1, max_value=10) | st.none(),
    increasing=st.booleans(),
)
def test_vander(
    *, dtype_and_x, N, increasing, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
        N=N,
        increasing=increasing,
    )
