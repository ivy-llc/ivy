# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
    _get_dtype_value1_value2_axis_for_tensordot,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_linalg import (
    _generate_multi_dot_dtype_and_arrays,
)


# outer
@handle_frontend_test(
    fn_tree="numpy.outer",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        shared_dtype=True,
    ),
)
def test_numpy_outer(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        b=xs[1],
    )


# inner
@handle_frontend_test(
    fn_tree="numpy.inner",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_inner(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        b=xs[1],
    )


# matmul
@handle_frontend_test(
    fn_tree="numpy.matmul",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[_get_first_matrix_and_dtype, _get_second_matrix_and_dtype],
        get_dtypes_kind="numeric",
    ),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="matmul"
    ),
)
def test_numpy_matmul(
    dtypes_values_casting,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        out=None,
        casting=casting,
        order="K",
        dtype=dtype,
        # The arguments below are currently unused.
        # subok=True,
    )


# matrix_power
@handle_frontend_test(
    fn_tree="numpy.linalg.matrix_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])),
    ),
    n=helpers.ints(min_value=1, max_value=8),
    test_with_out=st.just(False),
)
def test_numpy_matrix_power(
    dtype_and_x,
    n,
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
        a=x[0],
        n=n,
    )


# tensordot
@handle_frontend_test(
    fn_tree="numpy.tensordot",
    dtype_values_and_axes=_get_dtype_value1_value2_axis_for_tensordot(
        helpers.get_dtypes(kind="numeric")
    ),
    test_with_out=st.just(False),
)
def test_numpy_tensordot(
    dtype_values_and_axes,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, a, b, axes = dtype_values_and_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=a,
        b=b,
        axes=axes,
    )


# kron
@handle_frontend_test(
    fn_tree="numpy.kron",
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=2,
        allow_inf=True,
        allow_nan=True,
        shared_dtype=True,
    ),
)
def test_numpy_kron(
    *,
    dtype_and_x,
    frontend,
    fn_tree,
    on_device,
    test_flags,
):
    dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_flags=test_flags,
        a=xs[0],
        b=xs[1],
    )


# multi_dot
@handle_frontend_test(
    fn_tree="numpy.linalg.multi_dot",
    dtype_and_x=_generate_multi_dot_dtype_and_arrays(),
)
def test_numpy_multi_dot(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtypes, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_flags=test_flags,
        arrays=x,
        rtol=1e-3,
        atol=1e-3,
    )
