# global

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_elementwise import (  # noqa
    ldexp_args,
)


# exp
@handle_frontend_test(
    fn_tree="numpy.exp",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="exp"
    ),
)
def test_numpy_exp(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# expm1
@handle_frontend_test(
    fn_tree="numpy.expm1",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="expm1"
    ),
)
def test_numpy_expm1(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# exp2
@handle_frontend_test(
    fn_tree="numpy.exp2",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="exp2"
    ),
)
def test_numpy_exp2(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# log10
@handle_frontend_test(
    fn_tree="numpy.log10",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="log10"
    ),
)
def test_numpy_log10(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# log
@handle_frontend_test(
    fn_tree="numpy.log",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                small_abs_safety_factor=2,
                safety_factor_scale="linear",
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="log"
    ),
)
def test_numpy_log(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# log2
@handle_frontend_test(
    fn_tree="numpy.log2",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                small_abs_safety_factor=2,
                safety_factor_scale="linear",
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="log2"
    ),
)
def test_numpy_log2(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# log1p
@handle_frontend_test(
    fn_tree="numpy.log1p",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="log1p"
    ),
)
def test_numpy_log1p(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# logaddexp
@handle_frontend_test(
    fn_tree="numpy.logaddexp",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                num_arrays=2,
                shared_dtype=True,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="logaddexp"
    ),
)
def test_numpy_logaddexp(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# logaddexp2
@handle_frontend_test(
    fn_tree="numpy.logaddexp2",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                num_arrays=2,
                shared_dtype=True,
                min_value=-100,
                max_value=100,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="logaddexp2"
    ),
)
def test_numpy_logaddexp2(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# i0
@handle_frontend_test(
    fn_tree="numpy.i0",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_numpy_i0(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# frexp
@handle_frontend_test(
    fn_tree="numpy.frexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=1,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
    ),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="frexp"
    ),
)
def test_numpy_frexp(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# ldexp
@handle_frontend_test(
    fn_tree="numpy.ldexp",
    dtype_and_x=ldexp_args(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="ldexp"
    ),
)
def test_numpy_ldexp(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )
