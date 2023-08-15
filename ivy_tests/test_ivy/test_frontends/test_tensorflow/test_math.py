# global
import ivy
import numpy as np
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.helpers import handle_frontend_test


# imag
@handle_frontend_test(
    fn_tree="tensorflow.math.imag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_imag(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        input=x[0],
    )


# accumulate_n
@handle_frontend_test(
    fn_tree="tensorflow.math.accumulate_n",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.int64]),
        num_arrays=helpers.ints(min_value=2, max_value=5),
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_accumulate_n(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        inputs=x,
    )


# add
@handle_frontend_test(
    fn_tree="tensorflow.math.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_add(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# sin
@handle_frontend_test(
    fn_tree="tensorflow.math.sin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_sin(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# tan
@handle_frontend_test(
    fn_tree="tensorflow.math.tan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_tan(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# exp
@handle_frontend_test(
    fn_tree="tensorflow.math.exp",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_exp(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# expm1
@handle_frontend_test(
    fn_tree="tensorflow.math.expm1",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_expm1(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sqrt
@handle_frontend_test(
    fn_tree="tensorflow.math.sqrt",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_sqrt(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# multiply
@handle_frontend_test(
    fn_tree="tensorflow.math.multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_multiply(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# maximum
@handle_frontend_test(
    fn_tree="tensorflow.math.maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_maximum(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# subtract
@handle_frontend_test(
    fn_tree="tensorflow.math.subtract",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_subtract(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# squared_difference
@handle_frontend_test(
    fn_tree="tensorflow.math.squared_difference",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_squared_difference(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# logical_not
@handle_frontend_test(
    fn_tree="tensorflow.math.logical_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_logical_not(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# logical_xor
@handle_frontend_test(
    fn_tree="tensorflow.math.logical_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_logical_xor(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# divide
@handle_frontend_test(
    fn_tree="tensorflow.math.divide",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_divide(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# negative
@handle_frontend_test(
    fn_tree="tensorflow.math.negative",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(
            helpers.get_dtypes("signed_integer"),
            helpers.get_dtypes("float"),
        )
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_negative(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# logical_and
@handle_frontend_test(
    fn_tree="tensorflow.math.logical_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_logical_and(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# logical_or
@handle_frontend_test(
    fn_tree="tensorflow.math.logical_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_logical_or(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# log_sigmoid
@handle_frontend_test(
    fn_tree="tensorflow.math.log_sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=3,
        small_abs_safety_factor=3,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_log_sigmoid(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# log1p
@handle_frontend_test(
    fn_tree="tensorflow.math.log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_log1p(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# reciprocal
@handle_frontend_test(
    fn_tree="tensorflow.math.reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reciprocal(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        x=x[0],
    )


# reciprocal_no_nan
@handle_frontend_test(
    fn_tree="tensorflow.math.reciprocal_no_nan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reciprocal_no_nan(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# reduce_all()
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_all",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_all(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_any
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_any",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_any(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_euclidean_norm
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_euclidean_norm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_num_dims=2,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_euclidean_norm(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-01,
        atol=1e-01,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_logsumexp
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_logsumexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_logsumexp(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# argmax
@handle_frontend_test(
    fn_tree="tensorflow.math.argmax",
    dtype_and_x=_statistical_dtype_values(function="argmax"),
    output_type=st.sampled_from(["int16", "uint16", "int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_argmax(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    output_type,
):
    if backend_fw in ("torch", "paddle"):
        assume(output_type != "uint16")
    input_dtype, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        axis=axis,
        output_type=output_type,
    )


# reduce_max
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_max",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_max(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_min
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_min",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_min(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_prod
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_prod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_prod(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_std
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_std",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
)
def test_tensorflow_reduce_std(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# asinh
@handle_frontend_test(
    fn_tree="tensorflow.math.asinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_asinh(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# reduce_sum
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_sum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=25,
        small_abs_safety_factor=25,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_sum(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        atol=1e-03,
        input_tensor=x[0],
    )


# reduce_mean
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_mean",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_mean(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-2,
        rtol=1e-2,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_variance
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_variance",
    dtype_and_x=_statistical_dtype_values(
        function="var",
    ),
    test_with_out=st.just(False),
    keepdims=st.booleans(),
)
def test_tensorflow_reduce_variance(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    keepdims,
):
    input_dtype, x, axis, ddof = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
        axis=axis,
        atol=1e-2,
        rtol=1e-2,
        keepdims=keepdims,
    )


# scalar_mul
@handle_frontend_test(
    fn_tree="tensorflow.math.scalar_mul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.shared(
            helpers.get_dtypes("float", full=False),
            key="shared_dtype",
        ),
        min_num_dims=1,
        min_dim_size=2,
    ),
    scalar_val=helpers.dtype_and_values(
        available_dtypes=st.shared(
            helpers.get_dtypes("float", full=False),
            key="shared_dtype",
        ),
        shape=(1,),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_scalar_mul(
    *,
    dtype_and_x,
    scalar_val,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    scalar_dtype, scalar = scalar_val
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        scalar=scalar[0][0],
        x=x[0],
    )


# divide_no_nan
@handle_frontend_test(
    fn_tree="tensorflow.math.divide_no_nan",
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=2,
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_divide_no_nan(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xy[0],
        y=xy[1],
    )


# multiply_no_nan
@handle_frontend_test(
    fn_tree="tensorflow.math.multiply_no_nan",
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=2,
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_multiply_no_nan(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xy[0],
        y=xy[1],
    )


# erfcinv
@handle_frontend_test(
    fn_tree="tensorflow.math.erfcinv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_erfcinv(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# is_inf
@handle_frontend_test(
    fn_tree="tensorflow.math.is_inf",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_is_inf(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# is_non_decreasing
@handle_frontend_test(
    fn_tree="tensorflow.math.is_non_decreasing",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_is_non_decreasing(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# is_strictly_increasing
@handle_frontend_test(
    fn_tree="tensorflow.math.is_strictly_increasing",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_is_strictly_increasing(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# count_nonzero
@handle_frontend_test(
    fn_tree="tensorflow.math.count_nonzero",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        allow_neg_axes=False,
    ),
    keepdims=st.booleans(),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_count_nonzero(
    *,
    dtype_x_axis,
    dtype,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype[0],
    )


# confusion_matrix
@handle_frontend_test(
    fn_tree="tensorflow.math.confusion_matrix",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_value=0,
        max_value=4,
        shared_dtype=True,
    ),
    num_classes=st.integers(min_value=5, max_value=10),
    test_with_out=st.just(False),
)
def test_tensorflow_confusion_matrix(
    *,
    dtype_and_x,
    num_classes,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        labels=x[0],
        predictions=x[1],
        num_classes=num_classes,
    )


# polyval
@handle_frontend_test(
    fn_tree="tensorflow.math.polyval",
    dtype_and_coeffs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=0,
    ),
)
def test_tensorflow_polyval(
    *,
    dtype_and_coeffs,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype_x, x = dtype_and_x
    dtype_coeffs, coeffs = dtype_and_coeffs
    helpers.test_frontend_function(
        input_dtypes=dtype_coeffs + dtype_x,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        coeffs=coeffs,
        x=x,
    )


# unsorted_segment_mean
@handle_frontend_test(
    fn_tree="tensorflow.math.unsorted_segment_mean",
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype="int32", shape=(5,), min_value=0, max_value=4
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_unsorted_segment_mean(
    *,
    data,
    segment_ids,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int32", "int64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        data=data,
        segment_ids=segment_ids,
        num_segments=np.max(segment_ids) + 1,
    )


# unsorted_segment_sum
@handle_frontend_test(
    fn_tree="tensorflow.math.unsorted_segment_sum",
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_unsorted_segment_sum(
    *,
    data,
    segment_ids,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=["int32", "int64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        data=data,
        segment_ids=segment_ids,
        num_segments=np.max(segment_ids) + 1,
    )


# unsorted_segment_sqrt_n
@handle_frontend_test(
    fn_tree="tensorflow.math.unsorted_segment_sqrt_n",
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_unsorted_segment_sqrt_n(
    *,
    data,
    segment_ids,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        data=data,
        segment_ids=segment_ids,
        num_segments=np.max(segment_ids) + 1,
    )


# zero_fraction
@handle_frontend_test(
    fn_tree="tensorflow.math.zero_fraction",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_zero_fraction(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0],
    )


# truediv
@handle_frontend_test(
    fn_tree="tensorflow.math.truediv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_truediv(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
        rtol=1e-2,
        atol=1e-2,
    )


# pow
@handle_frontend_test(
    fn_tree="tensorflow.math.pow",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        num_arrays=2,
        min_value=1,
        max_value=7,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_pow(dtype_and_x, frontend, test_flags, backend_fw, fn_tree):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        y=x[1],
    )


# argmin
@handle_frontend_test(
    fn_tree="tensorflow.math.argmin",
    dtype_and_x=_statistical_dtype_values(function="argmin"),
    output_type=st.sampled_from(["int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_argmin(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    output_type,
):
    input_dtype, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        axis=axis,
        output_type=output_type,
    )


# equal
@handle_frontend_test(
    fn_tree="tensorflow.math.equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_equal(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# not_equal
@handle_frontend_test(
    fn_tree="tensorflow.math.not_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_not_equal(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# floor
@handle_frontend_test(
    fn_tree="tensorflow.math.floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_floor(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# ceil
@handle_frontend_test(
    fn_tree="tensorflow.math.ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_ceil(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# round
@handle_frontend_test(
    fn_tree="tensorflow.math.round",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_round(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# minimum
@handle_frontend_test(
    fn_tree="tensorflow.math.minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-20,
        max_value=20,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_minimum(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# sigmoid
@handle_frontend_test(
    fn_tree="tensorflow.math.sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        num_arrays=1,
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# tanh
@handle_frontend_test(
    fn_tree="tensorflow.math.tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_tanh(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# rsqrt
@handle_frontend_test(
    fn_tree="tensorflow.math.rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_rsqrt(
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
        rtol=1e-02,
        x=x[0],
    )


# nextafter
@handle_frontend_test(
    fn_tree="tensorflow.math.nextafter",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_nextafter(
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
        x1=x[0],
        x2=x[1],
    )


# log_softmax
@handle_frontend_test(
    fn_tree="tensorflow.math.log_softmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_log_softmax(
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
        logits=x[0],
    )


# abs
@handle_frontend_test(
    fn_tree="tensorflow.math.abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=25,
        small_abs_safety_factor=25,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_abs(
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
        rtol=1e-02,
        x=x[0],
    )


# asin
@handle_frontend_test(
    fn_tree="tensorflow.math.asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1,
        max_value=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_asin(
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
        x=x[0],
    )


# acos
@handle_frontend_test(
    fn_tree="tensorflow.math.acos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1,
        max_value=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_acos(
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
        x=x[0],
    )


# acosh
@handle_frontend_test(
    fn_tree="tensorflow.math.acosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        small_abs_safety_factor=3,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_acosh(
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
        rtol=1e-02,
        x=x[0],
    )


# square
@handle_frontend_test(
    fn_tree="tensorflow.math.square",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_square(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# is_nan
@handle_frontend_test(
    fn_tree="tensorflow.math.is_nan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_is_nan(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# is_finite


@handle_frontend_test(
    fn_tree="tensorflow.math.is_finite",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_is_finite(
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
        x=x[0],
    )


# atan
@handle_frontend_test(
    fn_tree="tensorflow.math.atan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_atan(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# log
@handle_frontend_test(
    fn_tree="tensorflow.math.log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_tensorflow_log(
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
        x=x[0],
    )


# add_n
@handle_frontend_test(
    fn_tree="tensorflow.math.add_n",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=helpers.ints(min_value=1, max_value=5),
        shared_dtype=True,
    ),
)
def test_tensorflow_add_n(
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
        inputs=x,
    )


# floormod
@handle_frontend_test(
    fn_tree="tensorflow.math.floormod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_floormod(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# greater
@handle_frontend_test(
    fn_tree="tensorflow.math.greater",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_greater(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.math.cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_cos(
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
        x=x[0],
    )


# sinh
@handle_frontend_test(
    fn_tree="tensorflow.math.sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_sinh(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# softmax
@handle_frontend_test(
    fn_tree="tensorflow.math.softmax",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        allow_inf=False,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_softmax(
    *,
    dtype_values_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logits=x[0],
        atol=1e-02,
        rtol=1e-2,
        axis=axis,
    )


# softplus
@handle_frontend_test(
    fn_tree="tensorflow.math.softplus",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_softplus(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        features=x[0],
    )


# xlogy
@handle_frontend_test(
    fn_tree="tensorflow.math.xlogy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_xlogy(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# cosh
@handle_frontend_test(
    fn_tree="tensorflow.math.cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_cosh(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# atan2
@handle_frontend_test(
    fn_tree="tensorflow.math.atan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_atan2(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y=x[0],
        x=x[1],
    )


# less_equal
@handle_frontend_test(
    fn_tree="tensorflow.math.less_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_less_equal(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# less
@handle_frontend_test(
    fn_tree="tensorflow.math.less",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_less(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# angle
@handle_frontend_test(
    fn_tree="tensorflow.math.angle",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=["float64", "complex64", "complex128"],
    ),
)
def test_tensorflow_angle(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# zeta
@handle_frontend_test(
    fn_tree="tensorflow.math.zeta",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_zeta(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        q=x[1],
    )


# greater_equal
@handle_frontend_test(
    fn_tree="tensorflow.math.greater_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_greater_equal(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# in_top_k
@handle_frontend_test(
    fn_tree="tensorflow.math.in_top_k",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    k=st.integers(min_value=0, max_value=5),
    test_with_out=st.just(False),
)
def test_tensorflow_in_top_k(
    *, dtype_and_x, frontend, test_flags, backend_fw, fn_tree, on_device, k
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        targets=x[0],
        pred=x[1],
        k=k,
    )


# conj
@handle_frontend_test(
    fn_tree="tensorflow.math.conj",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_tensorflow_conj(
    *,
    dtype_and_input,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# top_k
@handle_frontend_test(
    fn_tree="tensorflow.math.top_k",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shared_dtype=True,
    ),
    k=st.integers(min_value=0, max_value=5),
    sorted=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_top_k(
    *, dtype_and_x, frontend, test_flags, fn_tree, on_device, k, backend_fw
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
        k=k,
        sorted=sorted,
    )


# real
@handle_frontend_test(
    fn_tree="tensorflow.math.real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_real(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# atanh
@handle_frontend_test(
    fn_tree="tensorflow.math.atanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_atanh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.math.rint",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_rint(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        x=x[0],
    )


# bincount
@handle_frontend_test(
    fn_tree="tensorflow.math.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_bincount(
    *,
    dtype_and_x,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arr=x[0],
        weights=None,
        minlength=0,
    )


# xdivy
@handle_frontend_test(
    fn_tree="tensorflow.math.xdivy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_xdivy(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# Xlog1py
@handle_frontend_test(
    fn_tree="tensorflow.math.xlog1py",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_xlog1py(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.math.igamma",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        abs_smallest_val=1e-5,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=3,
        max_dim_size=3,
        min_value=2,
        max_value=100,
        allow_nan=False,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_igamma(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-04,
        a=xs[0],
        x=xs[1],
    )


# l2_normalize
@handle_frontend_test(
    fn_tree="tensorflow.math.l2_normalize",
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
    backend_fw,
):
    input_dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        x=x[0],
        axis=axis,
    )


# cumsum
@handle_frontend_test(
    fn_tree="tensorflow.math.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    exclusive=st.booleans(),
    reverse=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_cumsum(  # NOQA
    *,
    dtype_x_axis,
    exclusive,
    reverse,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
    )


# cumprod
@handle_frontend_test(
    fn_tree="tensorflow.math.cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    exclusive=st.booleans(),
    reverse=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_cumprod(  # NOQA
    *,
    dtype_x_axis,
    exclusive,
    reverse,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
    )


# softsign
@handle_frontend_test(
    fn_tree="tensorflow.math.softsign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_softsign(
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
        features=x[0],
    )
