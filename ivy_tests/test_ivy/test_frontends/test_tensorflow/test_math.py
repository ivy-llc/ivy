# global
import ivy
import numpy as np
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
from ivy_tests.test_ivy.helpers import handle_frontend_test


# accumulate_n
@handle_frontend_test(
    fn_tree="tensorflow.math.accumulate_n",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.int64]),
        num_arrays=2,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
    
    
# negative
@handle_frontend_test(
    fn_tree="tensorflow.math.negative",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_tensorflow_negative(
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_euclidean_norm(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# argmax
@handle_frontend_test(
    fn_tree="tensorflow.math.argmax",
    dtype_and_x=statistical_dtype_values(function="argmax"),
    output_type=st.sampled_from(["int16", "uint16", "int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_argmax(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    output_type,
):
    if ivy.current_backend_str() == "torch":
        assume(output_type != "uint16")
    input_dtype, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    ),
)
def test_tensorflow_reduce_std(
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_reduce_sum(
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_tensor=x[0],
    )


# reduce_variance
@handle_frontend_test(
    fn_tree="tensorflow.math.reduce_variance",
    dtype_and_x=statistical_dtype_values(
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
    on_device,
    keepdims,
):
    input_dtype, x, axis, ddof = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    scalar_dtype, scalar = scalar_val
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
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
    on_device,
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    keepdims=st.booleans(),
    dtype=helpers.get_dtypes("numeric"),
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
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    dtype_x, x = dtype_and_x
    dtype_coeffs, coeffs = dtype_and_coeffs
    helpers.test_frontend_function(
        input_dtypes=dtype_coeffs + dtype_x,
        frontend=frontend,
        test_flags=test_flags,
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
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
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
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
        frontend=frontend,
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
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_truediv(
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
        x=x[0],
        y=x[1],
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
def test_tensorflow_pow(dtype_and_x, frontend, test_flags, fn_tree):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        y=x[1],
    )


# argmin
@handle_frontend_test(
    fn_tree="tensorflow.math.argmin",
    dtype_and_x=statistical_dtype_values(function="argmin"),
    output_type=st.sampled_from(["int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_argmin(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    output_type,
):
    input_dtype, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    fn_tree,
    on_device,
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
    fn_tree,
    on_device,
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
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


#acos 
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
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend = frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
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
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
