# global
import numpy as np
from hypothesis import assume, given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# cross_entropy
@given(
    data=st.data(),
    true_dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    pred_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="cross_entropy"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_cross_entropy(
    data,
    true_dtype,
    pred_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and pred_dtype == "float16":
        return
    shape = data.draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, min_dim_size=2))
    true = data.draw(
        helpers.array_values(
            dtype=true_dtype,
            shape=shape,
            min_value=0,
            max_value=1,
        )
    )
    pred = data.draw(
        helpers.array_values(
            dtype=pred_dtype,
            shape=shape,
            min_value=0,
            max_value=1,
            exclude_min=True,
            exclude_max=True,
        )
    )
    axis = data.draw(helpers.integers(min_value=-1, max_value=0))
    epsilon = data.draw(
        helpers.array_values(
            dtype=pred_dtype, shape=(1,), min_value=0, max_value=1, allow_negative=False
        )
    )
    helpers.test_array_function(
        [true_dtype, pred_dtype],
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cross_entropy",
        true=np.asarray(true, dtype=true_dtype),
        pred=np.asarray(pred, dtype=pred_dtype),
        axis=axis,
        epsilon=epsilon[0],
    )


# binary_cross_entropy
@given(
    data=st.data(),
    true_dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    pred_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="binary_cross_entropy"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_binary_cross_entropy(
    data,
    true_dtype,
    pred_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and pred_dtype == "float16":
        return
    shape = data.draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, min_dim_size=2))
    true = data.draw(
        helpers.array_values(
            dtype=true_dtype,
            shape=shape,
            min_value=0,
            max_value=1,
        )
    )
    pred = data.draw(
        helpers.array_values(
            dtype=pred_dtype,
            shape=shape,
            min_value=0,
            max_value=1,
            exclude_min=True,
            exclude_max=True,
        )
    )
    epsilon = data.draw(
        helpers.array_values(
            dtype=pred_dtype, shape=(1,), min_value=0, max_value=1, allow_negative=False
        )
    )
    helpers.test_array_function(
        [true_dtype, pred_dtype],
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "binary_cross_entropy",
        true=np.asarray(true, dtype=true_dtype),
        pred=np.asarray(pred, dtype=pred_dtype),
        epsilon=epsilon[0],
    )


# sparse_cross_entropy
@given(
    data=st.data(),
    true_dtype=st.sampled_from(ivy_np.valid_int_dtypes),
    pred_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="sparse_cross_entropy"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_sparse_cross_entropy(
    data,
    true_dtype,
    pred_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and pred_dtype == "float16":
        return
    if fw == "tensorflow" and true_dtype not in ["uint8", "int32", "int64"]:
        return
    shape = data.draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, min_dim_size=2))
    pred = data.draw(
        helpers.array_values(
            dtype=pred_dtype,
            shape=shape,
            min_value=0,
            max_value=1,
            exclude_min=True,
            exclude_max=True,
        )
    )
    true = data.draw(
        helpers.array_values(
            dtype=true_dtype,
            shape=(1,),
            min_value=0,
            max_value=shape[0],
            exclude_max=True,
        )
    )
    axis = data.draw(helpers.integers(min_value=-1, max_value=0))
    epsilon = data.draw(
        helpers.array_values(
            dtype=pred_dtype, shape=(1,), min_value=0, max_value=1, allow_negative=False
        )
    )
    assume(all([v < len(pred) for v in true]))
    helpers.test_array_function(
        [true_dtype, pred_dtype],
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "sparse_cross_entropy",
        true=np.asarray(true, dtype=true_dtype),
        pred=np.asarray(pred, dtype=pred_dtype),
        axis=axis,
        epsilon=epsilon[0],
    )
