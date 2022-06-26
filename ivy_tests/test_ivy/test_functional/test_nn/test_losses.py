# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# cross_entropy
@given(
    data=st.data(),
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="cross_entropy"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
)
def test_cross_entropy(
    data,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if fw == "torch" and "float16" in input_dtype:
        return
    shape = data.draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, min_dim_size=1))
    true = data.draw(helpers.array_values(dtype=input_dtype[0], shape=shape))
    shape = (len(true),)
    pred = data.draw(helpers.array_values(dtype=input_dtype[1], shape=shape))
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cross_entropy",
        true=np.asarray(true, dtype=input_dtype[0]),
        pred=np.asarray(pred, dtype=input_dtype[1]),
    )


# binary_cross_entropy
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes, 2),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     with_out=st.booleans(),
#     num_positional_args=helpers.num_positional_args(fn_name="binary_cross
# _entropy"),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
#     instance_method=st.booleans(),
# )
# def test_binary_cross_entropy(
#     dtype_and_x,
#     as_variable,
#     with_out,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     input_dtype, x = dtype_and_x
#     if (v == [] for v in x):
#         return
#     if fw == "torch" and input_dtype == "float16":
#         return
#     helpers.test_array_function(
#         input_dtype,
#         as_variable,
#         with_out,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "binary_cross_entropy",
#         true=np.asarray(x[0], dtype=input_dtype[0]),
#         pred=np.asarray(x[1], dtype=input_dtype[1]),
#     )
#
#
# # sparse_cross_entropy
# @given(
#     data=st.data(),
#     true_dtype=st.sampled_from(ivy_np.valid_int_dtypes),
#     pred_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     with_out=st.booleans(),
#     num_positional_args=helpers.num_positional_args(fn_name="sparse_cross
# _entropy"),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
#     instance_method=st.booleans(),
# )
# def test_sparse_cross_entropy(
#     data,
#     true_dtype,
#     pred_dtype,
#     as_variable,
#     with_out,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     shape = data.draw(helpers.get_shape(min_num_dims=1, max_num_dims=1, min
# _dim_size=2))
#     pred = data.draw(helpers.array_values(dtype=pred_dtype, shape=shape))
#     true = data.draw(helpers.array_values(dtype=true_dtype, shape=(1,), min
# _value=0, max_value=len(pred)-1, allow_negative=False))
#     # if (v == [] for v in x):
#     #     return
#     if fw == "torch" and pred_dtype == "float16":
#         return
#     if fw == "tensorflow" and true_dtype not in ["uint8", "int32", "int64"]:
#         return
#     # if fw == "numpy":
#     #     return
#     # assume(true)
#     # assume(pred)
#     helpers.test_array_function(
#         [true_dtype, pred_dtype],
#         as_variable,
#         with_out,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "sparse_cross_entropy",
#         true=np.asarray(true, dtype=true_dtype),
#         pred=np.asarray(pred, dtype=pred_dtype),
#     )
