# global
# import numpy as np
from hypothesis import given

# local

import ivy_tests.test_ivy.helpers as helpers

# from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy.functional.frontends.onnx as onnx
import ivy.functional.frontends.torch as torch


# @handle_frontend_test(
#     fn_tree="onnx.abs",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("numeric", full=False),
#         large_abs_safety_factor=2.5,
#         small_abs_safety_factor=2.5,
#         safety_factor_scale="log",
#     ),
# )
# def test_onnx_abs(
#     *,
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#     )

# @handle_frontend_test(
#     fn_tree="onnx.acos",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("float"),
#     ),
# )
# def test_onnx_acos(
#     *,
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#     )
#
# @handle_frontend_test(
#     fn_tree="onnx.acosh",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("float"),
#     ),
# )
# def test_onnx_acosh(
#     *,
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#     )
#
# @handle_frontend_test(
#     fn_tree="onnx.add",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("numeric"),
#         num_arrays=2,
#         large_abs_safety_factor=2.5,
#         small_abs_safety_factor=2.5,
#         safety_factor_scale="log",
#     ),
#     alpha=st.integers(min_value=1, max_value=5),
# )
# def test_onnx_add(
#     *,
#     dtype_and_x,
#     alpha,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         rtol=1e-03,
#         input=x[0],
#         other=x[1],
#         alpha=alpha,
#     )
#
# @handle_frontend_test(
#     fn_tree="onnx.asin",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("float"),
#     ),
# )
# def test_onnx_asin(
#     *,
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         input=x[0],
#     )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", prune_function=False),
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    )
)
def test_onnx_abs_v2(dtype_x):
    _, data = dtype_x
    x_onnx = onnx.Tensor(data[0])
    x_torch = torch.Tensor(data[0])

    onnx_abs = onnx.abs(x_onnx)
    torch_abs = torch.abs(x_torch)

    ret = helpers.flatten_and_to_np(ret=onnx_abs)
    ret_gt = helpers.flatten_and_to_np(ret=torch_abs)

    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="torch",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", prune_function=False),
    ).filter(lambda x: "float16" not in x[0]),
)
def test_onnx_acos_v2(dtype_x):
    _, data = dtype_x
    x_onnx = onnx.Tensor(data[0])
    x_torch = torch.Tensor(data[0])

    onnx_acos = onnx.acos(x_onnx)
    torch_acos = torch.acos(x_torch)

    ret = helpers.flatten_and_to_np(ret=onnx_acos)
    ret_gt = helpers.flatten_and_to_np(ret=torch_acos)

    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="tensorflow",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", prune_function=False),
    ).filter(lambda x: "float16" not in x[0]),
)
def test_onnx_acosh_v2(dtype_x):
    _, data = dtype_x
    x_onnx = onnx.Tensor(data[0])
    x_torch = torch.Tensor(data[0])

    onnx_acosh = onnx.acosh(x_onnx)
    torch_acosh = torch.acosh(x_torch)

    ret = helpers.flatten_and_to_np(ret=onnx_acosh)
    ret_gt = helpers.flatten_and_to_np(ret=torch_acosh)

    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="tensorflow",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", prune_function=False),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_onnx_add_v2(dtype_x):
    _, data = dtype_x
    x_onnx_1 = onnx.Tensor(data[0])
    x_onnx_2 = onnx.Tensor(data[1])
    x_torch_1 = torch.Tensor(data[0])
    x_torch_2 = torch.Tensor(data[1])

    onnx_add = onnx.add(x_onnx_1, x_onnx_2)
    torch_add = torch.add(x_torch_1, x_torch_2)

    ret = helpers.flatten_and_to_np(ret=onnx_add)
    ret_gt = helpers.flatten_and_to_np(ret=torch_add)

    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="tensorflow",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", prune_function=False),
    ).filter(lambda x: "float16" not in x[0]),
)
def test_onnx_asin_v2(dtype_x):
    _, data = dtype_x
    x_onnx = onnx.Tensor(data[0])
    x_torch = torch.Tensor(data[0])

    onnx_asin = onnx.asin(x_onnx)
    torch_asin = torch.asin(x_torch)

    ret = helpers.flatten_and_to_np(ret=onnx_asin)
    ret_gt = helpers.flatten_and_to_np(ret=torch_asin)

    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="tensorflow",
    )
