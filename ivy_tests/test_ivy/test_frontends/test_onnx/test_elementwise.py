# global
# import numpy as np
# from hypothesis import strategies as st, assume

# local
import ivy

# import ivy_tests.test_ivy.helpers as helpers
# from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy.functional.frontends.onnx as onnx
import ivy.functional.frontends.torch as torch
import ivy.functional.frontends.jax as jax
import ivy.functional.frontends.paddle as paddle
import ivy.functional.frontends.tensorflow as tf
import ivy.functional.frontends.mxnet as mxnet


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


def test_onnx_abs_v2():
    x = ivy.random_uniform(low=-100.0, high=0, shape=10)

    onnx_abs = onnx.abs(x)
    torch_abs = torch.abs(x)
    jax_abs = jax.lax.abs(x)
    paddle_abs = paddle.abs(x)
    tf_abs = tf.abs(x)

    all_list = [onnx_abs, jax_abs, paddle_abs, tf_abs]

    assert torch.allclose(torch_abs, all_list)


def test_onnx_acos_v2():
    x = ivy.random_uniform(low=-1, high=1, shape=10)

    onnx_acos = onnx.acos(x)
    torch_acos = torch.acos(x)
    jax_acos = jax.lax.acos(x)
    paddle_acos = paddle.acos(x)
    tf_acos = tf.acos(x)

    all_list = [onnx_acos, jax_acos, paddle_acos, tf_acos]

    are_equal = torch.allclose(torch_acos, all_list)

    assert are_equal


def test_onnx_acosh_v2():
    x = ivy.random_uniform(low=1, high=1000000, shape=10)

    onnx_acosh = onnx.acosh(x)
    torch_acosh = torch.acosh(x)
    paddle_acosh = paddle.acosh(x)
    tf_acosh = tf.acosh(x)

    all_list = [onnx_acosh, paddle_acosh, tf_acosh]

    are_equal = torch.allclose(torch_acosh, all_list)

    assert are_equal


def test_onnx_add_v2():
    x = ivy.random_uniform(low=-1000000, high=1000000, shape=10)
    y = ivy.random_uniform(low=-1000000, high=1000000, shape=10)

    jax_numpy_add = jax.numpy.add(x, y)
    mxnet_add = mxnet.numpy.add(x, y)
    onnx_add = onnx.add(x, y)
    torch_add = torch.add(x, y)
    paddle_add = paddle.add(x, y)
    tf_add = tf.add(x, y)

    all_list = [jax_numpy_add, mxnet_add, onnx_add, paddle_add, tf_add]

    are_equal = torch.allclose(torch_add, all_list)

    assert are_equal


def test_onnx_asin_v2():
    x = ivy.random_uniform(low=-1, high=1, shape=8)

    jax_asin = jax.lax.asin(x)
    onnx_asin = onnx.asin(x)
    torch_asin = torch.asin(x)
    paddle_asin = paddle.asin(x)
    tf_asin = tf.asin(x)

    all_list = [jax_asin, onnx_asin, paddle_asin, tf_asin]

    are_equal = torch.allclose(torch_asin, all_list)

    assert are_equal
