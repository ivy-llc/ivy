"""Collection of tests for unified image-related functions."""

# global

from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers



# stack_images
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=8),
    ratio=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=2),
    dtype=st.sampled_from(ivy.valid_float_dtype_strs),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=st.integers(0, 2),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
)
def test_stack_images(
    shape,
    ratio,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and dtype == "float16":
        return
    images = [img for img in ivy.random_normal(shape=shape)]
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "stack_images",
        images=images,
        desired_aspect_ratio=ratio,
    )


# # bilinear_resample
# @given(
#     shape=st.lists(st.integers(min_value=1, max_value=8),
#                    min_size=4, max_size=8),
#     n_samples=st.integers(min_value=1, max_value=8),
#     dtype=st.sampled_from(ivy.valid_float_dtype_strs),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     num_positional_args=st.integers(0, 2),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
# )
# def test_bilinear_resample(
#     shape,
#     n_samples,
#     dtype,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     fw,
# ):
#     if fw == "torch" and dtype == "float16":
#         return
#     x = ivy.random_normal(shape=shape)
#     warp = ivy.random_uniform(shape=shape[:-3] + [n_samples, 2])
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         False,
#         fw,
#         "bilinear_resample",
#         x=x,
#         warp=warp
#     )


# gradient_image
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=8),
    dtype=st.sampled_from(ivy.valid_float_dtype_strs),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_gradient_image(
    shape, dtype, as_variable, num_positional_args, native_array, container, fw
):
    if fw == "torch" and dtype == "float16":
        return
    x = ivy.random_normal(shape=shape)
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "gradient_image",
        x=x,
    )


# float_img_to_uint8_img
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=3, max_size=8),
    dtype=st.sampled_from(ivy.valid_float_dtype_strs),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_float_img_to_uint8_img(
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and dtype == "float16":
        return
    x = ivy.random_normal(shape=shape)
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "float_img_to_uint8_img",
        x=x,
    )


# uint8_img_to_float_img
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8),
                   min_size=3, max_size=8),
    dtype=st.sampled_from(ivy.valid_float_dtype_strs),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_uint8_img_to_float_img(
    shape,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and dtype == "float16":
        return
    x = ivy.randint(0, 256, shape=shape + [4])
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "uint8_img_to_float_img",
        x=x,
    )


# # random_crop
# @given(
#     shape=st.lists(st.integers(min_value=1, max_value=8),
#                    min_size=4, max_size=8),
#     dtype=st.sampled_from(ivy.valid_float_dtype_strs),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     num_positional_args=st.integers(0, 2),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
# )
# def test_random_crop(
#     shape,
#     dtype,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     fw,
# ):
#     if fw == "torch" and dtype == "float16":
#         return
#     x = ivy.random_normal(shape=shape)
#     crop_size = [random.randint(1, shape[-3]),
#                  random.randint(1, shape[-2])]
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         False,
#         fw,
#         "random_crop",
#         x=x,
#         crop_size=crop_size,
#     )
    
    
# # linear_resample
# @given(
#     shape=st.lists(st.integers(min_value=1, max_value=8),
#                    min_size=1, max_size=8),
#     num_samples=st.integers(min_value=2, max_value=8),
#     dtype=st.sampled_from(ivy.valid_float_dtype_strs),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     num_positional_args=st.integers(0, 2),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
# )
# def test_linear_resample(
#     shape,
#     num_samples,
#     dtype,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     fw,
# ):
#     if fw == "torch" and dtype == "float16":
#         return
#     x = ivy.random_normal(shape=shape)
#     axis = random.randint(0, len(shape)-1)
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         False,
#         fw,
#         "linear_resample",
#         x=x,
#         num_samples=num_samples,
#         axis=axis,
#     )


###############
# Smoke Tests #
###############

# # global
# import pytest
# import numpy as np
# from operator import mul
#
# # noinspection PyProtectedMember
# from functools import reduce
#
# # local
# import ivy
# import ivy.functional.backends.numpy
# import ivy_tests.test_ivy.helpers as helpers


# # stack_images
# @pytest.mark.parametrize(
#     "shp_n_num_n_ar_n_newshp",
#     [
#         ((1, 2, 3), 4, (2, 1), (2, 4, 3)),
#         ((8, 8, 3), 9, (1, 1), (24, 24, 3)),
#         ((3, 16, 12, 4), 10, (2, 5), (3, 80, 36, 4)),
#         ((5, 20, 9, 5), 10, (5, 2), (5, 40, 72, 5)),
#     ],
# )
# def test_stack_images(shp_n_num_n_ar_n_newshp, device, call):
#     # smoke test
#     shape, num, ar, new_shape = shp_n_num_n_ar_n_newshp
#     xs = [ivy.ones(shape)] * num
#     ret = ivy.stack_images(xs, ar)
#     # type test
#     assert ivy.is_ivy_array(ret)
#     # cardinality test
#     assert ret.shape == new_shape
#     # docstring test
#     helpers.docstring_examples_run(ivy.stack_images)


# # bilinear_resample
# @pytest.mark.parametrize(
#     "x_n_warp",
#     [
#         (
#             [[[[0.0], [1.0]], [[2.0], [3.0]]]],
#             [[[0.0, 1.0], [0.25, 0.25], [0.5, 0.5], [0.5, 1.0], [1.0, 0.5]]],
#         ),
#         (
#             [[[[0.0], [1.0]], [[2.0], [3.0]]]],
#             [[[0.0, 1.0], [0.5, 0.5], [0.5, 1.0], [1.0, 0.5]]],
#         ),
#         (
#             [[[[[0.0], [1.0]], [[2.0], [3.0]]]]],
#             [[[[0.0, 1.0], [0.5, 0.5], [0.5, 1.0], [1.0, 0.5]]]],
#         ),
#     ],
# )
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
# def test_bilinear_resample(x_n_warp, dtype, tensor_fn, device, call):
#     # smoke test
#     x, warp = x_n_warp
#     x = tensor_fn(x, dtype, device)
#     warp = tensor_fn(warp, dtype, device)
#     ret = ivy.bilinear_resample(x, warp)
#     # type test
#     assert ivy.is_ivy_array(ret)
#     # cardinality test
#     assert ret.shape == warp.shape[:-1] + x.shape[-1:]
#     # value test
#     assert np.allclose(
#         call(ivy.bilinear_resample, x, warp),
#         ivy.functional.backends.numpy.bilinear_resample(
#             ivy.to_numpy(x), ivy.to_numpy(warp)
#         ),
#     )
#     # compilation test
#     if call in [helpers.torch_call]:
#         # torch scripting does not support builtins
#         return


# # gradient_image
# @pytest.mark.parametrize(
#     "x_n_dy_n_dx",
#     [
#         (
#             [[[[0.0], [1.0], [2.0]], [[5.0], [4.0], [3.0]], [[6.0], [8.0], [7.0]]]],
#             [[[[5.0], [3.0], [1.0]], [[1.0], [4.0], [4.0]], [[0.0], [0.0], [0.0]]]],
#             [[[[1.0], [1.0], [0.0]], [[-1.0], [-1.0], [0.0]], [[2.0], [-1.0], [0.0]]]]
#         )
#     ],
# )
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
# def test_gradient_image(x_n_dy_n_dx, dtype, tensor_fn, device, call):
#     # smoke test
#     x, dy_true, dx_true = x_n_dy_n_dx
#     x = tensor_fn(x, dtype, device)
#     dy, dx = ivy.gradient_image(x)
#     # type test
#     assert ivy.is_ivy_array(dy)
#     assert ivy.is_ivy_array(dx)
#     # cardinality test
#     assert dy.shape == x.shape
#     assert dx.shape == x.shape
#     # value test
#     dy_np, dx_np = call(ivy.gradient_image, x)
#     dy_true = ivy.functional.backends.numpy.array(dy_true, dtype)
#     dx_true = ivy.functional.backends.numpy.array(dx_true, dtype)
#     assert np.allclose(dy_np, dy_true)
#     assert np.allclose(dx_np, dx_true)
#     # compilation test
#     if call in [helpers.torch_call]:
#         # torch device cannot be assigned value of string while scripting
#         return


# # float_img_to_uint8_img
# @pytest.mark.parametrize(
#     "fi_tui",
#     [
#         (
#             [[0.0, 1.0], [2.0, 3.0]],
#             [[[0, 0, 0, 0], [0, 0, 128, 63]], [[0, 0, 0, 64], [0, 0, 64, 64]]],
#         )
#     ],
# )
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
# def test_float_img_to_uint8_img(fi_tui, tensor_fn, device, call):
#     # smoke test
#     if call is helpers.tf_graph_call:
#         # tensorflow tensors cannot be cast to numpy arrays in graph mode
#         pytest.skip()
#     float_img, true_uint8_img = fi_tui
#     float_img = tensor_fn(float_img, "float32", device)
#     true_uint8_img = np.array(true_uint8_img)
#     uint8_img = ivy.float_img_to_uint8_img(float_img)
#     # type test
#     assert ivy.is_ivy_array(float_img)
#     # cardinality test
#     assert uint8_img.shape == true_uint8_img.shape
#     # value test
#     uint8_img_np = call(ivy.float_img_to_uint8_img, float_img)
#     assert np.allclose(uint8_img_np, true_uint8_img)
#     # compilation test
#     if call in [helpers.torch_call]:
#         # torch device cannot be assigned value of string while scripting
#         return


# # uint8_img_to_float_img
# @pytest.mark.parametrize(
#     "ui_tfi",
#     [
#         (
#             [[[0, 0, 0, 0], [0, 0, 128, 63]], [[0, 0, 0, 64], [0, 0, 64, 64]]],
#             [[0.0, 1.0], [2.0, 3.0]],
#         )
#     ],
# )
# def test_uint8_img_to_float_img(ui_tfi, device, call):
#     # smoke test
#     if call is helpers.tf_graph_call:
#         # tensorflow tensors cannot be cast to numpy arrays in graph mode
#         pytest.skip()
#     uint8_img, true_float_img = ui_tfi
#     uint8_img = ivy.array(uint8_img, "uint8", device)
#     true_float_img = np.array(true_float_img)
#     float_img = ivy.uint8_img_to_float_img(uint8_img)
#     # type test
#     assert ivy.is_ivy_array(float_img)
#     # cardinality test
#     assert float_img.shape == true_float_img.shape
#     # value test
#     float_img_np = call(ivy.uint8_img_to_float_img, uint8_img)
#     assert np.allclose(float_img_np, true_float_img)
#     # compilation test
#     if call in [helpers.torch_call]:
#         # torch device cannot be assigned value of string while scripting
#         return


# random_crop
# @pytest.mark.parametrize("xshp_n_cs", [([2, 5, 6, 3], [2, 2])])
# def test_random_crop(xshp_n_cs, device, call):
#     # seed
#     ivy.seed(0)
#     np.random.seed(0)
#     # smoke test
#     x_shape, crop_size = xshp_n_cs
#     batch_size = x_shape[0]
#     x_size = reduce(mul, x_shape[1:], 1)
#     x = ivy.einops_repeat(
#         ivy.reshape(ivy.arange(x_size), x_shape[1:]), "... -> b ...", b=batch_size
#     )
#     cropped = ivy.random_crop(x, crop_size)
#     # type test
#     assert ivy.is_ivy_array(cropped)
#     # cardinality test
#     true_shape = [item for item in x_shape]
#     true_shape[-3] = crop_size[0]
#     true_shape[-2] = crop_size[1]
#     assert list(cropped.shape) == true_shape
#     # value test
#     assert np.allclose(ivy.to_numpy(x[0]), ivy.to_numpy(x[1]))
#     cropped = call(ivy.random_crop, x, crop_size)
#     assert not np.allclose(cropped[0], cropped[1])
#     # compilation test
#     if call in [helpers.torch_call]:
#         # reduce(mul) used for flat batch size computation is not torch jit compilable
#         return


# # linear_resample
# @pytest.mark.parametrize(
#     "x_n_samples_n_axis_n_y_true",
#     [
#         (
#             [[10.0, 9.0, 8.0]],
#             9,
#             -1,
#             [[10.0, 9.75, 9.5, 9.25, 9.0, 8.75, 8.5, 8.25, 8.0]],
#         ),
#         (
#             [[[10.0, 9.0], [8.0, 7.0]]],
#             5,
#             -2,
#             [[[10.0, 9.0], [9.5, 8.5], [9.0, 8.0], [8.5, 7.5], [8.0, 7.0]]],
#         ),
#         (
#             [[[10.0, 9.0], [8.0, 7.0]]],
#             5,
#             -1,
#             [[[10.0, 9.75, 9.5, 9.25, 9.0], [8.0, 7.75, 7.5, 7.25, 7.0]]],
#         ),
#     ],
# )
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
# def test_linear_resample(x_n_samples_n_axis_n_y_true, dtype, tensor_fn, device, call):
#     # smoke test
#     x, samples, axis, y_true = x_n_samples_n_axis_n_y_true
#     x = tensor_fn(x, dtype, device)
#     ret = ivy.linear_resample(x, samples, axis)
#     # type test
#     assert ivy.is_ivy_array(ret)
#     # cardinality test
#     x_shape = list(x.shape)
#     num_x_dims = len(x_shape)
#     axis = axis % num_x_dims
#     x_pre_shape = x_shape[0:axis]
#     x_post_shape = x_shape[axis + 1 :]
#     assert list(ret.shape) == x_pre_shape + [samples] + x_post_shape
#     # value test
#     y_true = np.array(y_true)
#     y = call(ivy.linear_resample, x, samples, axis)
#     assert np.allclose(y, y_true)
