"""Collection of image Ivy functions."""

# local
import ivy
import numpy as np
import operator
import functools
from ivy.backend_handler import current_backend
from ivy.func_wrapper import to_native_arrays_and_back, handle_out_argument
from typing import Union, List, Tuple, Optional


# Extra #
# ------#

@handle_out_argument
@to_native_arrays_and_back
def stack_images(
    images: List[Union[ivy.Array, ivy.NativeArray]],
    desired_aspect_ratio: Tuple[int, int] = (1, 1)
) -> ivy.Array:
    """Stacks a group of images into a combined windowed image, fitting the desired
    aspect ratio as closely as possible.

    Parameters
    ----------
    images
        Sequence of image arrays to be stacked *[batch_shape,height,width,dims]*
    desired_aspect_ratio:
        desired aspect ratio of the stacked image

    Returns
    -------
    ret
        an array containing the stacked images in a specified aspect ratio/dimensions

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> shape, num = (1, 2, 3), 2
    >>> data = [ivy.ones(shape)] * num
    >>> stacked = ivy.stack_images(data, (2, 1))
    >>> print(stacked)
    ivy.array([[[1., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.],
            [0., 0., 0.]],
           [[1., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.],
            [0., 0., 0.]]])

    With :code:`ivy.NativeArray` input:

    >>> shape, num = (1, 1, 2), 2
    >>> data = [ivy.native_array(([[[1,2]]]))] * num
    >>> stacked = ivy.stack_images(data, (4,1))
    >>> print(stacked)
    ivy.array([[[1., 2.],
        [1., 2.],
        [0., 0.]]])

    With :code:`ivy.Container` input:

    >>> shape, num = (1, 1, 2), 2
    >>> data = [ivy.array(([[[3,4]]]))] * num
    >>> d = ivy.Container({'a': data, 'b': data})
    >>> stacked = ivy.stack_images(d, (2,3))
    >>> print(stacked)
    {
    a: ivy.array([[[3., 4.],
                   [0., 0.]],
                  [[3., 4.],
                   [0., 0.]]]),
    b: ivy.array([[[3., 4.],
                   [0., 0.]],
                  [[3., 4.],
                   [0., 0.]]])
    }

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> shape, num = (1, 1, 2), 2
    >>> data = [ivy.array(([[[1,2]]]))* num-1, ivy.native_array(([[[3,4]]]))]
    >>> stacked = ivy.stack_images(data, (2,3))
    >>> print(stacked)
    ivy.array([[[1., 3.],
        [0., 0.]],

       [[3., 4.],
        [0., 0.]]])

    With a mix of :code:`ivy.NativeArray` and :code:`ivy.Container` inputs:

    >>> shape, num = (1, 1, 2), 2
    >>> data = [ivy.native_array(([[[1,2]]]))* num-1, ivy.Container({'a':ivy.array([[[5,4]]])})]
    >>> print(len(data))
    >>> stacked = ivy.stack_images(data, (2,3))
    >>> print(stacked)
    {
    a: ivy.array([[[1., 3.],
                   [0., 0.]],
                  [[5., 4.],
                   [0., 0.]]])
    }

    #instance methods not suitable here

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(images[0]).stack_images(images, desired_aspect_ratio)


@to_native_arrays_and_back
@handle_out_argument
def bilinear_resample(
    x: Union[ivy.Array, ivy.NativeArray],
    warp: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Performs bilinearly re-sampling on input image.

    Parameters
    ----------
    x
        Input image *[batch_shape,h,w,dims]*.
    warp
        Warp array *[batch_shape,num_samples,2]*

    Returns
    -------
    ret
        Image after bilinear re-sampling.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([[[[0.34820361],\
         [0.10818656]],\
        [[0.2018713 ],\
         [0.73314588]]]])
    >>> warp = ivy.array([[[[0.11129423, 0.09569724],\
         [0.32680186, 0.34083896]],\
        [[0.28126204, 0.29169936],\
         [0.26754953, 0.08126624]]]])
    >>> y = ivy.bilinear_resample(x,warp)
    >>> print(y)
    ivy.array([[[0.316],
        [0.306],
        [0.301],
        [0.289]]])

    """
    return current_backend(x).bilinear_resample(x, warp)


@to_native_arrays_and_back
def gradient_image(
        x: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Computes image gradients (dy, dx) for each channel.

    Parameters
    ----------
    x
        Input image *[batch_shape, h, w, d]* .

    Returns
    -------
    ret
        Gradient images dy *[batch_shape,h,w,d]* and dx *[batch_shape,h,w,d]* .

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> batch_size = 1
    >>> h = 3
    >>> w = 3
    >>> d = 1
    >>> x = ivy.arange(batch_size * h * w * d, dtype=ivy.float32)
    >>> image = ivy.reshape(x,shape=(batch_size, h, w, d))
    >>> dy, dx = ivy.gradient_image(image)
    >>> print(image[0, :,:,0])
    ivy.array([[0., 1., 2.],
               [3., 4., 5.],
               [6., 7., 8.]])

    >>> print(dy[0, :,:,0])
     ivy.array([[3., 3., 3.],
               [3., 3., 3.],
               [0., 0., 0.]])

    >>> print(dx[0, :,:,0])
     ivy.array([[1., 1., 0.],
               [1., 1., 0.],
               [1., 1., 0.]])

    With :code:`ivy.Container` input:

    >>> batch_size = 1
    >>> h = 3
    >>> w = 3
    >>> d = 1
    >>> a = ivy.arange(batch_size * h * w * d, dtype=ivy.float32)
    >>> x = ivy.Container({'a': a})
    >>> image = ivy.reshape(x, shape=(batch_size, h, w, d))
    >>> res = ivy.gradient_image(image)
    >>> dx = res['a'][0]
    >>> print(dx)
    >>> dy = res['a'][1]
    >>> print(dy)
    """
    return current_backend(x).gradient_image(x)


@to_native_arrays_and_back
def float_img_to_uint8_img(
        x: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Converts an image of floats into a bit-cast 4-channel image of uint8s, which can
    be saved to disk.

    Parameters
    ----------
    x
        Input float image *[batch_shape,h,w]*.

    Returns
    -------
    ret
        The new encoded uint8 image *[batch_shape,h,w,4]* .

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> batch_size = 1
    >>> h = 1
    >>>  = 2
    >>> x = ivy.arange(batch_size * h * w, dtype=ivy.float32)
    >>> image = ivy.reshape(x, shape=(batch_size, h, w))
    >>> res = ivy.float_img_to_uint8_img(image)
    >>> print(res)
    ivy.array([[[[  0,   0,   0,   0],
         [  0,   0, 128,  63]]]])

    """
    x_np = ivy.to_numpy(x).astype("float32")
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_uint8 = np.frombuffer(x_bytes, np.uint8)
    return ivy.array(np.reshape(x_uint8, list(x_shape) + [4]).tolist())


@to_native_arrays_and_back
def uint8_img_to_float_img(
    x: Union[ivy.Array, ivy.NativeArray]
) -> ivy.Array:
    """Converts an image of uint8 values into a bit-cast float image.

    Parameters
    ----------
    x
        Input uint8 image *[batch_shape,h,w,4]*

    Returns
    -------
    ret
        The new float image *[batch_shape,h,w]*

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> batch_size = 1
    >>> h = 2
    >>> w = 3
    >>> d = 4
    >>> x = ivy.arange(batch_size * h * w * d)
    >>> image = ivy.reshape(x,(batch_size, h, w, d))
    >>> y = ivy.uint8_img_to_float_img(image)
    >>> print(y)
    ivy.array([[[3.82e-37, 1.01e-34, 2.66e-32],
        [7.00e-30, 1.84e-27, 4.85e-25]]])

    With :code:`ivy.NativeArray` input:

    >>> batch_size = 1
    >>> h = 1
    >>> w = 2
    >>> d = 4
    >>> x = ivy.native_array([[[[1,2,3,4], [4,5,2,1]]]])
    >>> image = ivy.reshape(x,(batch_size,h, w, d))
    >>> y = ivy.uint8_img_to_float_img(image)
    >>> print(y)
    ivy.array([[[1.54e-36, 2.39e-38]]])

    """
    x_np = ivy.to_numpy(x).astype("uint8")
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_float = np.frombuffer(x_bytes, np.float32)
    return ivy.array(np.reshape(x_float, x_shape[:-1]).tolist())


@to_native_arrays_and_back
def random_crop(
    x: Union[ivy.Array, ivy.NativeArray],
    crop_size: List[int],
    batch_shape: Optional[List[int]] = None,
    image_dims: Optional[List[int]] = None,
    seed: int = None
) -> ivy.Array:
    """Randomly crops the input images.

    Parameters
    ----------
    x
        Input images to crop *[batch_shape,h,w,f]*
    crop_size
        The 2D crop size.
    batch_shape
        Shape of batch. Inferred from inputs if None. (Default value = None)
    image_dims
        Image dimensions. Inferred from inputs in None. (Default value = None)
    seed
        Required for random number generator

    Returns
    -------
    ret
        The new cropped image *[batch_shape,nh,nw,f]*

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([[[[1,2,3,4], [4,5,2,1]]]])
    >>> print(x.shape)
    (1, 1, 2, 4)
    >>> crop_size = [1,1]
    >>> res = ivy.random_crop(x,crop_size)
    >>> print(res)
    ivy.array([[[[4, 5, 2, 1]]]])
    >>> print(res.shape)
    (1, 1, 1, 4)
    """
    x_shape = x.shape
    if batch_shape is None:
        batch_shape = x_shape[:-3]
    if image_dims is None:
        image_dims = x_shape[-3:-1]
    num_channels = x_shape[-1]
    flat_batch_size = functools.reduce(operator.mul, batch_shape, 1)
    crop_size[0] = min(crop_size[-2], x_shape[-3])
    crop_size[1] = min(crop_size[-1], x_shape[-2])

    # shapes as list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    margins = [img_dim - cs for img_dim, cs in zip(image_dims, crop_size)]

    # FBS x H x W x F
    x_flat = ivy.reshape(x, [flat_batch_size] + image_dims + [num_channels])

    # FBS x 1
    rng = np.random.default_rng(seed)
    x_offsets = rng.integers(0, margins[0] + 1, flat_batch_size).tolist()
    y_offsets = rng.integers(0, margins[1] + 1, flat_batch_size).tolist()

    # list of 1 x NH x NW x F
    cropped_list = [
        img[..., xo : xo + crop_size[0], yo : yo + crop_size[1], :]
        for img, xo, yo in zip(ivy.unstack(x_flat, 0, True), x_offsets, y_offsets)
    ]

    # FBS x NH x NW x F
    flat_cropped = ivy.concat(cropped_list, 0)

    # BS x NH x NW x F
    return ivy.reshape(flat_cropped, batch_shape + crop_size + [num_channels])


@to_native_arrays_and_back
@handle_out_argument
def linear_resample(
    x: Union[ivy.Array, ivy.NativeArray],
    num_samples: int,
    axis: int = -1
) -> Union[ivy.Array, ivy.NativeArray]:
    """Performs linear re-sampling on input image.

    Parameters
    ----------
    x
        Input image
    num_samples
        The number of interpolated samples to take.
    axis
        The axis along which to perform the resample. Default is last dimension.

    Returns
    -------
    ret
        The array after the linear resampling.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:
    >>> data = ivy.array([[1, 2],[3, 4]])
    >>> y = ivy.linear_resample(data, 5)
    >>> print(y)
    ivy.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

    With :code:`ivy.Container` input:

    >>> data = ivy.Container({'a':ivy.array([[0.0976, -0.3452,  1.2740], \
        [0.1047,  0.5886,  1.2732], \
        [0.7696, -1.7024, -2.2518]])})
    >>> y = ivy.linear_resample(data, 3, 0)
    >>> print(y)
    {
    a: ivy.array([[0.0976, -0.345, 1.27],
                  [0.105, 0.589, 1.27],
                  [0.77, -1.7, -2.25]])
    }
    """
    return current_backend(x).linear_resample(x, num_samples, axis)
