# local
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"2.9.0 and below": ("float",)}, "tensorflow")
@to_ivy_arrays_and_back
def extract_patches(images, sizes, strides, rates, padding):

    # approach 1

    kernels = ivy.ones(sizes[1:3]+[images.shape[-1], 1], dtype=images.dtype)
    patches = ivy.conv2d(images, kernels, strides[1:3], padding, dilations=rates[1:3])
    return patches

    # approach 2

    # sizes = sizes[1:3]
    # strides = strides[1:3]
    # rates = rates[1:3]
    # images = ivy.to_numpy(images)
    # patches = []
    # for image in images:
    #     if padding == 'SAME':
    #         image = np.pad(image, [(0, sizes[0]-1), (0, sizes[1]-1), (0, 0)], mode='reflect')
    #     for j in range(0, image.shape[0] - sizes[0] + 1, strides[0]*rates[0]):
    #         for k in range(0, image.shape[1] - sizes[1] + 1, strides[1]*rates[1]):
    #             patch = image[j:j+sizes[0]:rates[0], k:k+sizes[1]:rates[1], :]
    #             patches.append(patch)
    # return patches

    # approach 3

    # batch_size, image_height, image_width, num_channels = images.shape
    # ksize_rows, ksize_cols = sizes[1], sizes[2]
    # row_stride, col_stride = strides[1], strides[2]
    # row_rate, col_rate = rates[1], rates[2]
    #
    # if padding == 'VALID':
    #     row_start = 0
    #     row_end = image_height - ksize_rows + 1
    #     col_start = 0
    #     col_end = image_width - ksize_cols + 1
    # elif padding == 'SAME':
    #     row_start = ksize_rows // 2
    #     row_end = row_start + image_height
    #     col_start = ksize_cols // 2
    #     col_end = col_start + image_width
    # else:
    #     raise ValueError('Invalid padding type')
    #
    # patches = []
    # for b in range(batch_size):
    #     for i in range(row_start, row_end, row_stride):
    #         for j in range(col_start, col_end, col_stride):
    #             patch = images[b, i:ksize_rows:row_rate, j:ksize_cols:col_rate, :]
    #             if patch.size == 0:
    #                 break
    #             patches.append(patch)
    #
    # return ivy.stack(patches, axis=-1)
