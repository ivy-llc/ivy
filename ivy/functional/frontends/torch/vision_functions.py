def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False):
    input_shape = ivy.shape(input)
    ivy.assertions.check_equal(
        ivy.get_num_dims(input),
        3,
        message="pixel_shuffle expects 3D, 4D or 5D input, but got input with sizes "
                + str(input_shape),)

    ivy.assertions.check_equal(
        ivy.get_num_dims(input),
        4,
        message="pixel_shuffle expects 3D, 4D or 5D input, but got input with sizes "
                + str(input_shape), )

    ivy.assertions.check_equal(
        ivy.get_num_dims(input),
        5,
        message="pixel_shuffle expects 3D, 4D or 5D input, but got input with sizes "
                + str(input_shape), )