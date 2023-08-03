# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def channel_shuffle(x, groups, data_format="NCHW", name=None):
    # ToDo ivy.channel_shuffle  # 21235
    # Issue Link: https://github.com/unifyai/ivy/issues/21235
    # This implementation will be simplified when ivy.channel_shuffle is implemented
    if len(x.shape) != 4:
        raise ValueError(
            "Input x should be 4D tensor, but received x with the shape of {}".format(
                x.shape
            )
        )

    if not isinstance(groups, int):
        raise TypeError("groups must be int type")

    if groups <= 0:
        raise ValueError("groups must be positive")
        # assert (channels % groups == 0)

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'."
            "But recevie Attr(data_format): {} ".format(data_format)
        )

    if data_format == "NCHW":
        n, c, h, w = x.shape
        if c % groups != 0:
            raise ValueError('channels must be divisible by groups')
        new_shape = (n, groups, c // groups, h, w)
        result = ivy.reshape(x, ivy.Shape(new_shape))
        result = ivy.permute_dims(result, (0, 2, 1, 3, 4))
        oshape = [n, c, h, w]
        result = ivy.reshape(result,ivy.Shape(oshape))
        return result
    else:
        n, h, w, c = x.shape
        if c % groups != 0:
            raise ValueError('channels must be divisible by groups')
        new_shape = (n, h, w, groups, c // groups)
        result = ivy.reshape(x, ivy.Shape(new_shape))
        result = ivy.permute_dims(result,(0, 1, 2, 4, 3))
        oshape = [n, h, w, c]
        result = ivy.reshape(result, ivy.Shape(oshape))
        return result

