



#local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def cos(x: ivy.Array) -> ivy.Array:
    """
    Computes trigonometric cosine element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :type x: array of floats
    :return: The cosine of x element-wise.
    """
    return _cur_framework(x).cos(x)
