# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def bitwise_and(x1: ivy.Array, x2: ivy.Array, /) -> ivy.Array:
    """
    Computes the bitwise AND of the underlying binary representation of each element x1_i of the input array x1 with
    the respective element x2_i of the input array x2.

    :param x1: first input array. Should have an integer or boolean data type.
    :param x2: second input array. Must be compatible with x1 (see Broadcasting). Should have an integer or
                boolean data type.
    :return:  an array containing the element-wise results. The returned array must have a data type determined
              by Type Promotion Rules.
    """
    return _cur_framework(x1, x2).bitwise_and(x1, x2)
