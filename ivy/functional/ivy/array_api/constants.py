# local
from ivy.framework_handler import current_framework as _cur_framework

def e(f=None):
    """
    IEEE 754 floating-point representation of Euler's constant.
    """
    return _cur_framework(None,f).e


def pi(f=None):
    """
    IEEE 754 floating-point representation of the mathematical constant e
    """
    return _cur_framework(None,f).pi
