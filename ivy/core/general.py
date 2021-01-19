"""
Collection of general Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None, f=None):
    """
    Creates an array.
    
    :param object_in: An array_like object, which exposes the array interface,
            an object whose __array__ method returns an array, or any (nested) sequence.
    :type object_in: array_like
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype_str: data-type string, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array object satisfying the specified requirements, in the form of the selected framework.
    """
    return _get_framework(object_in, f=f).array(object_in, dtype_str, dev)
