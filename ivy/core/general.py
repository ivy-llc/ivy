"""
Collection of general Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev_str=None, f=None):
    """
    Creates an array.
    
    :param object_in: An array_like object, which exposes the array interface,
            an object whose __array__ method returns an array, or any (nested) sequence.
    :type object_in: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype_str: data-type string, optional
    :param dev_str: device string on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array object satisfying the specified requirements, in the form of the selected framework.
    """
    return _get_framework(object_in, f=f).array(object_in, dtype_str, dev_str)


def to_numpy(x, f=None):
    """
    Converts array into a numpy array.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A numpy array.
    """
    return _get_framework(x, f=f).to_numpy(x)


def to_list(x, f=None):
    """
    Creates a (possibly nested) list from input array.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A list representation of the input array.
    """
    return _get_framework(x, f=f).to_list(x)


def shape(x, as_array=False, f=None):
    """
    Returns the shape of the array x.

    :param x: Input array to infer the shape of.
    :type x: array
    :param as_array: Whether to return the shape as a array, default False.
    :type as_array: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Shape of the array
    """
    return _get_framework(x, f=f).shape(x, as_array)


def get_num_dims(x, as_array=False, f=None):
    """
    Returns the number of dimensions of the array x.

    :param x: Input array to infer the number of dimensions for.
    :type x: array
    :param as_array: Whether to return the shape as a array, default False.
    :type as_array: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Shape of the array
    """
    return _get_framework(x, f=f).get_num_dims(x, as_array)


def minimum(x, y, f=None):
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    :param x: Input array containing elements to minimum threshold.
    :type x: array
    :param y: Tensor containing minimum values, must be broadcastable to x.
    :type y: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the elements of x, but clipped to not exceed the y values.
    """
    return _get_framework(x, f=f).minimum(x, y)


def maximum(x, y, f=None):
    """
    Returns the max of x and y (i.e. x > y ? x : y) element-wise.

    :param x: Input array containing elements to maximum threshold.
    :type x: array
    :param y: Tensor containing maximum values, must be broadcastable to x.
    :type y: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the elements of x, but clipped to not be lower than the y values.
    """
    return _get_framework(x, f=f).maximum(x, y)


def clip(x, x_min, x_max, f=None):
    """
    Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges (element-wise).
    For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    :param x: Input array containing elements to clip.
    :type x: array
    :param x_min: Minimum value.
    :type x_min: scalar
    :param x_max: Maximum value.
    :type x_max: scalar
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the elements of x, but where values < x_min are replaced with x_min,
                and those > x_max with x_max.
    """
    return _get_framework(x, f=f).clip(x, x_min, x_max)


# noinspection PyShadowingBuiltins
def round(x, f=None):
    """
    Rounds the values of an array to the nearest integer, element-wise.

    :param x: Input array containing elements to round.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements rounded to integers.
    """
    return _get_framework(x, f=f).round(x)


def floormod(x, y, f=None):
    """
    Returns element-wise remainder of division.

    :param x: Input array to floormod.
    :type x: array
    :param y: Denominator input for floormod.
    :type y: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements floor modded.
    """
    return _get_framework(x, f=f).floormod(x, y)


def floor(x, f=None):
    """
    Returns element-wise largest integer not greater than x.

    :param x: Input array to floor.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements floored to integers.
    """
    return _get_framework(x, f=f).floor(x)


def ceil(x, f=None):
    """
    Returns element-wise smallest integer not less than x.

    :param x: Input array to ceil.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements ceiled to integers.
    """
    return _get_framework(x, f=f).ceil(x)


# noinspection PyShadowingBuiltins
def abs(x, f=None):
    """
    Returns the absolute value of each element in x.

    :param x: Input array containing elements to absolute value.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array of the same shape as input array a, with all values now positive.
    """
    return _get_framework(x, f=f).abs(x)


def argmax(x, axis=0, f=None):
    """
    Returns the index with the largest value across axes of an array.

    :param x: Input array containing elements to argmax.
    :type x: array
    :param axis: Axis to perform the argmax, default is 0.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor containing the indices of the maximum values across the specified axis.
    """
    return _get_framework(x, f=f).argmax(x, axis)


def argmin(x, axis=0, f=None):
    """
    Returns the index with the smallest value across axes of an array.

    :param x: Input array containing elements to argmin.
    :type x: array
    :param axis: Axis to perform the argmin, default is 0.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor containing the indices of the minimum values across the specified axis.
    """
    return _get_framework(x, f=f).argmin(x, axis)


# noinspection PyShadowingNames
def cast(x, dtype_str, f=None):
    """
    Casts an array to a specified type.

    :param x: Input array containing elements to cast.
    :type x: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
            If not given, then the type will be determined as the minimum type required to hold the objects in the
            sequence.
    :type dtype_str: data-type string
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array of the same shape as input array a, with data type given by dtype_str.
    """
    return _get_framework(x, f=f).cast(x, dtype_str)


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype_str=None, dev_str=None, f=None):
    """
    Returns evenly spaced values within a given interval, with the spacing being specified.

    Values are generated within the half-open interval [start, stop) (in other words, the interval including start but
    excluding stop). For integer arguments the function is equivalent to the Python built-in range function,
    but returns an array in the chosen ml_framework rather than a list.

    See :math:`linspace` for a certain number of evenly spaced values in an interval.

    :param stop: End of interval. The interval does not include this value, except in some cases where step is not an
                integer and floating point round-off affects the length of out.
    :type stop: number
    :param start: Start of interval. The interval includes this value. The default start value is 0.
    :type start: number, optional
    :param step: Spacing between values. For any output out, this is the distance between two adjacent values,
                    out[i+1] - out[i]. The default step size is 1. If step is specified as a position argument,
                    start must also be given.
    :type step: number, optional
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of evenly spaced values.

            For floating point arguments, the length of the result is ceil((stop - start)/step).
            Because of floating point overflow, this rule may result in the last element of out being greater than stop.
    """
    return _get_framework(f=f).arange(stop, start, step, dtype_str, dev_str)


# noinspection PyShadowingNames
def linspace(start, stop, num, axis=None, dev_str=None, f=None):
    """
    Generates a certain number of evenly-spaced values in an interval along a given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in an interval.

    :param start: First entry in the range.
    :type start: array
    :param stop: Final entry in the range.
    :type stop: array
    :param num: Number of values to generate.
    :type num: int
    :param axis: Axis along which the operation is performed.
    :type axis: int
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of evenly-spaced values.
    """
    return _get_framework(start, f=f).linspace(start, stop, num, axis, dev_str)


def concatenate(xs, axis=-1, f=None):
    """
    Casts an array to a specified type.

    :param xs: The input arrays must have the same shape, except in the dimension corresponding to axis
                        (the first, by default).
    :type xs: sequence of arrays
    :param axis: The axis along which the arrays will be joined. Default is -1.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The concatenated array.
    """
    return _get_framework(xs[0], f=f).concatenate(xs, axis)


def flip(x, axis=None, batch_shape=None, f=None):
    """
    Reverses the ord of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.

    :param x: Input array.
    :type x: array
    :param axis: Axis or axes along which to flip over. The default, axis=None, will flip over all axes.
    :type axis: None or int or sequence of ints, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the entries of axis reversed.
    """
    return _get_framework(x, f=f).flip(x, axis, batch_shape)


def stack(xs, axis=0, f=None):
    """
    Joins a sequence of arrays along a new axis.
    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if axis=0, it will be the first dimension and if axis=-1, it will be the last dimension.

    :param xs: Input arrays, each array must have the same shape.
    :type xs: sequence of arrays
    :param axis: The axis in the result array along which the input arrays are stacked.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The stacked array has one more dimension than the input arrays.
    """
    return _get_framework(xs[0], f=f).stack(xs, axis)


def unstack(x, axis, f=None):
    """
    Unpacks the given dimension of a rank-R array into rank-(R-1) arrays.

    :param x: Input array to unstack.
    :type x: array
    :param axis: Axis for which to unpack the array.
    :type axis: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: List of arrays, unpacked along specified dimensions.
    """
    return _get_framework(x, f=f).unstack(x, axis)


def split(x, num_sections=None, axis=0, f=None):
    """
    Splits an array into multiple sub-arrays.

    :param x: Tensor to be divided into sub-arrays.
    :type x: array
    :param num_sections: Number of equal arrays to divide the array into, along the given axis.
                         Size of axis dimension must be multiple of num_sections.
                         Default is to divide into as many arrays as the axis dimension.
    :type num_sections: int, optional
    :param axis: The axis along which to split, default is 0.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A list of sub-arrays.
    """
    return _get_framework(x, f=f).split(x, num_sections, axis)


def tile(x, reps, f=None):
    """
    Constructs an array by repeating x the number of times given by reps.

    :param x: Input array.
    :type x: array
    :param reps: The number of repetitions of x along each axis.
    :type reps: sequence
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The tiled output array.
    """
    return _get_framework(x, f=f).tile(x, reps)


def constant_pad(x, pad_width, value=0, x_shape=None, f=None):
    """
    Pads an array with a constant value.

    :param x: Input array to pad.
    :type x: array
    :param pad_width: Number of values padded to the edges of each axis.
                      Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    :type pad_width: sequence of tuples of ints
    :param value: The constant value to pad the array with.
    :type value: float or int, default zero
    :param x_shape: Shape of x. Required for mxnet symbolic.
    :type x_shape: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Padded array of rank equal to x with shape increased according to pad_width.
    """
    return _get_framework(x, f=f).constant_pad(x, pad_width, value, x_shape)


def zero_pad(x, pad_width, x_shape=None, f=None):
    """
    Pads an array with zeros.

    :param x: Input array to pad.
    :type x: array
    :param pad_width: Number of values padded to the edges of each axis.
                      Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    :type pad_width: sequence of tuples of ints
    :param x_shape: Shape of x. Required for mxnet symbolic.
    :type x_shape: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Padded array of rank equal to x with shape increased according to pad_width.
    """
    return _get_framework(x, f=f).zero_pad(x, pad_width, x_shape)


def swapaxes(x, axis0, axis1, f=None):
    """
    Interchange two axes of an array.

    :param x: Input array.
    :type x: array
    :param axis0: First axis to be swapped.
    :type axis0: int
    :param axis1: Second axis to be swapped.
    :type axis1: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: x with its axes permuted.
    """
    return _get_framework(x, f=f).swapaxes(x, axis0, axis1)


def transpose(x, axes=None, f=None):
    """
    Permutes the dimensions of an array.

    :param x: Input array.
    :type x: array
    :param axes: By default, reverse the dimensions, otherwise permute the axes according to the values given.
    :type axes: sequence of ints of length N
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: x with its axes permuted.
    """
    return _get_framework(x, f=f).transpose(x, axes)


def expand_dims(x, axis, f=None):
    """
    Expands the shape of an array.
    Inserts a new axis that will appear at the axis position in the expanded array shape.

    :param x: Input array.
    :type x: array
    :param axis: Position in the expanded axes where the new axis is placed.
    :type axis: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: array with the number of dimensions increased by onearray
    """
    return _get_framework(x, f=f).expand_dims(x, axis)


def where(condition, x1, x2, condition_shape=None, x_shape=None, f=None):
    """
    Returns elements chosen from x or y depending on condition.

    :param condition: Where True, yield x1, otherwise yield x2.
    :type condition: bool array
    :param x1: values from which to choose when condition is True.
    :type x1: array
    :param x2: values from which to choose when condition is False.
    :type x2: array
    :param condition_shape: Shape of condition. Required for mxnet symbolic.
    :type condition_shape: sequence of ints
    :param x_shape: Shape of x1 and x2. Required for mxnet symbolic.
    :type x_shape: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with elements from x1 where condition is True, and elements from x2 elsewhere.
    """
    return _get_framework(x1, f=f).where(condition, x1, x2, condition_shape, x_shape)


def indices_where(x, f=None):
    """
    Returns indices or true elements in an input boolean array.

    :param x: Boolean array, for which indices are desired.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Indices for where the boolean array is True.
    """
    return _get_framework(x, f=f).indices_where(x)


def reshape(x, newshape, f=None):
    """
    Gives a new shape to an array without changing its data.

    :param x: Tensor to be reshaped.
    :type x: array
    :param newshape: The new shape should be compatible with the original shape. One shape dimension can be -1.
                        In this case, the value is inferred from the length of the array and remaining dimensions.
    :type newshape: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Reshaped array.
    """
    return _get_framework(x, f=f).reshape(x, newshape)


def squeeze(x, axis=None, f=None):
    """
    Removes a single-dimensional entry from the shape of an array.

    :param x: Input data.
    :type x: array
    :param axis: Index for one of the single-dimensional entries in the shape.
                 If an axis is selected with shape entry greater than one, an error is raised.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array, but with all (axis=None) or one (axis is int) of the dimensions of length 1 removed.
    """
    return _get_framework(x, f=f).squeeze(x, axis)


# noinspection PyShadowingNames
def zeros(shape, dtype_str='float32', dev_str=None, f=None):
    """
    Return a new array of given shape and type, filled with zeros.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: sequence of ints
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the given shape and dtype_str.
    """
    return _get_framework(f=f).zeros(shape, dtype_str, dev_str)


# noinspection PyShadowingNames
def zeros_like(x, dtype_str=None, dev_str=None, f=None):
    """
    Returns an array of zeros with the same shape and type as x, unless dtype_str provided which overrides.

    :param x: The shape and data-type of x define these same attributes of the returned array.
    :type x: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                    If not given, then the type of the original array is used.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype_str provided which overrides.
    """
    return _get_framework(x, f=f).zeros_like(x, dtype_str, dev_str)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev_str=None, f=None):
    """
    Returns a new array of given shape and type, filled with ones.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: sequence of ints
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of ones with the given shape and dtype_str.
    """
    return _get_framework(f=f).ones(shape, dtype_str, dev_str)


# noinspection PyShadowingNames
def ones_like(x, dtype_str=None, dev_str=None, f=None):
    """
    Returns an array of ones with the same shape and type as x, unless dtype_str provided which overrides.

    :param x: The shape and data-type of a define these same attributes of the returned array.
    :type x: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                    If not given, then the type of the original array is used.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype_str provided which overrides.
    """
    return _get_framework(x, f=f).ones_like(x, dtype_str, dev_str)


# noinspection PyShadowingNames
def one_hot(indices, depth, dev_str=None, f=None):
    """
    Returns a one-hot array
    :param indices: Indices for where the ones should be scattered *[batch_shape, dim]*
    :type indices: array
    :param depth: Scalar defining the depth of the one-hot dimension.
    :type depth: int
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype provided which overrides.
    """
    return _get_framework(indices, f=f).one_hot(indices, depth, dev_str)


def cross(x1, x2, f=None):
    """
    Returns the cross product of two (arrays of) vectors in R^3.
    The cross product of x1 and x2 in R^3 is a vector perpendicular to both x1 and x2.
    If x1 and x2 are arrays of vectors, the vectors are defined by the last axis of x1 and x2 by default which must have
    dimension 3.

    :param x1: Components of the first vector(s).
    :type x1: array
    :param x2: Components of the second vector(s).
    :type x2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Vector cross product(s).
    """
    return _get_framework(x1, f=f).cross(x1, x2)


def matmul(x1, x2, batch_shape=None, f=None):
    """
    Computes the matrix product of two arrays x1 and x2.

    :param x1: Input array 1.
    :type x1: array
    :param x2: Input array 2.
    :type x2: array
    :param batch_shape: Shape of batch. Required for mxnet symbolic.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The matrix product of the input arrays.
    """
    return _get_framework(x1, f=f).matmul(x1, x2, batch_shape)


def cumsum(x, axis=0, f=None):
    """
    Returns the cumulative sum of the elements along a given axis.

    :param x: Input array.
    :type x: array
    :param axis: Axis along which the cumulative sum is computed. By default 0.
    :type axis: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Input array with cumulatively summed elements along axis.
    """
    return _get_framework(x, f=f).cumsum(x, axis)


# noinspection PyShadowingNames
def identity(n, dtype_str='float32', batch_shape=None, dev_str=None, f=None):
    """
    Returns the identity array.
    The identity array is a square array with ones on the main diagonal.

    :param n: Number of rows (and columns) in n x n output.
    :type n: int
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype_str: data-type string, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: n x n array of type dtype_str, with its main diagonal set to one, and all other elements 0.
    """
    return _get_framework(f=f).identity(n, dtype_str, batch_shape, dev_str)


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size, reduction='sum', dev_str=None, f=None):
    """
    Scatter flat updates into a new flat array according to flat indices.

    :param indices: Indices for the new values to occupy.
    :type indices: array
    :param updates: Values for the new array to hold.
    :type updates: array
    :param size: The size of the result.
    :type size: int
    :param reduction: The reduction method for the scatter, one of 'sum', 'min' or 'max'
    :type reduction: str
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as updates if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values scattered at the indices.
    """
    return _get_framework(indices, f=f).scatter_flat(indices, updates, size, reduction, dev_str)


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, num_idx_dims=None, reduction='sum', dev_str=None, f=None):
    """
    Scatter updates into a new array according to indices.

    :param indices: Indices for the new values to occupy.
    :type indices: array
    :param updates: Values for the new array to hold.
    :type updates: array
    :param shape: The shape of the result.
    :type shape: sequence of ints
    :param num_idx_dims: Number of dimensions for indices array. Required for MXNet symbolic.
    :type num_idx_dims: int
    :param reduction: The reduction method for the scatter, one of 'sum', 'min' or 'max'
    :type reduction: str
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as updates if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values scattered at the indices.
    """
    return _get_framework(indices, f=f).scatter_nd(indices, updates, shape, num_idx_dims, reduction, dev_str)


# noinspection PyShadowingNames
def gather_flat(params, indices, dev_str=None, f=None):
    """
    Gather slices from flat params into a flat array with size specified by indices size.

    :param params: The array from which to gather values.
    :type params: array
    :param indices: Index array.
    :type indices: array
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values gathered at the indices.
    """
    return _get_framework(params, f=f).gather_flat(params, indices, dev_str)


# noinspection PyShadowingNames
def gather_nd(params, indices, indices_shape=None, dev_str=None, f=None):
    """
    Gather slices from params into a array with shape specified by indices.

    :param params: The array from which to gather values.
    :type params: array
    :param indices: Index array.
    :type indices: array
    :param indices_shape: Shape of the indices, required for mxnet symbolic mode.
    :type indices_shape: sequence of ints.
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values gathered at the indices.
    """
    return _get_framework(params, f=f).gather_nd(params, indices, indices_shape, dev_str)


def dev(x, f=None):
    """
    Get the native device handle for input array x.

    :param x: Tensor for which to get the device handle.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device handle for the array, in native framework format.
    """
    return _get_framework(x, f=f).dev(x)


def dev_to_str(dev_in, f=None):
    """
    Convert native data type to string representation.

    :param dev_in: The device handle to convert to string.
    :type dev_in: device handle
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'cuda:0'.
    """
    return _get_framework(None, f=f).dev_to_str(dev_in)


def dev_str(x, f=None):
    """
    Get the device string for input array x.

    :param x: Tensor for which to get the device string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string for the array, e.g. 'cuda:0', 'cuda:1', 'cpu' etc..
    """
    return _get_framework(x, f=f).dev_str(x)


def gpu_is_available(f=None):
    """
    Determine whether a GPU is available to use, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, as to whether a gpu is available.
    """
    return _get_framework(f=f).gpu_is_available()


def tpu_is_available(f=None):
    """
    Determine whether a TPU is available to use, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, as to whether a tpu is available.
    """
    return _get_framework(f=f).tpu_is_available()


def dtype(x, f=None):
    """
    Get the data type for input array x.

    :param x: Tensor for which to get the data type.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Data type of the array
    """
    return _get_framework(x, f=f).dtype(x)


def dtype_to_str(dtype_in, f=None):
    """
    Convert native data type to string representation.

    :param dtype_in: The data type to convert to string.
    :type dtype_in: data type
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'float32'.
    """
    return _get_framework(None, f=f).dtype_to_str(dtype_in)


def dtype_str(x, f=None):
    """
    Get the data type string for input array x.

    :param x: Tensor for which to get the data type string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'float32'.
    """
    return _get_framework(None, f=f).dtype_str(x)


def compile_fn(func, dynamic=True, example_inputs=None, f=None):
    """
    Provide a function which should be compiled, for faster inference.
    The handle to the newly compiled function is returned.

    :param func: Function to be compiled.
    :type func: function
    :param dynamic: Whether to compile all conditional branches, regardless of inputs during first invocation.
    :type dynamic: bool, default True
    :param example_inputs: Example of inputs to the function to be compiled.
                            Required for torch in non-dynamic mode, unused by other frameworks.
    :type example_inputs: single input of tuple of inputs.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The handle to the newly compiled function.
    """
    return _get_framework(example_inputs, f=f).compile_fn(func, dynamic, example_inputs)
