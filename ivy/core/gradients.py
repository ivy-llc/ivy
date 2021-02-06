"""
Collection of gradient Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


def variable(object_in, f=None):
    """
    Creates a variable, which supports gradient computation.

    :param object_in: An ivy array.
    :type object_in: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An ivy variable, supporting gradient computation.
    """
    return _get_framework(object_in, f=f).variable(object_in)


def execute_with_gradients(func, xs, f=None):
    """
    Call function func with input of xs variables, and return func first output y, the gradients [dy/dx for x in xs],
    and any other function outputs after the returned y value

    :param func: Function for which we compute the gradients of the output with respect to xs input.
    :type func: function
    :param xs: Variables for which to compute the function gradients with respective to.
    :type xs: sequence of variables
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: the function first output y, the gradients [dy/dx for x in xs], and any other extra function outputs
    """
    return _get_framework(f.array(xs[0]), f=f).execute_with_gradients(func, xs)


def gradient_descent_update(ws, dcdws, lr, f=None):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws].

    :param ws: Weights of the function to be updated.
    :type ws: sequence of variables
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: sequence of arrays
    :param lr: Learning rate, the rate at which the weights should be updated relative to the gradient.
    :type lr: float
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The new function weights ws_new, following the gradient descent updates.
    """
    return _get_framework(f.array(ws[0]), f=f).gradient_descent_update(ws, dcdws, lr)


def stop_gradient(x, f=None):
    """
    Stops gradient computation.

    :param x: Array for which to stop the gradient.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The same array x, but with no gradient information.
    """
    return _get_framework(x, f=f).stop_gradient(x)
