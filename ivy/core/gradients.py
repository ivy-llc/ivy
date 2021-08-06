"""
Collection of gradient Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


def variable(x, f=None):
    """
    Creates a variable, which supports gradient computation.

    :param x: An ivy array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An ivy variable, supporting gradient computation.
    """
    return _get_framework(x, f=f).variable(x)


def is_variable(x, f=None):
    """
    Determines whether the input is a variable or not.

    :param x: An ivy array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, true if x is a trainable variable, false otherwise.
    """
    return _get_framework(x, f=f).is_variable(x)


def execute_with_gradients(func, xs, retain_grads=False, f=None):
    """
    Call function func with input of xs variables, and return func first output y, the gradients [dy/dx for x in xs],
    and any other function outputs after the returned y value

    :param func: Function for which we compute the gradients of the output with respect to xs input.
    :type func: function
    :param xs: Variables for which to compute the function gradients with respective to.
    :type xs: sequence of variables
    :param retain_grads: Whether to retain the gradients of the returned values.
    :type retain_grads: bool
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: the function first output y, the gradients [dy/dx for x in xs], and any other extra function outputs
    """
    return _get_framework(None, f=f).execute_with_gradients(func, xs, retain_grads)


def gradient_descent_update(ws, dcdws, lr, inplace=True, f=None):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws].

    :param ws: Weights of the function to be updated.
    :type ws: Ivy container
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: Ivy container
    :param lr: Learning rate, the rate at which the weights should be updated relative to the gradient.
    :type lr: float
    :param inplace: Whether to perform the operation inplace, for backends which support inplace variable updates,
                    and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
                    computation graph (i.e. higher order optimization), then this should be set to False.
                    Default is True.
    :type inplace: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The new function weights ws_new, following the gradient descent updates.
    """
    return _get_framework(None, f=f).gradient_descent_update(ws, dcdws, lr, inplace)


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7, inplace=True, f=None):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, using ADAM update.
    `[reference] <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    :param ws: Weights of the function to be updated.
    :type ws: container of variables
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: container of arrays
    :param lr: Learning rate, the rate at which the weights should be updated relative to the gradient.
    :type lr: float
    :param mw: running average of the gradients
    :type mw: container of arrays
    :param vw: running average of second moments of the gradients
    :type vw: container of arrays
    :param step: training step
    :type step: int
    :param beta1: gradient forgetting factor
    :type beta1: float
    :param beta2: second moment of gradient forgetting factor
    :type beta2: float
    :param epsilon: divisor during adam update, preventing division by zero
    :type epsilon: float
    :param inplace: Whether to perform the operation inplace, for backends which support inplace variable updates,
                    and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
                    computation graph (i.e. higher order optimization), then this should be set to False.
                    Default is True.
    :type inplace: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The new function weights ws_new, and also new mw and vw, following the gradient descent updates.
    """
    return _get_framework(None, f=f).adam_update(ws, dcdws, lr, mw, vw, step, beta1, beta2, epsilon, inplace)


def stop_gradient(x, preserve_type=True, f=None):
    """
    Stops gradient computation.

    :param x: Array for which to stop the gradient.
    :type x: array
    :param preserve_type: Whether to preserve the input type (ivy.Variable or ivy.Array),
                          otherwise an array is always returned. Default is True.
    :param preserve_type: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The same array x, but with no gradient information.
    """
    return _get_framework(x, f=f).stop_gradient(x, preserve_type)
