"""
Collection of gradient Ivy functions.
"""

# local
import ivy as _ivy
from ivy.framework_handler import current_framework as _cur_framework


# Variables #
# ----------#

def variable(x, f=None):
    """
    Creates a variable, which supports gradient computation.

    :param x: An ivy array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An ivy variable, supporting gradient computation.
    """
    return _cur_framework(x, f=f).variable(x)


def is_variable(x, exclusive=False, f=None):
    """
    Determines whether the input is a variable or not.

    :param x: An ivy array.
    :type x: array
    :param exclusive: Whether to check if the data type is exclusively a variable, rather than an array.
                      For frameworks like JAX that do not have exclusive variable types, the function will always return
                      False if this flag is set, otherwise the check is the same for general arrays. Default is False.
    :type exclusive: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, true if x is a trainable variable, false otherwise.
    """
    return _cur_framework(x, f=f).is_variable(x, exclusive)


def variable_data(x, f=None):
    """
    Some backends wrap arrays in a dedicated variable class. For those frameworks, this function returns that wrapped
    array. For frameworks which do not have a dedicated variable class, the function returns the data passed in.

    :param x: An ivy variable.
    :type x: variable
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The internal data stored by the variable
    """
    return _cur_framework(x, f=f).variable_data(x)


def inplace_update(x, val, f=None):
    """
    Perform in-place update for the input variable.

    :param x: The variable to update.
    :type x: variable
    :param val: The array to update the variable with.
    :type val: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The variable following the in-place update.
    """
    return _cur_framework(x, f=f).inplace_update(x, val)


def inplace_decrement(x, val, f=None):
    """
    Perform in-place decrement for the input variable.

    :param x: The variable to decrement.
    :type x: variable
    :param val: The array to decrement the variable with.
    :type val: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The variable following the in-place decrement.
    """
    return _cur_framework(x, f=f).inplace_decrement(x, val)


def inplace_increment(x, val, f=None):
    """
    Perform in-place increment for the input variable.

    :param x: The variable to increment.
    :type x: variable
    :param val: The array to increment the variable with.
    :type val: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The variable following the in-place increment.
    """
    return _cur_framework(x, f=f).inplace_increment(x, val)


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
    return _cur_framework(x, f=f).stop_gradient(x, preserve_type)


# AutoGrad #
# ---------#

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
    return _cur_framework(None, f=f).execute_with_gradients(func, xs, retain_grads)


# Optimizer Steps #
# ----------------#

def adam_step(dcdws, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7):
    """
    Compute adam step delta, given the derivatives of some cost c with respect to ws, using ADAM update.
    `[reference] <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: container of arrays
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
    :return: The adam step delta.
    """
    step = float(_ivy.to_scalar(step))
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw[kc] + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws ** 2
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw[kc] + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)
    return mw.map(lambda m, kc: (alpha * m / (vw[kc] ** 0.5 + epsilon))), mw, vw


# Optimizer Updates #
# ------------------#

def optimizer_update(ws, effective_grads, lr, inplace=True, stop_gradients=True):
    """
    Update weights ws of some function, given the true or effective derivatives of some cost c with respect to ws,
    [dc/dw for w in ws].

    :param ws: Weights of the function to be updated.
    :type ws: Ivy container
    :param effective_grads: Effective gradients of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type effective_grads: Ivy container
    :param lr: Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    :type lr: float or container of layer-wise rates.
    :param inplace: Whether to perform the operation inplace, for backends which support inplace variable updates,
                    and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
                    computation graph (i.e. higher order optimization), then this should be set to False.
                    Default is True.
    :type inplace: bool, optional
    :param stop_gradients: Whether to stop the gradients of the variables after each gradient step. Default is True.
    :type stop_gradients: bool, optional
    :return: The new function weights ws_new, following the optimizer updates.
    """
    layerwise_lr = isinstance(lr, _ivy.Container)
    deltas = effective_grads.map(lambda eff_grad, kc: ((lr[kc] if layerwise_lr else lr) * eff_grad))
    if inplace:
        ws = ws.map(lambda w, kc: _ivy.inplace_decrement(w, deltas[kc]))
    else:
        ws = ws.map(lambda w, kc: -deltas[kc] + w)
    if stop_gradients:
        return ws.stop_gradients(preserve_type=True)
    return ws


def gradient_descent_update(ws, dcdws, lr, inplace=True, stop_gradients=True):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws].

    :param ws: Weights of the function to be updated.
    :type ws: Ivy container
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: Ivy container
    :param lr: Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    :type lr: float or container of layer-wise rates.
    :param inplace: Whether to perform the operation inplace, for backends which support inplace variable updates,
                    and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
                    computation graph (i.e. higher order optimization), then this should be set to False.
                    Default is True.
    :type inplace: bool, optional
    :param stop_gradients: Whether to stop the gradients of the variables after each gradient step. Default is True.
    :type stop_gradients: bool, optional
    :return: The new function weights ws_new, following the gradient descent updates.
    """
    return optimizer_update(ws, dcdws, lr, inplace, stop_gradients)


def lars_update(ws, dcdws, lr, decay_lambda=0, inplace=True, stop_gradients=True):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws],
    by applying Layerwise Adaptive Rate Scaling (LARS) method.

    :param ws: Weights of the function to be updated.
    :type ws: Ivy container
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: Ivy container
    :param lr: Learning rate, the rate at which the weights should be updated relative to the gradient.
    :type lr: float
    :param decay_lambda: The factor used for weight decay. Default is zero.
    :type decay_lambda: float
    :param inplace: Whether to perform the operation inplace, for backends which support inplace variable updates,
                    and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
                    computation graph (i.e. higher order optimization), then this should be set to False.
                    Default is True.
    :type inplace: bool, optional
    :param stop_gradients: Whether to stop the gradients of the variables after each gradient step. Default is True.
    :type stop_gradients: bool, optional
    :return: The new function weights ws_new, following the LARS updates.
    """
    ws_norm = ws.vector_norm()
    lr = _ivy.stable_divide(ws_norm * lr, dcdws.vector_norm())
    if decay_lambda > 0:
        lr /= (ws_norm * decay_lambda)
    return gradient_descent_update(ws, dcdws, lr, inplace, stop_gradients)


def adam_update(ws, dcdws, lr, mw_tm1, vw_tm1, step, beta1=0.9, beta2=0.999, epsilon=1e-7, inplace=True,
                stop_gradients=True):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, using ADAM update.
    `[reference] <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    :param ws: Weights of the function to be updated.
    :type ws: container of variables
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: container of arrays
    :param lr: Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    :type lr: float or container of layer-wise rates.
    :param mw_tm1: running average of the gradients, from the previous time-step.
    :type mw_tm1: container of arrays
    :param vw_tm1: running average of second moments of the gradients, from the previous time-step.
    :type vw_tm1: container of arrays
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
    :param stop_gradients: Whether to stop the gradients of the variables after each gradient step. Default is True.
    :type stop_gradients: bool, optional
    :return: The new function weights ws_new, and also new mw and vw, following the adam updates.
    """
    effective_grads, mw, vw = adam_step(dcdws, mw_tm1, vw_tm1, step, beta1, beta2, epsilon)
    return optimizer_update(ws, effective_grads, lr, inplace, stop_gradients), mw, vw


def lamb_update(ws, dcdws, lr, mw_tm1, vw_tm1, step, beta1=0.9, beta2=0.999, epsilon=1e-7, max_trust_ratio=10,
                decay_lambda=0, inplace=True, stop_gradients=True):
    """
    Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws],
    by applying LAMB method.

    :param ws: Weights of the function to be updated.
    :type ws: container of variables
    :param dcdws: Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    :type dcdws: container of arrays
    :param lr: Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    :type lr: float or container of layer-wise rates.
    :param mw_tm1: running average of the gradients, from the previous time-step.
    :type mw_tm1: container of arrays
    :param vw_tm1: running average of second moments of the gradients, from the previous time-step.
    :type vw_tm1: container of arrays
    :param step: training step
    :type step: int
    :param beta1: gradient forgetting factor
    :type beta1: float
    :param beta2: second moment of gradient forgetting factor
    :type beta2: float
    :param epsilon: divisor during adam update, preventing division by zero
    :type epsilon: float
    :param max_trust_ratio: The maximum value for the trust ratio. Default is 10.
    :type max_trust_ratio: float, optional
    :param decay_lambda: The factor used for weight decay. Default is zero.
    :type decay_lambda: float
    :param inplace: Whether to perform the operation inplace, for backends which support inplace variable updates,
                    and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
                    computation graph (i.e. higher order optimization), then this should be set to False.
                    Default is True.
    :type inplace: bool, optional
    :param stop_gradients: Whether to stop the gradients of the variables after each gradient step. Default is True.
    :type stop_gradients: bool, optional
    :return: The new function weights ws_new, following the LARS updates.
    """
    r1 = ws.vector_norm()
    eff_grads, mw, vw = adam_step(dcdws, mw_tm1, vw_tm1, step, beta1, beta2, epsilon)
    if decay_lambda > 0:
        r2 = (eff_grads + decay_lambda*ws).norm()
    else:
        r2 = eff_grads.vector_norm()
    r = _ivy.stable_divide(r1, r2).minimum(max_trust_ratio)
    lr = r * lr
    return optimizer_update(ws, eff_grads, lr, inplace, stop_gradients), mw, vw
