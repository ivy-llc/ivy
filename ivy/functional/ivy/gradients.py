"""
Collection of gradient Ivy functions.
"""

# local
import ivy
import ivy as _ivy
from typing import Union
from ivy.framework_handler import current_framework as _cur_framework

# Extra #
# ------#

with_grads_stack = list()


class GradientTracking:

    # noinspection PyShadowingNames
    def __init__(self, with_grads):
        self._with_grads = with_grads

    def __enter__(self):
        set_with_grads(self._with_grads)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_with_grads()
        return self


# Gradient Mode #

# noinspection PyShadowingNames
def with_grads(with_grads=None):
    """
    Return the input dev if provided, otherwise return the global default device.
    Parameters
    ----------
    with_grads :
         (Default value = None)

    Returns
    -------
    type
        

    """
    if _ivy.exists(with_grads):
        assert with_grads in [True, False]
        return with_grads
    global with_grads_stack
    if not with_grads_stack:
        with_grads_stack = [True]
    return with_grads_stack[-1]


# noinspection PyShadowingNames
def set_with_grads(with_grads):
    """

    Parameters
    ----------
    with_grads :
        

    Returns
    -------

    """
    assert with_grads in [True, False]
    global with_grads_stack
    with_grads_stack.append(with_grads)


def unset_with_grads():
    global with_grads_stack
    if with_grads_stack:
        with_grads_stack.pop(-1)


# Variables #

def variable(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Variable:
    """Creates a variable, which supports gradient computation.

    Parameters
    ----------
    x : array
        An ivy array.

    Returns
    -------
    type
        An ivy variable, supporting gradient computation.

    """
    return _cur_framework(x).variable(x)


def is_variable(x, exclusive=False):
    """Determines whether the input is a variable or not.

    Parameters
    ----------
    x : array
        An ivy array.
    exclusive : bool, optional
        Whether to check if the data type is exclusively a variable, rather than an array.
        For frameworks like JAX that do not have exclusive variable types, the function will always return
        False if this flag is set, otherwise the check is the same for general arrays. Default is False.

    Returns
    -------
    type
        Boolean, true if x is a trainable variable, false otherwise.

    """
    return _cur_framework(x).is_variable(x, exclusive)


def variable_data(x):
    """Some backends wrap arrays in a dedicated variable class. For those frameworks, this function returns that wrapped
    array. For frameworks which do not have a dedicated variable class, the function returns the data passed in.

    Parameters
    ----------
    x : variable
        An ivy variable.

    Returns
    -------
    type
        The internal data stored by the variable

    """
    return _cur_framework(x).variable_data(x)


def stop_gradient(x, preserve_type=True):
    """Stops gradient computation.

    Parameters
    ----------
    x : array
        Array for which to stop the gradient.
    preserve_type :
        Whether to preserve the input type (ivy.Variable or ivy.Array),
        otherwise an array is always returned. Default is True.
    preserve_type :
        bool, optional (Default value = True)

    Returns
    -------
    type
        The same array x, but with no gradient information.

    """
    return _cur_framework(x).stop_gradient(x, preserve_type)


# AutoGrad #

def execute_with_gradients(func, xs, retain_grads=False):
    """Call function func with input of xs variables, and return func first output y, the gradients [dy/dx for x in xs],
    and any other function outputs after the returned y value

    Parameters
    ----------
    func : function
        Function for which we compute the gradients of the output with respect to xs input.
    xs : sequence of variables
        Variables for which to compute the function gradients with respective to.
    retain_grads : bool
        Whether to retain the gradients of the returned values. (Default value = False)

    Returns
    -------
    type
        the function first output y, the gradients [dy/dx for x in xs], and any other extra function outputs

    """
    return _cur_framework(None).execute_with_gradients(func, xs, retain_grads)


# Optimizer Steps #

def adam_step(dcdws, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7):
    """Compute adam step delta, given the derivatives of some cost c with respect to ws, using ADAM update.
    `[reference] <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    Parameters
    ----------
    dcdws : container of arrays
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    mw : container of arrays
        running average of the gradients
    vw : container of arrays
        running average of second moments of the gradients
    step : int
        training step
    beta1 : float
        gradient forgetting factor (Default value = 0.9)
    beta2 : float
        second moment of gradient forgetting factor (Default value = 0.999)
    epsilon : float
        divisor during adam update, preventing division by zero (Default value = 1e-7)

    Returns
    -------
    type
        The adam step delta.

    """
    step = float(_ivy.to_scalar(step))
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw[kc] + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws ** 2
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw[kc] + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = (1 - beta2_pow) ** 0.5 / (1 - beta1_pow + epsilon)
    return mw.map(lambda m, kc: (alpha * m / (vw[kc] ** 0.5 + epsilon))), mw, vw


# Optimizer Updates #

def optimizer_update(ws, effective_grads, lr, inplace=None, stop_gradients=True):
    """Update weights ws of some function, given the true or effective derivatives of some cost c with respect to ws,
    [dc/dw for w in ws].

    Parameters
    ----------
    ws : Ivy container
        Weights of the function to be updated.
    effective_grads : Ivy container
        Effective gradients of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr : float or container of layer-wise rates.
        Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    inplace : bool, optional
        Whether to perform the operation inplace, for backends which support inplace variable updates,
        and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
        computation graph (i.e. higher order optimization), then this should be set to False.
        Default is True, provided the backend framework supports it.
    stop_gradients : bool, optional
        Whether to stop the gradients of the variables after each gradient step. Default is True.

    Returns
    -------
    type
        The new function weights ws_new, following the optimizer updates.

    """
    inplace = _ivy.default(inplace, _ivy.inplace_variables_supported())
    layerwise_lr = isinstance(lr, _ivy.Container)
    deltas = effective_grads.map(lambda eff_grad, kc: ((lr[kc] if layerwise_lr else lr) * eff_grad))
    if inplace:
        ws = ws.map(lambda w, kc: _ivy.inplace_decrement(w, deltas[kc]))
    else:
        ws = ws.map(lambda w, kc: -deltas[kc] + w)
    if stop_gradients:
        return ws.stop_gradients(preserve_type=True)
    return ws


def gradient_descent_update(ws, dcdws, lr, inplace=None, stop_gradients=True):
    """Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws].

    Parameters
    ----------
    ws : Ivy container
        Weights of the function to be updated.
    dcdws : Ivy container
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr : float or container of layer-wise rates.
        Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    inplace : bool, optional
        Whether to perform the operation inplace, for backends which support inplace variable updates,
        and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
        computation graph (i.e. higher order optimization), then this should be set to False.
        Default is True, provided the backend framework supports it.
    stop_gradients : bool, optional
        Whether to stop the gradients of the variables after each gradient step. Default is True.

    Returns
    -------
    type
        The new function weights ws_new, following the gradient descent updates.

    """
    return optimizer_update(ws, dcdws, lr, inplace, stop_gradients)


def lars_update(ws, dcdws, lr, decay_lambda=0, inplace=None, stop_gradients=True):
    """Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws],
    by applying Layerwise Adaptive Rate Scaling (LARS) method.

    Parameters
    ----------
    ws : Ivy container
        Weights of the function to be updated.
    dcdws : Ivy container
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr : float
        Learning rate, the rate at which the weights should be updated relative to the gradient.
    decay_lambda : float
        The factor used for weight decay. Default is zero.
    inplace : bool, optional
        Whether to perform the operation inplace, for backends which support inplace variable updates,
        and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
        computation graph (i.e. higher order optimization), then this should be set to False.
        Default is True, provided the backend framework supports it.
    stop_gradients : bool, optional
        Whether to stop the gradients of the variables after each gradient step. Default is True.

    Returns
    -------
    type
        The new function weights ws_new, following the LARS updates.

    """
    ws_norm = ws.vector_norm()
    lr = _ivy.stable_divide(ws_norm * lr, dcdws.vector_norm())
    if decay_lambda > 0:
        lr /= (ws_norm * decay_lambda)
    return gradient_descent_update(ws, dcdws, lr, inplace, stop_gradients)


def adam_update(ws, dcdws, lr, mw_tm1, vw_tm1, step, beta1=0.9, beta2=0.999, epsilon=1e-7, inplace=None,
                stop_gradients=True):
    """Update weights ws of some function, given the derivatives of some cost c with respect to ws, using ADAM update.
    `[reference] <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    Parameters
    ----------
    ws : container of variables
        Weights of the function to be updated.
    dcdws : container of arrays
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr : float or container of layer-wise rates.
        Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    mw_tm1 : container of arrays
        running average of the gradients, from the previous time-step.
    vw_tm1 : container of arrays
        running average of second moments of the gradients, from the previous time-step.
    step : int
        training step
    beta1 : float
        gradient forgetting factor (Default value = 0.9)
    beta2 : float
        second moment of gradient forgetting factor (Default value = 0.999)
    epsilon : float
        divisor during adam update, preventing division by zero (Default value = 1e-7)
    inplace : bool, optional
        Whether to perform the operation inplace, for backends which support inplace variable updates,
        and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
        computation graph (i.e. higher order optimization), then this should be set to False.
        Default is True, provided the backend framework supports it.
    stop_gradients : bool, optional
        Whether to stop the gradients of the variables after each gradient step. Default is True.

    Returns
    -------
    type
        The new function weights ws_new, and also new mw and vw, following the adam updates.

    """
    effective_grads, mw, vw = adam_step(dcdws, mw_tm1, vw_tm1, step, beta1, beta2, epsilon)
    return optimizer_update(ws, effective_grads, lr, inplace, stop_gradients), mw, vw


def lamb_update(ws, dcdws, lr, mw_tm1, vw_tm1, step, beta1=0.9, beta2=0.999, epsilon=1e-7, max_trust_ratio=10,
                decay_lambda=0, inplace=None, stop_gradients=True):
    """Update weights ws of some function, given the derivatives of some cost c with respect to ws, [dc/dw for w in ws],
    by applying LAMB method.

    Parameters
    ----------
    ws : container of variables
        Weights of the function to be updated.
    dcdws : container of arrays
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr : float or container of layer-wise rates.
        Learning rate(s), the rate(s) at which the weights should be updated relative to the gradient.
    mw_tm1 : container of arrays
        running average of the gradients, from the previous time-step.
    vw_tm1 : container of arrays
        running average of second moments of the gradients, from the previous time-step.
    step : int
        training step
    beta1 : float
        gradient forgetting factor (Default value = 0.9)
    beta2 : float
        second moment of gradient forgetting factor (Default value = 0.999)
    epsilon : float
        divisor during adam update, preventing division by zero (Default value = 1e-7)
    max_trust_ratio : float, optional
        The maximum value for the trust ratio. Default is 10.
    decay_lambda : float
        The factor used for weight decay. Default is zero.
    inplace : bool, optional
        Whether to perform the operation inplace, for backends which support inplace variable updates,
        and handle gradients behind the scenes such as PyTorch. If the update step should form part of a
        computation graph (i.e. higher order optimization), then this should be set to False.
        Default is True, provided the backend framework supports it.
    stop_gradients : bool, optional
        Whether to stop the gradients of the variables after each gradient step. Default is True.

    Returns
    -------
    type
        The new function weights ws_new, following the LARS updates.

    """
    r1 = ws.vector_norm()
    eff_grads, mw, vw = adam_step(dcdws, mw_tm1, vw_tm1, step, beta1, beta2, epsilon)
    if decay_lambda > 0:
        r2 = (eff_grads + decay_lambda * ws).norm()
    else:
        r2 = eff_grads.vector_norm()
    r = _ivy.stable_divide(r1, r2).minimum(max_trust_ratio)
    lr = r * lr
    return optimizer_update(ws, eff_grads, lr, inplace, stop_gradients), mw, vw
