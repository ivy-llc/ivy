"""Collection of gradient Ivy functions."""

# local
import ivy
from typing import Union, Optional
from ivy.backend_handler import current_backend

from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    inputs_to_native_arrays,
    handle_nestable,
)


# Extra #
# ------#

with_grads_stack = list()


class GradientTracking:
    """"""

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
def with_grads(with_grads: bool = None) -> bool:
    """
    Enter a nested code space where gradients are computed. This method
    adds the with_grads component to the global list with_grads_stack

    Parameters
    ----------
    with_grads
        Boolean value denoting whether the current code block has gradient
        computation enabled or not.
        'True' or 'False' or 'None' (Default value = None)

    Returns
    -------
    ret
        If with_grads is boolean, it returns the boolean value representing
        if gradient computation is enabled or not.
        If with_grads is None, it returns the last element in the with_grads_stack
        representing the parent of the current nested code block. If with_grads_stack
        is empty, it returns True by default.
        If with_grads is neither None nor boolean, it will raise an AssertionError

    Examples
    --------
    >>> ivy.set_with_grads(True)
    >>> print(ivy.with_grads(with_grads=None))
    True

    >>> ivy.set_with_grads(False)
    >>> print(ivy.with_grads(with_grads=None))
    False

    >>> print(ivy.with_grads(with_grads=True))
    True

    >>> print(ivy.with_grads(with_grads=False))
    False

    """
    if ivy.exists(with_grads):
        assert with_grads in [True, False]
        return with_grads
    global with_grads_stack
    if not with_grads_stack:
        with_grads_stack = [True]
    return with_grads_stack[-1]


# noinspection PyShadowingNames
def set_with_grads(with_grads: bool):
    """
    Enter a nested code space where gradients are computed. This method
    adds the with_grads component to the global list with_grads_stack

    Parameters
    ----------
    with_grads
        Boolean value denoting whether the current code block has gradient
        computation enabled or not.
        'True' or 'False' or 'None' (Default value = None)

    Returns
    -------
    ret
        If with_grads is boolean, it returns the boolean value representing
        if gradient computation is enabled or not.
        If with_grads is None, it returns the last element in the with_grads_stack
        representing the parent of the current nested code block. If with_grads_stack
        is empty, it returns True by default.
        If with_grads is neither None nor boolean, it will raise an AssertionError

    Examples
    --------
    >>> ivy.set_with_grads(True)
    >>> print(ivy.with_grads(with_grads=None))
    True

    >>> ivy.set_with_grads(False)
    >>> print(ivy.with_grads(with_grads=None))
    False

    >>> print(ivy.with_grads(with_grads=True))
    True

    >>> print(ivy.with_grads(with_grads=False))
    False

    """
    assert with_grads in [True, False]
    global with_grads_stack
    with_grads_stack.append(with_grads)


def unset_with_grads():

    global with_grads_stack
    if with_grads_stack:
        with_grads_stack.pop(-1)


# Variables #


@to_native_arrays_and_back
@handle_nestable
def variable(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Variable:
    """Creates a variable, which supports gradient computation.

    Parameters
    ----------
    x
        An ivy array.

    Returns
    -------
    ret
        An ivy variable, supporting gradient computation.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> ivy.set_backend('torch')
    >>> x = ivy.array([1., 0.3, -4.5])
    >>> y = ivy.variable(x)
    >>> y
    ivy.array([ 1. ,  0.3, -4.5])
    >>> ivy.unset_backend()

    With :code:`ivy.NativeArray` input:

    >>> ivy.set_backend('jax')
    >>> x = ivy.native_array([0.2, 2., 3.])
    >>> y = ivy.variable(x)
    >>> y
    ivy.array([0.2, 2., 3.])
    >>> ivy.unset_backend()

    With :code:`ivy.Container` input:

    >>> ivy.set_backend('tensorflow')
    >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([-0.2, 4.]))
    >>> y = ivy.variable(x)
    >>> y
    {
        a: ivy.array([1., 2.]),
        b: ivy.array([-0.2, 4.])
    }
    >>> ivy.unset_backend()
    """
    return current_backend(x).variable(x)


@inputs_to_native_arrays
@handle_nestable
def is_variable(x: Union[ivy.Array, ivy.NativeArray], exclusive: bool = False) -> bool:
    """Determines whether the input is a variable or not.

    Parameters
    ----------
    x
        An ivy array.
    exclusive
        Whether to check if the data type is exclusively a variable, rather than an
        array. For frameworks like JAX that do not have exclusive variable types, the
        function will always return False if this flag is set, otherwise the check is
        the same for general arrays. Default is False.

    Returns
    -------
    ret
        Boolean, true if x is a trainable variable, false otherwise.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array(2.3)
    >>> is_var = ivy.is_variable(x)
    >>> print(is_var)
        False

    >>> x = ivy.zeros((3, 2))
    >>> is_var = ivy.is_variable(x)
    >>> print(is_var)
        False

    >>> x = ivy.array([[2], [3], [5]])
    >>> is_var = ivy.is_variable(x, True)
    >>> print(is_var)
        False

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([7])
    >>> is_var = ivy.is_variable(x)
    >>> print(is_var)
        False

    >>> x = ivy.native_array([2, 3, 4])
    >>> is_var = ivy.is_variable(x)
    >>> print(is_var)
        False

    >>> x = ivy.native_array([-1, 0., 0.8, 9])
    >>> is_var =  ivy.is_variable(x, True)
    >>> print(is_var)
        False

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array(3.2), b=ivy.array(2))
    >>> exclusive = True
    >>> is_var = ivy.is_variable(x, exclusive=exclusive)
    >>> print(is_var)
    {
        a: false,
        b: false
    }


    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
    >>> exclusive = ivy.Container(a=False, b=True)
    >>> is_var = ivy.is_variable(x, exclusive=exclusive)
    >>> print(is_var)
    {
        a: false,
        b: false
    }
    """
    return current_backend(x).is_variable(x, exclusive)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def variable_data(x):
    """Some backends wrap arrays in a dedicated variable class. For those frameworks,
    this function returns that wrapped array. For frameworks which do not have a
    dedicated variable class, the function returns the data passed in.

    Parameters
    ----------
    x
        An ivy variable.

    Returns
    -------
    ret
        The internal data stored by the variable

    """
    return current_backend(x).variable_data(x)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def stop_gradient(
    x: Union[ivy.Array, ivy.NativeArray],
    preserve_type: bool = True,
    *,
    out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Stops gradient computation.

    Parameters
    ----------
    x
        Array for which to stop the gradient.
    preserve_type
        Whether to preserve the input type (ivy.Variable or ivy.Array),
        otherwise an array is always returned. Default is True.
    preserve_type
        bool, optional (Default value = True)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The same array x, but with no gradient information.

    """
    return current_backend(x).stop_gradient(x, preserve_type, out=out)


# AutoGrad #


@to_native_arrays_and_back
def execute_with_gradients(func, xs, retain_grads=False):
    """Call function func with input of xs variables, and return func first output y,
    the gradients [dy/dx for x in xs], and any other function outputs after the returned
    y value.

    Parameters
    ----------
    func
        Function for which we compute the gradients of the output with respect to xs
        input.
    xs
        Variables for which to compute the function gradients with respective to.
    retain_grads
        Whether to retain the gradients of the returned values. (Default value = False)

    Returns
    -------
    ret
        the function first output y, the gradients [dy/dx for x in xs], and any other
        extra function outputs

    """
    return current_backend(None).execute_with_gradients(func, xs, retain_grads)


# Optimizer Steps #


@to_native_arrays_and_back
def adam_step(
    dcdw: Union[ivy.Array, ivy.NativeArray],
    mw: Union[ivy.Array, ivy.NativeArray],
    vw: Union[ivy.Array, ivy.NativeArray],
    step: Union[int, float],
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-7,
) -> ivy.Array:
    """Compute adam step delta, given the derivatives of some cost c with respect
    to weights ws, using ADAM update. `[reference]

    <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    Parameters
    ----------
    dcdw
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    mw
        running average of the gradients
    vw
        running average of second moments of the gradients
    step
        training step
    beta1
        gradient forgetting factor (Default value = 0.9)
    beta2
        second moment of gradient forgetting factor (Default value = 0.999)
    epsilon
        divisor during adam update, preventing division by zero (Default value = 1e-7)

    Returns
    -------
    ret
        The adam step delta.

    Functional Examples
    -------------------
    With :code:`ivy.Array` inputs:

    >>> dcdw = ivy.array([1, 2, 3])
    >>> mw = ivy.zeros(3)
    >>> vw = ivy.zeros(1)
    >>> step = ivy.array(3)
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step)
    >>> print(adam_step_delta)
        (ivy.array([0.639, 0.639, 0.639]),
        ivy.array([0.1, 0.2, 0.3]),
        ivy.array([0.001, 0.004, 0.009]))

    >>> dcdw = ivy.array([[1., 4., -3.], [2., 3., 0.5]])
    >>> mw = ivy.zeros((2,3))
    >>> vw = ivy.zeros(3)
    >>> step = ivy.array(1)
    >>> beta1 = 0.86
    >>> beta2 = 0.95
    >>> epsilon = 1e-6
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1, beta2, epsilon)
    >>> print(adam_step_delta)
        (ivy.array([[1., 1., -1.],
                    [1., 1., 1.]]),
        ivy.array([[ 0.14, 0.56, -0.42],
                  [ 0.28, 0.42, 0.07]]),
        ivy.array([[0.05, 0.8, 0.45],
                  [0.2, 0.45, 0.0125]]))

    >>> dcdw = ivy.array([1, -2, 3])
    >>> mw = ivy.zeros(1)
    >>> vw = ivy.zeros(1)
    >>> step = ivy.array(3.6)
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step)
    >>> print(adam_step_delta)
        (ivy.array([ 0.601, -0.601, 0.601]),
        ivy.array([ 0.1, -0.2, 0.3]),
        ivy.array([0.001, 0.004, 0.009]))

    With :code:`ivy.NativeArray` inputs:

    >>> dcdw = ivy.native_array([2, 3, 5])
    >>> mw = ivy.native_array([0, 0, 0])
    >>> vw = ivy.native_array([0])
    >>> step = ivy.native_array([4])
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step)
    >>> print(adam_step_delta)
        (ivy.array([0.581, 0.581, 0.581]),
        ivy.array([0.2, 0.3, 0.5]),
        ivy.array([0.004, 0.009, 0.025]))

    >>> dcdw = ivy.native_array([3., -4., 1., 0., 2., -3., 2.6,])
    >>> mw = ivy.zeros([7])
    >>> vw = ivy.native_array([1])
    >>> step = ivy.native_array([2])
    >>> beta1 = 0.76
    >>> beta2 = 0.992
    >>> epsilon = 1e-5
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1, beta2, epsilon)
    >>> print(adam_step_delta)
        (ivy.array([0.209, -0.271, 0.0717, 0., 0.142, -0.209, 0.182]),
         ivy.array([ 0.72, -0.96, 0.24, 0., 0.48, -0.72, 0.624]),
         ivy.array([1.06, 1.12, 1., 0.992, 1.02, 1.06, 1.05]))

    with mixture of both :code:`ivy.NativeArray`  and :code:'ivy.Array' inputs:

    >>> dcdw = ivy.array([1, 2, 3])
    >>> mw = ivy.native_array([0, 0, 0])
    >>> vw = ivy.zeros(1)
    >>> step = ivy.native_array([2])
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step)
    >>> print(adam_step_delta)
        (ivy.array([0.744, 0.744, 0.744]),
        ivy.array([0.1, 0.2, 0.3]),
        ivy.array([0.001, 0.004, 0.009]))

    with :code: `ivy.container` inputs:

    >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),\
                             b=ivy.array([3., 4., 5.]))
    >>> mw = ivy.Container(a=ivy.array([0., 0., 0.]),\
                           b=ivy.array([0., 0., 0.]))
    >>> vw = ivy.Container(a=ivy.array([0.,]),\
                           b=ivy.array([0.,]))
    >>> step = ivy.array([3.4])
    >>> beta1 = 0.87
    >>> beta2 = 0.976
    >>> epsilon = 1e-5
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1, beta2, epsilon)
    >>> print(adam_step_delta)
    ({
        a: ivy.array([0., 0.626, 0.626]),
        b: ivy.array([0.626, 0.626, 0.626])
    }, {
        a: ivy.array([0., 0.13, 0.26]),
        b: ivy.array([0.39, 0.52, 0.65])
    }, {
        a: ivy.array([0., 0.024, 0.096]),
        b: ivy.array([0.216, 0.384, 0.6])
    })
    """
    step = float(ivy.to_scalar(step))
    mw = beta1 * mw + (1 - beta1) * dcdw
    dcdw_sqrd = dcdw**2
    vw = beta2 * vw + (1 - beta2) * dcdw_sqrd
    beta1_pow = beta1**step
    beta2_pow = beta2**step
    alpha = (1 - beta2_pow) ** 0.5 / (1 - beta1_pow + epsilon)
    return ((alpha * mw) / (vw**0.5 + epsilon)), mw, vw


# Optimizer Updates #


@to_native_arrays_and_back
def optimizer_update(
    w: Union[ivy.Array, ivy.NativeArray],
    effective_grad: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    inplace=None,
    stop_gradients=True,
) -> ivy.Array:
    """Update weights ws of some function, given the true or effective derivatives of
    some cost c with respect to ws, [dc/dw for w in ws].

    Parameters
    ----------
    w
        Weights of the function to be updated.
    effective_grad
        Effective gradients of the cost c with respect to the weights ws,
        [dc/dw for w in ws].
    lr
        Learning rate(s), the rate(s) at which the weights should be updated relative to
        the gradient.
    inplace
        Whether to perform the operation inplace, for backends which support inplace
        variable updates, and handle gradients behind the scenes such as PyTorch. If the
        update step should form part of a computation graph (i.e. higher order
        optimization), then this should be set to False. Default is True, provided the
        backend framework supports it.
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.

    Returns
    -------
    ret
        The new function weights ws_new, following the optimizer updates.

    """
    inplace = ivy.default(inplace, ivy.inplace_variables_supported())
    deltas = effective_grad * lr
    if inplace:
        w = ivy.inplace_decrement(w, deltas)
    else:
        w = w - deltas

    if stop_gradients:
        return ivy.stop_gradient(w, preserve_type=True)
    return w


@to_native_arrays_and_back
def gradient_descent_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    inplace=None,
    stop_gradients=True,
) -> ivy.Array:
    """Update weights ws of some function, given the derivatives of some cost c with
    respect to ws, [dc/dw for w in ws].

    Parameters
    ----------
    w
        Weights of the function to be updated.
    dcdw
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr
        Learning rate(s), the rate(s) at which the weights should be updated relative to
        the gradient.
    inplace
        Whether to perform the operation inplace, for backends which support inplace
        variable updates, and handle gradients behind the scenes such as PyTorch. If the
        update step should form part of a computation graph (i.e. higher order
        optimization), then this should be set to False. Default is True, provided the
        backend framework supports it.
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.

    Returns
    -------
    ret
        The new function weights ws_new, following the gradient descent updates.

    """
    return ivy.optimizer_update(w, dcdw, lr, inplace, stop_gradients)


@to_native_arrays_and_back
def lars_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    decay_lambda=0,
    inplace=None,
    stop_gradients=True,
) -> ivy.Array:
    """Update weights ws of some function, given the derivatives of some cost c with
    respect to ws, [dc/dw for w in ws], by applying Layerwise Adaptive Rate Scaling
    (LARS) method.

    Parameters
    ----------
    w
        Weights of the function to be updated.
    dcdw
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr
        Learning rate, the rate at which the weights should be updated relative to the
        gradient.
    decay_lambda
        The factor used for weight decay. Default is zero.
    inplace
        Whether to perform the operation inplace, for backends which support inplace
        variable updates, and handle gradients behind the scenes such as PyTorch. If the
        update step should form part of a computation graph (i.e. higher order
        optimization), then this should be set to False. Default is True, provided the
        backend framework supports it.
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.

    Returns
    -------
    ret
        The new function weights ws_new, following the LARS updates.

    """
    w_norm = ivy.vector_norm(w)
    lr = ivy.stable_divide(w_norm * lr, ivy.vector_norm(dcdw))
    if decay_lambda > 0:
        lr /= w_norm * decay_lambda
    return ivy.gradient_descent_update(w, dcdw, lr, inplace, stop_gradients)


@to_native_arrays_and_back
def adam_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    mw_tm1: Union[ivy.Array, ivy.NativeArray],
    vw_tm1: Union[ivy.Array, ivy.NativeArray],
    step: int,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-7,
    inplace=None,
    stop_gradients=True,
) -> ivy.Array:
    """Update weights ws of some function, given the derivatives of some cost c with
    respect to ws, using ADAM update. `[reference]

    <https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam>`_

    Parameters
    ----------
    w
        Weights of the function to be updated.
    dcdw
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr
        Learning rate(s), the rate(s) at which the weights should be updated relative to
        the gradient.
    mw_tm1
        running average of the gradients, from the previous time-step.
    vw_tm1
        running average of second moments of the gradients, from the previous time-step.
    step
        training step
    beta1
        gradient forgetting factor (Default value = 0.9)
    beta2
        second moment of gradient forgetting factor (Default value = 0.999)
    epsilon
        divisor during adam update, preventing division by zero (Default value = 1e-7)
    inplace
        Whether to perform the operation inplace, for backends which support inplace
        variable updates, and handle gradients behind the scenes such as PyTorch. If the
        update step should form part of a computation graph (i.e. higher order
        optimization), then this should be set to False. Default is True, provided the
        backend framework supports it.
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.

    Returns
    -------
    ret
        The new function weights ws_new, and also new mw and vw, following the adam
        updates.

    """
    effective_grads, mw, vw = adam_step(
        dcdw, mw_tm1, vw_tm1, step, beta1, beta2, epsilon
    )
    return ivy.optimizer_update(w, effective_grads, lr, inplace, stop_gradients), mw, vw


@to_native_arrays_and_back
def lamb_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    mw_tm1: Union[ivy.Array, ivy.NativeArray],
    vw_tm1: Union[ivy.Array, ivy.NativeArray],
    step: int,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-7,
    max_trust_ratio=10,
    decay_lambda=0,
    inplace=None,
    stop_gradients=True,
) -> ivy.Array:
    """Update weights ws of some function, given the derivatives of some cost c with
    respect to ws, [dc/dw for w in ws], by applying LAMB method.

    Parameters
    ----------
    w
        Weights of the function to be updated.
    dcdw
        Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
    lr
        Learning rate(s), the rate(s) at which the weights should be updated relative to
        the gradient.
    mw_tm1
        running average of the gradients, from the previous time-step.
    vw_tm1
        running average of second moments of the gradients, from the previous time-step.
    step
        training step
    beta1
        gradient forgetting factor (Default value = 0.9)
    beta2
        second moment of gradient forgetting factor (Default value = 0.999)
    epsilon
        divisor during adam update, preventing division by zero (Default value = 1e-7)
    max_trust_ratio
        The maximum value for the trust ratio. Default is 10.
    decay_lambda
        The factor used for weight decay. Default is zero.
    inplace
        Whether to perform the operation inplace, for backends which support inplace
        variable updates, and handle gradients behind the scenes such as PyTorch. If the
        update step should form part of a computation graph (i.e. higher order
        optimization), then this should be set to False. Default is True, provided the
        backend framework supports it.
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.

    Returns
    -------
    ret
        The new function weights ws_new, following the LARS updates.

    """
    r1 = ivy.vector_norm(w)
    eff_grads, mw, vw = ivy.adam_step(dcdw, mw_tm1, vw_tm1, step, beta1, beta2, epsilon)
    if decay_lambda > 0:
        r2 = ivy.vector_norm(eff_grads + decay_lambda * w)
    else:
        r2 = ivy.vector_norm(eff_grads)
    r = ivy.stable_divide(r1, r2).minimum(max_trust_ratio)
    lr = r * lr
    return ivy.optimizer_update(w, eff_grads, lr, inplace, stop_gradients), mw, vw


@to_native_arrays_and_back
def jac(func):
    """Call function func, and return func's Jacobian partial derivatives.

    Parameters
    ----------
    func
        Function for which we compute the gradients of the output with respect to xs
        input.

    Returns
    -------
    ret
        the Jacobian function

    """
    return current_backend(None).jac(func)


@to_native_arrays_and_back
def grad(func):
    """Call function func, and return func's gradients.

    Parameters
    ----------
    func
        Function for which we compute the gradients of the output with respect to xs
        input.

    Returns
    -------
    ret
        the grad function

    """
    return current_backend(None).grad(func)
