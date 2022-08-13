"""Collection of gradient Ivy functions."""

# local
import ivy
from typing import Union, Optional
from ivy.backend_handler import current_backend

from ivy.func_wrapper import (
    inputs_to_ivy_arrays,
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
def with_grads(*, with_grads: bool = None) -> bool:
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
    """
    Enter a nested code space where gradients are computed. This method
    deletes the with_grads component from the global list with_grads_stack

    Parameters
    ----------
        No Paramters(Void function)

    Returns
    -------
    ret
        Remove and return item at index (default last).

    Examples
    --------
    >>> ivy.set_with_grads(True)
    >>> ivy.unset_with_grads()
    >>> print(ivy.with_grads(with_grads=None))
    False

    >>> ivy.set_with_grads(True)
    >>> ivy.unset_with_grads()
    Returns last deleted value

    >>> ivy.set_with_grads(False)
    >>> ivy.unset_with_grads()
    Raises IndexError if list is empty or index is out of range.

    """
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

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([1., 0.3, -4.5])
    >>> y = ivy.variable(x)
    >>> print(y)
    ivy.array([ 1. ,  0.3, -4.5])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([-0.2, 4.]))
    >>> y = ivy.variable(x)
    >>> print(y)
    {
        a: ivy.array([1., 2.]),
        b: ivy.array([-0.2, 4.])
    }
    """
    return current_backend(x).variable(x)


@inputs_to_native_arrays
@handle_nestable
def is_variable(
    x: Union[ivy.Array, ivy.NativeArray], /, *, exclusive: bool = False
) -> bool:
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

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.variable(ivy.array(2.3))
    >>> is_var = ivy.is_variable(x)
    >>> print(is_var)
    True

    >>> x = ivy.array([[2], [3], [5]])
    >>> is_var = ivy.is_variable(x, exclusive=True)
    >>> print(is_var)
    False

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array(3.2), b=ivy.array(2))
    >>> is_var = ivy.is_variable(x, exclusive=True)
    >>> print(is_var)
    {
        a: false,
        b: false
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.variable(ivy.array([2.0, -1.0, 0.0])),\
                          b=ivy.array([0., -0.4, 8]))
    >>> exclusive = ivy.Container(a=False, b=True)
    >>> is_var = ivy.is_variable(x, exclusive=exclusive)
    >>> print(is_var)
    {
        a: true,
        b: false
    }

    """
    return current_backend(x).is_variable(x, exclusive)


is_variable.computes_gradients = True


@to_native_arrays_and_back
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

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).variable_data(x)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def stop_gradient(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    preserve_type: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Stops gradient computation.

    Parameters
    ----------
    x
        Array for which to stop the gradient.
    preserve_type
        Whether to preserve the input type (ivy.Variable or ivy.Array),
        otherwise an array is always returned. Default is True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The same array x, but with no gradient information.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` inputs:

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.stop_gradient(x, preserve_type=True)
    >>> print(y)
    ivy.array([1., 2., 3.])

    >>> x = ivy.zeros((2, 3))
    >>> ivy.stop_gradient(x, preserve_type=False, out=x)
    >>> print(x)
    ivy.array([[0., 0., 0.],
               [0., 0., 0.]])

    With one :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),\
                          b=ivy.array([3., 4., 5.]))
    >>> y = ivy.stop_gradient(x, preserve_type=False)
    >>> print(y)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),\
                          b=ivy.array([3., 4., 5.]))
    >>> ivy.stop_gradient(x, preserve_type=True, out=x)
    >>> print(x)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    """
    return current_backend(x).stop_gradient(x, preserve_type=preserve_type, out=out)


# AutoGrad #


@inputs_to_native_arrays
def execute_with_gradients(func, xs, /, *, retain_grads=False):
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
        extra function outputs.

    """
    return current_backend(None).execute_with_gradients(func, xs, retain_grads)


execute_with_gradients.computes_gradients = True


@to_native_arrays_and_back
def value_and_grad(func):
    """
    Create a function that evaluates both func and the gradient of func.

    Parameters
    ----------
    func
        Function for which we compute the gradients of the output with respect to xs
        input.

    Returns
    -------
    ret
        A function that returns both func and the gradient of func.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.variable(ivy.array([[4.6, 2.1, 5], [2.8, 1.3, 6.2]]))
    >>> func = lambda x: ivy.mean(ivy.square(x))
    >>> grad_fn = ivy.value_and_grad(func)
    >>> value_grad = grad_fn(x)
    >>> print(value_grad)
    (ivy.array(16.423332), ivy.array([[1.53, 0.7, 1.67], [0.933, 0.433, 2.07]]))

    """
    return current_backend(None).value_and_grad(func)


value_and_grad.computes_gradients = True


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
    
    Examples
    --------
    With :code:`ivy.Array` input:
    
    >>> x = ivy.array([[4.6, 2.1, 5], [2.8, 1.3, 6.2]])
    >>> func = lambda x: ivy.mean(ivy.square(x))
    >>> jac_fn = ivy.jac(func)
    >>> jacobian = jac_fn(x)
    >>> print(jacobian)
    ivy.array([[1.53 , 0.7  , 1.67 ],\
               [0.933, 0.433, 2.07 ]])

    """
    return current_backend(None).jac(func)


jac.computes_gradients = True


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
    
    Examples
    --------
    >>> x = ivy.array([[4.6, 2.1, 5], [2.8, 1.3, 6.2]])
    >>> func = lambda x: ivy.mean(ivy.square(x))
    >>> grad_fn = ivy.grad(func)
    >>> grad = grad_fn(x)
    >>> print(grad)
    ivy.array([[1.53 , 0.7  , 1.67 ],\
               [0.933, 0.433, 2.07 ]])

    """
    return current_backend(None).grad(func)


grad.computes_gradients = True


# Optimizer Steps #


@inputs_to_ivy_arrays
def adam_step(
    dcdw: Union[ivy.Array, ivy.NativeArray],
    mw: Union[ivy.Array, ivy.NativeArray],
    vw: Union[ivy.Array, ivy.NativeArray],
    step: Union[int, float],
    /,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the effective grad of adam_step to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The adam step delta.

    Examples
    --------
    With :code:`ivy.Array` inputs:

    >>> dcdw = ivy.array([1, 2, 3])
    >>> mw = ivy.ones(3)
    >>> vw = ivy.ones(1)
    >>> step = ivy.array(3)
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step)
    >>> print(adam_step_delta)
    (ivy.array([0.182, 0.182, 0.182]),\
     ivy.array([0.9, 0.9, 0.9]),\
     ivy.array([0.999, 0.999, 0.999]))

    >>> dcdw = ivy.array([[1., 4., -3.], [2., 3., 0.5]])
    >>> mw = ivy.zeros((2,3))
    >>> vw = ivy.zeros(3)
    >>> step = ivy.array(1)
    >>> beta1 = 0.86
    >>> beta2 = 0.95
    >>> epsilon = 1e-6
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1=beta1, beta2=beta2,\
                                        epsilon=epsilon)
    >>> print(adam_step_delta)
    (ivy.array([[ 1.,  1., -1.],\
                [ 1.,  1.,  1.]]),\
     ivy.array([[ 0.14,  0.56, -0.42],\
                [ 0.28,  0.42,  0.07]]),\
     ivy.array([[0.05  , 0.8   , 0.45  ],\
                [0.2   , 0.45  , 0.0125]]))

    >>> dcdw = ivy.array([1, -2, 3])
    >>> mw = ivy.ones(1)
    >>> vw = ivy.ones(1)
    >>> step = ivy.array(3.6)
    >>> out = ivy.zeros_like(dcdw)
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, out=out)
    >>> print(out)
        ivy.array([0.171, 0.171, 0.171])

    With one :code:`ivy.Container` input:

    >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),\
                             b=ivy.array([3., 4., 5.]))
    >>> mw = ivy.array([1., 4., 9.])
    >>> vw = ivy.array([0.,])
    >>> step = ivy.array([3.4])
    >>> beta1 = 0.87
    >>> beta2 = 0.976
    >>> epsilon = 1e-5
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1=beta1, beta2=beta2,\
                                        epsilon=epsilon)
    >>> print(adam_step_delta)
    ({
        a: ivy.array([6.49e+04, 1.74e+01, 1.95e+01]),
        b: ivy.array([2.02, 4.82, 8.17])
    }, {
        a: ivy.array([0.87, 3.61, 8.09]),
        b: ivy.array([1.26, 4., 8.48])
    }, {
        a: ivy.array([0., 0.024, 0.096]),
        b: ivy.array([0.216, 0.384, 0.6])
    })
    
    With multiple :code:`ivy.Container` inputs:

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
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1=beta1, beta2=beta2,\
                                        epsilon=epsilon)
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
    step = float(step)
    mw = ivy.add(beta1 * mw, (1 - beta1) * dcdw)
    dcdw_sqrd = dcdw**2
    vw = ivy.add(beta2 * vw, (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1**step
    beta2_pow = beta2**step
    alpha = (1 - beta2_pow) ** 0.5 / (1 - beta1_pow + epsilon)
    return ivy.divide(alpha * mw, vw**0.5 + epsilon, out=out), mw, vw


adam_step.out_index = 0


# Optimizer Updates #


@inputs_to_ivy_arrays
def optimizer_update(
    w: Union[ivy.Array, ivy.NativeArray],
    effective_grad: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
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
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The new function weights ws_new, following the optimizer updates.
    
    Examples
    --------
    With :code:`ivy.Array` inputs:

    >>> w = ivy.array([1., 2., 3.])
    >>> effective_grad = ivy.zeros(3)
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr)
    >>> print(ws_new)
    ivy.array([1., 2., 3.])

    >>> w = ivy.array([1., 2., 3.])
    >>> effective_grad = ivy.zeros(3)
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr,\
                                    out=None, stop_gradients=True)
    >>> print(ws_new)
    ivy.array([1., 2., 3.])
    
    >>> w = ivy.array([[1., 2.], [4., 5.]])
    >>> out = ivy.zeros_like(w)
    >>> effective_grad = ivy.array([[4., 5.], [7., 8.]])
    >>> lr = ivy.array([3e-4, 1e-2])
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr, out=out)
    >>> print(out)
    ivy.array([[0.999, 1.95],
               [4., 4.92]])

    >>> w = ivy.array([1., 2., 3.])
    >>> out = ivy.zeros_like(w)
    >>> effective_grad = ivy.array([4., 5., 6.])
    >>> lr = ivy.array([3e-4])
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr,\
                                      stop_gradients=False, out=out)
    >>> print(out)
    ivy.array([0.999, 2.   , 3.   ])
    
    With one :code:`ivy.Container` input:
        
    >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),\
                          b=ivy.array([3., 4., 5.]))
    >>> effective_grad = ivy.array([0., 0., 0.])
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr)
    >>> print(ws_new)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    With multiple :code:`ivy.Container` inputs:
    
    >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),\
                          b=ivy.array([3., 4., 5.]))
    >>> effective_grad = ivy.Container(a=ivy.array([0., 0., 0.]),\
                                       b=ivy.array([0., 0., 0.]))
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr, out=w)
    >>> print(w)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }
    
    >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),\
                        b=ivy.array([3., 4., 5.]))
    >>> effective_grad = ivy.Container(a=ivy.array([0., 0., 0.]),\
                                    b=ivy.array([0., 0., 0.]))
    >>> lr = ivy.array([3e-4])
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr,\
                                      stop_gradients=False)
    >>> print(ws_new)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    """
    deltas = effective_grad * lr
    w = ivy.subtract(w, deltas, out=out)
    if stop_gradients:
        return ivy.stop_gradient(w, preserve_type=True, out=out)
    return w


@inputs_to_ivy_arrays
def gradient_descent_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
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
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The new weights, following the gradient descent updates.

    Examples
    --------
    With :code:`ivy.Array` inputs:

    >>> w = ivy.array([[1., 2, 3],\
                       [4, 6, 1],\
                       [1, 0, 7]])
    >>> dcdw = ivy.array([[0.5, 0.2, 0.1],\
                          [0.3, 0.6, 0.4],\
                          [0.4, 0.7, 0.2]])
    >>> lr = ivy.array(0.1)
    >>> new_weights = ivy.gradient_descent_update(w, dcdw, lr, stop_gradients=True)
    >>> print(new_weights)
    ivy.array([[ 0.95,  1.98,  2.99],\
               [ 3.97,  5.94,  0.96],\
               [ 0.96, -0.07,  6.98]])
    
    >>> w = ivy.array([1., 2., 3.])
    >>> dcdw = ivy.array([0.5, 0.2, 0.1])
    >>> lr = ivy.array(0.3)
    >>> out = ivy.zeros_like(w)
    >>> ivy.gradient_descent_update(w, dcdw, lr, out=out)
    >>> print(out)
    ivy.array([0.85, 1.94, 2.97])
    
    With one :code:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                          b=ivy.array([3.48, 5.72, 1.98]))
    >>> dcdw = ivy.array([0.5, 0.2, 0.1])
    >>> lr = ivy.array(0.3)
    >>> w_new = ivy.gradient_descent_update(w, dcdw, lr)
    >>> print(w_new)
    {
        a: ivy.array([0.85, 1.94, 2.97]),
        b: ivy.array([3.33, 5.66, 1.95])
    }
    
    With multiple :code:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),\
                          b=ivy.array([3.48, 5.72, 1.98]))
    >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),\
                             b=ivy.array([2., 3.42, 1.69]))
    >>> lr = ivy.array(0.3)
    >>> w_new = ivy.gradient_descent_update(w, dcdw, lr)
    >>> print(w_new)
    {
        a: ivy.array([0.85, 1.94, 2.97]),
        b: ivy.array([2.88, 4.69, 1.47])
    }

    """
    return ivy.optimizer_update(w, dcdw, lr, stop_gradients=stop_gradients, out=out)


@inputs_to_ivy_arrays
def lars_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    decay_lambda: float = 0,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
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
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The new function weights ws_new, following the LARS updates.

    """
    w_norm = ivy.vector_norm(w)
    lr = ivy.stable_divide(w_norm * lr, ivy.vector_norm(dcdw))
    if decay_lambda > 0:
        lr /= w_norm * decay_lambda
    return ivy.gradient_descent_update(
        w, dcdw, lr, stop_gradients=stop_gradients, out=out
    )


@inputs_to_ivy_arrays
def adam_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    mw_tm1: Union[ivy.Array, ivy.NativeArray],
    vw_tm1: Union[ivy.Array, ivy.NativeArray],
    step: int,
    /,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
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
        training step.
    beta1
        gradient forgetting factor (Default value = 0.9).
    beta2
        second moment of gradient forgetting factor (Default value = 0.999).
    epsilon
        divisor during adam update, preventing division by zero (Default value = 1e-7).
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.
    out
        optional output array, for writing the new function weights ws_new to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The new function weights ws_new, and also new mw and vw, following the adam
        updates.

    """
    effective_grads, mw, vw = ivy.adam_step(
        dcdw, mw_tm1, vw_tm1, step, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    return (
        ivy.optimizer_update(
            w, effective_grads, lr, stop_gradients=stop_gradients, out=out
        ),
        mw,
        vw,
    )


adam_update.out_index = 0


@inputs_to_ivy_arrays
def lamb_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    mw_tm1: Union[ivy.Array, ivy.NativeArray],
    vw_tm1: Union[ivy.Array, ivy.NativeArray],
    step: int,
    /,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    max_trust_ratio: Union[int, float] = 10,
    decay_lambda: float = 0,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
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
        training step.
    beta1
        gradient forgetting factor (Default value = 0.9).
    beta2
        second moment of gradient forgetting factor (Default value = 0.999).
    epsilon
        divisor during adam update, preventing division by zero (Default value = 1e-7).
    max_trust_ratio
        The maximum value for the trust ratio. (Default value = 10)
    decay_lambda
        The factor used for weight decay. (Default value = 0).
    stop_gradients
        Whether to stop the gradients of the variables after each gradient step.
        Default is True.
    out
        optional output array, for writing the new function weights ws_new to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The new function weights ws_new, following the LAMB updates.

    Examples
    --------
    With :code:`ivy.Array` inputs:

    >>> w = ivy.array([1., 2, 3])
    >>> dcdw = ivy.array([0.5,0.2,0.1])
    >>> lr = ivy.array(0.1)
    >>> vw_tm1 = ivy.zeros(1)
    >>> mw_tm1 = ivy.zeros(3)
    >>> step = ivy.array(1)
    >>> new_weights = ivy.lamb_update(w, dcdw, lr, mw_tm1, vw_tm1, step)
    >>> print(new_weights)
    (ivy.array([0.784, 1.78 , 2.78 ]), \
     ivy.array([0.05, 0.02, 0.01]), \
     ivy.array([2.5e-04, 4.0e-05, 1.0e-05]))

    >>> w = ivy.array([[1., 2, 3],[4, 6, 1],[1, 0, 7]])
    >>> dcdw = ivy.array([[0.5, 0.2, 0.1],[0.3, 0.6, 0.4],[0.4, 0.7, 0.2]])
    >>> lr = ivy.array(0.1)
    >>> mw_tm1 = ivy.zeros((3,3))
    >>> vw_tm1 = ivy.zeros(3)
    >>> step = ivy.array(1)
    >>> beta1 = 0.9
    >>> beta2 = 0.999
    >>> epsilon = 1e-7
    >>> max_trust_ratio = 10
    >>> decay_lambda = 0
    >>> out = ivy.zeros_like(w)
    >>> stop_gradients = True
    >>> new_weights = ivy.lamb_update(w, dcdw, lr, mw_tm1, vw_tm1, step, beta1=beta1,\
                                      beta2=beta2, epsilon=epsilon,\
                                      max_trust_ratio=max_trust_ratio,\
                                      decay_lambda=decay_lambda, out=out,\
                                      stop_gradients=stop_gradients)  
    >>> print(out)
    ivy.array([[ 0.639,  1.64 ,  2.64 ],\
               [ 3.64 ,  5.64 ,  0.639],\
               [ 0.639, -0.361,  6.64 ]])

    With one :code:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([4., 5., 6.]))
    >>> dcdw = ivy.array([3., 4., 5.])
    >>> mw_tm1 = ivy.array([0., 0., 0.])
    >>> vw_tm1 = ivy.array([0.])
    >>> lr = ivy.array(1.)
    >>> step = ivy.array([2])
    >>> new_weights = ivy.lamb_update(w, dcdw, mw_tm1, vw_tm1, lr, step)
    >>> print(new_weights)
    ({
        a: ivy.array([1., 2., 3.]),
        b: ivy.array([4., 5., 6.])
    }, ivy.array([0.3, 0.4, 0.5]), ivy.array([1.01, 1.01, 1.02]))

    With multiple :code:`ivy.Container` inputs:
    
    >>> w = ivy.Container(a=ivy.array([1.,3.,5.]),\
                          b=ivy.array([3.,4.,2.]))
    >>> dcdw = ivy.Container(a=ivy.array([0.2,0.3,0.6]),\
                             b=ivy.array([0.6,0.4,0.7]))
    >>> mw_tm1 = ivy.Container(a=ivy.array([0.,0.,0.]),\
                               b=ivy.array([0.,0.,0.]))

    >>> vw_tm1 = ivy.Container(a=ivy.array([0.,]),\
                               b=ivy.array([0.,]))
    >>> step = ivy.array([3.4])
    >>> beta1 = 0.9
    >>> beta2 = 0.999
    >>> epsilon = 1e-7
    >>> max_trust_ratio = 10
    >>> decay_lambda = 0
    >>> stop_gradients = True
    >>> lr = ivy.array(0.5)
    >>> new_weights = ivy.lamb_update(w, dcdw, lr, mw_tm1, vw_tm1, step, beta1=beta1,\
                                      beta2=beta2, epsilon=epsilon,\
                                      max_trust_ratio=max_trust_ratio,\
                                      decay_lambda=decay_lambda,\
                                      stop_gradients=stop_gradients)
    >>> print(new_weights)
    ({
        a: ivy.array([-0.708, 1.29, 3.29]),
        b: ivy.array([1.45, 2.45, 0.445])
    }, {
        a: ivy.array([0.02, 0.03, 0.06]),
        b: ivy.array([0.06, 0.04, 0.07])
    }, {
        a: ivy.array([4.0e-05, 9.0e-05, 3.6e-04]),
        b: ivy.array([0.00036, 0.00016, 0.00049])
    })

    """
    r1 = ivy.vector_norm(w)
    eff_grads, mw, vw = ivy.adam_step(
        dcdw, mw_tm1, vw_tm1, step, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    if decay_lambda > 0:
        r2 = ivy.vector_norm(eff_grads + decay_lambda * w)
    else:
        r2 = ivy.vector_norm(eff_grads)
    r = ivy.stable_divide(r1, r2).minimum(max_trust_ratio)
    lr = r * lr
    return (
        ivy.optimizer_update(w, eff_grads, lr, stop_gradients=stop_gradients, out=out),
        mw,
        vw,
    )


lamb_update.out_index = 0
