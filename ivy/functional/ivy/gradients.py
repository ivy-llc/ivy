"""Collection of gradient Ivy functions."""

# global
from typing import Union, Optional, Tuple
import numpy as np
import itertools

# local
import ivy
from ivy.backend_handler import current_backend

from ivy.func_wrapper import (
    inputs_to_ivy_arrays,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like,
)
from ivy.exceptions import handle_exceptions


# Helpers #
# ------- #


def _arrays_to_float_variables(xs, xs_grad_idxs=None):
    def map_fn(x):
        def inner_fn(x):
            if ivy.is_array(x, exclusive=True):
                if ivy.is_int_dtype(x.dtype):
                    x = ivy.astype(x, ivy.default_float_dtype())
                elif _is_variable(x):
                    x = stop_gradient(x, preserve_type=False)

                return _variable(x)
            return x

        return ivy.nested_map(x, fn=inner_fn, include_derived=True)

    if xs_grad_idxs is not None:
        if isinstance(xs, ivy.Container):
            xs = xs.to_dict()
            ivy.map_nest_at_indices(xs, xs_grad_idxs, map_fn)
            xs = ivy.Container(xs)
        else:
            ivy.map_nest_at_indices(xs, xs_grad_idxs, map_fn)
        return xs
    else:
        return ivy.nested_map(xs, map_fn, include_derived=True)


def _get_required_native_variables(xs, xs_grad_idxs):
    xs = ivy.to_ivy(xs)
    if xs_grad_idxs is not None:
        ivy.map_nest_at_indices(xs, xs_grad_idxs, ivy.to_native)
    else:
        xs = ivy.nested_map(xs, ivy.to_native)

    def map_fn(x):
        if ivy.is_native_array(x):
            return x
        return None

    xs = ivy.nested_map(xs, map_fn, include_derived=True, to_mutable=True)
    none_idxs = ivy.nested_argwhere(xs, lambda x: x is None)
    if not _check_if_empty(none_idxs):
        none_idxs.reverse()
        ivy.prune_nest_at_indices(xs, none_idxs)
    if ivy.is_array(xs):
        return xs
    elif isinstance(xs, ivy.Container):
        xs = xs.prune_empty()
    else:
        xs = _remove_empty(xs)
    if len(xs) == 1 and isinstance(xs, list):
        return xs[0]
    return xs


def _remove_empty(xs):
    valid = False
    if isinstance(xs, dict):
        keys = [k for k in xs]
        for k in keys:
            xs[k] = _remove_empty(xs[k])
            if xs[k] is not None:
                valid = True
        for k in keys:
            if xs[k] is None:
                del xs[k]
    elif isinstance(xs, (list, tuple)):
        xs = list(xs)
        for i in range(len(xs)):
            xs[i] = _remove_empty(xs[i])
            if xs[i] is not None:
                valid = True
        for i in range(len(xs) - 1, -1, -1):
            if xs[i] is None:
                del xs[i]
    if not valid and not ivy.is_array(xs):
        return None
    return xs


def _check_if_empty(idxs):
    return not isinstance(idxs, list) or np.asarray(idxs, dtype="object").size == 0


def _remove_zeros_and_nones(grads, x, idx=[]):
    if ivy.is_array(x):
        abs_val = ivy.abs(x)
        if ivy.all(abs_val.astype("float64") < 1e-10) and len(idx):
            ivy.prune_nest_at_index(grads, idx)
        return grads
    if x is None:
        ivy.prune_nest_at_index(grads, idx)
    else:
        keys = [k for k in x]
        for k in keys:
            idx.append(k)
            grads = _remove_zeros_and_nones(grads, x[k], idx)
            idx.pop()

        keys = [k for k in x]
        if len(keys) == 0 and len(idx) and _check_if_empty(idx):
            ivy.prune_nest_at_index(grads, idx)
    return grads


def _idxs_to_str(idxs):
    final_idxs = []
    for i in range(len(idxs)):
        final_idxs.append([str(x) for x in idxs[i]])
        final_idxs[i] = "_".join(final_idxs[i])
    return final_idxs


def _get_native_variables_and_indices(x, reshape=True, idxs=None, create_var=False):
    def map_fn(x_):
        if ivy.is_array(x_):
            x_ = ivy.to_ivy(x_) if ivy.is_native_array(x_) else x_
            if create_var:
                x_ = _variable(x_) if not _is_variable(x_, exclusive=True) else x_
            if len(x_.shape) == 0:
                return ivy.to_native(x_)
            if reshape:
                if x_.size == 1:
                    if reshape:
                        return ivy.to_native(ivy.reshape(x_, []))
                    return ivy.to_native(x_)
                else:
                    return ivy.to_ivy(x_)
            else:
                return ivy.to_native(x_)
        return x_

    if ivy.is_array(x):
        return [], map_fn(x)

    x = ivy.nested_map(x, map_fn, include_derived=True)
    arr_idxs = ivy.nested_argwhere(x, lambda x: ivy.is_native_array(x))
    if _check_if_empty(arr_idxs):
        return arr_idxs, []
    else:
        if idxs is not None:
            arr_idxs = [
                arr_idx
                for arr_idx in arr_idxs
                if "_".join(str(x) for x in arr_idx) in _idxs_to_str(idxs)
            ]
        arr_values = ivy.multi_index_nest(x, arr_idxs)
        arr_idxs = _idxs_to_str(arr_idxs)
        return arr_idxs, arr_values


def _set_duplicates(xs, duplicate_index_chains):
    originals = [
        [key_chains[0]] * (len(key_chains) - 1) for key_chains in duplicate_index_chains
    ]
    originals = ivy.multi_index_nest(xs, list(itertools.chain(*originals)))
    duplicates = [list(index_chains[1:]) for index_chains in duplicate_index_chains]
    nullifying_index_chains = (
        [index_chain.split("/") for index_chain in list(itertools.chain(*duplicates))]
        if isinstance(xs, ivy.Container)
        else list(itertools.chain(*duplicates))
    )
    ivy.set_nest_at_indices(xs, nullifying_index_chains, originals)
    return xs


def _stop_grad_and_index(func_ret, retain_grads, grads):
    if not retain_grads:
        if ivy.is_array(func_ret):
            func_ret = ivy.stop_gradient(func_ret)
        else:
            func_ret = ivy.nested_map(
                func_ret,
                lambda x: ivy.stop_gradient(x) if ivy.is_array(x) else x,
                include_derived=True,
            )
    if isinstance(grads, dict):
        grads = ivy.Container(grads)
    return func_ret, grads


# Private Variable Helpers #
# -------------------------#


def _variable(x):
    x = ivy.to_native(x, nested=True)
    ret = ivy.nested_map(x, current_backend(x).variable, include_derived=True)
    return ivy.nested_map(ret, ivy.to_ivy, include_derived=True)


def _is_variable(x, exclusive=False) -> bool:
    x = ivy.to_native(x, nested=True)
    return ivy.nested_map(
        x,
        lambda x: current_backend(x).is_variable(x, exclusive=exclusive),
        include_derived=True,
    )


def _variable_data(x):
    x = ivy.to_native(x, nested=True)
    return ivy.nested_map(
        x, lambda x: current_backend(x).variable_data(x), include_derived=True
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
@handle_exceptions
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
        If with_grads is neither None nor boolean, it will raise an IvyException

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
        ivy.assertions.check_elem_in_list(with_grads, [True, False])
        return with_grads
    global with_grads_stack
    if not with_grads_stack:
        with_grads_stack = [True]
    return with_grads_stack[-1]


# noinspection PyShadowingNames
@handle_exceptions
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
        If with_grads is neither None nor boolean, it will raise an IvyException

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
    ivy.assertions.check_elem_in_list(with_grads, [True, False])
    global with_grads_stack
    with_grads_stack.append(with_grads)


@handle_exceptions
def unset_with_grads():
    """
    Enter a nested code space where gradients are computed. This method
    deletes the with_grads component from the global list with_grads_stack

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
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
        Whether to preserve gradient computation on ivy.Array instances. Default is
        True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The same array x, but with no gradient information.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.stop_gradient(x, preserve_type=True)
    >>> print(y)
    ivy.array([1., 2., 3.])

    >>> x = ivy.zeros((2, 3))
    >>> ivy.stop_gradient(x, preserve_type=False, out=x)
    >>> print(x)
    ivy.array([[0., 0., 0.],
               [0., 0., 0.]])

    With one :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> y = ivy.stop_gradient(x, preserve_type=False)
    >>> print(y)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> ivy.stop_gradient(x, preserve_type=True, out=x)
    >>> print(x)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    """
    return current_backend(x).stop_gradient(x, preserve_type=preserve_type, out=out)


# AutoGrad #


@inputs_to_ivy_arrays
@handle_exceptions
@handle_array_like
def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    """Call function func with input of xs variables, and return the function result
    func_ret and the gradients of each output variable w.r.t each input variable,

    Parameters
    ----------
    func
        Function for which we compute the gradients of the output with respect to xs
        input.
    xs
        Variables for which to compute the function gradients with respective to. This
        can be a single array or an arbitrary nest of arrays.
    retain_grads
        Whether to retain the gradients of the returned values. (Default value = False)
    xs_grad_idxs
        Indices of the input arrays to compute gradients with respect to. If None,
        gradients are returned with respect to all input arrays. (Default value = None)
    ret_grad_idxs
        Indices of the returned arrays for which to return computed gradients. If None,
        gradients are returned for all returned arrays. (Default value = None)

    Returns
    -------
    ret
        the function result func_ret and a dictionary of gradients of each output
        variable w.r.t each input variable.

    """
    return current_backend(None).execute_with_gradients(
        func,
        xs,
        retain_grads=retain_grads,
        xs_grad_idxs=xs_grad_idxs,
        ret_grad_idxs=ret_grad_idxs,
    )


execute_with_gradients.computes_gradients = True


@to_native_arrays_and_back
@handle_exceptions
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
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[4.6, 2.1, 5], [2.8, 1.3, 6.2]])
    >>> func = lambda x: ivy.mean(ivy.square(x))
    >>> grad_fn = ivy.value_and_grad(func)
    >>> value_grad = grad_fn(x)
    >>> print(value_grad)
    (ivy.array(16.423332), ivy.array([[1.53, 0.7, 1.67], [0.933, 0.433, 2.07]]))

    """
    return current_backend(None).value_and_grad(func)


value_and_grad.computes_gradients = True


@to_native_arrays_and_back
@handle_exceptions
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
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[4.6, 2.1, 5], [2.8, 1.3, 6.2]])
    >>> func = lambda x: ivy.mean(ivy.square(x))
    >>> jac_fn = ivy.jac(func)
    >>> jacobian = jac_fn(x)
    >>> print(jacobian)
    ivy.array([[1.53 , 0.7  , 1.67 ],
    ...        [0.933, 0.433, 2.07 ]])

    """
    return current_backend(None).jac(func)


jac.computes_gradients = True


@to_native_arrays_and_back
@handle_exceptions
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
    ivy.array([[1.53 , 0.7  , 1.67 ],
    ...        [0.933, 0.433, 2.07 ]])

    """
    return current_backend(None).grad(func)


grad.computes_gradients = True


# Optimizer Steps #


@inputs_to_ivy_arrays
@handle_exceptions
@handle_array_like
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
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
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
    With :class:`ivy.Array` inputs:

    >>> dcdw = ivy.array([1, 2, 3])
    >>> mw = ivy.ones(3)
    >>> vw = ivy.ones(1)
    >>> step = ivy.array(3)
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step)
    >>> print(adam_step_delta)
    (ivy.array([0.2020105,0.22187898,0.24144873]),
        ivy.array([1.,1.10000002,1.20000005]),
        ivy.array([1.,1.00300002,1.00800002]))

    >>> dcdw = ivy.array([[1., 4., -3.], [2., 3., 0.5]])
    >>> mw = ivy.zeros((2,3))
    >>> vw = ivy.zeros(3)
    >>> step = ivy.array(1)
    >>> beta1 = 0.86
    >>> beta2 = 0.95
    >>> epsilon = 1e-6
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1=beta1, beta2=beta2,
    ...                                 epsilon=epsilon)
    >>> print(adam_step_delta)
    (ivy.array([[ 1.,  1., -1.],
    ...         [ 1.,  1.,  1.]]),
    ... ivy.array([[ 0.14,  0.56, -0.42],
    ...            [ 0.28,  0.42,  0.07]]),
     ivy.array([[0.05  , 0.8   , 0.45  ],
                [0.2   , 0.45  , 0.0125]]))

    >>> dcdw = ivy.array([1, -2, 3])
    >>> mw = ivy.ones(1)
    >>> vw = ivy.ones(1)
    >>> step = ivy.array(3.6)
    >>> out = ivy.zeros_like(dcdw)
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, out=out)
    >>> print(out)
        ivy.array([0.171, 0.171, 0.171])

    With one :class:`ivy.Container` input:

    >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                      b=ivy.array([3., 4., 5.]))
    >>> mw = ivy.array([1., 4., 9.])
    >>> vw = ivy.array([0.,])
    >>> step = ivy.array([3.4])
    >>> beta1 = 0.87
    >>> beta2 = 0.976
    >>> epsilon = 1e-5
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1=beta1, beta2=beta2,
    ...                                 epsilon=epsilon)
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

    With multiple :class:`ivy.Container` inputs:

    >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                      b=ivy.array([3., 4., 5.]))
    >>> mw = ivy.Container(a=ivy.array([0., 0., 0.]),
    ...                    b=ivy.array([0., 0., 0.]))
    >>> vw = ivy.Container(a=ivy.array([0.,]),
    ...                    b=ivy.array([0.,]))
    >>> step = ivy.array([3.4])
    >>> beta1 = 0.87
    >>> beta2 = 0.976
    >>> epsilon = 1e-5
    >>> adam_step_delta = ivy.adam_step(dcdw, mw, vw, step, beta1=beta1, beta2=beta2,
    ...                                 epsilon=epsilon)
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
    vw_sqrt = ivy.maximum(vw, 0.0) ** 0.5
    beta1_pow = beta1**step
    beta2_pow = beta2**step
    alpha = (1 - beta2_pow) ** 0.5 / (1 - beta1_pow + epsilon)
    return ivy.divide(alpha * mw, vw_sqrt + epsilon, out=out), mw, vw


adam_step.out_index = 0


# Optimizer Updates #


@inputs_to_ivy_arrays
@handle_exceptions
@handle_array_like
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
        Default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The new function weights ws_new, following the optimizer updates.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> w = ivy.array([1., 2., 3.])
    >>> effective_grad = ivy.zeros(3)
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr)
    >>> print(ws_new)
    ivy.array([1., 2., 3.])

    >>> w = ivy.array([1., 2., 3.])
    >>> effective_grad = ivy.zeros(3)
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr,
    ...                               out=None, stop_gradients=True)
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
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr,
    ...                               stop_gradients=False, out=out)
    >>> print(out)
    ivy.array([0.999, 2.   , 3.   ])

    With one :class:`ivy.Container` input:

    >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> effective_grad = ivy.array([0., 0., 0.])
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr)
    >>> print(ws_new)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> effective_grad = ivy.Container(a=ivy.array([0., 0., 0.]),
    ...                                b=ivy.array([0., 0., 0.]))
    >>> lr = 3e-4
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr, out=w)
    >>> print(w)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    >>> w = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> effective_grad = ivy.Container(a=ivy.array([0., 0., 0.]),
    ...                                b=ivy.array([0., 0., 0.]))
    >>> lr = ivy.array([3e-4])
    >>> ws_new = ivy.optimizer_update(w, effective_grad, lr,
    ...                               stop_gradients=False)
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
@handle_exceptions
@handle_array_like
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
        Default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The new weights, following the gradient descent updates.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> w = ivy.array([[1., 2, 3],
    ...                [4, 6, 1],
    ...                [1, 0, 7]])
    >>> dcdw = ivy.array([[0.5, 0.2, 0.1],
    ...                   [0.3, 0.6, 0.4],
    ...                   [0.4, 0.7, 0.2]])
    >>> lr = ivy.array(0.1)
    >>> new_weights = ivy.gradient_descent_update(w, dcdw, lr, stop_gradients=True)
    >>> print(new_weights)
    ivy.array([[ 0.95,  1.98,  2.99],
    ...        [ 3.97,  5.94,  0.96],
    ...        [ 0.96, -0.07,  6.98]])

    >>> w = ivy.array([1., 2., 3.])
    >>> dcdw = ivy.array([0.5, 0.2, 0.1])
    >>> lr = ivy.array(0.3)
    >>> out = ivy.zeros_like(w)
    >>> ivy.gradient_descent_update(w, dcdw, lr, out=out)
    >>> print(out)
    ivy.array([0.85, 1.94, 2.97])

    With one :class:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),
    ...                   b=ivy.array([3.48, 5.72, 1.98]))
    >>> dcdw = ivy.array([0.5, 0.2, 0.1])
    >>> lr = ivy.array(0.3)
    >>> w_new = ivy.gradient_descent_update(w, dcdw, lr)
    >>> print(w_new)
    {
        a: ivy.array([0.85, 1.94, 2.97]),
        b: ivy.array([3.33, 5.66, 1.95])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([1., 2., 3.]),
    ...                   b=ivy.array([3.48, 5.72, 1.98]))
    >>> dcdw = ivy.Container(a=ivy.array([0.5, 0.2, 0.1]),
    ...                      b=ivy.array([2., 3.42, 1.69]))
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
@handle_exceptions
@handle_array_like
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
        Default is ``True``.
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
@handle_exceptions
@handle_array_like
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
        Default is ``True``.
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
@handle_exceptions
@handle_array_like
@handle_array_like
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
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
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
        Default is ``True``.
    out
        optional output array, for writing the new function weights ws_new to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The new function weights ws_new, following the LAMB updates.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> w = ivy.array([1., 2, 3])
    >>> dcdw = ivy.array([0.5,0.2,0.1])
    >>> lr = ivy.array(0.1)
    >>> vw_tm1 = ivy.zeros(1)
    >>> mw_tm1 = ivy.zeros(3)
    >>> step = ivy.array(1)
    >>> new_weights = ivy.lamb_update(w, dcdw, lr, mw_tm1, vw_tm1, step)
    >>> print(new_weights)
    (ivy.array([0.784, 1.78 , 2.78 ]),
    ... ivy.array([0.05, 0.02, 0.01]),
    ... ivy.array([2.5e-04, 4.0e-05, 1.0e-05]))

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
    >>> new_weights = ivy.lamb_update(w, dcdw, lr, mw_tm1, vw_tm1, step, beta1=beta1,
    ...                               beta2=beta2, epsilon=epsilon,
    ...                               max_trust_ratio=max_trust_ratio,
    ...                               decay_lambda=decay_lambda, out=out,
    ...                               stop_gradients=stop_gradients)
    >>> print(out)
    ivy.array([[ 0.639,  1.64 ,  2.64 ],
    ...        [ 3.64 ,  5.64 ,  0.639],
    ...        [ 0.639, -0.361,  6.64 ]])

    With one :class:`ivy.Container` inputs:

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

    With multiple :class:`ivy.Container` inputs:

    >>> w = ivy.Container(a=ivy.array([1.,3.,5.]),
    ...                   b=ivy.array([3.,4.,2.]))
    >>> dcdw = ivy.Container(a=ivy.array([0.2,0.3,0.6]),
    ...                      b=ivy.array([0.6,0.4,0.7]))
    >>> mw_tm1 = ivy.Container(a=ivy.array([0.,0.,0.]),
    ...                        b=ivy.array([0.,0.,0.]))

    >>> vw_tm1 = ivy.Container(a=ivy.array([0.,]),
    ...                        b=ivy.array([0.,]))
    >>> step = ivy.array([3.4])
    >>> beta1 = 0.9
    >>> beta2 = 0.999
    >>> epsilon = 1e-7
    >>> max_trust_ratio = 10
    >>> decay_lambda = 0
    >>> stop_gradients = True
    >>> lr = ivy.array(0.5)
    >>> new_weights = ivy.lamb_update(w, dcdw, lr, mw_tm1, vw_tm1, step, beta1=beta1,
    ...                               beta2=beta2, epsilon=epsilon,
    ...                               max_trust_ratio=max_trust_ratio,
    ...                               decay_lambda=decay_lambda,
    ...                               stop_gradients=stop_gradients)
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
    r = ivy.minimum(ivy.stable_divide(r1, r2), ivy.array(max_trust_ratio))
    lr = r * lr
    return (
        ivy.optimizer_update(w, eff_grads, lr, stop_gradients=stop_gradients, out=out),
        mw,
        vw,
    )


lamb_update.out_index = 0
