"""Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from typing import Sequence, Union, Optional, Callable

# local
import ivy
from ivy.functional.ivy.gradients import (
    _get_required_float_variables,
    _get_y_and_ret_idxs,
    _get_native_y,
    _set_duplicates,
    _process_func_ret_and_grads,
)


def variable(x, /):
    with tf.device(ivy.dev(x, as_native=True)):
        return tf.Variable(x, trainable=True)


def is_variable(x, /, *, exclusive=False):
    return isinstance(x, tf.Variable)


def variable_data(x: tf.Variable, /) -> tf.Variable:
    return x.value()


def _grad_func(y, xs, xs_required, tape):
    """Gradient calculation function."""
    # Creating a zero gradient nest for the case where no gradients are computed
    grads_ = ivy.nested_map(
        xs_required,
        lambda x: ivy.to_native(ivy.zeros_like(x)),
        include_derived=True,
        shallow=False,
    )

    # Gradient calculation
    grads = tape.gradient(y, xs_required)

    # Returning zeros if no gradients are computed for consistent results
    if isinstance(xs, ivy.NativeArray):
        grads = grads_ if grads is None else grads
    else:
        grads = ivy.nested_map(
            grads,
            lambda x: 0 if x is None else x,
            include_derived=True,
        )
        if isinstance(grads, ivy.Container):
            grads += grads_
        else:
            grads = ivy.nested_multi_map(lambda x, _: (x[0] + x[1]), [grads, grads_])
    return grads


def execute_with_gradients(
    func,
    xs: Union[tf.Tensor, tf.Variable],
    /,
    *,
    retain_grads: bool = False,
    xs_grad_idxs: Optional[Sequence[Sequence[Union[str, int]]]] = None,
    ret_grad_idxs: Optional[Sequence[Sequence[Union[str, int]]]] = None,
):
    # Conversion of required arrays to float variables and duplicate index chains
    xs, xs_required, required_duplicate_index_chains, _ = _get_required_float_variables(
        xs, xs_grad_idxs
    )

    # Creating a tape to record operations
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(xs_required)
        func_ret = func(xs)

    # Getting the relevant outputs from the function return for gradient calculation
    y, ret_idxs = _get_y_and_ret_idxs(func_ret, ret_grad_idxs, reshape=False)

    if isinstance(y, ivy.NativeArray):
        # Gradient calculation for a single output
        grads = _set_duplicates(
            ivy.to_ivy(_grad_func(y, xs, xs_required, tape)),
            required_duplicate_index_chains,
        )
    else:
        # Gradient calculation for multiple outputs
        y = _get_native_y(y)
        grads_ = ivy.nested_map(
            y,
            lambda x: _grad_func(x, xs, xs_required, tape),
            include_derived=True,
            shallow=False,
        )
        grads = grads_
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {
                ret_idxs[i]: _set_duplicates(grad, required_duplicate_index_chains)
                for i, grad in enumerate(grads_)
            }

    # Deleting the tape if not retaining gradients
    if not retain_grads:
        del tape

    # Stop further gradient propagation if not retaining gradients
    return _process_func_ret_and_grads(func_ret, grads, retain_grads)


def value_and_grad(func):
    def grad_fn(xs):
        grads = ivy.nested_map(
            xs, lambda x: ivy.zeros_like(x), include_derived=True, shallow=False
        )
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            xs = ivy.nested_map(xs, lambda x: ivy.to_native(x), include_derived=True)
            tape.watch(xs)
            y = func(xs)
        y = y.to_native(y)
        grads_ = tape.gradient(y, xs)
        grads_ = ivy.nested_map(
            grads_,
            lambda x: ivy.to_ivy(x),
            include_derived=True,
        )
        grads_ = ivy.to_ivy(grads_)
        grad_idxs = ivy.nested_argwhere(grads_, lambda x: ivy.is_ivy_array(x))
        grad_array_vals = list(ivy.multi_index_nest(grads_, grad_idxs))
        xs = ivy.to_ivy(xs)
        if isinstance(xs, ivy.Array):
            grads = grads_
        else:
            ivy.set_nest_at_indices(grads, grad_idxs, grad_array_vals)
        y = ivy.to_ivy(y)
        return y, grads

    return grad_fn


def stop_gradient(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    preserve_type: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    is_var = is_variable(x)
    x = tf.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        with tf.GradientTape() as tape:
            x_in = ivy.to_native(x_in)
            tape.watch(x_in)
            y = grad_fn(x_in)
        return ivy.to_ivy(tape.jacobian(y, x_in))

    return callback_fn


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        with tf.GradientTape() as tape:
            x_in = ivy.to_native(ivy.array(x_in))
            tape.watch(x_in)
            y = grad_fn(x_in)
        return ivy.to_ivy(tape.gradient(y, x_in))

    return callback_fn
