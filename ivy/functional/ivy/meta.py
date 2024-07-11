# global
import ivy
from ivy.func_wrapper import handle_array_function
from ivy.functional.ivy.gradients import gradient_descent_update
from ivy.utils.exceptions import handle_exceptions

# local
from typing import Optional, Union, Callable, Tuple, Any

# Extra #
# ------#

# Private #


def _compute_cost_and_update_grads(
    cost_fn,
    order,
    batch,
    variables,
    outer_v,
    keep_outer_v,
    average_across_steps_or_final,
    all_grads,
    unique_outer,
    batched,
    num_tasks,
):
    """Compute cost and update gradients.

    This function computes the cost and updates gradients for optimization.

    Parameters
    ----------
    cost_fn : function
        The cost function.
    order : int
        The order of computation.
    batch : object
        The batch data.
    variables : ivy.Container
        The variables for optimization.
    outer_v : object
        Outer variable.
    keep_outer_v : bool
        Whether to keep outer variable.
    average_across_steps_or_final : bool
        Whether to average across steps or final.
    all_grads : list
        List to accumulate gradients.
    unique_outer : bool
        Whether outer variables are unique.
    batched : bool
        Whether the data is batched.
    num_tasks : int
        Number of tasks.

    Returns
    -------
    object
        The computed cost.

    Examples
    --------
    >>> # Example usage here
    >>> pass
    """
    if order == 1:

        def cost_fn_with_variable(v):
            return cost_fn(
                batch, v=variables.cont_set_at_key_chains(v) if unique_outer else v
            )

        cost, inner_grads = ivy.execute_with_gradients(
            cost_fn_with_variable,
            (
                variables.cont_at_key_chains(outer_v, ignore_none=True)
                if keep_outer_v
                else variables.cont_prune_key_chains(outer_v, ignore_none=True)
            ),
            retain_grads=False,
        )

        var = (
            variables.cont_at_key_chains(outer_v, ignore_none=True)
            if keep_outer_v
            else variables.cont_prune_key_chains(outer_v, ignore_none=True)
        )

        inner_grads = ivy.Container(
            {
                k: ivy.zeros_like(v) if k not in inner_grads else inner_grads[k]
                for k, v in var.cont_to_iterator()
            }
        )

        if batched:
            inner_grads = ivy.multiply(inner_grads, num_tasks)

        if average_across_steps_or_final:
            all_grads.append(inner_grads)
    else:
        cost = cost_fn(batch, v=variables)

    return cost


def _train_task(
    inner_batch,
    outer_batch,
    inner_cost_fn,
    outer_cost_fn,
    variables,
    inner_grad_steps,
    inner_learning_rate,
    inner_optimization_step,
    order,
    average_across_steps,
    inner_v,
    keep_innver_v,
    outer_v,
    keep_outer_v,
    batched,
    num_tasks,
    stop_gradients,
):
    # init
    total_cost = 0
    all_grads = []

    # inner and outer
    unique_inner = inner_v is not None
    unique_outer = outer_v is not None

    # iterate through inner loop training steps
    for i in range(inner_grad_steps):
        # compute inner gradient for update the inner variables
        cost, inner_update_grads = ivy.execute_with_gradients(
            lambda v: inner_cost_fn(
                inner_batch,
                v=variables.cont_set_at_key_chains(v) if unique_inner else v,
            ),
            (
                variables.cont_at_key_chains(inner_v, ignore_none=True)
                if keep_innver_v
                else variables.cont_prune_key_chains(inner_v, ignore_none=True)
            ),
            retain_grads=order > 1,
        )
        var = (
            variables.cont_at_key_chains(inner_v, ignore_none=True)
            if keep_innver_v
            else variables.cont_prune_key_chains(inner_v, ignore_none=True)
        )
        inner_update_grads = ivy.Container(
            {
                k: (
                    ivy.zeros_like(v)
                    if k not in inner_update_grads
                    else inner_update_grads[k]
                )
                for k, v in var.cont_to_iterator()
            }
        )
        if batched:
            inner_update_grads = ivy.multiply(inner_update_grads, num_tasks)

        # compute the cost to be optimized, and update all_grads if fist order method
        if outer_cost_fn is None and not unique_inner and not unique_outer:
            all_grads.append(inner_update_grads)
        else:
            cost = _compute_cost_and_update_grads(
                inner_cost_fn if outer_cost_fn is None else outer_cost_fn,
                order,
                outer_batch,
                variables,
                outer_v,
                keep_outer_v,
                average_across_steps,
                all_grads,
                unique_outer,
                batched,
                num_tasks,
            )

        # update cost and update parameters
        total_cost = total_cost + cost
        if unique_inner:
            variables = variables.cont_set_at_key_chains(
                inner_optimization_step(
                    (
                        variables.cont_at_key_chains(inner_v)
                        if keep_innver_v
                        else variables.cont_prune_key_chains(inner_v)
                    ),
                    inner_update_grads,
                    inner_learning_rate,
                    stop_gradients=stop_gradients,
                )
            )
        else:
            variables = inner_optimization_step(
                variables,
                inner_update_grads,
                inner_learning_rate,
                stop_gradients=stop_gradients,
            )

    # once training is finished, compute the final cost, and update
    # all_grads if fist order method
    final_cost = _compute_cost_and_update_grads(
        inner_cost_fn if outer_cost_fn is None else outer_cost_fn,
        order,
        outer_batch,
        variables,
        outer_v,
        keep_outer_v,
        True,
        all_grads,
        unique_outer,
        batched,
        num_tasks,
    )

    # update variables
    if stop_gradients:
        variables = variables.stop_gradient()
    if not batched:
        variables = variables.expand_dims(axis=0)

    # average the cost or gradients across all timesteps if this option is chosen
    if average_across_steps:
        total_cost = total_cost + final_cost
        if order == 1:
            all_grads = sum(all_grads) / max(len(all_grads), 1)
        return total_cost / (inner_grad_steps + 1), variables, all_grads

    # else return only the final values
    if order == 1:
        all_grads = all_grads[-1]
    return final_cost, variables, all_grads


def _train_tasks_batched(
    batch,
    inner_batch_fn,
    outer_batch_fn,
    inner_cost_fn,
    outer_cost_fn,
    variables,
    inner_grad_steps,
    inner_learning_rate,
    inner_optimization_step,
    order,
    average_across_steps,
    inner_v,
    keep_innver_v,
    outer_v,
    keep_outer_v,
    return_inner_v,
    num_tasks,
    stop_gradients,
):
    """Train tasks in a batched manner.

    This function trains tasks in a batched manner with optional inner and outer batch functions.

    Parameters
    ----------
    batch : object
        The batch data.
    inner_batch_fn : function or None
        The inner batch function.
    outer_batch_fn : function or None
        The outer batch function.
    inner_cost_fn : function
        The inner cost function.
    outer_cost_fn : function
        The outer cost function.
    variables : ivy.Container
        The variables for optimization.
    inner_grad_steps : int
        Number of inner gradient steps.
    inner_learning_rate : float
        Inner learning rate.
    inner_optimization_step : function
        The inner optimization step function.
    order : int
        The order of computation.
    average_across_steps : bool
        Whether to average across steps.
    inner_v : object
        Inner variable.
    keep_inner_v : bool
        Whether to keep inner variable.
    outer_v : object
        Outer variable.
    keep_outer_v : bool
        Whether to keep outer variable.
    return_inner_v : str or bool
        Whether and which inner variables to return.
    num_tasks : int
        Number of tasks.
    stop_gradients : bool
        Whether to stop gradients during optimization.

    Returns
    -------
    object or tuple
        The computed cost and, optionally, gradients and updated inner variables.

    Examples
    --------
    >>> # Example usage here
    >>> pass
    """  # noqa: E501 (Line too long)
    inner_batch = batch
    outer_batch = batch
    if inner_batch_fn is not None:
        inner_batch = inner_batch_fn(inner_batch)
    if outer_batch_fn is not None:
        outer_batch = outer_batch_fn(outer_batch)

    cost, updated_ivs, grads = _train_task(
        inner_batch,
        outer_batch,
        inner_cost_fn,
        outer_cost_fn,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        inner_optimization_step,
        order,
        average_across_steps,
        inner_v,
        keep_innver_v,
        outer_v,
        keep_outer_v,
        True,
        num_tasks,
        stop_gradients,
    )
    grads = grads.mean(axis=0) if isinstance(grads, ivy.Container) else grads
    if order == 1:
        if return_inner_v in ["all", True]:
            return cost, grads, updated_ivs
        elif return_inner_v == "first":
            return cost, grads, updated_ivs[0:1]
        return cost, grads
    if return_inner_v in ["all", True]:
        return cost, updated_ivs
    elif return_inner_v == "first":
        return cost, updated_ivs[0:1]
    return cost


def _train_tasks_with_for_loop(
    batch,
    inner_sub_batch_fn,
    outer_sub_batch_fn,
    inner_cost_fn,
    outer_cost_fn,
    variables,
    inner_grad_steps,
    inner_learning_rate,
    inner_optimization_step,
    order,
    average_across_steps,
    inner_v,
    keep_innver_v,
    outer_v,
    keep_outer_v,
    return_inner_v,
    num_tasks,
    stop_gradients,
):
    total_cost = 0
    updated_ivs_to_return = []
    all_grads = []
    if isinstance(inner_v, (list, tuple)) and isinstance(
        inner_v[0], (list, tuple, dict, type(None))
    ):
        inner_v_seq = True
    else:
        inner_v_seq = False
    if isinstance(outer_v, (list, tuple)) and isinstance(
        outer_v[0], (list, tuple, dict, type(None))
    ):
        outer_v_seq = True
    else:
        outer_v_seq = False
    for i, sub_batch in enumerate(batch.cont_unstack_conts(0, True, num_tasks)):
        if inner_sub_batch_fn is not None:
            inner_sub_batch = inner_sub_batch_fn(sub_batch)
        else:
            inner_sub_batch = sub_batch
        if outer_sub_batch_fn is not None:
            outer_sub_batch = outer_sub_batch_fn(sub_batch)
        else:
            outer_sub_batch = sub_batch
        iv = inner_v[i] if inner_v_seq else inner_v
        ov = outer_v[i] if outer_v_seq else outer_v
        cost, updated_iv, grads = _train_task(
            inner_sub_batch,
            outer_sub_batch,
            inner_cost_fn,
            outer_cost_fn,
            variables,
            inner_grad_steps,
            inner_learning_rate,
            inner_optimization_step,
            order,
            average_across_steps,
            iv,
            keep_innver_v,
            ov,
            keep_outer_v,
            False,
            num_tasks,
            stop_gradients,
        )
        if (return_inner_v == "first" and i == 0) or return_inner_v in ["all", True]:
            updated_ivs_to_return.append(updated_iv)
        total_cost = total_cost + cost
        all_grads.append(grads)
    if order == 1:
        if return_inner_v:
            return (
                total_cost / num_tasks,
                sum(all_grads) / num_tasks,
                ivy.concat(updated_ivs_to_return, axis=0),
            )
        return total_cost / num_tasks, sum(all_grads) / num_tasks
    if return_inner_v:
        return total_cost / num_tasks, ivy.concat(updated_ivs_to_return, axis=0)
    return total_cost / num_tasks


def _train_tasks(
    batch,
    inner_batch_fn,
    outer_batch_fn,
    inner_cost_fn,
    outer_cost_fn,
    variables,
    inner_grad_steps,
    inner_learning_rate,
    inner_optimization_step,
    order,
    average_across_steps,
    batched,
    inner_v,
    keep_innver_v,
    outer_v,
    keep_outer_v,
    return_inner_v,
    num_tasks,
    stop_gradients,
):
    if batched:
        return _train_tasks_batched(
            batch,
            inner_batch_fn,
            outer_batch_fn,
            inner_cost_fn,
            outer_cost_fn,
            variables,
            inner_grad_steps,
            inner_learning_rate,
            inner_optimization_step,
            order,
            average_across_steps,
            inner_v,
            keep_innver_v,
            outer_v,
            keep_outer_v,
            return_inner_v,
            num_tasks,
            stop_gradients,
        )
    return _train_tasks_with_for_loop(
        batch,
        inner_batch_fn,
        outer_batch_fn,
        inner_cost_fn,
        outer_cost_fn,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        inner_optimization_step,
        order,
        average_across_steps,
        inner_v,
        keep_innver_v,
        outer_v,
        keep_outer_v,
        return_inner_v,
        num_tasks,
        stop_gradients,
    )


# Public #

# First Order


@handle_exceptions
@handle_array_function
def fomaml_step(
    batch: ivy.Container,
    inner_cost_fn: Callable,
    outer_cost_fn: Callable,
    variables: ivy.Container,
    inner_grad_steps: int,
    inner_learning_rate: float,
    /,
    *,
    inner_optimization_step: Callable = gradient_descent_update,
    inner_batch_fn: Optional[Callable] = None,
    outer_batch_fn: Optional[Callable] = None,
    average_across_steps: bool = False,
    batched: bool = True,
    inner_v: Optional[ivy.Container] = None,
    keep_inner_v: bool = True,
    outer_v: Optional[ivy.Container] = None,
    keep_outer_v: bool = True,
    return_inner_v: Union[str, bool] = False,
    num_tasks: Optional[int] = None,
    stop_gradients: bool = True,
) -> Tuple[ivy.Array, ivy.Container, Any]:
    """Perform step of first order MAML.

    Parameters
    ----------
    batch
        The input batch
    inner_cost_fn
        callable for the inner loop cost function, receiving sub-batch, inner vars and
        outer vars
    outer_cost_fn
        callable for the outer loop cost function, receiving task-specific sub-batch,
        inner vars and outer vars. If None, the cost from the inner loop will also be
        optimized in the outer loop.
    variables
        Variables to be optimized during the meta step
    inner_grad_steps
        Number of gradient steps to perform during the inner loop.
    inner_learning_rate
        The learning rate of the inner loop.
    inner_optimization_step
        The function used for the inner loop optimization.
        Default is ivy.gradient_descent_update.
    inner_batch_fn
        Function to apply to the task sub-batch, before passing to the inner_cost_fn.
        Default is ``None``.
    outer_batch_fn
        Function to apply to the task sub-batch, before passing to the outer_cost_fn.
        Default is ``None``.
    average_across_steps
        Whether to average the inner loop steps for the outer loop update.
        Default is ``False``.
    batched
        Whether to batch along the time dimension, and run the meta steps in batch.
        Default is ``True``.
    inner_v
        Nested variable keys to be optimized during the inner loop, with same keys and
        boolean values. (Default value = None)
    keep_inner_v
        If True, the key chains in inner_v will be kept, otherwise they will be removed.
        Default is ``True``.
    outer_v
        Nested variable keys to be optimized during the inner loop, with same keys and
        boolean values.
        (Default value = None)
    keep_outer_v
        If True, the key chains in inner_v will be kept, otherwise they will be removed.
        Default is ``True``.
    return_inner_v
        Either 'first', 'all', or False. 'first' means the variables for the first task
        inner loop will also be returned. variables for all tasks will be returned with
        'all'.
        Default is ``False``.
    num_tasks
        Number of unique tasks to inner-loop optimize for the meta step. Determined from
        batch by default.
    stop_gradients
        Whether to stop the gradients of the cost.
        Default is ``True``.

    Returns
    -------
    ret
        The cost and the gradients with respect to the outer loop variables.
    """
    if num_tasks is None:
        num_tasks = batch.cont_shape[0]
    rets = _train_tasks(
        batch,
        inner_batch_fn,
        outer_batch_fn,
        inner_cost_fn,
        outer_cost_fn,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        inner_optimization_step,
        1,
        average_across_steps,
        batched,
        inner_v,
        keep_inner_v,
        outer_v,
        keep_outer_v,
        return_inner_v,
        num_tasks,
        stop_gradients,
    )
    cost = rets[0]
    if stop_gradients:
        cost = ivy.stop_gradient(cost, preserve_type=False)
    grads = rets[1]
    if return_inner_v:
        return cost, grads, rets[2]
    return cost, grads


fomaml_step.computes_gradients = True


@handle_exceptions
@handle_array_function
def reptile_step(
    batch: ivy.Container,
    cost_fn: Callable,
    variables: ivy.Container,
    inner_grad_steps: int,
    inner_learning_rate: float,
    /,
    *,
    inner_optimization_step: Callable = gradient_descent_update,
    batched: bool = True,
    return_inner_v: Union[str, bool] = False,
    num_tasks: Optional[int] = None,
    stop_gradients: bool = True,
) -> Tuple[ivy.Array, ivy.Container, Any]:
    """Perform a step of Reptile.

    Parameters
    ----------
    batch
        The input batch.
    cost_fn
        The cost function that receives the task-specific sub-batch and variables, and
        returns the cost.
    variables
        Variables to be optimized.
    inner_grad_steps
        Number of gradient steps to perform during the inner loop.
    inner_learning_rate
        The learning rate of the inner loop.
    inner_optimization_step
        The function used for the inner loop optimization. It takes the learnable
        weights,the derivative of the cost with respect to the weights, and the learning
        rate as arguments, and returns the updated variables.
        Default is `gradient_descent_update`.
    batched
        Whether to batch along the time dimension and run the meta steps in batch.
        Default is `True`.
    return_inner_v
        Either `'first'`, `'all'`, or `False`. If `'first'`, the variables for the first
        task inner loop will also be returned. If `'all'`, variables for all tasks will
        be returned. Default is `False`.
    num_tasks
        Number of unique tasks to inner-loop optimize for the meta step. Determined from
        the batch by default.
    stop_gradients
        Whether to stop the gradients of the cost. Default is `True`.

    Returns
    -------
    ret
        The cost, the gradients with respect to the outer loop variables, and additional
        information from the inner loop optimization.

    Examples
    --------
    With :class:`ivy.Container` input:

    >>> from ivy.functional.ivy.gradients import gradient_descent_update
    >>> import ivy
    >>> from ivy.functional.ivy.gradients import _variable

    >>> ivy.set_backend("torch")

    >>> def inner_cost_fn(batch_in, v):
    ...     return batch_in.mean().x / v.mean().latent

    >>> num_tasks = 2
    >>> batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})
    >>> variables = ivy.Container({
    ...     "latent": _variable(ivy.repeat(ivy.array([[1.0]]), num_tasks, axis=0))
    ... })

    >>> cost, gradients = ivy.reptile_step(batch, inner_cost_fn, variables, 5, 0.01,
    ...                                    num_tasks=num_tasks)
    >>> print(cost)
    ivy.array(1.4485182)
    >>> print(gradients)
    {
        latent: ivy.array([-139.9569855])
    }

    >>> batch = ivy.Container({"x": ivy.arange(1, 4, dtype="float32")})
    >>> variables = ivy.Container({
    ...     "latent": _variable(ivy.array([1.0, 2.0]))
    ... })

    >>> cost, gradients, firsts = ivy.reptile_step(batch, inner_cost_fn, variables, 4,
    ...                                            0.025, batched=False, num_tasks=2,
    ...                                            return_inner_v='first')
    >>> print(cost)
    ivy.array(0.9880483)
    >>> print(gradients)
    {
        latent: ivy.array([-13.01766968, -13.01766968])
    }
    >>> print(firsts)
    {
        latent: ivy.array([[1.02197957, 2.02197981]])
    }
    """
    if num_tasks is None:
        num_tasks = batch.cont_shape[0]

    rets = _train_tasks(
        batch,
        None,
        None,
        cost_fn,
        None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        inner_optimization_step,
        1,
        True,
        batched,
        None,
        True,
        None,
        True,
        return_inner_v,
        num_tasks,
        stop_gradients,
    )
    cost = rets[0]
    if stop_gradients:
        cost = ivy.stop_gradient(cost, preserve_type=False)
    grads = rets[1] / inner_learning_rate
    if return_inner_v:
        return cost, grads, rets[2]
    return cost, grads


reptile_step.computes_gradients = True


# Second Order


@handle_exceptions
@handle_array_function
def maml_step(
    batch: ivy.Container,
    inner_cost_fn: Callable,
    outer_cost_fn: Callable,
    variables: ivy.Container,
    inner_grad_steps: int,
    inner_learning_rate: float,
    /,
    *,
    inner_optimization_step: Callable = gradient_descent_update,
    inner_batch_fn: Optional[Callable] = None,
    outer_batch_fn: Optional[Callable] = None,
    average_across_steps: bool = False,
    batched: bool = True,
    inner_v: Optional[ivy.Container] = None,
    keep_inner_v: bool = True,
    outer_v: Optional[ivy.Container] = None,
    keep_outer_v: bool = True,
    return_inner_v: Union[str, bool] = False,
    num_tasks: Optional[int] = None,
    stop_gradients: bool = True,
) -> Tuple[ivy.Array, ivy.Container, Any]:
    """Perform step of vanilla second order MAML.

    Parameters
    ----------
    batch
        The input batch
    inner_cost_fn
        callable for the inner loop cost function, receiving sub-batch, inner vars and
        outer vars
    outer_cost_fn
        callable for the outer loop cost function, receiving task-specific sub-batch,
        inner vars and outer vars. If None, the cost from the inner loop will also be
        optimized in the outer loop.
    variables
        Variables to be optimized during the meta step
    inner_grad_steps
        Number of gradient steps to perform during the inner loop.
    inner_learning_rate
        The learning rate of the inner loop.
    inner_optimization_step
        The function used for the inner loop optimization.
        Default is ivy.gradient_descent_update.
    inner_batch_fn
        Function to apply to the task sub-batch, before passing to the inner_cost_fn.
        Default is ``None``.
    outer_batch_fn
        Function to apply to the task sub-batch, before passing to the outer_cost_fn.
        Default is ``None``.
    average_across_steps
        Whether to average the inner loop steps for the outer loop update.
        Default is ``False``.
    batched
        Whether to batch along the time dimension, and run the meta steps in batch.
        Default is ``True``.
    inner_v
        Nested variable keys to be optimized during the inner loop, with same keys and
        boolean values. (Default value = None)
    keep_inner_v
        If True, the key chains in inner_v will be kept, otherwise they will be removed.
        Default is ``True``.
    outer_v
        Nested variable keys to be optimized during the inner loop, with same keys and
        boolean values. (Default value = None)
    keep_outer_v
        If True, the key chains in inner_v will be kept, otherwise they will be removed.
        Default is ``True``.
    return_inner_v
        Either 'first', 'all', or False. 'first' means the variables for the first task
        inner loop will also be returned. variables for all tasks will be returned with
        'all'. Default is ``False``.
    num_tasks
        Number of unique tasks to inner-loop optimize for the meta step. Determined from
        batch by default.
    stop_gradients
        Whether to stop the gradients of the cost. Default is ``True``.

    Returns
    -------
    ret
        The cost and the gradients with respect to the outer loop variables.

    Examples
    --------
    With :class:`ivy.Container` input:

    >>> import ivy
    >>> from ivy.functional.ivy.gradients import _variable

    >>> ivy.set_backend("torch")

    >>> def inner_cost_fn(sub_batch, v):
    ...     return sub_batch.mean().x / v.mean().latent
    >>> def outer_cost_fn(sub_batch,v):
    ...     return sub_batch.mean().x / v.mean().latent

    >>> num_tasks = 2
    >>> batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})
    >>> variables = ivy.Container({
    ...     "latent": _variable(ivy.repeat(ivy.array([[1.0]]), num_tasks, axis=0))
    ... })

    >>> cost = ivy.maml_step(batch, inner_cost_fn, outer_cost_fn, variables, 5, 0.01)
    >>> print(cost)
    (ivy.array(1.40069818), {
    latent: ivy.array([-1.13723135])
    }, ())
    """
    if num_tasks is None:
        num_tasks = batch.cont_shape[0]
    unique_outer = outer_v is not None
    func_ret, grads = ivy.execute_with_gradients(
        lambda v: _train_tasks(
            batch,
            inner_batch_fn,
            outer_batch_fn,
            inner_cost_fn,
            outer_cost_fn,
            variables.cont_set_at_key_chains(v) if unique_outer else v,
            inner_grad_steps,
            inner_learning_rate,
            inner_optimization_step,
            2,
            average_across_steps,
            batched,
            inner_v,
            keep_inner_v,
            outer_v,
            keep_outer_v,
            return_inner_v,
            num_tasks,
            False,
        ),
        (
            variables.cont_at_key_chains(outer_v, ignore_none=True)
            if keep_outer_v
            else variables.cont_prune_key_chains(outer_v, ignore_none=True)
        ),
    )
    if isinstance(func_ret, tuple):
        grads = grads["0"] if "0" in grads else grads
        cost = func_ret[0]
        rest = func_ret[1]
    else:
        cost = func_ret
        rest = ()
    if stop_gradients:
        cost = ivy.stop_gradient(cost, preserve_type=False)
    return cost, grads.sum(axis=0), rest


maml_step.computes_gradients = True
