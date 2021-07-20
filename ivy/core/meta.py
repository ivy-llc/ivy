# global
import ivy
from ivy.core.gradients import gradient_descent_update


# Private #
# --------#

# unique variables for inner and outer loop

def _train_task_w_unique_v(sub_batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, inner_grad_steps,
                           inner_learning_rate, inner_optimization_step, order, average_across_steps):

    # init
    total_cost = 0
    all_grads = list()

    # iterate through inner loop training steps
    for i in range(inner_grad_steps):

        # compute inner gradient for update the inner variables
        _, inner_update_grads = ivy.execute_with_gradients(lambda v: inner_cost_fn(sub_batch, v, outer_v), inner_v,
                                                           retain_grads=order > 1)

        # store inner gradients if inner cost
        if outer_cost_fn is None:
            if order == 1:
                cost, inner_grads = ivy.execute_with_gradients(lambda v: inner_cost_fn(sub_batch, inner_v, v), outer_v,
                                                               retain_grads=order > 1)
                if average_across_steps:
                    all_grads.append(inner_grads)
            else:
                cost = inner_cost_fn(sub_batch, inner_v, outer_v)

        # else, update cost if using outer cost function
        else:

            # compute gradients if first order
            if order == 1:
                cost, outer_grads = ivy.execute_with_gradients(lambda v: outer_cost_fn(sub_batch, inner_v, v), outer_v,
                                                               retain_grads=False)
                if average_across_steps:
                    all_grads.append(outer_grads)
            # otherwise just compute the cost, and an outer optimizer will compute higher order gradients
            else:
                cost = outer_cost_fn(sub_batch, inner_v, outer_v)

        # update cost and update parameters
        total_cost = total_cost + cost
        inner_v = inner_optimization_step(inner_v, inner_update_grads, inner_learning_rate, inplace=False)

        # stop gradient in variables if first order
        if order == 1:
            inner_v = inner_v.map(lambda x, kc: ivy.variable(ivy.stop_gradient(x)))

    # once training is finished, compute the final cost
    if outer_cost_fn is None:
        if order == 1:
            final_cost, inner_grads = ivy.execute_with_gradients(
                lambda v: inner_cost_fn(sub_batch, inner_v, v), outer_v, retain_grads=False)
            all_grads.append(inner_grads)
        else:
            final_cost = inner_cost_fn(sub_batch, inner_v, outer_v)
    else:
        if order == 1:
            final_cost, outer_grads = ivy.execute_with_gradients(
                lambda v: outer_cost_fn(sub_batch, inner_v, v), outer_v, retain_grads=False)
            all_grads.append(outer_grads)
        else:
            final_cost = outer_cost_fn(sub_batch, inner_v, outer_v)

    # average the cost or gradients across all timesteps if this option is chosen
    if average_across_steps:
        total_cost = total_cost + final_cost
        if order == 1:
            all_grads = sum(all_grads) / max(len(all_grads), 1)
        return total_cost / (inner_grad_steps + 1), inner_v, all_grads

    # else return only the final values
    if order == 1:
        all_grads = all_grads[-1]
    return final_cost, inner_v, all_grads


def _train_tasks_w_unique_v(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, num_tasks, inner_grad_steps,
                            inner_learning_rate, inner_optimization_step, order, average_across_steps):
    total_cost = 0
    all_grads = list()
    inner_v_orig = inner_v.map(lambda x, kc: ivy.stop_gradient(x))
    for sub_batch in batch.unstack(0, num_tasks):
        inner_v = inner_v_orig.map(lambda x, kc: ivy.variable(x))
        cost, inner_v, grads = _train_task_w_unique_v(sub_batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v,
                                                      inner_grad_steps, inner_learning_rate, inner_optimization_step,
                                                      order, average_across_steps)
        total_cost = total_cost + cost
        all_grads.append(grads)
    if order == 1:
        return total_cost / num_tasks, sum(all_grads) / len(all_grads)
    return total_cost / num_tasks


# shared variables for inner and outer loop

def _train_task_w_shared_v(sub_batch, inner_cost_fn, outer_cost_fn, v, inner_grad_steps,
                           inner_learning_rate, inner_optimization_step, order, average_across_steps):

    # init
    total_cost = 0
    all_grads = list()

    # iterate through inner loop training steps
    for i in range(inner_grad_steps):

        # compute cost and inner gradient, retaining grads for higher order optimization
        cost, inner_grads = ivy.execute_with_gradients(lambda v_: inner_cost_fn(sub_batch, v_), v,
                                                       retain_grads=order > 1)

        # store inner gradients if inner cost
        if outer_cost_fn is None:
            if order == 1 and average_across_steps:
                all_grads.append(inner_grads)

        # else, update cost if using outer cost function
        else:

            # compute gradients if first order
            if order == 1:
                cost, outer_grads = ivy.execute_with_gradients(lambda v_: outer_cost_fn(sub_batch, v_), v,
                                                               retain_grads=False)
                if average_across_steps:
                    all_grads.append(outer_grads)
            # otherwise just compute the cost, and an outer optimizer will compute higher order gradients
            else:
                cost = outer_cost_fn(sub_batch, v)

        # update cost and update parameters
        total_cost = total_cost + cost
        v = inner_optimization_step(v, inner_grads, inner_learning_rate, inplace=False)

        # stop gradient in variables if first order
        if order == 1:
            v = v.map(lambda x, kc: ivy.variable(ivy.stop_gradient(x)))

    # once training is finished, compute the final cost
    if outer_cost_fn is None:
        if order == 1:
            final_cost, inner_grads = ivy.execute_with_gradients(lambda v_: inner_cost_fn(sub_batch, v_), v,
                                                                 retain_grads=False)
            all_grads.append(inner_grads)
        else:
            final_cost = inner_cost_fn(sub_batch, v)
    else:
        if order == 1:
            final_cost, outer_grads = ivy.execute_with_gradients(lambda v_: outer_cost_fn(sub_batch, v_), v,
                                                                 retain_grads=False)
            all_grads.append(outer_grads)
        else:
            final_cost = outer_cost_fn(sub_batch, v)

    # average the cost or gradients across all timesteps if this option is chosen
    if average_across_steps:
        total_cost = total_cost + final_cost
        if order == 1:
            all_grads = sum(all_grads) / max(len(all_grads), 1)
        return total_cost / (inner_grad_steps + 1), v, all_grads

    # else return only the final values
    if order == 1:
        all_grads = all_grads[-1]
    return final_cost, v, all_grads


def _train_tasks_w_shared_v(batch, inner_cost_fn, outer_cost_fn, v, num_tasks, inner_grad_steps,
                            inner_learning_rate, inner_optimization_step, order, average_across_steps):
    total_cost = 0
    all_grads = list()
    for sub_batch in batch.unstack(0, num_tasks):
        if order == 1:
            v_copy = v.map(lambda x, kc: ivy.variable(ivy.stop_gradient(x)))
        else:
            v_copy = v * 1
        cost, v_copy, grads = _train_task_w_shared_v(
            sub_batch, inner_cost_fn, outer_cost_fn, v_copy, inner_grad_steps, inner_learning_rate,
            inner_optimization_step, order, average_across_steps)
        total_cost = total_cost + cost
        all_grads.append(grads)
    if order == 1:
        return total_cost / num_tasks, sum(all_grads) / len(all_grads)
    return total_cost / num_tasks


# Public #
# -------#

# First Order

def fomaml_step(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, num_tasks, inner_grad_steps,
                inner_learning_rate, inner_optimization_step=gradient_descent_update, average_across_steps=False):
    """
    Perform step of first order MAML.

    :param batch: The input batch
    :type batch: ivy.Container
    :param inner_cost_fn: callable for the inner loop cost function, receving task-specific sub-batch,
                            inner vars and outer vars
    :type inner_cost_fn: callable
    :param outer_cost_fn: callable for the outer loop cost function, receving task-specific sub-batch,
                            inner vars and outer vars. If None, the cost from the inner loop will also be
                            optimized in the outer loop.
    :type outer_cost_fn: callable, optional
    :param inner_v: Variables to be optimized during the inner loop
    :type inner_v: ivy.Container
    :param outer_v: Variables to be optimized during the outer loop. If None, the same variables are optimized for
                    in the inner loop and outer loop.
    :type outer_v: ivy.Container, optional
    :param num_tasks: Number of unique tasks to inner-loop optimize for during the meta step.
                        This must be the leading size of the input batch.
    :type num_tasks: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    :param inner_optimization_step: The function used for the inner loop optimization.
                                    Default is ivy.gradient_descent_update.
    :type inner_optimization_step: callable, optional
    :param average_across_steps: Whether to average the inner loop steps for the outer loop update. Default is False.
    :type average_across_steps: bool, optional
    :return: The cost and the gradients with respect to the outer loop variables.
    """

    # iterate through batches and training task, collecting gradients
    # update the parameters in the direction of these gradients

    if outer_v is None:
        return _train_tasks_w_shared_v(
            batch, inner_cost_fn, outer_cost_fn, inner_v, num_tasks, inner_grad_steps, inner_learning_rate,
            inner_optimization_step, order=1, average_across_steps=average_across_steps)
    return _train_tasks_w_unique_v(
        batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, order=1, average_across_steps=average_across_steps)


def reptile_step(batch, cost_fn, variables, num_tasks, inner_grad_steps, inner_learning_rate,
                 inner_optimization_step=gradient_descent_update, average_across_steps=True):
    """
    Perform step of Reptile.

    :param batch: The input batch
    :type batch: ivy.Container
    :param cost_fn: callable for the cost function, receivng the task-specific sub-batch and variables
    :type cost_fn: callable
    :param variables: Variables to be optimized
    :type variables: ivy.Container
    :param num_tasks: Number of unique tasks to inner-loop optimize for during the meta step.
                        This must be the leading size of the input batch.
    :type num_tasks: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    :param inner_optimization_step: The function used for the inner loop optimization.
                                    Default is ivy.gradient_descent_update.
    :type inner_optimization_step: callable, optional
    :param average_across_steps: Whether to average the inner loop steps for the outer loop update. Default is True.
    :type average_across_steps: bool, optional
    :return: The cost and the gradients with respect to the outer loop variables.
    """
    return _train_tasks_w_shared_v(
        batch, cost_fn, None, variables, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, order=1, average_across_steps=average_across_steps)


# Second Order

def maml_step(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, num_tasks, inner_grad_steps, inner_learning_rate,
              inner_optimization_step=gradient_descent_update, average_across_steps=False):
    """
    Perform step of vanilla second order MAML.

    :param batch: The input batch
    :type batch: ivy.Container
    :param inner_cost_fn: callable for the inner loop cost function, receing sub-batch, inner vars and outer vars
    :type inner_cost_fn: callable
    :param outer_cost_fn: callable for the outer loop cost function, receving task-specific sub-batch,
                            inner vars and outer vars. If None, the cost from the inner loop will also be
                            optimized in the outer loop.
    :type outer_cost_fn: callable, optional
    :param inner_v: Variables to be optimized during the inner loop
    :type inner_v: ivy.Container
    :param outer_v: Variables to be optimized during the outer loop. If None, the same variables are optimized for
                    in the inner loop and outer loop.
    :type outer_v: ivy.Container, optional
    :param num_tasks: Number of unique tasks to inner-loop optimize for during the meta step.
                        This must be the leading size of the input batch.
    :type num_tasks: int
    :param inner_grad_steps: Number of gradient steps to perform during the inner loop.
    :type inner_grad_steps: int
    :param inner_learning_rate: The learning rate of the inner loop.
    :type inner_learning_rate: float
    :param inner_optimization_step: The function used for the inner loop optimization.
                                    Default is ivy.gradient_descent_update.
    :type inner_optimization_step: callable, optional
    :param average_across_steps: Whether to average the inner loop steps for the outer loop update. Default is False.
    :type average_across_steps: bool, optional
    :return: The cost and the gradients with respect to the outer loop variables.
    """
    if outer_v is None:
        return ivy.execute_with_gradients(lambda v: _train_tasks_w_shared_v(
            batch, inner_cost_fn, outer_cost_fn, v, num_tasks, inner_grad_steps, inner_learning_rate,
            inner_optimization_step, order=2, average_across_steps=average_across_steps), inner_v)
    return ivy.execute_with_gradients(lambda v: _train_tasks_w_unique_v(
        batch, inner_cost_fn, outer_cost_fn, inner_v, v, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, order=2, average_across_steps=average_across_steps), outer_v)
