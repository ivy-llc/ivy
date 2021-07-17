# global
import ivy
from ivy.core.gradients import gradient_descent_update


# Private #
# --------#

# unique variables for inner and outer loop

def _train_task_w_unique_v(sub_batch, inner_cost_fn, inner_v, outer_v, inner_grad_steps, inner_learning_rate,
                           inner_optimization_step, order, average_across_steps):
    total_cost = 0
    for i in range(inner_grad_steps):
        cost, inner_grads = ivy.execute_with_gradients(lambda v: inner_cost_fn(sub_batch, v, outer_v), inner_v,
                                                       retain_grads=order > 1 or average_across_steps)
        total_cost = total_cost + cost
        inner_v = inner_optimization_step(inner_v, inner_grads, inner_learning_rate, inplace=False)
        inner_v = inner_v.map(lambda x, kc: ivy.variable(ivy.stop_gradient(x))) if order == 1 else inner_v
    final_cost = inner_cost_fn(sub_batch, inner_v, outer_v)
    if average_across_steps:
        total_cost = total_cost + final_cost
        return total_cost / (inner_grad_steps + 1), inner_v
    return final_cost, inner_v


def _train_tasks_w_unique_v(batch, inner_cost_fn, outer_cost_fn, inner_v, outer_v, num_tasks, inner_grad_steps,
                            inner_learning_rate, inner_optimization_step, order, average_across_steps):
    total_cost = 0
    inner_v_orig = inner_v.map(lambda x, kc: ivy.stop_gradient(x))
    for sub_batch in batch.unstack(0, num_tasks):
        inner_v = inner_v_orig.map(lambda x, kc: ivy.variable(x))
        cost, inner_v = _train_task_w_unique_v(sub_batch, inner_cost_fn, inner_v, outer_v, inner_grad_steps,
                                               inner_learning_rate, inner_optimization_step, order,
                                               average_across_steps)
        if outer_cost_fn is not None:
            cost = outer_cost_fn(sub_batch, inner_v, outer_v)
        total_cost = total_cost + cost
    return total_cost / num_tasks


# shared variables for inner and outer loop

def _train_task_w_shared_v(sub_batch, inner_cost_fn, inner_v, outer_v, inner_grad_steps, inner_learning_rate,
                           inner_optimization_step, order, average_across_steps):
    total_cost = 0
    outer_v_ones = outer_v / outer_v.map(lambda x, kc: ivy.stop_gradient(x))
    for i in range(inner_grad_steps):
        cost, inner_grads = ivy.execute_with_gradients(lambda v: inner_cost_fn(sub_batch, v), inner_v,
                                                       retain_grads=order > 1 or average_across_steps)
        total_cost = total_cost + cost
        inner_v = inner_optimization_step(inner_v, inner_grads, inner_learning_rate, inplace=False)
        if order == 1:
            inner_v = inner_v.map(lambda x, kc: ivy.variable(ivy.stop_gradient(x)))
        if average_across_steps or i == inner_grad_steps - 1:
            inner_v = inner_v * outer_v_ones

    final_cost = inner_cost_fn(sub_batch, inner_v)
    if average_across_steps:
        total_cost = total_cost + final_cost
        return total_cost / (inner_grad_steps + 1), inner_v
    return final_cost, inner_v


def _train_tasks_w_shared_v(batch, inner_cost_fn, outer_cost_fn, v, num_tasks, inner_grad_steps,
                            inner_learning_rate, inner_optimization_step, order, average_across_steps):
    total_cost = 0
    for sub_batch in batch.unstack(0, num_tasks):
        if average_across_steps:
            inner_v = v * 1
        else:
            inner_v = v.map(lambda x, kc: ivy.variable(ivy.stop_gradient(x)))
        cost, inner_v = _train_task_w_shared_v(sub_batch, inner_cost_fn, inner_v, v, inner_grad_steps,
                                               inner_learning_rate, inner_optimization_step, order,
                                               average_across_steps)
        if outer_cost_fn is not None:
            cost = outer_cost_fn(sub_batch, inner_v, v)
        total_cost = total_cost + cost
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
    if outer_v is None:
        return ivy.execute_with_gradients(lambda v: _train_tasks_w_shared_v(
            batch, inner_cost_fn, outer_cost_fn, v, num_tasks, inner_grad_steps, inner_learning_rate,
            inner_optimization_step, order=1, average_across_steps=average_across_steps), inner_v)
    return ivy.execute_with_gradients(lambda v: _train_tasks_w_unique_v(
        batch, inner_cost_fn, outer_cost_fn, inner_v, v, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, order=1, average_across_steps=average_across_steps), outer_v)


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
    return ivy.execute_with_gradients(lambda v: _train_tasks_w_shared_v(
        batch, cost_fn, None, v, num_tasks, inner_grad_steps, inner_learning_rate,
        inner_optimization_step, order=1, average_across_steps=average_across_steps), variables)


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
