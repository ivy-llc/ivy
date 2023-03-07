"""Collection of tests for unified meta functions."""

# global
import pytest
import numpy as np
from hypothesis import strategies as st

# local
import ivy
from ivy.functional.ivy.gradients import _variable, _is_variable
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# ToDo: replace dict checks for verifying costs with analytic calculations


# First Order #
# ------------#

# fomaml step unique vars
@handle_test(
    fn_tree="functional.ivy.fomaml_step",
    inner_grad_steps=helpers.ints(min_value=1, max_value=3),
    with_outer_cost_fn=st.booleans(),
    average_across_steps=st.booleans(),
    batched=st.booleans(),
    stop_gradients=st.booleans(),
    num_tasks=helpers.ints(min_value=1, max_value=2),
    return_inner_v=st.sampled_from(["first", "all", False]),
)
def test_fomaml_step_unique_vars(
    on_device,
    inner_grad_steps,
    with_outer_cost_fn,
    average_across_steps,
    batched,
    stop_gradients,
    num_tasks,
    return_inner_v,
    backend_fw,
):

    # Numpy does not support gradients, and jax does not support gradients on
    # custom nested classes
    if backend_fw.current_backend_str() == "numpy":
        return

    # config
    inner_learning_rate = 1e-2

    # create variables
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[0.0]], device=on_device), num_tasks, axis=0)
                ),
                "weight": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                ),
            }
        )
    else:
        variables = ivy.Container(
            {
                "latent": _variable(ivy.array([0.0], device=on_device)),
                "weight": _variable(ivy.array([1.0], device=on_device)),
            }
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_v["latent"] * sub_batch_in["x"] * sub_v["weight"])[0]
        return cost / batch_size

    # outer cost function
    def outer_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost + (sub_v["latent"] * sub_batch_in["x"] * sub_v["weight"])[0]
        return cost / batch_size

    # numpy
    weight_np = ivy.to_numpy(variables.weight[0:1])
    latent_np = ivy.to_numpy(variables.latent[0:1])
    batch_np = batch.cont_map(lambda x, kc: ivy.to_numpy(x))

    # true gradient
    all_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        all_outer_grads.append(
            [
                (
                    -i * inner_learning_rate * weight_np * sub_batch["x"][0] ** 2
                    - sub_batch["x"][0] * latent_np
                )
                * (-1 if with_outer_cost_fn else 1)
                for i in range(inner_grad_steps + 1)
            ]
        )
    if average_across_steps:
        true_weight_grad = (
            sum([sum(og) / len(og) for og in all_outer_grads]) / num_tasks
        )
    else:
        true_weight_grad = sum([og[-1] for og in all_outer_grads]) / num_tasks

    # true cost
    true_cost_dict = {
        1: {
            True: {True: {1: 0.005, 2: 0.0125}, False: {1: 0.01, 2: 0.025}},
            False: {True: {1: -0.005, 2: -0.0125}, False: {1: -0.01, 2: -0.025}},
        },
        2: {
            True: {True: {1: 0.01, 2: 0.025}, False: {1: 0.02, 2: 0.05}},
            False: {True: {1: -0.01, 2: -0.025}, False: {1: -0.02, 2: -0.05}},
        },
        3: {
            True: {True: {1: 0.015, 2: 0.0375}, False: {1: 0.03, 2: 0.075}},
            False: {True: {1: -0.015, 2: -0.0375}, False: {1: -0.03, 2: -0.075}},
        },
    }
    true_cost = true_cost_dict[inner_grad_steps][with_outer_cost_fn][
        average_across_steps
    ][num_tasks]

    # meta update
    rets = ivy.fomaml_step(
        batch,
        inner_cost_fn,
        outer_cost_fn if with_outer_cost_fn else None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        average_across_steps=average_across_steps,
        batched=batched,
        inner_v="latent",
        outer_v="weight",
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_weight_grad))
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# fomaml step shared vars
@handle_test(
    fn_tree="functional.ivy.fomaml_step",
    inner_grad_steps=helpers.ints(min_value=1, max_value=3),
    with_outer_cost_fn=st.booleans(),
    average_across_steps=st.booleans(),
    batched=st.booleans(),
    stop_gradients=st.booleans(),
    num_tasks=helpers.ints(min_value=1, max_value=2),
    return_inner_v=st.sampled_from(["first", "all", False]),
)
def test_fomaml_step_shared_vars(
    on_device,
    inner_grad_steps,
    with_outer_cost_fn,
    average_across_steps,
    batched,
    stop_gradients,
    num_tasks,
    return_inner_v,
    backend_fw,
):
    # Numpy does not support gradients, jax does not support gradients on custom
    # nested classes
    if backend_fw.current_backend_str() == "numpy":
        return

    # config
    inner_learning_rate = 1e-2

    # create variable
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                )
            }
        )
    else:
        variables = ivy.Container(
            {"latent": _variable(ivy.array([1.0], device=on_device))}
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_batch_in["x"] * sub_v["latent"] ** 2)[0]
        return cost / batch_size

    # outer cost function
    def outer_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost + (sub_batch_in["x"] * sub_v["latent"] ** 2)[0]
        return cost / batch_size

    # numpy
    latent_np = ivy.to_numpy(variables.latent[0:1])
    batch_np = batch.cont_map(lambda x, kc: ivy.to_numpy(x))

    # loss grad function
    def loss_grad_fn(sub_batch_in, w_in, outer=False):
        return (
            (1 if (with_outer_cost_fn and outer) else -1)
            * 2
            * sub_batch_in["x"][0]
            * w_in
        )

    # true gradient
    true_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        ws = list()
        grads = list()
        ws.append(latent_np)
        for step in range(inner_grad_steps):
            update_grad = loss_grad_fn(sub_batch, ws[-1])
            w = ws[-1] - inner_learning_rate * update_grad
            if with_outer_cost_fn:
                grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
            else:
                grads.append(update_grad)
            ws.append(w)
        if with_outer_cost_fn:
            grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
        else:
            grads.append(loss_grad_fn(sub_batch, ws[-1]))

        # true outer grad
        if average_across_steps:
            true_outer_grad = sum(grads) / len(grads)
        else:
            true_outer_grad = grads[-1]
        true_outer_grads.append(true_outer_grad)
    true_outer_grad = sum(true_outer_grads) / len(true_outer_grads)

    # true cost
    true_cost_dict = {
        1: {
            True: {True: {1: 1.0202, 2: 1.5509}, False: {1: 1.0404, 2: 1.6018}},
            False: {True: {1: -1.0202, 2: -1.5509}, False: {1: -1.0404, 2: -1.6018}},
        },
        2: {
            True: {
                True: {1: 1.0409441, 2: 1.6042916},
                False: {1: 1.0824323, 2: 1.7110746},
            },
            False: {
                True: {1: -1.0409441, 2: -1.6042916},
                False: {1: -1.0824323, 2: -1.7110746},
            },
        },
        3: {
            True: {
                True: {1: 1.0622487, 2: 1.6603187},
                False: {1: 1.1261624, 2: 1.8284001},
            },
            False: {
                True: {1: -1.0622487, 2: -1.6603187},
                False: {1: -1.1261624, 2: -1.8284001},
            },
        },
    }
    true_cost = true_cost_dict[inner_grad_steps][with_outer_cost_fn][
        average_across_steps
    ][num_tasks]

    # meta update
    rets = ivy.fomaml_step(
        batch,
        inner_cost_fn,
        outer_cost_fn if with_outer_cost_fn else None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        average_across_steps=average_across_steps,
        batched=batched,
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(ivy.to_numpy(outer_grads.latent[0]), np.array(true_outer_grad))
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# fomaml step overlapping vars
@handle_test(
    fn_tree="functional.ivy.fomaml_step",
    inner_grad_steps=helpers.ints(min_value=1, max_value=3),
    with_outer_cost_fn=st.booleans(),
    average_across_steps=st.booleans(),
    batched=st.booleans(),
    stop_gradients=st.booleans(),
    num_tasks=helpers.ints(min_value=1, max_value=2),
    return_inner_v=st.sampled_from(["first", "all", False]),
)
def test_fomaml_step_overlapping_vars(
    on_device,
    inner_grad_steps,
    with_outer_cost_fn,
    average_across_steps,
    batched,
    stop_gradients,
    num_tasks,
    return_inner_v,
    backend_fw,
):
    # Numpy does not support gradients, jax does not support gradients on custom
    # nested classes
    if backend_fw.current_backend_str() == "numpy":
        return

    # config
    inner_learning_rate = 1e-2

    # create variables
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[0.0]], device=on_device), num_tasks, axis=0)
                ),
                "weight": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                ),
            }
        )
    else:
        variables = ivy.Container(
            {
                "latent": _variable(ivy.array([0.0], device=on_device)),
                "weight": _variable(ivy.array([1.0], device=on_device)),
            }
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_batch_in["x"] * sub_v["latent"] * sub_v["weight"])[0]
        return cost / batch_size

    # outer cost function
    def outer_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost + (sub_batch_in["x"] * sub_v["latent"] * sub_v["weight"])[0]
        return cost / batch_size

    # numpy
    latent_np = ivy.to_numpy(variables.latent[0:1])
    weight_np = ivy.to_numpy(variables.weight[0:1])
    batch_np = batch.cont_map(lambda x, kc: ivy.to_numpy(x))

    # true gradient
    all_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        all_outer_grads.append(
            [
                (
                    -i * inner_learning_rate * weight_np * sub_batch["x"][0] ** 2
                    - sub_batch["x"][0] * latent_np
                )
                * (-1 if with_outer_cost_fn else 1)
                for i in range(inner_grad_steps + 1)
            ]
        )
    if average_across_steps:
        true_weight_grad = (
            sum([sum(og) / len(og) for og in all_outer_grads]) / num_tasks
        )
    else:
        true_weight_grad = sum([og[-1] for og in all_outer_grads]) / num_tasks

    # true latent gradient
    true_latent_grad = np.array(
        [(-1 - (num_tasks - 1) / 2) * (-1 if with_outer_cost_fn else 1)]
    )

    # true cost
    true_cost_dict = {
        1: {
            True: {True: {1: 0.005, 2: 0.0125}, False: {1: 0.01, 2: 0.025}},
            False: {True: {1: -0.005, 2: -0.0125}, False: {1: -0.01, 2: -0.025}},
        },
        2: {
            True: {True: {1: 0.01, 2: 0.025}, False: {1: 0.02, 2: 0.05}},
            False: {True: {1: -0.01, 2: -0.025}, False: {1: -0.02, 2: -0.05}},
        },
        3: {
            True: {True: {1: 0.015, 2: 0.0375}, False: {1: 0.03, 2: 0.075}},
            False: {True: {1: -0.015, 2: -0.0375}, False: {1: -0.03, 2: -0.075}},
        },
    }
    true_cost = true_cost_dict[inner_grad_steps][with_outer_cost_fn][
        average_across_steps
    ][num_tasks]

    # meta update
    rets = ivy.fomaml_step(
        batch,
        inner_cost_fn,
        outer_cost_fn if with_outer_cost_fn else None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        average_across_steps=average_across_steps,
        batched=batched,
        inner_v="latent",
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(ivy.to_numpy(outer_grads.weight[0]), np.array(true_weight_grad))
    assert np.allclose(ivy.to_numpy(outer_grads.latent[0]), np.array(true_latent_grad))
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# reptile step
@pytest.mark.parametrize("inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("stop_gradients", [True, False])
@pytest.mark.parametrize("num_tasks", [1, 2])
@pytest.mark.parametrize("return_inner_v", ["first", "all", False])
def test_reptile_step(
    on_device, inner_grad_steps, batched, stop_gradients, num_tasks, return_inner_v
):
    if ivy.current_backend_str() == "numpy":
        # Numpy does not support gradients, jax does not support gradients on custom
        # nested classes,
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variable
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                )
            }
        )
    else:
        variables = ivy.Container(
            {"latent": _variable(ivy.array([1.0], device=on_device))}
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_batch_in["x"] * sub_v["latent"] ** 2)[0]
        return cost / batch_size

    # numpy
    latent_np = ivy.to_numpy(variables.latent[0:1])
    batch_np = batch.cont_map(lambda x, kc: ivy.to_numpy(x))

    # loss grad function
    def loss_grad_fn(sub_batch_in, w_in):
        return -2 * sub_batch_in["x"][0] * w_in

    # true gradient
    true_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        ws = list()
        grads = list()
        ws.append(latent_np)
        for step in range(inner_grad_steps):
            update_grad = loss_grad_fn(sub_batch, ws[-1])
            w = ws[-1] - inner_learning_rate * update_grad
            grads.append(update_grad)
            ws.append(w)
        grads.append(loss_grad_fn(sub_batch, ws[-1]))

        # true outer grad
        true_outer_grad = sum(grads) / len(grads)
        true_outer_grads.append(true_outer_grad)
    true_outer_grad = (
        sum(true_outer_grads) / len(true_outer_grads)
    ) / inner_learning_rate

    # true cost
    true_cost_dict = {
        1: {1: -1.0202, 2: -1.5509},
        2: {1: -1.0409441, 2: -1.6042916},
        3: {1: -1.0622487, 2: -1.6603187},
    }
    true_cost = true_cost_dict[inner_grad_steps][num_tasks]

    # meta update
    rets = ivy.reptile_step(
        batch,
        inner_cost_fn,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        batched=batched,
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(ivy.to_numpy(outer_grads.latent[0]), np.array(true_outer_grad))
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# Second Order #
# -------------#

# maml step unique vars
@pytest.mark.parametrize("inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize("with_outer_cost_fn", [True, False])
@pytest.mark.parametrize("average_across_steps", [True, False])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("stop_gradients", [True, False])
@pytest.mark.parametrize("num_tasks", [1, 2])
@pytest.mark.parametrize("return_inner_v", ["first", "all", False])
def test_maml_step_unique_vars(
    on_device,
    inner_grad_steps,
    with_outer_cost_fn,
    average_across_steps,
    batched,
    stop_gradients,
    num_tasks,
    return_inner_v,
):
    if ivy.current_backend_str() == "numpy":
        # Numpy does not support gradients, jax does not support gradients on custom
        # nested classes
        pytest.skip()

    if ivy.current_backend_str() == "tensorflow":
        # ToDo: work out why MAML does not work for tensorflow
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variables
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[0.0]], device=on_device), num_tasks, axis=0)
                ),
                "weight": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                ),
            }
        )
    else:
        variables = ivy.Container(
            {
                "latent": _variable(ivy.array([0.0], device=on_device)),
                "weight": _variable(ivy.array([1.0], device=on_device)),
            }
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_batch_in["x"] * sub_v["latent"] * sub_v["weight"])[0]
        return cost / batch_size

    # outer cost function
    def outer_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost + (sub_batch_in["x"] * sub_v["latent"] * sub_v["weight"])[0]
        return cost / batch_size

    # numpy
    weight_np = ivy.to_numpy(variables.weight[0:1])
    latent_np = ivy.to_numpy(variables.latent[0:1])
    batch_np = batch.cont_map(lambda x, kc: ivy.to_numpy(x))

    # true gradient
    all_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        all_outer_grads.append(
            [
                (
                    -2 * i * inner_learning_rate * weight_np * sub_batch["x"][0] ** 2
                    - sub_batch["x"][0] * latent_np
                )
                * (-1 if with_outer_cost_fn else 1)
                for i in range(inner_grad_steps + 1)
            ]
        )
    if average_across_steps:
        true_outer_grad = sum([sum(og) / len(og) for og in all_outer_grads]) / num_tasks
    else:
        true_outer_grad = sum([og[-1] for og in all_outer_grads]) / num_tasks

    # true cost
    true_cost_dict = {
        1: {
            True: {True: {1: 0.005, 2: 0.0125}, False: {1: 0.01, 2: 0.025}},
            False: {True: {1: -0.005, 2: -0.0125}, False: {1: -0.01, 2: -0.025}},
        },
        2: {
            True: {True: {1: 0.01, 2: 0.025}, False: {1: 0.02, 2: 0.05}},
            False: {True: {1: -0.01, 2: -0.025}, False: {1: -0.02, 2: -0.05}},
        },
        3: {
            True: {True: {1: 0.015, 2: 0.0375}, False: {1: 0.03, 2: 0.075}},
            False: {True: {1: -0.015, 2: -0.0375}, False: {1: -0.03, 2: -0.075}},
        },
    }
    true_cost = true_cost_dict[inner_grad_steps][with_outer_cost_fn][
        average_across_steps
    ][num_tasks]

    # meta update
    rets = ivy.maml_step(
        batch,
        inner_cost_fn,
        outer_cost_fn if with_outer_cost_fn else None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        average_across_steps=average_across_steps,
        batched=batched,
        inner_v="latent",
        outer_v="weight",
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(ivy.to_numpy(outer_grads.weight), np.array(true_outer_grad))
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# maml step shared vars
@pytest.mark.parametrize("inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize("with_outer_cost_fn", [True, False])
@pytest.mark.parametrize("average_across_steps", [True, False])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("stop_gradients", [True, False])
@pytest.mark.parametrize("num_tasks", [1, 2])
@pytest.mark.parametrize("return_inner_v", ["first", "all", False])
def test_maml_step_shared_vars(
    on_device,
    inner_grad_steps,
    with_outer_cost_fn,
    average_across_steps,
    batched,
    stop_gradients,
    num_tasks,
    return_inner_v,
):
    if ivy.current_backend_str() == "numpy":
        # Numpy does not support gradients, jax does not support gradients on custom
        # nested classes
        pytest.skip()

    if ivy.current_backend_str() == "tensorflow":
        # ToDo: work out why MAML does not work for tensorflow
        pytest.skip()

    # config
    inner_learning_rate = 1e-2

    # create variable
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                )
            }
        )
    else:
        variables = ivy.Container(
            {"latent": _variable(ivy.array([1.0], device=on_device))}
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_batch_in["x"] * sub_v["latent"] ** 2)[0]
        return cost / batch_size

    # outer cost function
    def outer_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost + (sub_batch_in["x"] * sub_v["latent"] ** 2)[0]
        return cost / batch_size

    # numpy
    variables_np = variables.cont_map(lambda x, kc: ivy.to_ivy(x))
    batch_np = batch.cont_map(lambda x, kc: ivy.to_ivy(x))

    # loss grad function
    def loss_grad_fn(sub_batch_in, w_in, outer=False):
        return (
            (1 if (with_outer_cost_fn and outer) else -1)
            * 2
            * sub_batch_in["x"][0]
            * w_in
        )

    # update grad function
    def update_grad_fn(w_init, sub_batch_in, num_steps, average=False):
        terms = [0] * num_steps + [1]
        collection_of_terms = [terms]
        for s in range(num_steps):
            rhs = [t * 2 * sub_batch_in["x"][0] for t in terms]
            rhs.pop(0)
            rhs.append(0)
            terms = [t + rh for t, rh in zip(terms, rhs)]
            collection_of_terms.append([t for t in terms])
        if average:
            return [
                sum(
                    [
                        t * inner_learning_rate ** (num_steps - i)
                        for i, t in enumerate(tms)
                    ]
                )
                * w_init.latent
                for tms in collection_of_terms
            ]
        return (
            sum(
                [
                    t * inner_learning_rate ** (num_steps - i)
                    for i, t in enumerate(terms)
                ]
            )
            * w_init.latent
        )

    # true gradient
    true_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        ws = list()
        grads = list()
        ws.append(variables_np)
        for step in range(inner_grad_steps):
            update_grad = loss_grad_fn(sub_batch, ws[-1])
            w = ws[-1] - inner_learning_rate * update_grad
            if with_outer_cost_fn:
                grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
            else:
                grads.append(update_grad)
            ws.append(w)
        if with_outer_cost_fn:
            grads.append(loss_grad_fn(sub_batch, ws[-1], outer=True))
        else:
            grads.append(loss_grad_fn(sub_batch, ws[-1]))

        # true outer grad
        if average_across_steps:
            true_outer_grad = sum(
                [
                    ig.latent * ug
                    for ig, ug in zip(
                        grads,
                        update_grad_fn(
                            variables_np, sub_batch, inner_grad_steps, average=True
                        ),
                    )
                ]
            ) / len(grads)
        else:
            true_outer_grad = ivy.multiply(
                update_grad_fn(variables_np, sub_batch, inner_grad_steps),
                grads[-1].latent,
            )
        true_outer_grads.append(true_outer_grad)
    true_outer_grad = sum(true_outer_grads) / len(true_outer_grads)

    # true cost
    true_cost_dict = {
        1: {
            True: {True: {1: 1.0202, 2: 1.5509}, False: {1: 1.0404, 2: 1.6018}},
            False: {True: {1: -1.0202, 2: -1.5509}, False: {1: -1.0404, 2: -1.6018}},
        },
        2: {
            True: {
                True: {1: 1.0409441, 2: 1.6042916},
                False: {1: 1.0824323, 2: 1.7110746},
            },
            False: {
                True: {1: -1.0409441, 2: -1.6042916},
                False: {1: -1.0824323, 2: -1.7110746},
            },
        },
        3: {
            True: {
                True: {1: 1.0622487, 2: 1.6603187},
                False: {1: 1.1261624, 2: 1.8284001},
            },
            False: {
                True: {1: -1.0622487, 2: -1.6603187},
                False: {1: -1.1261624, 2: -1.8284001},
            },
        },
    }
    true_cost = true_cost_dict[inner_grad_steps][with_outer_cost_fn][
        average_across_steps
    ][num_tasks]

    # meta update
    rets = ivy.maml_step(
        batch,
        inner_cost_fn,
        outer_cost_fn if with_outer_cost_fn else None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        average_across_steps=average_across_steps,
        batched=batched,
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(
        ivy.to_numpy(outer_grads.latent), ivy.to_numpy(true_outer_grad[0])
    )
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# maml step overlapping vars
@pytest.mark.parametrize("inner_grad_steps", [1, 2, 3])
@pytest.mark.parametrize("with_outer_cost_fn", [True, False])
@pytest.mark.parametrize("average_across_steps", [True, False])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("stop_gradients", [True, False])
@pytest.mark.parametrize("num_tasks", [1, 2])
@pytest.mark.parametrize("return_inner_v", ["first", "all", False])
def test_maml_step_overlapping_vars(
    on_device,
    inner_grad_steps,
    with_outer_cost_fn,
    average_across_steps,
    batched,
    stop_gradients,
    num_tasks,
    return_inner_v,
):
    if ivy.current_backend_str() == "numpy":
        # Numpy does not support gradients, jax does not support gradients on custom
        # nested classes
        pytest.skip()

    if ivy.current_backend_str() == "tensorflow":
        # ToDo: work out why MAML does not work for tensorflow in wrapped mode
        pytest.skip()
    # config
    inner_learning_rate = 1e-2

    # create variables
    if batched:
        variables = ivy.Container(
            {
                "latent": _variable(
                    ivy.repeat(ivy.array([[0.0]], device=on_device), num_tasks, axis=0)
                ),
                "weight": _variable(
                    ivy.repeat(ivy.array([[1.0]], device=on_device), num_tasks, axis=0)
                ),
            }
        )
    else:
        variables = ivy.Container(
            {
                "latent": _variable(ivy.array([0.0], device=on_device)),
                "weight": _variable(ivy.array([1.0], device=on_device)),
            }
        )

    # batch
    batch = ivy.Container({"x": ivy.arange(1, num_tasks + 1, dtype="float32")})

    # inner cost function
    def inner_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost - (sub_batch_in["x"] * sub_v["latent"] * sub_v["weight"])[0]
        return cost / batch_size

    # outer cost function
    def outer_cost_fn(batch_in, v):
        cost = 0
        batch_size = batch_in.cont_shape[0]
        for sub_batch_in, sub_v in zip(
            batch_in.cont_unstack_conts(0, keepdims=True),
            v.cont_unstack_conts(0, keepdims=True),
        ):
            cost = cost + (sub_batch_in["x"] * sub_v["latent"] * sub_v["weight"])[0]
        return cost / batch_size

    # numpy
    latent_np = ivy.to_numpy(variables.latent)
    weight_np = ivy.to_numpy(variables.weight)
    batch_np = batch.cont_map(lambda x, kc: ivy.to_numpy(x))

    # true weight gradient
    all_outer_grads = list()
    for sub_batch in batch_np.cont_unstack_conts(0, True, num_tasks):
        all_outer_grads.append(
            [
                (
                    -2 * i * inner_learning_rate * weight_np * sub_batch["x"][0] ** 2
                    - sub_batch["x"][0] * latent_np
                )
                * (-1 if with_outer_cost_fn else 1)
                for i in range(inner_grad_steps + 1)
            ]
        )
    if average_across_steps:
        true_weight_grad = (
            sum([sum(og) / len(og) for og in all_outer_grads]) / num_tasks
        )
    else:
        true_weight_grad = sum([og[-1] for og in all_outer_grads]) / num_tasks

    # true latent gradient
    true_latent_grad = np.array(
        [(-1 - (num_tasks - 1) / 2) * (-1 if with_outer_cost_fn else 1)]
    )

    # true cost
    true_cost_dict = {
        1: {
            True: {True: {1: 0.005, 2: 0.0125}, False: {1: 0.01, 2: 0.025}},
            False: {True: {1: -0.005, 2: -0.0125}, False: {1: -0.01, 2: -0.025}},
        },
        2: {
            True: {True: {1: 0.01, 2: 0.025}, False: {1: 0.02, 2: 0.05}},
            False: {True: {1: -0.01, 2: -0.025}, False: {1: -0.02, 2: -0.05}},
        },
        3: {
            True: {True: {1: 0.015, 2: 0.0375}, False: {1: 0.03, 2: 0.075}},
            False: {True: {1: -0.015, 2: -0.0375}, False: {1: -0.03, 2: -0.075}},
        },
    }
    true_cost = true_cost_dict[inner_grad_steps][with_outer_cost_fn][
        average_across_steps
    ][num_tasks]

    # meta update
    rets = ivy.maml_step(
        batch,
        inner_cost_fn,
        outer_cost_fn if with_outer_cost_fn else None,
        variables,
        inner_grad_steps,
        inner_learning_rate,
        average_across_steps=average_across_steps,
        batched=batched,
        inner_v="latent",
        return_inner_v=return_inner_v,
        stop_gradients=stop_gradients,
    )
    calc_cost = rets[0]
    if stop_gradients:
        assert ivy.equal(_is_variable(calc_cost, exclusive=True), False)
    assert np.allclose(ivy.to_scalar(calc_cost), true_cost)
    outer_grads = rets[1]
    assert np.allclose(ivy.to_numpy(outer_grads.weight), np.array(true_weight_grad))
    assert np.allclose(ivy.to_numpy(outer_grads.latent), np.array(true_latent_grad))
    if return_inner_v:
        inner_v_rets = rets[2]
        assert isinstance(inner_v_rets, ivy.Container)
        if return_inner_v == "all":
            assert list(inner_v_rets.cont_shape) == [num_tasks, 1]
        elif return_inner_v == "first":
            assert list(inner_v_rets.cont_shape) == [1, 1]


# Still to Add #
# ---------------#

# _compute_cost_and_update_grads
# _train_tasks
# _train_tasks_batched
# _train_tasks_with_for_loop
# _fomaml_step
