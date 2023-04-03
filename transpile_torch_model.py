import functools

import ivy
import jax
import torch
import torchvision
import timm
import haiku as hk
import optax
import numpy as np
import jax.numpy as jnp
import graph_compiler.compiler as gc
import transpiler.transpiler as tc
import logging
from typing import *

jax.config.update("jax_enable_x64", True)

model = torch.hub.load(
    "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=False
)

ivy.set_backend("jax")
x = ivy.random_uniform(shape=(1, 3, 224, 224))

# model.blocks = torch.nn.Sequential(*[model.blocks[i] for i in range(2)])
# model = torch.nn.Sequential(*(list(model.children())[:-3]))
# model.eval()

jax_deit = tc.transpile(model, source="torch", to="jax", args=(x,))

# ivy.set_backend("jax")


class DenseBlock(hk.Module):
    def __init__(
        self,
        init_scale: float,
        widening_factor: int = 4,
        num_classes: int = 10,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._num_classes = num_classes
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x):
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(scale=self._init_scale)
        x = hk.Flatten(preserve_dims=-2)(x)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(self._num_classes, w_init=initializer)(x)


def layer_norm(x, name: Optional[str] = None):
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def build_forward_function(num_classes):
    def forward_fn(data):
        x, _ = data
        # print(x.dtype)
        h_dense = jax_deit()(x)
        # h_norm = layer_norm(x)
        # x = h_att + h_norm
        # h_dense = DenseBlock(1.0, num_classes=num_classes)(x)
        return h_dense

    return forward_fn


def lm_loss_fn(
    forward_fn,
    num_classes,
    params,
    rng,
    data,
):
    print(data[0].shape, data[1].shape)
    logits = forward_fn(params, rng, data)
    labels = jax.nn.one_hot(data[1], num_classes)
    print(logits.shape, labels.shape)

    assert logits.shape == labels.shape
    return -jnp.sum(labels * jax.nn.log_softmax(logits)) / labels.shape[0]


class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """

    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state["rng"])
        params = state["params"]
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state["opt_state"])
        params = optax.apply_updates(params, updates)

        new_state = {
            "step": state["step"] + 1,
            "rng": new_rng,
            "opt_state": opt_state,
            "params": params,
        }

        metrics = {
            "step": state["step"],
            "loss": loss,
        }
        return new_state, metrics


def create_dataset(num_classes):
    images = []
    targets = []
    for i in range(num_classes):
        images.append(ivy.random_uniform(shape=(3, 224, 224)))
        targets.append(i)
    return images, targets


def generate_batches(images, classes, dataset_size, batch_size=32):
    targets = {k: v for v, k in enumerate(np.unique(classes))}
    y_train = [targets[classes[i]] for i in range(len(classes))]
    if batch_size > dataset_size:
        raise ivy.utils.exceptions.IvyError("Use a smaller batch size")
    for idx in range(0, dataset_size, batch_size):
        yield ivy.stack(images[idx : min(idx + batch_size, dataset_size)]), ivy.array(
            y_train[idx : min(idx + batch_size, dataset_size)]
        )


def fit(images, classes, size, batch_size):
    forward_fn = build_forward_function(num_classes)
    forward_fn = hk.transform(forward_fn)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, num_classes)

    optimizer = optax.adam(learning_rate=1e-4)

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    # Initialize parameters.
    logging.info("Initializing parameters...")
    rng = jax.random.PRNGKey(428)
    data = next(generate_batches(images, classes, size, batch_size))
    state = updater.init(rng, data)

    logging.info("Starting train loop...")
    for step in range(100):
        data = next(generate_batches(images, classes, size, batch_size))
        state, metrics = updater.update(state, data)
        print(f"step - {step}, loss -{metrics['loss']}")


num_classes = 10

images, classes = create_dataset(num_classes)
fit(images, classes, num_classes, 4)
