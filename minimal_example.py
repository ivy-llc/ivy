import jax
import ivy
import torch


rng = jax.random.PRNGKey(0)


def dummy_test_fn(inputs):
    batch_size = inputs.shape[2]
    return batch_size


x1 = jax.random.uniform(rng, shape=(1, 224, 224, 3))

torch_fn = ivy.transpile(dummy_test_fn, source="jax", to="torch", args=(x1,))

xt1 = torch.rand((1, 224, 224, 3))
xt2 = torch.rand((2, 224, 512, 3))

print(torch_fn(xt1))
print(torch_fn(xt2))
