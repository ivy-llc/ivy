
## Ivy as a transpiler

Ivy\'s transpiler allows you to use code from any other framework (or
from any other version of the same framework!) in your own code, by just
adding one line of code. Under the hood, Ivy traces a computational
graph and leverages the frontends and backends to link one framework to
another.

This way, Ivy makes all ML-related projects available for you,
independently of the framework you want to use to research, develop, or
deploy systems. Feel free to head over to the docs for the full API
reference, but the functions you\'d most likely want to use are:

``` python
# Compiles a function into an efficient fully-functional graph, removing all wrapping and redundant code
ivy.compile()

# Converts framework-specific code to a different framework
ivy.transpile()

# Converts framework-specific code to Ivy
ivy.unify()
```

These functions can be used eagerly or lazily. If you pass the necessary
arguments for function tracing, the compilation/transpilation step will
happen instantly (eagerly). Otherwise, the compilation/transpilation
will happen only when the returned function is first invoked.

``` python
import ivy
import jax
ivy.set_backend("jax")

# Simple JAX function to transpile
def test_fn(x):
    return jax.numpy.sum(x)

x1 = ivy.array([1., 2.])
```

``` python
# Arguments are available -> transpilation happens eagerly
eager_graph = ivy.transpile(test_fn, source="jax", to="torch", args=(x1,))

# eager_graph is now torch code and runs efficiently
ret = eager_graph(x1)
```

``` python
# Arguments are not available -> transpilation happens lazily
lazy_graph = ivy.transpile(test_fn, source="jax", to="torch")

# The transpiled graph is initialized, transpilation will happen here
ret = lazy_graph(x1)

# lazy_graph is now torch code and runs efficiently
ret = lazy_graph(x1)
```

If you want to learn more, you can find more information in the [Ivy as
a transpiler section of the
docs!](https://unify.ai/docs/ivy/overview/design/ivy_as_a_transpiler.html)

### When should I use Ivy as a transpiler?

If you want to use building blocks published in other frameworks (neural
networks, layers, array computing libraries, training pipelines\...),
you want to integrate code developed in various frameworks, or maybe
straight up move code from one framework to another, the transpiler is
definitely the tool 🔧 for the job! As the output of transpilation is
native code in the target framework, you can use the converted code just
as if it was code originally developed in that framework, applying
framework-specific optimizations or tools, instantly exposing your
project to all of the unique perks of a different framework.

## Ivy as a framework

The Ivy framework is built on top of various essential components,
mainly the [Backend
Handler](https://unify.ai/docs/ivy/overview/design/building_blocks.html#backend-handler),
which manages what framework is being used behind the scenes and the
[Backend Functional
APIs](https://unify.ai/docs/ivy/overview/design/building_blocks.html#backend-functional-apis),
which provide framework-specific implementations of the Ivy functions.
Likewise, classes such as `ivy.Container` or `ivy.Array` are also
available, facilitating the use of structured data and array-like
objects (learn more about them
[here!](https://unify.ai/docs/ivy/overview/design/ivy_as_a_framework.html)).

All of the functionalities in Ivy are exposed through the
`Ivy functional API` and the `Ivy stateful API`. All functions in the
[Functional
API](https://unify.ai/docs/ivy/overview/design/building_blocks.html#ivy-functional-api)
are **Framework Agnostic Functions**, which means that we can use them
like this:

``` python
import ivy
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
import torch

def mse_loss(y, target):
    return ivy.mean((y - target)**2)

jax_mse   = mse_loss(jnp.ones((5,)), jnp.ones((5,)))
tf_mse    = mse_loss(tf.ones((5,)), tf.ones((5,)))
np_mse    = mse_loss(np.ones((5,)), np.ones((5,)))
torch_mse = mse_loss(torch.ones((5,)), torch.ones((5,)))
```

In the example above we show how Ivy\'s functions are compatible with
tensors from different frameworks. This is the same for ALL Ivy
functions. They can accept tensors from any framework and return the
correct result.

The [Ivy Stateful
API](https://unify.ai/docs/ivy/overview/design/ivy_as_a_framework/ivy_stateful_api.html),
on the other hand, allows you to define trainable modules and layers,
which you can use alone or as a part of any other framework code!

``` python
import ivy


class Regressor(ivy.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__()

    def _build(self, *args, **kwargs):
        self.linear0 = ivy.Linear(self.input_dim, 128)
        self.linear1 = ivy.Linear(128, self.output_dim)

    def _forward(self, x):
        x = self.linear0(x)
        x = ivy.functional.relu(x)
        x = self.linear1(x)
        return x
```

If we put it all together, we\'ll have something like this. This example
uses PyTorch as the backend, but this can easily be changed to your
favorite frameworks, such as TensorFlow, or JAX.

``` python
import ivy


class Regressor(ivy.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__()

    def _build(self, *args, **kwargs):
        self.linear0 = ivy.Linear(self.input_dim, 128)
        self.linear1 = ivy.Linear(128, self.output_dim)

    def _forward(self, x):
        x = self.linear0(x)
        x = ivy.functional.relu(x)
        x = self.linear1(x)
        return x

ivy.set_backend('torch')  # set backend to PyTorch (or any other backend!)

model = Regressor(input_dim=1, output_dim=1)
optimizer = ivy.Adam(0.3)

n_training_examples = 2000
noise = ivy.random.random_normal(shape=(n_training_examples, 1), mean=0, std=0.1)
x = ivy.linspace(-6, 3, n_training_examples).reshape((n_training_examples, 1))
y = 0.2 * x ** 2 + 0.5 * x + 0.1 + noise


def loss_fn(v, x, target):
    pred = model(x, v=v)
    return ivy.mean((pred - target) ** 2)

for epoch in range(40):
    # forward pass
    pred = model(x)

    # compute loss and gradients
    loss, grads = ivy.execute_with_gradients(lambda params: loss_fn(*params), (model.v, x, y))

    # update parameters
    model.v = optimizer.step(model.v, grads)

    # print current loss
    print(f'Epoch: {epoch + 1:2d} --- Loss: {ivy.to_numpy(loss).item():.5f}')

print('Finished training!')
```

The model\'s output can be visualized as follows:

<div align="center">
   <img width="50%" class="dark-light" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/regressor_lq.gif">
</div>


As always, you can find more information about [Ivy as a framework in
the
docs!](https://unify.ai/docs/ivy/overview/design/ivy_as_a_framework.html)

### When should I use Ivy as a framework?

As Ivy supports multiple backends, writing code in Ivy breaks you free
from framework limitations. If you want to publish highly flexible code
for everyone to use, independently of the framework they are using, or
you plan to develop ML-related tools and want them to be interoperable
with not only the already existing frameworks, but also with future
frameworks, then Ivy is for you!

## Setting up Ivy

There are various ways to use Ivy, depending on your preferred
environment:

### Installing using pip

The easiest way to set up Ivy is to install it using pip with the
following command:

``` bash
pip install ivy
```

or alternatively:

``` bash
python3 -m pip install ivy
```

### Docker

If you prefer to use containers, we also have pre-built Docker images
with all the supported frameworks and some relevant packages already
installed, which you can pull from:

``` bash
docker pull unifyai/ivy:latest
```

If you are working on a GPU device, you can pull from:

``` bash
docker pull unifyai/ivy:latest-gpu
```

### Installing from source

You can also install Ivy from source if you want to take advantage of
the latest changes, but we can\'t ensure everything will work as
expected. :sweat_smile:

``` bash
git clone https://github.com/unifyai/ivy.git
cd ivy
pip install --user -e .
```

or alternatively, for the last step:

``` bash
python3 -m pip install --user -e .
```

If you want to set up testing and various frameworks it\'s probably best
to check out the [Contributing - Setting
Up](https://unify.ai/docs/ivy/overview/contributing/setting_up.html#setting-up)
page, where OS-specific and IDE-specific instructions and video
tutorials to do so are available!

### Using Ivy

You can find quite a lot more examples in the corresponding section
below, but using Ivy is as simple as:

#### Multi-backend Support

``` python
import ivy
import torch
import jax

ivy.set_backend("jax")

x = jax.numpy.array([1, 2, 3])
y = jax.numpy.array([3, 2, 1])
z = ivy.add(x, y)

ivy.set_backend('torch')

x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 2, 1])
z = ivy.add(x, y)
```

#### Transpilation API

``` python
import ivy
import torch
import jax

def jax_fn(x):
    a = jax.numpy.dot(x, x)
    b = jax.numpy.mean(x)
    return x * a + b

jax_x = jax.numpy.array([1, 2, 3])
torch_x = torch.tensor([1, 2, 3])
torch_fn = ivy.transpile(jax_fn, source="jax", to="torch", args=(jax_x,))
ret = torch_fn(torch_x)
```

## 📚 Documentation

You can find our extensive documentation from this [Ivy Docs page](https://unify.ai/docs/ivy/)  which includes: 
- [Docs](https://unify.ai/docs/ivy/): the full extensive documentation 
    - [Design](https://unify.ai/docs/ivy/overview/design.html): the design decision, architecture, and building blocks (layers and nodes)  of Ivy. 
    - [Deep
dive](https://unify.ai/docs/ivy/overview/deep_dive.html)explains our code base and how to contribute to a specific field. 
    - [Background](https://unify.ai/docs/ivy/overview/background.html): This contextualizes the problem Ivy is trying to solve through
        - the current [ML
Explosion](https://unify.ai/docs/ivy/overview/background/ml_explosion.html#ml-explosion),, 
        - explaining both why is important [to solve this
problem](https://unify.ai/docs/ivy/overview/background/why_unify.html#why-unify) and 
        - how we adhere to existing [standards](https://unify.ai/docs/ivy/overview/background/standardization.html#standardization) to make this happen.
    - [Related
Work](https://unify.ai/docs/ivy/overview/related_work.html)   which paints a clear picture of the role Ivy plays in the ML stack, comparing it to other existing solutions in terms of functionalities and level.