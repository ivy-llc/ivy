Ivy as a Transpiler
===================

On the :ref:`Building Blocks` page, we explored the role of the backend functional APIs, the Ivy functional API, the framework handler and the graph compiler. These are parts are labelled as (a) in the image below.

Here, we explain the role of the framework-specific frontends in Ivy, and how these enable automatic code conversions between different ML frameworks. This part is labelled as (b) in the image below.

The code conversion tools described on this page are works in progress, as indicated by the the construction signs 🚧. This is in keeping with the rest of the documentation.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

Frontend Functional APIs 🚧
------------------------

While the backend API, Ivy API and framework handler enable all Ivy code to be framework-agnostic, they do not for example enable PyTorch code to be framework agnostic. But with frontend APIs, we can also achieve this!

Let’s revisit the :code:`ivy.clip` method we explored when learning about the backend APIs. The backend code is as follows:

.. code-block:: python

   # ivy/functional/backends/jax/elementwise.py:
   def clip(x, x_min, x_max):
       return jax.numpy.clip(x, x_min, x_max)
.. code-block:: python

   # ivy/functional/backends/mxnet/elementwise.py:
   def clip(x, x_min, x_max):
       return mxnet.nd.clip(x, x_min, x_max)
.. code-block:: python

   # ivy/functional/backends/numpy/elementwise.py:
   def clip(x, x_min, x_max):
       return numpy.clip(x, x_min, x_max)
.. code-block:: python

   # ivy/functional/backends/tensorflow/elementwise.py:
   def clip(x, x_min, x_max):
       return tensorflow.clip_by_value(x, x_min, x_max)
.. code-block:: python

   # ivy/functional/backends/torch/elementwise.py:
   def clip(x, x_min, x_max):
       return torch.clamp(x, x_min, x_max)

The Ivy API for the :code:`clip` method is as follows:

.. code-block:: python

   # ivy/functional/ivy/elementwise.py:
   def clip(x: Union[ivy.Array, ivy.NativeArray],
         x_min: Union[Number, Union[ivy.Array, ivy.NativeArray]],
         x_max: Union[Number, Union[ivy.Array, ivy.NativeArray]],
         f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
        """
        Clips (limits) the values in an array.
        Given an interval, values outside the interval are clipped
        to the interval edges (element-wise). For example, if an
        interval of [0, 1] is specified, values smaller than 0
        become 0,and values larger than 1 become 1.

        :param x: Input array containing elements to clip.
        :type x: array
        :param x_min: Minimum value.
        :type x_min: scalar or array
        :param x_max: Maximum value.
        :type x_max: scalar or array
        :param f: ML framework. Inferred from inputs if None.
        :type f: ml_framework, optional
        :return: An array with the elements of x, but where values
            < x_min are replaced with x_min, and those > x_max with
            x_max.
        """
        return ivy.current_framework(x, f=f).clip(x, x_min, x_max)

Now, the frontend APIs are as follows:

.. code-block:: python

   # ivy/functional/frontends/jax/numpy/general.py
   def clip(x, x_min, x_max):
       return ivy.clip(x, x_min, x_max)

.. code-block:: python

   # ivy/functional/frontends/mxnet/nd/general.py
   def clip(x, x_min, x_max):
       return ivy.clip(x, x_min, x_max)

.. code-block:: python

   # ivy/functional/frontends/numpy/general.py
   def clip(x, x_min, x_max):
       return ivy.clip(x, x_min, x_max)

.. code-block:: python

   # ivy/functional/frontends/tensorflow/general.py
   def clip_by_value(x, x_min, x_max):
       return ivy.clip(x, x_min, x_max)

.. code-block:: python

   # ivy/functional/frontends/torch/general.py
   def clamp(x, x_min, x_max):
       return ivy.clip(x, x_min, x_max)

combined, we have the following situation:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/clip_backends_n_frontends.png?raw=true
   :align: center
   :width: 100%

Importantly, we can select the backend and frontend **independently** from one another. For example, this means we can select a JAX backend, but also select the PyTorch frontend and write Ivy code which fully adheres to the PyTorch functional API. In the reverse direction: we can take pre-written pure PyTorch code, replace each PyTorch function with the equivalent function using Ivy’s PyTorch frontend, and then run this PyTorch code using JAX:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/clip_conversion.png?raw=true
   :align: center
   :width: 100%

For this example it’s very simple, the differences are only syntactic, but the above process works for **any** function. If there are semantic differences then these will be captured (a) in the wrapped frontend code which expresses the frontend method as a composition of Ivy functions, and (b) in the wrapped backend code which expressed the Ivy functions as compositions of backend methods.

Let’s take a more complex example and convert PyTorch method :code:`torch.nn.functional.one_hot()` into NumPy code. The frontend is implemented by wrapping a single Ivy method :code:`ivy.one_hot()` as follows:

.. code-block:: python

   # ivy/functional/frontends/torch/nn/sparse_functions.py
   def one_hot(tensor, num_classes=-1):
       return ivy.one_hot(tensor, num_classes)

Let’s look at the NumPy backend code for this Ivy method:

.. code-block:: python

   # ivy/functional/backends/numpy/general.py
   def one_hot(indices, depth):
       res = np.eye(depth)[np.array(indices).reshape(-1)]
       return res.reshape(list(indices.shape) + [depth])

By chaining these method together, we can now call :code:`torch.nn.functional.one_hot()` using NumPy:

.. code-block:: python

   import ivy
   import ivy.frontends.torch as torch

   ivy.set_framework('numpy')

   x = np.array([0., 1., 2.])
   ret = torch.nn.functional.one_hot(x, 3)

Let’s take one more example and convert TensorFlow method :code:`tf.cumprod()` into PyTorch code. This time, the frontend is implemented by wrapping two Ivy methods :code:`ivy.cumprod()`, and :code:`ivy.flip()` as follows:

.. code-block:: python

   # ivy/functional/frontends/tensorflow/math/general.py
   def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
       ret = ivy.cumprod(x, axis, exclusive)
       if reverse:
           return ivy.flip(ret, axis)
       return ret

Let’s look at the PyTorch backend code for both of these Ivy methods:

.. code-block:: python

   # ivy/functional/backends/torch/general.py
   def cumprod(x, axis: int = 0, exclusive: bool = False):
       if exclusive:
           x = torch.transpose(x, axis, -1)
           x = torch.cat(
               (torch.ones_like(x[..., -1:]), x[..., :-1]), -1)
           res = torch.cumprod(x, -1)
           return torch.transpose(res, axis, -1)
       return torch.cumprod(x, axis)

.. code-block:: python

   # ivy/functional/backends/torch/manipulation.py
   def flip(x, axis: Optional[List[int]] = None,
            batch_shape: Optional[List[int]] = None):
       num_dims: int = len(batch_shape) \
           if batch_shape is not None else len(x.shape)
       if not num_dims:
           return x
       if axis is None:
           new_axis: List[int] = list(range(num_dims))
       else:
           new_axis: List[int] = axis
       if isinstance(new_axis, int):
           new_axis = [new_axis]
       else:
           new_axis = new_axis
       new_axis = [item + num_dims if item < 0
                   else item for item in new_axis]
       return torch.flip(x, new_axis)

Again, by chaining these methods together, we can now call :code:`tf.math.cumprod()` using PyTorch:

.. code-block:: python

   import ivy
   import ivy.frontends.tensorflow as tf

   ivy.set_framework('torch')

   x = torch.tensor([[0., 1., 2.]])
   ret = tf.math.cumprod(x, -1)

Role of the Graph Compiler 🚧
-------------------------

The very simple example above worked well, but what about even more complex PyTorch code involving Modules, Optimizers, and other higher level objects? This is where the graph compiler plays a vital role. The graph compiler can convert any code into it’s constituent functions at the functional API level for any ML framework.

For example, let’s take the following PyTorch code and run it using JAX:

.. code-block:: python

   import torch

   class Network(torch.nn.Module):

       def __init__(self):
        super(Network, self).__init__()
        self._linear = torch.nn.Linear(3, 3)

       def forward(self, x):
        return self._linear(x)

   x = torch.tensor([1., 2., 3.])
   net = Network()
   net(x)

We cannot simply :code:`import ivy.frontends.torch` in place of :code:`import torch` as we did in the previous examples. This is because the Ivy frontend only supports the functional API for each framework, whereas the code above makes use of higher level classes through the use of the :code:`torch.nn` namespace.

In general, the way we convert code is by first compiling the code into it’s constituent functions in the core API using Ivy’s graph compiler, and then we convert this executable graph into the new framework. For the example above, this would look like:

.. code-block:: python

   import jax
   import ivy

   graph = ivy.compile_graph(net, x).to_backend('jax')
   x = jax.numpy.array([1., 2., 3.])
   jax_graph(x)

However, when calling :code:`ivy.compile_graph()` the graph only connects the inputs to the outputs. Any other tensors or variables which are not listed in the inputs are treated as constants in the graph. In this case, this means the learnable weights in the Module will be treated as constants. This works fine if we only care about running inference on our graph post-training, but this won’t enable training of the Module in JAX.

Converting Network Models 🚧
-------------------------

In order to convert a model from PyTorch to JAX, we first must convert the :code:`torch.nn.Module` instance to an :code:`ivy.Module` instance using the method :code:`ivy.to_ivy_module()` like so:

.. code-block:: python

   net = ivy.to_ivy_module(net)

In it’s current form, the :code:`ivy.Module` instance thinly wraps the PyTorch model into the :code:`ivy.Module` interface, whilst preserving the pure PyTorch backend. We can compile this network into a graph using Ivy’s graph compiler like so:

.. code-block:: python

   net = net.compile_graph()

In this case, the learnable weights are treated as inputs to the graph rather than constants.

Now, with a compiled graph under the hood of our model, we can call :code:`.to_backend()` directly on the :code:`ivy.Module` instance to convert it to any backend of our choosing, like so:

.. code-block:: python

   net = net.to_backend('jax')

The network can now be trained using Ivy’s optimizer classes with a JAX backend like so:

.. code-block:: python

   optimizer = ivy.Adam(1e-4)
   x_in = ivy.array([1., 2., 3.])
   target = ivy.array([0.])

   def loss_fn(v):
       out = model(x_in, v=v)
       return ivy.reduce_mean((out - target)**2)

   for step in range(100):
       loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
       model.v = optimizer.step(model.v, grads)

To convert this :code:`ivy.Module` instance to a :code:`haiku.Module` instance, we can call :code:`.to_haiku_module()` like so:

.. code-block:: python

   net = net.to_haiku_module()

If we want to remove Ivy from the pipeline entirely, we can then train the model in haiku like so:

.. code-block:: python

   import haiku as hk
   import jax.numpy as jnp

   x_in = jnp.array([1., 2., 3.])
   target = jnp.array([0.])

   def loss_fn():
       out = net(x_in)
       return jnp.mean((out - target)**2)

   loss_fn_t = hk.transform(loss_fn)
   loss_fn_t = hk.without_apply_rng(loss_fn_t)

   rng = jax.random.PRNGKey(42)
   params = loss_fn_t.init(rng)

   def update_rule(param, update):
       return param - 0.01 * update

   for i in range(100):
       grads = jax.grad(loss_fn_t.apply)(params)
       params = jax.tree_multimap(update_rule, params, grads)


Other JAX-specific network libraries such as Flax, Trax and Objax are also supported.

Overall, we have taken a :code:`torch.nn.Module` instance, which can be trained using PyTorch’s optimizer classes, and converted this to a :code:`haiku.Module` instance which can be trained using Haiku’s optimizer classes. The same is true for any combination of frameworks, and for any network architecture, regardless of it’s complexity!

**Round Up**

Hopefully this has explained how, with the addition of framework-specific frontends, Ivy will be able to easily convert code between different ML frameworks 🙂 works in progress, as indicated by the the construction signs 🚧. This is in keeping with the rest of the documentation.

Please check out the discussions on the `repo <https://github.com/unifyai/ivy>`_ for FAQs, and reach out on `discord <https://discord.gg/ZVQdvbzNQJ>`_ if you have any questions!
