Ivy as a Transpiler
===================

Here, we explain the Ivy's source-to-source transpiler and computational graph tracer, and the roles they play.


**Supported Frameworks**

+-------------+-+--------+--------+
| Framework   | | Source | Target |
+=============+=+========+========+
| PyTorch     | | ✅     | 🚧     |
+-------------+-+--------+--------+
| TensorFlow  | | 🚧     | ✅     |
+-------------+-+--------+--------+
| JAX         | | 🚧     | ✅     |
+-------------+-+--------+--------+
| NumPy       | | 🚧     | 🚧     |
+-------------+-+--------+--------+


Source-to-Source Transpiler ✅
------------------------------

Ivy's source-to-source transpiler enables seamless conversion of code between different machine learning frameworks.

Let's have a look at a brief example:

.. code-block:: python

   import ivy
   import tensorflow as tf
   import torch

   class Network(torch.nn.Module):

       def __init__(self):
        super().__init__()
        self._linear = torch.nn.Linear(3, 3)

       def forward(self, x):
        return self._linear(x)

   TFNetwork = ivy.transpile(Network, source="torch", target="tensorflow")

   x = tf.convert_to_tensor([1., 2., 3.])
   net = TFNetwork()
   net(x)

| Here **ivy.transpile** takes three arguments:
| - The object to be converted
| - The framework we are converting from
| - The framework we are converting to

The transpiled TensorFlow class is immediately available for use after the ivy.transpile call, as shown in this example, but the
generated source code is also saved into the **Translated_Outputs/** directory, meaning you can edit the source code manually after the fact,
or use it just as if the model had been originally written in TensorFlow.

Graph Tracer ✅
---------------

The tracer extracts a computational graph of functions from any given framework functional API.
The dependency graph for this process looks like this:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/compiler_dependency_graph.png?raw=true
   :align: center
   :width: 75%

Let's look at a few examples, and observe the traced graph of the Ivy code against the native backend code.
First, let's set our desired backend as PyTorch.
When we trace the three functions below, despite the fact that each
has a different mix of Ivy and PyTorch code, they all trace to the same graph:

+----------------------------------------+-----------------------------------------+-----------------------------------------+
|.. code-block:: python                  |.. code-block:: python                   |.. code-block:: python                   |
|                                        |                                         |                                         |
| def pure_ivy(x):                       | def pure_torch(x):                      | def mix(x):                             |
|     y = ivy.mean(x)                    |     y = torch.mean(x)                   |     y = ivy.mean(x)                     |
|     z = ivy.sum(x)                     |     z = torch.sum(x)                    |     z = torch.sum(x)                    |
|     f = ivy.var(y)                     |     f = torch.var(y)                    |     f = ivy.var(y)                      |
|     k = ivy.cos(z)                     |     k = torch.cos(z)                    |     k = torch.cos(z)                    |
|     m = ivy.sin(f)                     |     m = torch.sin(f)                    |     m = ivy.sin(f)                      |
|     o = ivy.tan(y)                     |     o = torch.tan(y)                    |     o = torch.tan(y)                    |
|     return ivy.concatenate(            |     return torch.cat(                   |     return ivy.concatenate(             |
|         [k, m, o], -1)                 |         [k, m, o], -1)                  |         [k, m, o], -1)                  |
|                                        |                                         |                                         |
| # input                                | # input                                 | # input                                 |
| x = ivy.array([[1., 2., 3.]])          | x = torch.tensor([[1., 2., 3.]])        | x = ivy.array([[1., 2., 3.]])           |
|                                        |                                         |                                         |
| # create graph                         | # create graph                          | # create graph                          |
| graph = ivy.trace_graph(               | graph = ivy.trace_graph(                | graph = ivy.trace_graph(                |
|     pure_ivy, x)                       |     pure_torch, x)                      |     mix, x)                             |
|                                        |                                         |                                         |
| # call graph                           | # call graph                            | # call graph                            |
| ret = graph(x)                         | ret = graph(x)                          | ret = graph(x)                          |
+----------------------------------------+-----------------------------------------+-----------------------------------------+

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/compiled_graph_a.png?raw=true
   :align: center
   :width: 75%

For all existing ML frameworks, the functional API is the backbone that underpins all higher level functions and classes.
This means that under the hood, any code can be expressed as a composition of ops in the functional API.
The same is true for Ivy.
Therefore, when compiling the graph with Ivy, any higher-level classes or extra code which does not directly contribute towards the computation graph is excluded.
For example, the following 3 pieces of code all result in the exact same computation graph when traced as shown:

+----------------------------------------+-----------------------------------------+-----------------------------------------+
|.. code-block:: python                  |.. code-block:: python                   |.. code-block:: python                   |
|                                        |                                         |                                         |
| class Network(ivy.module)              | def clean(x, w, b):                     | def unclean(x, w, b):                   |
|                                        |     return w*x + b                      |     y = b + w + x                       |
|     def __init__(self):                |                                         |     print('message')                    |
|         self._layer = ivy.Linear(3, 3) |                                         |     wx = w * x                          |
|         super().__init__()             |                                         |     ret = wx + b                        |
|                                        |                                         |     temp = y * wx                       |
|     def _forward(self, x):             |                                         |     return ret                          |
|         return self._layer(x)          |                                         |                                         |
|                                        | # input                                 | # input                                 |
| # build network                        | x = ivy.array([1., 2., 3.])             | x = ivy.array([1., 2., 3.])             |
| net = Network()                        | w = ivy.random_uniform(                 | w = ivy.random_uniform(                 |
|                                        |     -1, 1, (3, 3))                      |     -1, 1, (3, 3))                      |
| # input                                | b = ivy.zeros((3,))                     | b = ivy.zeros((3,))                     |
| x = ivy.array([1., 2., 3.])            |                                         |                                         |
|                                        | # trace graph                           | # trace graph                           |
| # trace graph                          | graph = ivy.trace_graph(                | graph = ivy.trace_graph(                |
| net.trace_graph(x)                     |     clean, x, w, b)                     |     unclean, x, w, b)                   |
|                                        |                                         |                                         |
| # execute graph                        | # execute graph                         | # execute graph                         |
| net(x)                                 | graph(x, w, b)                          | graph(x, w, b)                          |
+----------------------------------------+-----------------------------------------+-----------------------------------------+

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/compiled_graph_b.png?raw=true
   :align: center
   :width: 75%

This tracing is not restricted to just PyTorch.
Let's take another example, but trace to Tensorflow, NumPy, and JAX:

+------------------------------------+
|.. code-block:: python              |
|                                    |
| def ivy_func(x, y):                |
|     w = ivy.diag(x)                |
|     z = ivy.matmul(w, y)           |
|     return z                       |
|                                    |
| # input                            |
| x = ivy.array([[1., 2., 3.]])      |
| y = ivy.array([[2., 3., 4.]])      |
| # create graph                     |
| graph = ivy.trace_graph(           |
|     ivy_func, x, y)                |
|                                    |
| # call graph                       |
| ret = graph(x, y)                  |
+------------------------------------+

Converting this code to a graph, we get a slightly different graph for each backend:

Tensorflow:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/compiled_graph_tf.png?raw=true
   :align: center
   :width: 75%

|

Numpy:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/compiled_graph_numpy.png?raw=true
   :align: center
   :width: 75%

|

Jax:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/compiled_graph_jax.png?raw=true
   :align: center
   :width: 75%
|

The example above further emphasizes that the tracer creates a computation graph consisting of backend functions, not Ivy functions.
Specifically, the same Ivy code is traced to different graphs depending on the selected backend.
However, when compiling native framework code, we are only able to trace a graph for that same framework.
For example, we cannot take torch code and trace this into tensorflow code.
However, we can transpile torch code into tensorflow code (see `Ivy as a Transpiler <ivy_as_a_transpiler.rst>`_ for more details).

The tracer is not a compiler and does not compile to C++, CUDA, or any other lower level language.
It simply traces the backend functional methods in the graph, stores this graph, and then efficiently traverses this graph at execution time, all in Python.
Compiling to lower level languages (C++, CUDA, TorchScript etc.) is supported for most backend frameworks via :func:`ivy.compile`, which wraps backend-specific compilation code, for example:

.. code-block:: python

    # ivy/functional/backends/tensorflow/compilation.py
    compile = lambda fn, dynamic=True, example_inputs=None,\
    static_argnums=None, static_argnames=None:\
        tf.function(fn)

.. code-block:: python

    # ivy/functional/backends/torch/compilation.py
    def compile(fn, dynamic=True, example_inputs=None,
            static_argnums=None, static_argnames=None):
    if dynamic:
        return torch.jit.script(fn)
    return torch.jit.trace(fn, example_inputs)

.. code-block:: python

    # ivy/functional/backends/jax/compilation.py
    compile = lambda fn, dynamic=True, example_inputs=None,\
                static_argnums=None, static_argnames=None:\
    jax.jit(fn, static_argnums=static_argnums,
            static_argnames=static_argnames)

Therefore, the backend code can always be run with maximal efficiency by compiling into an efficient low-level backend-specific computation graph.

**Round Up**

Hopefully, this has explained how, with the addition of backend-specific frontends, Ivy will be able to easily convert code between different ML frameworks 🙂 works in progress, as indicated by the construction signs 🚧.
This is in keeping with the rest of the documentation.

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
