Building Blocks
===============

Here we explain the components of Ivy which are fundamental to it’s usage either as a code converter or as a fully-fledged framework-agnostic ML framework.
These are the 4 parts labelled as (a) in the image below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

Backend Functional APIs ✅
-----------------------

The first important point to make is that, Ivy does not implement it’s own C++ or CUDA backend.
Instead, Ivy **wraps** the functional APIs of existing frameworks, bringing them into syntactic and semantic alignment.
Let’s take the function :func:`ivy.stack` as an example.

There are separate backend modules for JAX, TensorFlow, PyTorch and NumPy, and so we implement the :code:`stack` method once for each backend, each in separate backend files like so:

.. code-block:: python

   # ivy/functional/backends/jax/manipulation.py:
    def stack(
        x: Union[Tuple[JaxArray], List[JaxArray]],
        axis: Optional[int] = None,
        *,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:
        if axis is None:
            axis = 0
        ret = jnp.stack(x, axis=axis)
        return ret

.. code-block:: python

   # ivy/functional/backends/numpy/manipulation.py:
    def stack(
        x: Union[Tuple[np.ndarray], List[np.ndarray]],
        axis: int =0,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.stack(x, axis, out=out)

.. code-block:: python

   # ivy/functional/backends/tensorflow/manipulation.py:
    def stack(
        x: Union[Tuple[tf.Tensor], List[tf.Tensor]],
        axis: int =0,
    ) -> Union[tf.Tensor, tf.Variable]:
        ret = tf.experimental.numpy.stack(x, axis)
        return ret

.. code-block:: python

   # ivy/functional/backends/torch/manipulation.py:
    def stack(
        x: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        axis: int =0,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ret = torch.stack(x, axis, out=out)
        return ret

There were no changes required for this function except a minor axis check for JAX.

For more complicated functions, we need to do more than simply wrap and maybe change the name.
For functions with differing behavior then we must modify the function to fit the unified in-out behavior of Ivy’s API.
For example, the APIs of JAX, PyTorch and NumPy all have a :code:`logspace` method, but TensorFlow does not at the time of writing.
Therefore, we need to construct it using a composition of existing TensorFlow ops like so:

.. code-block:: python

   # ivy/functional/backends/tensorflow/creation.py:
    def logspace(
        start: Union[tf.Tensor, tf.Variable, int],
        stop: Union[tf.Tensor, tf.Variable, int],
        num: int,
        base: float = 10.0,
        axis: Optional[int] = None,
        *,
        dtype: tf.DType,
        device: str,
    ) -> Union[tf.Tensor, tf.Variable]:
        power_seq = ivy.linspace(start, stop, num, axis, dtype=dtype, device=device)
        return base**power_seq

Ivy Functional API ✅
------------------

Calling the different backend files explicitly would work okay, but it would mean we need to :code:`import ivy.functional.backends.torch as ivy` to use a PyTorch backend or :code:`import ivy.functional.backends.tensorflow as ivy` to use a TensorFlow backend.
Instead, we allow these backends to be bound to the single shared namespace ivy.
The backend can then be changed by calling :code:`ivy.set_backend(‘torch’)` for example.

:mod:`ivy.functional.ivy` is the submodule where all the doc strings and argument typing reside for the functional Ivy API.
For example, The function :func:`prod`  is shown below:

.. code-block:: python

   # ivy/functional/ivy/elementwise.py:
    @to_native_arrays_and_back
    @handle_out_argument
    @handle_nestable
    def prod(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Calculates the product of input array x elements.

        x
            input array. Should have a numeric data type.
        axis
            axis or axes along which products must be computed. By default, the product must
            be computed over the entire array. If a tuple of integers, products must be
            computed over multiple axes. Default: ``None``.
        keepdims
            bool, if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False, the reduced axes
            (dimensions) must not be included in the result. Default: ``False``.
        dtype
            data type of the returned array. If None,
            if the default data type corresponding to the data type “kind” (integer or
            floating-point) of x has a smaller range of values than the data type of x
            (e.g., x has data type int64 and the default data type is int32, or x has data
            type uint64 and the default data type is int64), the returned array must have
            the same data type as x. if x has a floating-point data type, the returned array
            must have the default floating-point data type. if x has a signed integer data
            type (e.g., int16), the returned array must have the default integer data type.
            if x has an unsigned integer data type (e.g., uint16), the returned array must
            have an unsigned integer data type having the same number of bits as the default
            integer data type (e.g., if the default integer data type is int32, the returned
            array must have a uint32 data type). If the data type (either specified or
            resolved) differs from the data type of x, the input array should be cast to the
            specified data type before computing the product. Default: ``None``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            array,  if the product was computed over the entire array, a zero-dimensional
            array containing the product; otherwise, a non-zero-dimensional array containing
            the products. The returned array must have a data type as described by the dtype
            parameter above.

        >>> x = ivy.array([1, 2, 3])
        >>> z = ivy.prod(x)
        >>> print(z)
        ivy.array(6)

        >>> x = ivy.array([1, 0, 3])
        >>> z = ivy.prod(x)
        >>> print(z)
        ivy.array(0)

        """
        return current_backend(x).prod(
            x, axis=axis, dtype=dtype, keepdims=keepdims, out=out
        )

Implicitly, Ivy sets numpy as the default backend or operates with the backend corresponding to the specified data inputs
until the user explicitly sets a different backend.
The examples can be seen below:


+----------------------------------------+----------------------------------------------------+
|                                        |                                                    |
|.. code-block:: python                  |.. code-block:: python                              |
|                                        |                                                    |
|   # implicit                           |   # explicit                                       |
|   import ivy                           |   import ivy                                       |
|   x = ivy.array([1, 2, 3])             |   ivy.set_backend("jax")                           |
|   (type(ivy.to_native(x)))             |                                                    |
|   # -> <class 'numpy.ndarray'>         |   z = ivy.array([1, 2, 3]))                        |
|                                        |   type(ivy.to_native(z))                           |
|   import torch                         |   # ->  <class 'jaxlib.xla_extension.DeviceArray'> |
|   t = torch.tensor([23,42, -1])        |                                                    |
|   type(ivy.to_native(ivy.sum(t)))      |                                                    |
|   # -> <class 'torch.Tensor'>          |                                                    |
+----------------------------------------+----------------------------------------------------+

This implicit backend selection, and the use of a shared global ivy namespace for all backends, are both made possible via the backend handler.

Backend Handler ✅
-----------------

All code for setting and unsetting backend resides in the submodule at :mod:`ivy/backend_handler.py`, and the front facing function is :func:`ivy.current_backend`.
The contents of this function are as follows:

.. code-block:: python

   # ivy/backend_handler.py
    def current_backend(*args, **kwargs):
        global implicit_backend
        # if a global backend has been set with set_backend then this will be returned
        if backend_stack:
            f = backend_stack[-1]
            if verbosity.level > 0:
                verbosity.cprint("Using backend from stack: {}".format(f))
            return f

        # if no global backend exists, we try to infer the backend from the arguments
        f = _determine_backend_from_args(list(args) + list(kwargs.values()))
        if f is not None:
            implicit_backend = f.current_backend_str()
            return f
        if verbosity.level > 0:
            verbosity.cprint("Using backend from type: {}".format(f))
        return importlib.import_module(_backend_dict[implicit_backend])

If a global backend framework has been previously set using for example :code:`ivy.set_backend(‘tensorflow’)`, then this globally set backend is returned.
Otherwise, the input arguments are type-checked to infer the backend, and this is returned from the function as a callable module with all bound functions adhering to the specific backend.

The functions in this returned module are populated by iterating through the global :attr:`ivy.__dict__` (or a non-global copy of :attr:`ivy.__dict__` if non-globally-set), and overwriting every function which is also directly implemented in the backend-specific namespace.
The following is a slightly simplified version of this code for illustration, which updates the global :attr:`ivy.__dict__` directly:

.. code-block:: python

   # ivy/backend_handler.py
   def set_backend(backend: str):

       # un-modified ivy.__dict__
       global ivy_original_dict
       if not backend_stack:
           ivy_original_dict = ivy.__dict__.copy()

       # add the input backend to global stack
       backend_stack.append(backend)

       # iterate through original ivy.__dict__
       for k, v in ivy_original_dict.items():

           # if method doesn't exist in the backend
           if k not in backend.__dict__:
               # add the original ivy method to backend
               backend.__dict__[k] = v
           # update global ivy.__dict__ with this method
           ivy.__dict__[k] = backend.__dict__[k]

       # maybe log to terminal
       if verbosity.level > 0:
           verbosity.cprint(
               'Backend stack: {}'.format(backend_stack))

The functions implemented by the backend-specific backend such as :code:`ivy.functional.backends.torch` only constitute a subset of the full Ivy API.
This is because many higher level functions are written as a composition of lower level Ivy functions.
These functions therefore do not need to be written independently for each backend framework.
A good example is :func:`ivy.lstm_update`, as shown:

.. code-block:: python

    # ivy/functional/ivy/layers.py
    @to_native_arrays_and_back
    @handle_nestable
    def lstm_update(
        x: Union[ivy.Array, ivy.NativeArray],
        init_h: Union[ivy.Array, ivy.NativeArray],
        init_c: Union[ivy.Array, ivy.NativeArray],
        kernel: Union[ivy.Array, ivy.NativeArray],
        recurrent_kernel: Union[ivy.Array, ivy.NativeArray],
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        recurrent_bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Tuple[ivy.Array, ivy.Array]:
        """Perform long-short term memory update by unrolling time dimension of input array.
        Parameters
        ----------
        x
            input tensor of LSTM layer *[batch_shape, t, in]*.
        init_h
            initial state tensor for the cell output *[batch_shape, out]*.
        init_c
            initial state tensor for the cell hidden state *[batch_shape, out]*.
        kernel
            weights for cell kernel *[in, 4 x out]*.
        recurrent_kernel
            weights for cell recurrent kernel *[out, 4 x out]*.
        bias
            bias for cell kernel *[4 x out]*. (Default value = None)
        recurrent_bias
            bias for cell recurrent kernel *[4 x out]*. (Default value = None)
        Returns
        -------
        ret
            hidden state for all timesteps *[batch_shape,t,out]* and cell state for last
            timestep *[batch_shape,out]*
        """
        # get shapes
        x_shape = list(x.shape)
        batch_shape = x_shape[:-2]
        timesteps = x_shape[-2]
        input_channels = x_shape[-1]
        x_flat = ivy.reshape(x, (-1, input_channels))

        # input kernel
        Wi = kernel
        Wi_x = ivy.reshape(
            ivy.matmul(x_flat, Wi) + (bias if bias is not None else 0),
            batch_shape + [timesteps, -1],
        )
        Wii_x, Wif_x, Wig_x, Wio_x = ivy.split(Wi_x, 4, -1)

        # recurrent kernel
        Wh = recurrent_kernel

        # lstm states
        ht = init_h
        ct = init_c

        # lstm outputs
        hts_list = list()

        # unrolled time dimension with lstm steps
        for Wii_xt, Wif_xt, Wig_xt, Wio_xt in zip(
            ivy.unstack(Wii_x, axis=-2),
            ivy.unstack(Wif_x, axis=-2),
            ivy.unstack(Wig_x, axis=-2),
            ivy.unstack(Wio_x, axis=-2),
        ):
            htm1 = ht
            ctm1 = ct

            Wh_htm1 = ivy.matmul(htm1, Wh) + (
                recurrent_bias if recurrent_bias is not None else 0
            )
            Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = ivy.split(
                Wh_htm1, num_or_size_splits=4, axis=-1
            )

            it = ivy.sigmoid(Wii_xt + Whi_htm1)
            ft = ivy.sigmoid(Wif_xt + Whf_htm1)
            gt = ivy.tanh(Wig_xt + Whg_htm1)
            ot = ivy.sigmoid(Wio_xt + Who_htm1)
            ct = ft * ctm1 + it * gt
            ht = ot * ivy.tanh(ct)

            hts_list.append(ivy.expand_dims(ht, -2))

        return ivy.concat(hts_list, -2), ct

We *could* find and wrap the functional LSTM update methods for each backend framework which might bring a small performance improvement, but in this case there are no functional LSTM methods exposed in the official functional APIs of the backend frameworks, and therefore the functional LSTM code which does exist for the backends is much less stable and less reliable for wrapping into Ivy.
Generally, we have made decisions so that Ivy is as stable and scalable as possible, minimizing dependencies to backend framework code where possible with minimal sacrifices in performance.

Graph Compiler 🚧
--------------

“What about performance?” I hear you ask.
This is a great point to raise!

With the design as currently presented, there would be a small performance hit every time we call an Ivy function by virtue of the added Python wrapping.
One reason we created the graph compiler was to address this issue.

The compiler takes in any Ivy function, backend function, or composition, and returns the computation graph using the backend functional API only.
The dependency graph for this process looks like this:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/compiler_dependency_graph.png?raw=true
   :align: center
   :width: 75%

Let's look at a few examples, and observe the compiled graph of the Ivy code against the native backend code.
First, let's set our desired backend as PyTorch.
When we compile the three functions below, despite the fact that each
has a different mix of Ivy and PyTorch code, they all compile to the same graph:

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
| graph = ivy.compile_graph(             | graph = ivy.compile_graph(              | graph = ivy.compile_graph(              |
|     pure_ivy, x)                       |     pure_torch, x)                      |     mix, x)                             |
|                                        |                                         |                                         |
| # call graph                           | # call graph                            | # call graph                            |
| ret = graph(x)                         | ret = graph(x)                          | ret = graph(x)                          |
+----------------------------------------+-----------------------------------------+-----------------------------------------+

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/compiled_graph_a.png?raw=true
   :align: center
   :width: 75%

For all existing ML frameworks, the functional API is the backbone which underpins all higher level functions and classes.
This means that under the hood, any code can be expressed as a composition of ops in the functional API.
The same is true for Ivy.
Therefore, when compiling the graph with Ivy, any higher-level classes or extra code which does not directly contribute towards the computation graph is excluded.
For example, the following 3 pieces of code all compile to the exact same computation graph as shown:

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
| net = Network()                        | w = ivy.random_unifrom(                 | w = ivy.random_unifrom(                 |
|                                        |     -1, 1, (3, 3))                      |     -1, 1, (3, 3))                      |
| # input                                | b = ivy.zeros((3,))                     | b = ivy.zeros((3,))                     |
| x = ivy.array([1., 2., 3.])            |                                         |                                         |
|                                        | # compile graph                         | # compile graph                         |
| # compile graph                        | graph = ivy.compile_graph(              | graph = ivy.compile_graph(              |
| net.compile_graph(x)                   |     clean, x, w, b)                     |     unclean, x, w, b)                   |
|                                        |                                         |                                         |
| # execute graph                        | # execute graph                         | # execute graph                         |
| net(x)                                 | graph(x, w, b)                          | graph(x, w, b)                          |
+----------------------------------------+-----------------------------------------+-----------------------------------------+

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/compiled_graph_b.png?raw=true
   :align: center
   :width: 75%

This compilation is not restricted to just PyTorch.
Let's take another example, but compile to Tensorflow, NumPy and JAX:

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
| graph = ivy.compile_graph(         |
|     ivy_func, x, y)                |
|                                    |
| # call graph                       |
| ret = graph(x, y)                  |
+------------------------------------+

Converting this code to a graph, we get a slightly different graph for each backend:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/compiled_graph_tf.png?raw=true
   :align: center
   :width: 75%

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/compiled_graph_numpy.png?raw=true
   :align: center
   :width: 75%

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/design/compiled_graph_jax.png?raw=true
   :align: center
   :width: 75%

The example above further emphasizes that the graph compiler creates a computation graph consisting of backend functions, not Ivy functions.
Specifically, the same Ivy code compiles to different graphs depending on the selected backend.
However, when compiling native framework code, we are only able to compile a graph for that same framework.
For example, we cannot take torch code and compile this into tensorflow code.
However, we can transpile torch code into tensorflow code (see :ref:Ivy as a Transpiler for more details).

The graph compiler does not compile to C++, CUDA or any other lower level language.
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

Hopefully this has painted a clear picture of the fundamental building blocks underpinning the Ivy framework, being the backend functional APIs, Ivy functional API, backend handler and graph compiler 🙂

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
