Building Blocks
===============

Here we explain the components of Ivy which are fundamental to it’s usage either as a code converter or as a fully-fledged framework-agnostic ML framework. These are the 4 parts labelled as (a) in the image below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

Backend Functional APIs ✅
-----------------------

The first important point to make is that, Ivy does not implement it’s own C++ or CUDA backend. Instead, Ivy **wraps** the functional APIs of existing frameworks, bringing them into syntactic and semantic alignment. Let’s take the function *ivy.clip* as an example.

There are separate backend modules for JAX, TensorFlow, PyTorch, MXNet and NumPy, and so we implement the clip method once for each backend, each in separate backend files like so:

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

For NumPy, MXNet, and JAX there are no changes at all in this case, and for TensorFlow and PyTorch there is just a simple name change.

For more complicated functions, we need to do more than simply wrap and maybe change the name. For functions with differing behavior then we must modify the function to fit the unified in-out behavior of Ivy’s API. For example, the APIs of JAX, PyTorch, MXNet and NumPy all have a *logspace* method, but TensorFlow does not at the time of writing. Therefore, we need to construct it using a composition of existing TensorFlow ops like so:

.. code-block:: python

   # ivy/functional/backends/tensorflow/creation.py:
   def logspace(start, stop, num, base=10., axis=None, dev=None):
       power_seq = linspace(start, stop, num, axis, default_device(dev))
       return base ** power_seq

Ivy Functional API ✅
------------------

Calling the different backend files explicitly would work okay, but it would mean we need to *import ivy.functional.backends.torch as ivy* to use a PyTorch backend or *import ivy.functional.backends.tensorflow as ivy* to use a TensorFlow backend. Instead, we allow these backends to be bound to the single shared namespace ivy. The backend can then be changed by calling *ivy.set_framework(‘torch’)* for example.

ivy.api is the submodule where all the doc strings and argument typing reside for the functional Ivy API. The *clip* function is shown below:

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

Furthermore, this *ivy.api* submodule enables implicit framework selection, where a function can be called before a backend framework is explicitly set, and the backend framework is then inferred automatically by checking the input types to the function.


+----------------------------------------+-----------------------------------------+
|                                        |                                         |
|.. code-block:: python                  |.. code-block:: python                   |
|                                        |                                         |
|   # implicit                           |   # explicit                            |
|   import ivy                           |   import ivy                            |
|   import torch                         |   import torch                          |
|                                        |                                         |
|   a = torch.ones((1, ))                |   a = torch.ones((1,))                  |
|   b = torch.zeros((1,))                |   b = torch.zeros((1,))                 |
|                                        |                                         |
|   print(ivy.current_framework())       |   print(ivy.current_framework())        |
|   -> 'No backed framework selected'    |   -> 'No backed framework selected'     |
|                                        |                                         |
|                                        |   ivy.set_framework('torch')            |
|                                        |                                         |
|   # ivy/api/core/general.py            |   # ivy/backends/torch/core/general.py  |
|   c = ivy.concatenate([a, b], 0)       |   c = ivy.concatenate([a, b], 0)        |
|                                        |                                         |
|   print(type(c))                       |   print(type(c))                        |
|   -> <class 'torch.Tensor'>            |   -> <class 'torch.Tensor'>             |
|                                        |                                         |
|   print(ivy.current_framework())       |   print(ivy.current_framework())        |
|   -> 'No backed framework selected'    |   -> <module 'ivy.backends.torch'>      |
+----------------------------------------+-----------------------------------------+

This implicit framework selection, and the use of a shared global ivy namespace for all backends, are both made possible via the framework handler.

Framework Handler ✅
-----------------

All code for setting and unsetting frameworks resides in the submodule at *ivy/framework_handler.py*, and the front facing function is *ivy.current_framework()*. The contents of this function are as follows:

.. code-block:: python

   # ivy/framework_handler.py
   def current_framework(*args, f=None, **kwargs):

       if f:
           if verbosity.level > 0:
               verbosity.cprint(
                   'Using provided framework: {}'.format(f))
           return f

       if framework_stack:
           f = framework_stack[-1]
           if verbosity.level > 0:
               verbosity.cprint(
                   'Using framework from stack: {}'.format(f))
           return f

       f = _determine_framework_from_args(
           list(args) + list(kwargs.values()))
       if f is None:
           raise ValueError(
               'get_framework failed to find a valid library'
               'from the inputs: {} {}'.format(args, kwargs))
       if verbosity.level > 0:
           verbosity.cprint(
               'Using framework from type: {}'.format(f))
       return f

When the backend framework is provided explicitly as an argument (for example *f=ivy.functional.backends.torch*), then this framework is returned directly without setting it as the global framework. Otherwise, if a global framework has been previously added to the framework stack using for example *ivy.set_framework(‘tensorflow’)*, then this globally set framework is returned. Finally if neither of these cases apply then the input arguments are type-checked to infer the framework, and this is returned from the function without setting as the global framework. In all cases, a callable module is returned with all bound functions adhering to the specific backend.

The functions in this returned module are populated by iterating through the global *ivy.__dict__* (or a non-global copy of *ivy.__dict__* if non-globally-set), and overwriting every function which is also directly implemented in the framework-specific namespace. The following is a slightly simplified version of this code for illustration, which updates the global *ivy.__dict__* directly:

.. code-block:: python

   # ivy/framework_handler.py
   def set_framework(f):

       # un-modified ivy.__dict__
       global ivy_original_dict
       if not framework_stack:
           ivy_original_dict = ivy.__dict__.copy()

       # add the input framework to global stack
       framework_stack.append(f)

       # iterate through original ivy.__dict__
       for k, v in ivy_original_dict.items():

           # if method doesn't exist in backend module f
           if k not in f.__dict__:
               # add the original ivy method to f
               f.__dict__[k] = v
           # update global ivy.__dict__ with this method
           ivy.__dict__[k] = f.__dict__[k]

       # maybe log to terminal
       if verbosity.level > 0:
           verbosity.cprint(
               'framework stack: {}'.format(framework_stack))

The functions implemented by the framework-specific backend such as *ivy.functional.backends.torch* only constitute a subset of the full Ivy API. This is because many higher level functions are written as a composition of lower level Ivy functions. These functions therefore do not need to be written independently for each backend framework. A good example is *ivy.lstm_update*, as shown:

.. code-block:: python

    # ivy/functional/ivy/nn/layers.py
    def lstm_update(x, init_h, init_c, kernel, recurrent_kernel,
                    bias=None, recurrent_bias=None):

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
                           batch_shape + [timesteps, -1])
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
                ivy.unstack(Wii_x, axis=-2), ivy.unstack(Wif_x, axis=-2),
                ivy.unstack(Wig_x, axis=-2), ivy.unstack(Wio_x, axis=-2)):

            htm1 = ht
            ctm1 = ct

            Wh_htm1 = ivy.matmul(htm1, Wh) + \
                (recurrent_bias if recurrent_bias is not None else 0)
            Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = \
                ivy.split(Wh_htm1, num_or_size_splits=4, axis=-1)

            it = ivy.sigmoid(Wii_xt + Whi_htm1)
            ft = ivy.sigmoid(Wif_xt + Whf_htm1)
            gt = ivy.tanh(Wig_xt + Whg_htm1)
            ot = ivy.sigmoid(Wio_xt + Who_htm1)
            ct = ft * ctm1 + it * gt
            ht = ot * ivy.tanh(ct)

            hts_list.append(ivy.expand_dims(ht, -2))

        return ivy.concatenate(hts_list, -2), ct

We *could* find and wrap the functional LSTM update methods for each backend framework which might bring a small performance improvement, but in this case there are no functional LSTM methods exposed in the official functional APIs of the backend frameworks, and therefore the functional LSTM code which does exist for these frameworks is much less stable and less reliable for wrapping into Ivy.
Generally, we have made decisions so that Ivy is as stable and scalable as possible, minimizing dependencies to backend framework code where possible with minimal sacrifices in performance.

Graph Compiler 🚧
--------------

“What about performance?” I hear you ask. This is a great point to raise!

With the design as currently presented, there would be a small performance hit every time we call an Ivy function by virtue of the added Python wrapping. One reason we created the graph compiler was to address this issue.

The compiler takes in any Ivy function, backend function, or composition, and returns the computation graph using the backend functional API only. The dependency graph for this process looks like this:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler_dependency_graph.png?raw=true
   :align: center
   :width: 75%

As an example, the following 3 pieces of code all compile to the exact same computation graph as shown:

CODE

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiled_graph_a.png?raw=true
   :align: center
   :width: 75%

For all existing ML frameworks, the functional API is the backbone which underpins all higher level functions and classes. This means that under the hood, any code can be expressed as a composition of ops in the functional API. The same is true for Ivy. Therefore, when compiling the graph with Ivy, any higher-level classes or extra code which does not directly contribute towards the computation graph is excluded. For example, the following 3 pieces of code all compile to the exact same computation graph as shown:

CODE

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiled_graph_b.png?raw=true
   :align: center
   :width: 75%

The graph compiler does not compile to C++, CUDA or any other lower level language. It simply traces the backend functional methods in the graph, stores this graph, and then efficiently traverses this graph at execution time, all in Python. Compiling to lower level languages (C++, CUDA, TorchScript etc.) is supported for most backend frameworks via *ivy.compile()*, which wraps backend-specific compilation code, for example:

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

Hopefully this has painted a clear picture of the fundamental building blocks underpinning the Ivy framework, being the backend functional APIs, Ivy functional API, framework handler and graph compiler 🙂

Please check out the discussions on the `repo <https://github.com/unifyai/ivy>`_ for FAQs, and reach out on `discord <https://discord.gg/ZVQdvbzNQJ>`_ if you have any questions!