Ivy Frontends
=============

.. _`tensorflow.tan`: https://github.com/unifyai/ivy/blob/f52457a7bf3cfafa30a7c1a29a708ade017a735f/ivy_tests/test_ivy/test_frontends/test_tensorflow/test_math.py#L109
.. _`aliases`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`jax.lax.add`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html
.. _`jax.lax`: https://jax.readthedocs.io/en/latest/jax.lax.html
.. _`jax.lax.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.tan.html
.. _`numpy.add`: https://numpy.org/doc/stable/reference/generated/numpy.add.html
.. _`numpy mathematical functions`: https://numpy.org/doc/stable/reference/index.html
.. _`numpy.tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`tf`: https://www.tensorflow.org/api_docs/python/tf
.. _`tf.math.tan`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`torch.add`: https://pytorch.org/docs/stable/generated/torch.add.html#torch.add
.. _`torch`: https://pytorch.org/docs/stable/torch.html#math-operations
.. _`torch.tan`: https://pytorch.org/docs/stable/generated/torch.tan.html#torch.tan
.. _`YouTube tutorial series`: https://www.youtube.com/watch?v=72kBVJTpzIw&list=PLwNuX3xB_tv-wTpVDMSJr7XW6IP_qZH0t
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`ivy frontends channel`: https://discord.com/channels/799879767196958751/998782045494976522
.. _`ivy frontends forum`: https://discord.com/channels/799879767196958751/1028297849735229540
.. _`open task`: https://lets-unify.ai/ivy/contributing/open_tasks.html#open-tasks

Introduction
------------

On top of the Ivy functional API and backend functional APIs, Ivy has another set of framework-specific frontend functional APIs, which play an important role in code transpilations, as explained `here <https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html>`_.

Let's start with some examples to have a better idea on Ivy Frontends!


The Frontend Basics
-------------------

When using functions and methods of Ivy Frontends, in addition to importing ivy itself like :code:`import ivy` please also import the corresponding Frontend module.
For example, to use ivy's tensorflow frontend:

    :code:`import ivy.functional.frontends.tensorflow as tf_frontend`

----

When testing the frontend functions, we can sometimes call the function directly from the root frontend namespace.
For example, we call `tensorflow.tan`_ rather than :func:`tensorflow.math.tan`.
In this particular case both are fine, and in fact are `aliases`_.

However, sometimes an extra namespace path is necessary.
Taking JAX as an example, the functions :func:`jax.numpy.abs` and :func:`jax.lax.abs` both exist, while :func:`jax.abs` does not exist.
In our JAX frontend, if we add both of these to the root namespace, it would be possible to call :func:`jax.abs` in our frontend.

This would result in :func:`jax.numpy.abs` or :func:`jax.lax.abs` overwriting the other one in an arbitrary manner.
In fact, neither of these should be added to the root namespace, as it does not exist in the native :mod:`jax` framework.

If you accidentally test a function with :code:`fn_tree="<func_name>"` instead of :code:`fn_tree="<lax|numpy>.<func_name>"`, you will see an error since the wrong frontend function is being tested.

Therefore, in order to avoid this potential conflict:

* All frontend tests should use the full namespace path when calling the frontend function.
  In the case of TensorFlow, this would mean writing :code:`fn_tree="math.tan"` instead of :code:`fn_tree="tan"` in the frontend test.

* The :mod:`__init__.py` file in all frontends should be carefully checked, and you should verify that you are not adding aliases into the frontend which should not exist, such as the case of :func:`jax.abs` explained above.

* You should ensure that the tests are passing before merging any frontend PRs.
  The only exception to this rule is if the test is failing due to a bug in the Ivy functional API, which does not need to be solved as part of the frontend task.

There will be some implicit discussion of the locations of frontend functions in these examples, however an explicit explanation of how to place a frontend function can be found in a sub-section of the Frontend APIs `open task`_.


**NOTE:** Type hints, docstrings and examples are not required when working on frontend functions.


**Jax**

JAX has two distinct groups of functions, those in the :mod:`jax.lax` namespace and those in the :mod:`jax.numpy` namespace.
The former set of functions map very closely to the API for the Accelerated Linear Algebra (`XLA <https://www.tensorflow.org/xla>`_) compiler, which is used under the hood to run high performance JAX code.
The latter set of functions map very closely to NumPy's well known API.
In general, all functions in the :mod:`jax.numpy` namespace are themselves implemented as a composition of the lower-level functions in the :mod:`jax.lax` namespace.

When transpiling between frameworks, the first step is to compile the computation graph into low level python functions for the source framework using Ivy's graph compiler, before then replacing these nodes with the associated functions in Ivy's frontend API.
Given that all jax code can be decomposed into :mod:`jax.lax` function calls, when transpiling JAX code it should always be possible to express the computation graph as a composition of only :mod:`jax.lax` functions.
Therefore, arguably these are the *only* functions we should need to implement in the JAX frontend.
However, in general we wish to be able to compile a graph in the backend framework with varying levels of dynamicism.
A graph of only :mod:`jax.lax` functions chained together in general is more *static* and less *dynamic* than a graph which chains :mod:`jax.numpy` functions together.
We wish to enable varying extents of dynamicism when compiling a graph with our graph compiler, and therefore we also implement the functions in the :mod:`jax.numpy` namespace in our frontend API for JAX.

Thus, both :mod:`lax` and :mod:`numpy` modules are created in the JAX frontend API.
We start with the function :func:`lax.add` as an example.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def add(x, y):
        return ivy.add(x, y)

:func:`lax.add` is categorised under :code:`operators` as shown in the `jax.lax`_ package directory.
We organize the functions using the same categorizations as the original framework, and also mimic the importing behaviour regarding modules and namespaces etc.

For the function arguments, these must be identical to the original function in Jax.
In this case, `jax.lax.add`_ has two arguments, and so we will also have the same two arguments in our Jax frontend :func:`lax.add`.
In this case, the function will then simply return :func:`ivy.add`, which in turn will link to the backend-specific implementation :func:`ivy.add` according to the framework set in the backend.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def tan(x):
        return ivy.tan(x)

Using :func:`lax.tan` as a second example, we can see that this is placed under :mod:`operators`, again in the `jax.lax`_ directory.
By referring to the `jax.lax.tan`_ documentation, we can see that it has only one argument.
In the same manner as our :func:`add` function, we simply link its return to :func:`ivy.tan`, and again the computation then depends on the backend framework.

**NumPy**

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/arithmetic_operations.py
    def add(
        x1,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
    ):
        if dtype:
            x1 = ivy.astype(ivy.array(x1), ivy.as_ivy_dtype(dtype))
            x2 = ivy.astype(ivy.array(x2), ivy.as_ivy_dtype(dtype))
        ret = ivy.add(x1, x2, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

In NumPy, :func:`add` is categorised under :mod:`mathematical_functions` with a sub-category of :mod:`arithmetic_operations` as shown in the `numpy mathematical functions`_ directory.

The function arguments for this function are slightly more complex due to the extra optional arguments.
Additional handling code is added to recover the behaviour according to the `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_ documentation.
For example, if :code:`dtype` is specified, the arguments will be cast to the desired type through :func:`ivy.astype`.
The returned result is then obtained through :func:`ivy.add` just like the other examples.

However, the arguments :code:`casting`, :code:`order` and :code:`subok` are completely unhandled here.
This is for two reasons.

In the case of :code:`casting`, support will be added for this via the inclusion of a decorator at some point in future, and so this is simply being deferred for the time being.

In the case of :code:`order` and :code:`subok`, this is because the aspects which these arguments seek to control are simply not controllable when using Ivy.
:code:`order` controls the low-level memory layout of the stored array.
Similarly, :code:`subok` controls whether or not subclasses of the :class:`numpy.ndarray` should be permitted as inputs to the function.
Again, this is a very framework-specific argument.
All ivy functions by default do enable subclasses of the :class:`ivy.Array` to be passed, and the frontend function will be operating with :class:`ivy.Array` instances rather than :class:`numpy.ndarray` instances, and so we omit this argument.
Again, it has no bearing on input-output behaviour and so this is not a problem when transpiling between frameworks.

See the section "Unused Arguments" below for more details.

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/trigonometric_functions.py
    @from_zero_dim_arrays_to_float
    def tan(
        x,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if dtype:
            x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
        ret = ivy.tan(x, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

For the second example, :func:`tan` has a sub-category of :mod:`trigonometric_functions` according to the `numpy mathematical functions`_ directory.
By referring to the `numpy.tan`_ documentation, we can see it has the same additional arguments as the :func:`add` function.
In the same manner as :func:`add`, we handle the argument :code:`out`, :code:`where` and :code:`dtype`, but we omit support for :code:`casting`, :code:`order` and :code:`subok`.

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    def add(x, y, name=None):
        return ivy.add(x, y)

The :func:`add` function is categorised under the :mod:`math` folder in the TensorFlow frontend.
There are three arguments according to the `tf.math.add <https://www.tensorflow.org/api_docs/python/tf/math/add>`_ documentation, which are written accordingly as shown above.
Just like the previous examples, the implementation wraps :func:`ivy.add`, which itself defers to backend-specific functions depending on which framework is set in Ivy's backend.

The arguments :code:`x` and :code:`y` are both used in the implementation, but the argument :code:`name` is not used.
Similar to the omitted arguments in the NumPy example above, the :code:`name` argument does not change the input-output behaviour of the function.
Rather, this argument is added purely for the purpose of operation logging and retrieval, and also graph visualization in TensorFlow.
Ivy does not support the unique naming of individual operations, and so we omit support for this particular argument.

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    def tan(x, name=None):
        return ivy.tan(x)

Likewise, :code:`tan` is also placed under :mod:`math`.
By referring to the `tf.math.tan`_ documentation, we add the same arguments, and simply wrap :func:`ivy.tan` in this case.
Again, we do not support the :code:`name` argument for the reasons outlined above.

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def add(input, other, *, alpha=None, out=None):
        return ivy.add(input, other, alpha=alpha, out=out)

For PyTorch, :func:`add` is categorised under :mod:`pointwise_ops` as is the case in the `torch`_ framework.

In this case, the native `torch.add`_ has both positional and keyword arguments, and we therefore use the same for our PyTorch frontend :func:`add`.
We wrap :func:`ivy.add` as usual.

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def tan(input, *, out=None):
        return ivy.tan(input, out=out)

:func:`tan` is also placed under :mod:`pointwise_ops` as is the case in the `torch`_ framework.
Looking at the `torch.tan`_ documentation, we can mimic the same arguments, and again simply wrap :func:`ivy.tan`, also making use of the :code:`out` argument in this case.

Unused Arguments
----------------

As can be seen from the examples above, there are often cases where we do not add support for particular arguments in the frontend function.
Generally, we can omit support for a particular argument only if: the argument **does not** fundamentally affect the input-output behaviour of the function in a mathematical sense.
The only two exceptions to this rule are arguments related to either the data type or the device on which the returned array(s) should reside.
Examples of arguments which can be omitted, on account that they do not change the mathematics of the function are arguments which relate to:

* the layout of the array in memory, such as :code:`order` in `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_.

* the algorithm or approximations used under the hood, such as :code:`precision` and :code:`preferred_element_type` in `jax.lax.conv_general_dilated <https://github.com/google/jax/blob/1338864c1fcb661cbe4084919d50fb160a03570e/jax/_src/lax/convolution.py#L57>`_.

* the specific array class in the original framework, such as :code:`subok` in `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_.

* the labelling of functions for organizational purposes, such as :code:`name` in `tf.math.add <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/python/ops/math_ops.py#L3926-L4004>`_.

There are likely to be many other examples of arguments which do not fundamentally affect the input-output behaviour of the function in a mathematical sense, and so can also be omitted from Ivy's frontend implementation.

The reason we omit these arguments in Ivy is because Ivy is not designed to provide low-level control to functions that extend beyond the pure mathematics of the function.
This is a requirement because Ivy abstracts the backend framework, and therefore also abstracts everything below the backend framework's functional API, including the backend array class, the low-level language compiled to, the device etc.
Most ML frameworks do not offer per-array control of the memory layout, and control for the finer details of the algorithmic approximations under the hood, and so we cannot in general offer this level of control at the Ivy API level, nor the frontend API level as a direct result.
As explained above, this is not a problem, as the memory layout has no bearing at all on the input-output behaviour of the function.
In contrast, the algorithmic approximation may have a marginal bearing on the final results in some cases, but Ivy is only designed to unify to within a reasonable numeric approximation in any case, and so omitting these arguments also very much fits within Ivy's design.


Compositions
------------

In many cases, frontend functions meet the following criteria:

* the function is unique to a particular frontend framework, and does not exist in the other frameworks
* the function has extra features and/or arguments on top of the most similar ivy function that is available

In such cases, compositions are required to replicate the function behaviour.

**Examples**

In the native TensorFlow function :func:`tf.cumprod`, it supports an extra argument - :code:`reverse`, which returns a flipped result if :code:`True`.
However, the backend :func:`ivy.cumprod` does not support this argument or behaviour.

**Ivy**

.. code-block:: python

    # in ivy/functional/ivy/statistical.py
    def cumprod(
        x: Union[ivy.Array, ivy.NativeArray],
        axis: int = 0,
        exclusive: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return current_backend(x).cumprod(x, axis, exclusive, dtype=dtype, out=out)

To enable this behaviour, we need to incorporate other Ivy functions which are compositionally able to mimic the required behaviour.
For example, we can simply reverse the result by calling :func:`ivy.flip` on the result of :func:`ivy.cumprod`.

**TensorFlow Frontend**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
        ret = ivy.cumprod(x, axis, exclusive)
        if reverse:
            return ivy.flip(ret, axis)
        return ret

Through compositions, we can easily meet the required input-output behaviour for the TensorFlow frontend function.

Missing Ivy Functions
---------------------

Sometimes, there is a clear omission of an Ivy function, which would make the frontend implementation much simpler.
For example, at the time of writing, implementing :func:`median` for the NumPy frontend would require a very manual and heavily compositional implementation.
However, if the function :func:`ivy.median` was added to Ivy's functional API, then this frontend implementation would become very simple, with some light wrapping around :func:`ivy.median`.

Adding :func:`ivy.median` would be a sensible decision, as many frameworks support this function.
When you come across such a function which is missing from Ivy, you should create a new issue on the Ivy repo, with the title :func:`ivy.func_name` and with the labels :code:`Suggestion`, :code:`Experimental`, :code:`Ivy API` and :code:`Next Release`.
A member of our team will then review this issue, and if the proposed addition is deemed to be timely and sensible, then we will add this function to the "Extend Ivy Functional API" `ToDo list issue <https://github.com/unifyai/ivy/issues/3856>`_.
At this point in time, you can reserve the function for yourself and get it implemented in a unique PR.
Once merged, you can then resume working on the frontend function, which will now be a much easier task with the new addition to Ivy.

Temporary Compositions
----------------------

Alternatively, if after creating the new issue you would rather not wait around for a member of our team to review and possibly add to the "Extend Ivy Functional API" `ToDo list issue <https://github.com/unifyai/ivy/issues/3856>`_, you can instead go straight ahead add the frontend function as a heavy composition of the existing Ivy functions, with a :code:`#ToDo` comment included, explaining that this frontend implementation will be simplified if/when :func:`ivy.func_name` is add to Ivy.

The entire workflow for extending the Ivy Frontends as an external contributor is explained in more detail in the `Open Tasks <https://lets-unify.ai/ivy/contributing/open_tasks.html#frontend-apis>`_ section.


Supported Data Types and Devices
--------------------------------

Sometimes, the corresponding function in the original framework might only support a subset of data types.
For example, :func:`tf.math.logical_and` only supports inputs of type :code:`tf.bool`.
However, Ivy's `implementation <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L84>`_ is as follows, with direct wrapping around :func:`ivy.logical_and`:

.. code-block:: python

    def logical_and(x, y, name="LogicalAnd"):
        return ivy.logical_and(x, y)

:func:`ivy.logical_and` supports all data types, and so :func:`ivy.functional.frontends.tensorflow.math.logical_and` can also easily support all data types.
However, the primary purpose of these frontend functions is for code transpilations, and in such cases it would never be useful to support extra data types beyond :code:`tf.bool`, as the tensorflow code being transpiled would not support this.
Additionally, the unit tests for all frontend functions use the original framework function as the ground truth, and so we can only test :func:`ivy.functional.frontends.tensorflow.math.logical_and` with boolean inputs anyway.


For these reasons, all frontend functions which correspond to functions with limited data type support in the native framework (in other words, which have even more restrictions than the data type limitations of the framework itself) should be flagged `as such <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L88>`_ in a manner like the following:

.. code-block:: python

   logical_and.supported_dtypes = ("bool",)

The same logic applies to unsupported devices.
Even if the wrapped Ivy function supports more devices, we should still flag the frontend function supported devices to be the same as those supported by the function in the native framework.
Again, this is only needed if the limitations go beyond those of the framework itself.
For example, it is not necessary to uniquely flag every single NumPy function as supporting only CPU, as this is a limitation of the entire framework, and this limitation is already `globally flagged <https://github.com/unifyai/ivy/blob/6eb2cadf04f06aace9118804100b0928dc71320c/ivy/functional/backends/numpy/__init__.py#L21>`_.

It could also be the case that a frontend function supports a data type, but one or more of the backend frameworks does not, and therefore the frontend function may not support the data type due to backend limitation.
For example, the frontend function `jax.lax.cumprod <https://github.com/unifyai/ivy/blob/6e80b20d27d26b67a3876735c3e4cd9a1d38a0e9/ivy/functional/frontends/jax/lax/operators.py#L111>`_ do support all data types, but PyTorch does not support :code:`bfloat16` for the function :func:`cumprod`, even though the framework generally supports handling :code:`bfloat16` data type.
In that case, we should flag that the backend function does not support :code:`bfloat16` as this is done `here <https://github.com/unifyai/ivy/blob/6e80b20d27d26b67a3876735c3e4cd9a1d38a0e9/ivy/functional/backends/torch/statistical.py#L234>`_.

Classes and Instance Methods
----------------------------

Most frameworks include instance methods and special methods on their array class for common array processing functions, such as :func:`reshape`, :func:`expand_dims` and :func:`add`.
This simple design choice comes with many advantages, some of which are explained in our :ref:`Ivy Array` section.

**Important Note**
Before implementing the instance method or special method, make sure that the regular function in the specific frontend is already implemented.

In order to implement Ivy's frontend APIs to the extent that is required for arbitrary code transpilations, it's necessary for us to also implement these instance methods and special methods of the framework-specific array classes (:class:`tf.Tensor`, :class:`torch.Tensor`, :class:`numpy.ndarray`, :class:`jax.numpy.ndarray` etc).

**Instance Method**

**numpy.ndarray**

For an example of how these are implemented, we first show the instance method for :meth:`np.ndarray.add`, which is implemented in the frontend `ndarray class <https://github.com/unifyai/ivy/blob/master/ivy/functional/frontends/numpy/ndarray/ndarray.py#L23>`_:


.. code-block:: python

    # ivy/functional/frontends/numpy/ndarray/ndarray.py
    def add(
        self,
        value,
    ):
        return np_frontend.add(
            self.data,
            value,
        )

Under the hood, this simply calls the frontend :func:`np_frontend.add` function, which itself is implemented as follows:

.. code-block:: python

    # ivy/functional/frontends/numpy/mathematical_functions/arithmetic_operations.py
    def add(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
    ):
    if dtype:
        x1 = ivy.astype(ivy.array(x1), ivy.as_ivy_dtype(dtype))
        x2 = ivy.astype(ivy.array(x2), ivy.as_ivy_dtype(dtype))
    ret = ivy.add(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

**Special Method**

Some examples referring to the special methods would make things more clear.
For example lets take a look at how :meth:`tf_frontend.tensor.__add__` is implemented and how it's reverse :meth:`tf_frontend.tensor.__radd__` is implemented.

.. code-block:: python

    # ivy/functional/frontends/tensorflow/tensor.py
    def __add__(self, y, name="add"):
        return tf_frontend.add(self.data, y, name=name)

For the reverse operator of :func:`add`.

.. code-block:: python

    # ivy/functional/frontends/tensorflow/tensor.py
    def __radd__(self, x, name="radd"):
        return tf_frontend.add(x, self.data, name=name)


Here also, both of them simply call the frontend :func:`tf_frontend.math.add` under the hood.
The functions with reverse operators should call the same frontend function as shown in the examples above.
The implementation for the :func:`tf_frontend.math.add` is shown as follows:

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    def add(x, y, name=None):
    return ivy.add(x, y)

**numpy.matrix**

To support special classes and their instance methods, the equivalent classes are created in their respective frontend so that the useful instance methods are supported for transpilation.

For instance, the :class:`numpy.matrix` class is supported in the Ivy NumPy frontend.
Part of the code is shown below as an example:

.. code-block:: python

    # ivy/functional/frontends/numpy/matrix/methods.py
    class matrix:
        def __init__(self, data, dtype=None, copy=True):
            self._init_data(data, dtype)

        def _init_data(self, data, dtype):
            if isinstance(data, str):
                self._process_str_data(data, dtype)
            elif isinstance(data, list) or ivy.is_array(data):
                data = (
                    ivy.array(data, dtype=dtype) if ivy.exists(dtype) else ivy.array(data)
                )
                ivy.assertions.check_equal(len(ivy.shape(data)), 2)
                self._data = data
            else:
                raise ivy.exceptions.IvyException("data must be a 2D array, list, or str")
            self._shape = ivy.shape(self._data)
            self._dtype = self._data.dtype

With this class available, the supported instance methods can now be included in the class.
For example, :class:`numpy.matrix` has an instance method of :meth:`any`:

.. code-block:: python

    # ivy/functional/frontends/numpy/matrix/methods.py
    def any(self, axis=None, out=None):
        if ivy.exists(axis):
            return ivy.any(self.A, axis=axis, keepdims=True, out=out)
        return ivy.any(self.A, axis=axis, out=out)

We need to create these frontend array classes and all of their instance methods and also their special methods such that we are able to transpile code which makes use of these methods.
As explained in :ref:`Ivy as a Transpiler`, when transpiling code we first extract the computation graph in the source framework.
In the case of instance methods, we then replace each of the original instance methods in the extracted computation graph with these new instance methods defined in the Ivy frontend class.

Frontend Data Type Promotion Rules
----------------------------------

Each frontend framework has its own rules governing the common result type for two array operands during an arithmetic operation.

In order to ensure that each frontend framework implemented in Ivy has the same data type promotion behaviors as the native framework does, we have implemented data type promotion rules according to framework-specific data type promotion tables for these we are currently supporting as frontends.
The function can be accessed through calling :func:`promote_types_of_<frontend>_inputs` and pass in both array operands.

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    from ivy.functional.frontends.tensorflow import promote_types_of_tensorflow_inputs
    ...
    def add(x, y, name=None):
        x, y = promote_types_of_tensorflow_inputs(x, y)
        return ivy.add(x, y)

Although under most cases, array operands being passed into an arithmetic operation function should be the same data type, using the data type promotion rules can add a layer of sanity check to prevent data precision losses or exceptions from further arithmetic operations.

NumPy Special Argument - Casting
--------------------------------

NumPy supports an additional, special argument - :code:`casting`, which allows user to determine the kind of dtype casting that fits their objectives.
The :code:`casting` rules are explained in the `numpy.can_cast documentation <https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html>`_.
While handling this argument, the :code:`dtype` argument is used to state the desired return dtype.

To handle this, a decorator - :code:`handle_numpy_casting` is used to simplify the handling logic and reduce code redundancy.
It is located in the `ivy/functional/frontends/numpy/func_wrapper.py <https://github.com/unifyai/ivy/blob/45d443187678b33dd2b156f29a18b84efbc48814/ivy/functional/frontends/numpy/func_wrapper.py#L39>`_.

This decorator is then added to the numpy frontend functions with the :code:`casting` argument.
An example of the :func:`add` function is shown below.

.. code-block:: python

    # ivy/functional/frontends/numpy/mathematical_functions/arithmetic_operations.py
    @handle_numpy_casting
    @to_ivy_arrays_and_back
    def add(
        x1,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
    ):
        ret = ivy.add(x1, x2, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

There is a special case for the :code:`casting` argument, where the allowed dtype must be :code:`bool`, therefore a :code:`handle_numpy_casting_special` is included to handle this.

.. code-block:: python

    # ivy/functional/frontends/numpy/func_wrapper.py
    def handle_numpy_casting_special(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def new_fn(*args, casting="same_kind", dtype=None, **kwargs):
            ivy.assertions.check_elem_in_list(
                casting,
                ["no", "equiv", "safe", "same_kind", "unsafe"],
                message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
            )
            if ivy.exists(dtype):
                ivy.assertions.check_equal(
                    ivy.as_ivy_dtype(dtype),
                    "bool",
                    message="output is compatible with bool only",
                )
            return fn(*args, **kwargs)
        new_fn.handle_numpy_casting_special = True
        return new_fn


An example function using this is the :func:`numpy.isfinite` function.

.. code-block:: python

    # ivy/functional/frontends/numpy/logic/array_type_testing.py
    @handle_numpy_casting_special
    @to_ivy_arrays_and_back
    def isfinite(
        x,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        ret = ivy.isfinite(x, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

**Round Up**

This should hopefully have given you a better grasp on the what the Ivy Frontend APIs are for, how they should be implemented, and the things to watch out for!
We also have a short `YouTube tutorial series`_ on this as well if you prefer a video explanation!

If you have any questions, please feel free to reach out on `discord`_ in the `ivy frontends channel`_ or in the `ivy frontends forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/_9KeK-idaFs" class="video">
    </iframe>
