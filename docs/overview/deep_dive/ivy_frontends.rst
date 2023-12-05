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
.. _`open task`: https://unify.ai/docs/ivy/overview/contributing/open_tasks.html#frontend-apis
.. _`Array manipulation routines`: https://numpy.org/doc/stable/reference/routines.array-manipulation.html#
.. _`Array creation routines`: https://numpy.org/doc/stable/reference/routines.array-creation.html

Introduction
------------

On top of the Ivy functional API and backend functional APIs, Ivy has another set of framework-specific frontend functional APIs, which play an important role in code transpilations, as explained `here <https://lets-unify.ai/docs/ivy/overview/design/ivy_as_a_transpiler.html>`_.




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
In our JAX frontend, if we add both of these to the root namespace, it would not be possible to call :func:`jax.abs` in our frontend.

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


**Frontend Arrays**

The native arrays of each framework have their own attributes and instance methods which differ from the attributes and instance methods of :class:`ivy.Array`.
As such we have implemented framework-specific array classes: :class:`tf_frontend.Tensor`, :class:`torch_frontend.Tensor`, :class:`numpy_frontend.ndarray`, and :class:`jax_frontend.DeviceArray`.
These classes simply wrap an :class:`ivy.Array`, which is stored in the :code:`ivy_array` attribute, and behave as closely as possible to the native framework array classes.
This is explained further in the `Classes and Instance Methods <https://unify.ai/docs/ivy/overview/deep_dive/ivy_frontends.html#id6>`_ section.

As we aim to replicate the frontend frameworks as closely as possible, all functions accept their frontend array class (as well as :class:`ivy.Array` and :class:`ivy.NativeArray`) and return a frontend array.
However, since most logic in each function is handled by Ivy, the :class:`ivy.Array` must be extracted from any frontend array inputs.
Therefore we add the wrapper :code:`@to_ivy_arrays_and_back` to virtually all functions in the frontends.

There are more framework-specific classes we support in the frontends such as NumPy and Tensorflow :class:`Dtype` classes, NumPy and Jax :class:`Scalars`, NumPy :class:`Matrix`, etc.
All these increase the fidelity of our frontends.


Writing Frontend Functions
-------------------

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
    @to_ivy_arrays_and_back
    def add(x, y):
        return ivy.add(x, y)

:func:`lax.add` is categorised under :code:`operators` as shown in the `jax.lax`_ package directory.
We organize the functions using the same categorizations as the original framework, and also mimic the importing behaviour regarding modules and namespaces etc.

For the function arguments, these must be identical to the original function in Jax.
In this case, `jax.lax.add`_ has two arguments, and so we will also have the same two arguments in our Jax frontend :func:`lax.add`.
In this case, the function will then simply return :func:`ivy.add`, which in turn will link to the backend-specific implementation :func:`ivy.add` according to the framework set in the backend.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    @to_ivy_arrays_and_back
    def tan(x):
        return ivy.tan(x)

Using :func:`lax.tan` as a second example, we can see that this is placed under :mod:`operators`, again in the `jax.lax`_ directory.
By referring to the `jax.lax.tan`_ documentation, we can see that it has only one argument.
In the same manner as our :func:`add` function, we simply link its return to :func:`ivy.tan`, and again the computation then depends on the backend framework.

**NumPy**

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/arithmetic_operations.py
    @handle_numpy_out
    @handle_numpy_dtype
    @to_ivy_arrays_and_back
    @handle_numpy_casting
    @from_zero_dim_arrays_to_scalar
    def _add(
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
        x1, x2 = promote_types_of_numpy_inputs(x1, x2)
        ret = ivy.add(x1, x2, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

In NumPy, :func:`add` is categorised under :mod:`mathematical_functions` with a sub-category of :mod:`arithmetic_operations` as shown in the `numpy mathematical functions`_ directory.
It is important to note that :func:`add` is a universal function (`ufunc <https://numpy.org/doc/stable/reference/ufuncs.html>`_) in NumPy, thus the function is actually an object with instance methods like :code:`.at` and :code:`.reduce`, etc.
We deal with this in the NumPy frontend by including a :class:`ufunc` class and initialising it in the :mod:`__init__` file:

.. code-block:: python

    # in ivy/functional/frontends/numpy/__init__.py
    from ivy.functional.frontends.numpy.mathematical_functions.arithmetic_operations import _add
    add = ufunc("_add")

As shown, we import the above function :func:`_add` and use it to initialise the :class:`ufunc` object which corresponds to the NumPy :func:`add` function.
Practically the :func:`add` object calls the :func:`_add` under the hood, but it has all the extra instance methods of the :class:`ufunc` class.
All other functions which are :class:`ufunc` objects in NumPy are implemented in the same way.
Of course if the :class:`ufunc` object and its respective function have the same name, we would run into problems where one would overwrite the other, to prevent this we make the actual function private by adding an underscore in the front of its name.
Since only the :class:`ufunc` object should be accessible to the user, this approach is sufficient.
When adding new NumPy functions which are :class:`ufuncs`, it's important to implement them in this way in order to properly replicate their functionality.
Namely, a private function needs to be created in the respective sub-category, this function needs to be imported in the :mod:`__init__` file, and a :class:`ufunc` object needs to be created that shares the name of the function.
For functions which are not :class:`ufuncs`, they are named normally without the underscore and are implemented as any other function.

The function arguments for this function are slightly more complex due to the extra optional arguments.
Additional handling code is added to recover the behaviour according to the `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_ documentation.
For example, :code:`@handle_numpy_out` is added to functions with an :code:`out` argument and it handles the inplace update of the :class:`ivy.Array` specified by :code:`out`, or the :class:`ivy.Array` wrapped by a frontend :class:`ndarray`.
This wrapper was added because :code:`out` can be either a positional or keyword argument in most functions, thus it required some additional logic for proper handling.
Additionally, :code:`casting` and :code:`dtype` are handled in the :code:`@handle_numpy_casting` wrapper, which casts the input arguments to the desired dtype as specified by :code:`dtype` and the chosen :code:`casting` rules.
There's an additional wrapper for the :code:`dtype` argument :code:`@handle_numpy_dtype`.
This wrapper is included to handle the various formats of the :code:`dtype` argument which NumPy `accepts <https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types>`_, such as type strings, :class:`numpy.Dtype` objects, characters, etc.
In NumPy, most functions which can return a scalar value return it as a NumPy `Scalar <https://numpy.org/doc/stable/reference/arrays.scalars.html>`_.
To replicate this we add the wrapper :code:`@from_zero_dim_arrays_to_scalar` which converts outputs that would normally be 0-dim arrays from Ivy functions, to a NumPy scalar.
Of course the returned scalar object is actually an Ivy frontend equivalent object which behaves very similarly to the frontend :class:`ndarray`.
Finally, :code:`order` is handled in the :code:`@to_ivy_arrays_and_back` decorator.
The returned result is then obtained through :func:`ivy.add` just like the other examples.

However, the argument :code:`subok` is completely unhandled here because it controls whether or not subclasses of the :class:`numpy.ndarray` should be permitted as inputs to the function.
All ivy functions by default do enable subclasses of the :class:`ivy.Array` to be passed, and the frontend function will be operating with :class:`ivy.Array` instances rather than :class:`numpy.ndarray` instances, and so we omit this argument.
Again, it has no bearing on input-output behaviour and so this is not a problem when transpiling between frameworks.

See the section "Unused Arguments" below for more details.

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/trigonometric_functions.py
    @handle_numpy_out
    @handle_numpy_dtype
    @to_ivy_arrays_and_back
    @handle_numpy_casting
    @from_zero_dim_arrays_to_scalar
    def _tan(
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
        ret = ivy.tan(x, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

For the second example, :func:`tan` has a sub-category of :mod:`trigonometric_functions` according to the `numpy mathematical functions`_ directory.
By referring to the `numpy.tan`_ documentation, we can see it has the same additional arguments as the :func:`add` function and it's also a :class:`ufunc`.
In the same manner as :func:`add`, we handle the argument :code:`out`, :code:`where`, :code:`dtype`, :code:`casting`, and :code:`order` but we omit support for :code:`subok`.

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    @to_ivy_arrays_and_back
    def add(x, y, name=None):
        x, y = check_tensorflow_casting(x, y)
        return ivy.add(x, y)

The :func:`add` function is categorised under the :mod:`math` folder in the TensorFlow frontend.
There are three arguments according to the `tf.math.add <https://www.tensorflow.org/api_docs/python/tf/math/add>`_ documentation, which are written accordingly as shown above.
Just like the previous examples, the implementation wraps :func:`ivy.add`, which itself defers to backend-specific functions depending on which framework is set in Ivy's backend.

The arguments :code:`x` and :code:`y` are both used in the implementation, but the argument :code:`name` is not used.
Similar to the omitted argument in the NumPy example above, the :code:`name` argument does not change the input-output behaviour of the function.
Rather, this argument is added purely for the purpose of operation logging and retrieval, and also graph visualization in TensorFlow.
Ivy does not support the unique naming of individual operations, and so we omit support for this particular argument.

Additionally TensorFlow only allows explicit casting, therefore there are no promotion rules in the TensorFlow frontend, except in the case of array like or scalar inputs, which get casted to the dtype of the other argument if it's a :class:`Tensor`, or the default dtype if both arguments are array like or scalar.
The function :func:`check_tensorflow_casting` is added to functions with multiple arguments such as :func:`add`, and it ensures the second argument is the same type as the first, just as TensorFlow does.

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    @to_ivy_arrays_and_back
    def tan(x, name=None):
        return ivy.tan(x)

Likewise, :code:`tan` is also placed under :mod:`math`.
By referring to the `tf.math.tan`_ documentation, we add the same arguments, and simply wrap :func:`ivy.tan` in this case.
Again, we do not support the :code:`name` argument for the reasons outlined above.

**NOTE**

Many of the functions in the :mod:`tf.raw_ops` module have identical behaviour to functions in the general TensorFlow namespace e.g :func:`tf.argmax`.
However, these functions are specified to have key-word only arguments and in some cases they have different argument names.
In order to tackle these variations in behaviour, the :code:`map_raw_ops_alias` decorator was designed to wrap the functions that exist in the TensorFlow namespace, thus reducing unnecessary re-implementations.

.. code-block:: python
    
    # in ivy/functional/frontends/tensorflow/math.py
    @to_ivy_arrays_and_back
    def argmax(input, axis, output_type=None, name=None):
        if output_type in ["uint16", "int16", "int32", "int64"]:
            return ivy.astype(ivy.argmax(input, axis=axis), output_type)
        else:
            return ivy.astype(ivy.argmax(input, axis=axis), "int64")

This function :func:`argmax` is implemented in the :mod:`tf.math` module of the TensorFlow framework, there exists an identical function in the :mod:`tf.raw_ops` module implemented as :func:`ArgMax`.
Both the functions have identical behaviour except for the fact that all arguments are passed as key-word only for :func:`tf.raw_ops.ArgMax`.
In some corner cases, arguments are renamed such as :func:`tf.math.argmax`, the :code:`dimension` argument replaces the :code:`axis` argument.
Let's see how the :code:`map_raw_ops_alias` decorator can be used to tackle these variations.

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/raw_ops.py
    ArgMax = to_ivy_arrays_and_back(
        map_raw_ops_alias(
            tf_frontend.math.argmax,
            kwargs_to_update={"dimension": "axis"},
        )
    )

The decorator :code:`map_raw_ops_alias` here, takes the existing behaviour of :func:`tf_frontend.math.argmax` as its first parameter, and changes all its arguments to key-word only. The argument :code:`kwargs_to_update` is a dictionary indicating all updates in arguments names to be made, in the case of :func:`tf.raw_ops.ArgMax`, :code:`dimension` is replacing :code:`axis`.
The wrapper mentioned above is implemnted here `map_raw_ops_alias <https://github.com/unifyai/ivy/blob/54cc9cd955b84c50a1743dddddaf6e961f688dd5/ivy/functional/frontends/tensorflow/func_wrapper.py#L127>`_  in the ivy codebase.

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    @to_ivy_arrays_and_back
    def add(input, other, *, alpha=None, out=None):
        return ivy.add(input, other, alpha=alpha, out=out)

For PyTorch, :func:`add` is categorised under :mod:`pointwise_ops` as is the case in the `torch`_ framework.

In this case, the native `torch.add`_ has both positional and keyword arguments, and we therefore use the same for our PyTorch frontend :func:`add`.
We wrap :func:`ivy.add` as usual.

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    @to_ivy_arrays_and_back
    def tan(input, *, out=None):
        return ivy.tan(input, out=out)

:func:`tan` is also placed under :mod:`pointwise_ops` as is the case in the `torch`_ framework.
Looking at the `torch.tan`_ documentation, we can mimic the same arguments, and again simply wrap :func:`ivy.tan`, also making use of the :code:`out` argument in this case.

Short Frontend Implementations
-----------------------------

Ideally all frontend functions should call the equivalent Ivy function and only be one line long. This is mainly because compositional implementations are bound to be slower than direct backend implementation calls.

In case a frontend function is complex and there is no equivalent Ivy function to use, it is strongly advised to add that function to our Experimental API. To do so, you are invited to open a *Missing Function Suggestion* issue as described in the `Open Tasks <https://unify.ai/docs/ivy/overview/contributing/the_basics.html#id4>`_ section. A member of our team will then review your issue, and if the proposed addition is deemed to be timely and sensible, we will add the function to the "Extend Ivy Functional API" `ToDo list issue <https://github.com/unifyai/ivy/issues/3856>`_.

If you would rather not wait around for a member of our team to review your suggestion, you can instead go straight ahead and add the frontend function as a heavy composition of the existing Ivy functions, with a :code:`#ToDo` comment included, explaining that this frontend implementation will be simplified when :func:`ivy.func_name` is added.

**Examples**

The native TensorFlow function :func:`tf.reduce_logsumexp` does not have an equivalent function in Ivy, therefore it can be composed of multiple Ivy functions instead.

**TensorFlow Frontend**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    @to_ivy_arrays_and_back
    def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name="reduce_logsumexp"):
        # stable logsumexp trick
        max_input_tensor = ivy.max(input_tensor, axis=axis, keepdims=True)
        return (
            ivy.log(
                ivy.sum(
                    ivy.exp(input_tensor - max_input_tensor),
                    axis=axis,
                    keepdims=keepdims,
                )
            )
            + max_input_tensor
        ).astype(input_tensor.dtype)

Through compositions, we can easily meet the required input-output behaviour for the TensorFlow frontend function.

The entire workflow for extending the Ivy Frontends as an external contributor is explained in more detail in the `Open Tasks <https://unify.ai/docs/ivy/overview/contributing/open_tasks.html#frontend-apis>`_ section.

Unused Arguments
----------------

As can be seen from the examples above, there are often cases where we do not add support for particular arguments in the frontend function.
Generally, we can omit support for a particular argument only if: the argument **does not** fundamentally affect the input-output behaviour of the function in a mathematical sense.
The only two exceptions to this rule are arguments related to either the data type or the device on which the returned array(s) should reside.
Examples of arguments which can be omitted, on account that they do not change the mathematics of the function are arguments which relate to:

* the algorithm or approximations used under the hood, such as :code:`precision` and :code:`preferred_element_type` in `jax.lax.conv_general_dilated <https://github.com/google/jax/blob/1338864c1fcb661cbe4084919d50fb160a03570e/jax/_src/lax/convolution.py#L57>`_.

* the specific array class in the original framework, such as :code:`subok` in `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_.

* the labelling of functions for organizational purposes, such as :code:`name` in `tf.math.add <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/python/ops/math_ops.py#L3926-L4004>`_.

There are likely to be many other examples of arguments which do not fundamentally affect the input-output behaviour of the function in a mathematical sense, and so can also be omitted from Ivy's frontend implementation.

The reason we omit these arguments in Ivy is because Ivy is not designed to provide low-level control to functions that extend beyond the pure mathematics of the function.
This is a requirement because Ivy abstracts the backend framework, and therefore also abstracts everything below the backend framework's functional API, including the backend array class, the low-level language compiled to, the device etc.
Most ML frameworks do not offer per-array control of the memory layout, and control for the finer details of the algorithmic approximations under the hood, and so we cannot in general offer this level of control at the Ivy API level, nor the frontend API level as a direct result.
As explained above, this is not a problem, as the memory layout has no bearing at all on the input-output behaviour of the function.
In contrast, the algorithmic approximation may have a marginal bearing on the final results in some cases, but Ivy is only designed to unify to within a reasonable numeric approximation in any case, and so omitting these arguments also very much fits within Ivy's design.

Supported Data Types and Devices
--------------------------------

Sometimes, the corresponding function in the original framework might only support a subset of data types.
For example, :func:`tf.math.logical_and` only supports inputs of type :code:`tf.bool`.
However, Ivy's `implementation <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L84>`_ is as follows, with direct wrapping around :func:`ivy.logical_and`:

.. code-block:: python

    @to_ivy_arrays_and_back
    def logical_and(x, y, name="LogicalAnd"):
        return ivy.logical_and(x, y)

:func:`ivy.logical_and` supports all data types, and so :func:`ivy.functional.frontends.tensorflow.math.logical_and` can also easily support all data types.
However, the primary purpose of these frontend functions is for code transpilations, and in such cases it would never be useful to support extra data types beyond :code:`tf.bool`, as the tensorflow code being transpiled would not support this.
Additionally, the unit tests for all frontend functions use the original framework function as the ground truth, and so we can only test :func:`ivy.functional.frontends.tensorflow.math.logical_and` with boolean inputs anyway.


For these reasons, all frontend functions which correspond to functions with limited data type support in the native framework (in other words, which have even more restrictions than the data type limitations of the framework itself) should be flagged `as such <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L88>`_ in a manner like the following:

.. code-block:: python

   @with_unsupported_dtypes({"2.13.0 and below": ("float16", "bfloat16")}, "tensorflow")

The same logic applies to unsupported devices.
Even if the wrapped Ivy function supports more devices, we should still flag the frontend function supported devices to be the same as those supported by the function in the native framework.
Again, this is only needed if the limitations go beyond those of the framework itself.
For example, it is not necessary to uniquely flag every single NumPy function as supporting only CPU, as this is a limitation of the entire framework, and this limitation is already `globally flagged <https://github.com/unifyai/ivy/blob/6eb2cadf04f06aace9118804100b0928dc71320c/ivy/functional/backends/numpy/__init__.py#L21>`_.

It could also be the case that a frontend function supports a data type, but one or more of the backend frameworks does not, and therefore the frontend function may not support the data type due to backend limitation.
For example, the frontend function `jax.lax.cumprod <https://github.com/unifyai/ivy/blob/6e80b20d27d26b67a3876735c3e4cd9a1d38a0e9/ivy/functional/frontends/jax/lax/operators.py#L111>`_ does support all data types, but PyTorch does not support :code:`bfloat16` for the function :func:`cumprod`, even though the framework generally supports handling :code:`bfloat16` data type.
In that case, we should flag that the backend function does not support :code:`bfloat16` as this is done `here <https://github.com/unifyai/ivy/blob/6e80b20d27d26b67a3876735c3e4cd9a1d38a0e9/ivy/functional/backends/torch/statistical.py#L234>`_.

Classes and Instance Methods
----------------------------

Most frameworks include instance methods and special methods on their array class for common array processing functions, such as :func:`reshape`, :func:`expand_dims` and :func:`add`.
This simple design choice comes with many advantages, some of which are explained in our :ref:`Ivy Array` section.

**Important Note**
Before implementing the instance method or special method, make sure that the regular function in the specific frontend is already implemented.

In order to implement Ivy's frontend APIs to the extent that is required for arbitrary code transpilations, it's necessary for us to also implement these instance methods and special methods of the framework-specific array classes (:class:`tf.Tensor`, :class:`torch.Tensor`, :class:`numpy.ndarray`, :class:`jax.DeviceArray` etc).

**Instance Method**

**numpy.ndarray**

For an example of how these are implemented, we first show the instance method for :meth:`np.ndarray.argsort`, which is implemented in the frontend `ndarray class <https://github.com/unifyai/ivy/blob/94679019a8331cf9d911c024b9f3e6c9b09cad02/ivy/functional/frontends/numpy/ndarray/ndarray.py#L8>`_:


.. code-block:: python

    # ivy/functional/frontends/numpy/ndarray/ndarray.py
    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self._ivy_array, axis=axis, kind=kind, order=order)

Under the hood, this simply calls the frontend :func:`np_frontend.argsort` function, which itself is implemented as follows:

.. code-block:: python

    # ivy/functional/frontends/numpy/mathematical_functions/arithmetic_operations.py
    @to_ivy_arrays_and_back
    def argsort(
        x,
        /,
        *,
        axis=-1,
        kind=None,
        order=None,
    ):
        return ivy.argsort(x, axis=axis)

**Special Method**

Some examples referring to the special methods would make things more clear.
For example lets take a look at how :meth:`tf_frontend.tensor.__add__` is implemented and how it's reverse :meth:`tf_frontend.tensor.__radd__` is implemented.

.. code-block:: python

    # ivy/functional/frontends/tensorflow/tensor.py
    def __radd__(self, x, name="radd"):
        return tf_frontend.math.add(x, self._ivy_array, name=name)

    def __add__(self, y, name="add"):
        return self.__radd__(y)

Here also, both of them simply call the frontend :func:`tf_frontend.math.add` under the hood.
The functions with reverse operators should call the same frontend function as shown in the examples above.
The implementation for the :func:`tf_frontend.math.add` is shown as follows:

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    @to_ivy_arrays_and_back
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
            elif isinstance(data, (list, ndarray)) or ivy.is_array(data):
                if isinstance(data, ndarray):
                    data = data.ivy_array
                if ivy.is_array(data) and dtype is None:
                    dtype = data.dtype
                data = ivy.array(data, dtype=dtype)
                self._data = data
            else:
                raise ivy.exceptions.IvyException("data must be an array, list, or str")
            ivy.assertions.check_equal(
                len(ivy.shape(self._data)), 2, message="data must be 2D"
            )
            self._dtype = self._data.dtype
            self._shape = ivy.shape(self._data)

With this class available, the supported instance methods can now be included in the class.
For example, :class:`numpy.matrix` has an instance method of :meth:`any`:

.. code-block:: python

    # ivy/functional/frontends/numpy/matrix/methods.py
    from ivy.functional.frontends.numpy import any
    ...
    def any(self, axis=None, out=None):
        if ivy.exists(axis):
            return any(self.A, axis=axis, keepdims=True, out=out)
        return any(self.A, axis=axis, out=out)

We need to create these frontend array classes and all of their instance methods and also their special methods such that we are able to transpile code which makes use of these methods.
As explained in :ref:`Ivy as a Transpiler`, when transpiling code we first extract the computation graph in the source framework.
In the case of instance methods, we then replace each of the original instance methods in the extracted computation graph with these new instance methods defined in the Ivy frontend class.

Frontend Data Type Promotion Rules
----------------------------------

Each frontend framework has its own rules governing the common result type for two array operands during an arithmetic operation.

In order to ensure that each frontend framework implemented in Ivy has the same data type promotion behaviors as the native framework does, we have implemented data type promotion rules according to framework-specific data type promotion tables for these we are currently supporting as frontends.
The function can be accessed through calling :func:`promote_types_of_<frontend>_inputs` and pass in both array operands.

.. code-block:: python

    # ivy/functional/frontends/torch/pointwise_ops.py
    @to_ivy_arrays_and_back
    def add(input, other, *, alpha=1, out=None):
        input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
        return ivy.add(input, other, alpha=alpha, out=out)

Although under most cases, array operands being passed into an arithmetic operation function should be the same data type, using the data type promotion rules can add a layer of sanity check to prevent data precision losses or exceptions from further arithmetic operations.

TensorFlow is a framework where casting is completely explicit, except for array likes and scalars.
As such there are not promotion rules we replicate for the TensorFlow frontend, instead we check if the two arguments of the function are the same type using :func:`check_tensorflow_casting`.

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    @to_ivy_arrays_and_back
    def add(x, y, name=None):
        x, y = check_tensorflow_casting(x, y)
        return ivy.add(x, y)

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
    @handle_numpy_out
    @handle_numpy_dtype
    @to_ivy_arrays_and_back
    @handle_numpy_casting
    @from_zero_dim_arrays_to_scalar
    def _add(
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
        x1, x2 = promote_types_of_numpy_inputs(x1, x2)
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
    @handle_numpy_out
    @handle_numpy_dtype
    @to_ivy_arrays_and_back
    @handle_numpy_casting_special
    @from_zero_dim_arrays_to_scalar
    def _isfinite(
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


Frontends Duplicate Policy
--------------------------
Some frontend functions appear in multiple namespaces within the original framework that the frontend is replicating.
For example the :func:`np.asarray` function appears in `Array manipulation routines`_ and also in `Array creation routines`_.
This section outlines a policy that should serve as a guide for handling duplicate functions. The following sub-headings outline the policy:

**Listing duplicate frontend functions on the ToDo lists**

Essentially, there are two types of duplicate functions;

1. Functions that are listed in multiple namespaces but are callable from the same path, for example :func:`asarray` is listed in `manipulation routines` and `creation routines` however this function called from the same path as :func:`np.asarray`.

2. Functions that are listed in multiple namespaces but are callable from different paths, for example the function :func:`tf.math.tan` and :func:`tf.raw_ops.Tan`.

When listing frontend functions, extra care should be taken to keep note of these two type of duplicate functions.

* For duplicate functions of the first type, we should list the function once in any namespace where it exists and leave it out of all other namespaces.

* For duplicates of the second type, we should list the function in each namespace where it exists but there should be a note to highlight that the function(s) on the list are duplicates and should therefore be implemented as aliases. For example, most of the functions in `tf.raw_ops` are aliases and this point is made clear when listing the functions on the ToDo list `here <https://github.com/unifyai/ivy/issues/1565>`_.

**Contributing duplicate frontend functions**

Before working on a frontend function, contributors should check if the function is designated as an alias on the ToDo list.
If the function is an alias, you should check if there is an implementation that can be aliased.

* If an implementation exist then simply create an alias of the implementation, for example many functions in `ivy/functional/frontends/tensorflow/raw_ops` are implemented as aliases `here <https://github.com/unifyai/ivy/blob/main/ivy/functional/frontends/tensorflow/raw_ops.py>`_.

* If there is no implementation to be aliased then feel free to contribute the implementation first, then go ahead to create the alias.

**Testing duplicate functions**

Unit tests should be written for all aliases. This is arguably a duplication, but having a unique test for each alias helps us to keep the testing code organised and aligned with the groupings in the frontend API.

**Round Up**

This should hopefully have given you a better grasp on what the Ivy Frontend APIs are for, how they should be implemented, and the things to watch out for!
We also have a short `YouTube tutorial series`_ on this as well if you prefer a video explanation!

If you have any questions, please feel free to reach out on `discord`_ in the `ivy frontends channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/SdiyetRNey8" class="video">
    </iframe>
