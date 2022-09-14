Ivy Frontends
=============

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html
.. _`jax.lax.add`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html
.. _`jax.lax`: https://jax.readthedocs.io/en/latest/jax.lax.html
.. _`jax.lax.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.tan.html
.. _`numpy.add`: https://numpy.org/doc/stable/reference/generated/numpy.add.html
.. _`numpy mathematical functions`: https://numpy.org/doc/stable/reference/index.html
.. _`numpy.tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`tf.add`: https://www.tensorflow.org/api_docs/python/tf/math/add
.. _`tf`: https://www.tensorflow.org/api_docs/python/tf
.. _`tf.tan`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`torch.add`: https://pytorch.org/docs/stable/generated/torch.add.html#torch.add
.. _`torch`: https://pytorch.org/docs/stable/torch.html#math-operations
.. _`torch.tan`: https://pytorch.org/docs/stable/generated/torch.tan.html#torch.tan
.. _`YouTube tutorial series`: https://www.youtube.com/watch?v=72kBVJTpzIw&list=PLwNuX3xB_tv-wTpVDMSJr7XW6IP_qZH0t
.. _`ivy frontends discussion`: https://github.com/unifyai/ivy/discussions/2051
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`ivy frontends channel`: https://discord.com/channels/799879767196958751/998782045494976522

Introduction
------------

On top of the Ivy functional API and backend functional APIs, Ivy has another set of
framework-specific frontend functional APIs, which play an important role in code
transpilations, as explained `here`_.

Let's start with some examples to have a better idea on Ivy Frontends!

Basic
-----

**NOTE:** Type hints, docstrings and examples are not required when working on
frontend functions.

**Jax**

In general, all functions in the :code:`jax.numpy` namespace are themselves implemented
as a composition of the lower-level functions in the :code:`jax.lax` namespace,
which maps very closely to the API for the Accelerated Linear Algebra (XLA) compiler
which is used under the hood to run high performance JAX code.

When transpiling between frameworks, the first step is to compile the computation graph
into the lowest level python functions for the source framework using Ivy's graph
compiler, before then replacing these nodes with the associated functions in Ivy's
frontend API. Given that almost all jax code can be decomposed into :code:`jax.lax`
functions, these are the most important functions to implement in Ivy's frontend API.
Thus, a :code:`lax` module is created in the frontend API, and most functions are placed
there. We start with the function :code:`add` as an example.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def add(x, y):
        return ivy.add(x, y)

:code:`add` is categorised under :code:`operators` as shown in the `jax.lax`_ package
directory. We organize the functions using the same categorizations as the original
framework, and also mimick the importing behaviour regarding modules and namespaces etc.

For the function arguments, these must be identical to the original function in
Jax. In this case, `jax.lax.add`_ has two arguments,
and so we will also have the same two arguments in our Jax frontend :code:`add`.
In this case, the function will then simply return :code:`ivy.add`,
which in turn will link to the backend-specific implementation :code:`add`
according to the framework set in the backend.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def tan(x):
        return ivy.tan(x)

Using :code:`tan` as a second example, we can see that this is placed under
:code:`operators`, again in the `jax.lax`_ directory.
By referring to the `jax.lax.tan`_ documentation, we can see that it has only one
argument. In the same manner as our :code:`add` function, we simply link its return to
:code:`ivy.tan`, and again the computation then depends on the backend framework.

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

In NumPy, :code:`add` is categorised under :code:`mathematical_functions` with a
sub-category of :code:`arithmetic_operations` as shown in the
`numpy mathematical functions`_ directory.

The function arguments for this function are slightly more complex due to the extra
optional arguments. Additional handling code is added to recover the behaviour
according to the `numpy.add`_ documentation. For example, if :code:`dtype` is specified,
the arguments will be cast to the desired type through :code:`ivy.astype`.
The returned result is then obtained through :code:`ivy.add` just like the other
examples.

However, the arguments :code:`casting`, :code:`order` and :code:`subok` are completely
unhandled here. This is for two reasons.

In the case of :code:`casting`, support will be added for this via the inclusion of a
decorator at some point in future, and so this is simply being deferred for the time
being.

In the case of :code:`order` and :code:`subok`, this is because the aspects which these
arguments seek to control are simply not controllable when using Ivy.
:code:`order` controls the low-level memory layout of the stored array.
Ivy abstracts the backend framework, and therefore also abstracts everything below Ivy's
functional API, including the backend array class, the low-level language compiled to,
the device etc. Most ML frameworks do not offer per-array control of the memory layout,
and so we cannot offer this control at the Ivy API level, nor the frontend API level
either. This is not a problem, as the memory layout has no bearing at all on the
input-output behaviour of the function. Similarly, :code:`subok` controls whether or not
subclasses of the :code:`numpy.ndarray` should be permitted as inputs to the function.
Again, this is a very framework-specific argument. All ivy functions by default do
enable subclasses of the :code:`ivy.Array` to be passed, and the frontend function will
be operating with :code:`ivy.Array` instances rather than :code:`numpy.ndarray`
instances, and so we omit this argument. Again, it has no bearing on input-output
behaviour and so this is not a problem when transpiling between frameworks.

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/trigonometric_functions.py
    def tan(
        x,
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
            x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
        ret = ivy.tan(x, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

For the second example, :code:`tan` has a sub-category of
:code:`trigonometric_functions` according to the `numpy mathematical functions`_
directory. By referring to the `numpy.tan`_ documentation, we can see it has the same
additional arguments as the :code:`add` function. In the same manner as :code:`add`,
we handle the argument :code:`out`, :code:`where` and :code:`dtype`,
but we omit support for :code:`casting`, :code:`order` and :code:`subok`.

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    def add(x, y, name=None):
        return ivy.add(x, y)

The :code:`add` function is categorised under the :code:`math` folder in the TensorFlow
frontend. There are three arguments according to the `tf.add`_ documentation, which are
written accordingly as shown above. Just like the previous examples, the implementation
wraps :code:`ivy.add`, which itself defers to backend-specific functions depending on
which framework is set in Ivy's backend.

The arguments :code:`x` and :code:`y` are both used in the implementation,
but the argument :code:`name` is not used. Similar to the omitted arguments in the NumPy
example above, the :code:`name` argument does not change the input-output behaviour of
the function. Rather, this argument is added purely for the purpose of operation logging
and retrieval, and also graph visualization in TensorFlow. Ivy does not support the
unique naming of individual operations, and so we omit support for this particular
argument.

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    def tan(x, name=None):
        return ivy.tan(x)

Likewise, :code:`tan` is also placed under :code:`math`.
By referring to the `tf.tan`_ documentation, we add the same arguments,
and simply wrap :code:`ivy.tan` in this case.
Again, we do not support the :code:`name` argument for the reasons outlined above.

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def add(input, other, *, alpha=1, out=None):
        return ivy.add(input, other * alpha, out=out)

For PyTorch, :code:`add` is categorised under :code:`pointwise_ops` as is the case in
the `torch`_ framework.

In this case, the native `torch.add`_ has both positional and keyword arguments,
and we therefore use the same for our PyTorch frontend :code:`add`.
We wrap :code:`ivy.add` as usual, but the arguments work slightly different in this
example. Looking at the PyTorch `torch.add`_ documentation,
we can see that :code:`alpha` acts as a scale for the :code:`other` argument.
Thus, we can mimick the original behaviour by simply passing :code:`other * alpha`
into :code:`ivy.add`.

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def tan(input, *, out=None):
        return ivy.tan(input, out=out)

:code:`tan` is also placed under :code:`pointwise_ops` as is the case in the `torch`_
framework. Looking at the `torch.tan`_ documentation, we can mimick the same arguments,
and again simply wrap :code:`ivy.tan`,
also making use of the :code:`out` argument in this case.

Compositions
------------

In many cases, frontend functions meet the following criteria:

* the function is unique to a particular frontend framework, and does not exist in the
  other frameworks
* the function has extra features and/or arguments on top of the most similar ivy
  function that is available

In such cases, compositions are required to replicate the function behaviour.

**Examples**

In the native TensorFlow function :code:`tf.cumprod()`, it supports an extra
argument - :code:`reverse`, which returns a flipped result if :code:`True`. However,
the backend :code:`ivy.cumprod()` does not support this argument or behaviour.

**Ivy**

.. code-block:: python

    # in ivy/functional/ivy/general.py
    def cumprod(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        axis: int = 0,
        *,
        exclusive: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        return current_backend(x).cumprod(x, axis, exclusive, out=out)

To enable this behaviour, we need to incorporate other Ivy functions which are
compositionally able to mimick the required behaviour.
For example, we can simply reverse the result by calling :code:`ivy.flip()` on the
result of :code:`ivy.cumprod()`.

**TensorFlow Frontend**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
        ret = ivy.cumprod(x, axis, exclusive)
        if reverse:
            return ivy.flip(ret, axis)
        return ret

Through compositions, we can easily meet the required input-output behaviour for the
TensorFlow frontend function.

Temporary Compositions
----------------------

Sometimes, there is a clear omission of an Ivy function, which would make the frontend
implementation much simpler. For example, at the time of writing,
implementing :code:`median` for the NumPy frontend would require a very manual and
heavily compositional implementation.
However, if the function :code:`ivy.median` was added to Ivy's functional API, then this
frontend implementation would become very simple, with some light wrapping around
:code:`ivy.median`.

Adding :code:`ivy.median` would be a sensible decision, as many frameworks support this
function. When you come across such a function which is missing from Ivy, you should
create a new issue on the Ivy repo, with the title :code:`ivy.<func_name>` and with the
labels :code:`Suggestion`, :code:`Extension`, :code:`Ivy API` and :code:`Next Release`.
A member of our team will then review this issue, and if the proposed addition is deemed
to be timely and sensible, then we will add this function to the
"Extend Ivy Functional API"
`ToDo list issue <https://github.com/unifyai/ivy/issues/3856>`_.
At this point in time, you can reserve the function for yourself and get it implemented
in a unique PR. Once merged, you can then resume working on the frontned function,
which will now be a much easier task with the new addition to Ivy.

Alternatively, if after creating the new issue you would rather not wait around for a
member of our team to review and possibly add to the "Extend Ivy Functional API"
`ToDo list issue <https://github.com/unifyai/ivy/issues/3856>`_,
you can instead go straight ahead add the frontend function as a heavy composition of
the existing Ivy functions, with a :code:`#ToDo` comment included, explaining that this
frontend implementation will be simplified if/when :code:`ivy.<func_name>` is add to
Ivy.

The entire workflow for extending the Ivy Frontends as an external contributor is
explained in more detail in the
`Open Tasks <https://lets-unify.ai/ivy/contributing/4_open_tasks.html#frontend-apis>`_
section.


Supported Data Types and Devices
--------------------------------

Sometimes, the corresponding function in the original framework might only support a
subset of data types. For example, :code:`tf.math.logical_and` only supports inputs of
type :code:`tf.bool`. However, Ivy's
`implementation <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L84>`_
is as follows, with direct wrapping around :code:`ivy.logical_and`:

.. code-block:: python

    def logical_and(x, y, name="LogicalAnd"):
        return ivy.logical_and(x, y)

:code:`ivy.logical_and` supports all data types, and so
:code:`ivy.functional.frontends.tensorflow.math.logical_and` can also easily support all
data types. However, the primary purpose of these frontend functions is for code
transpilations, and in such cases it would never be useful to support extra data types
beyond :code:`tf.bool`, as the tensorflow code being transpiled would not support this.
Additionally, the unit tests for all frontend functions use the original framework
function as the ground truth, and so we can only test
:code:`ivy.functional.frontends.tensorflow.math.logical_and` with boolean inputs anyway.


For these reasons, all frontend functions which correspond to functions with limited
data type support in the native framework (in other words, which have even more
restrictions than the data type limitations of the framework itself) should be flagged
`as such <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L88>`_
in a manner like the following:

.. code-block:: python
    
   logical_and.supported_dtypes = ("bool",)

The same logic applies to unsupported devices. Even if the wrapped Ivy function supports
more devices, we should still flag the frontend function supported devices to be the
same as those supported by the function in the native framework. Again, this is only
needed if the limitations go beyond those of the framework itself. For example, it is
not necessary to uniquely flag every single NumPy function as supporting only CPU,
as this is a limitation of the entire framework, and this limitation is already
`globally flagged <>`_.

Instance Methods
----------------

Most frameworks include instance methods on their array class for common array
processing functions, such as :code:`reshape`, :code:`expand_dims` etc.
This simple design choice comes with many advantages,
some of which are explained in our :ref:`Ivy Array` section.

In order to implement Ivy's frontend APIs to the extent that is required for arbitrary
code transpilations, it's necessary for us to also implement these instance methods of
the framework-specific array classes (:code:`tf.Tensor`, :code:`torch.Tensor`,
:code:`numpy.ndarray`, :code:`jax.numpy.ndarray` etc).

For an example of how these are implemented, we first show the instance method for
:code:`np.ndarray.reshape`, which is implemented in the frontend
`ndarray class <https://github.com/unifyai/ivy/blob/2e3ffc0f589791c7afc9d0384ce77fad4e0658ff/ivy/functional/frontends/numpy/ndarray/ndarray.py#L8>`_:

.. code-block:: python

    # ivy/functional/frontends/numpy/ndarray/ndarray.py
    def reshape(self, newshape, copy=None):
        return np_frontend.reshape(self.data, newshape, copy=copy)

Under the hood, this simply calls the frontend :code:`np_frontend.reshape` function,
which itself is implemented as follows:

.. code-block:: python

    # ivy/functional/frontends/numpy/manipulation_routines/changing_array_shape.py
    def reshape(x, /, shape, *, copy=None):
        return ivy.reshape(x, shape, copy=copy)

We need to create these frontend array classes and all of their instance methods such
that we are able to transpile code which makes use of instance methods.
As explained in :ref:`Ivy as a Transpiler`, when transpiling code we first extract the
computation graph in the source framework. In the case of instance methods, we then
replace each of the original instance methods in the extracted computation graph with
these new instance methods defined in the Ivy frontend class.


Framework-Specific Classes
--------------------------

Some of the frontend functions that we need to implement for our frontend functional API 
include framework-specific classes, which do not have a counterpart in other frameworks 
or Ivy, as the types for their arguments or as the default values for the arguments. 
When re-implementing these functions in Ivy's frontend, we would like to still include
those arguments without directly using those special classes as they do not exist in Ivy.
In this section of the deep dive we are going to introduce how to deal with 
Framework-Specific Classes.

For each backend framework, there is a dictionary named `<backend>_classes_to_ivy_classes`
in `ivy/ivy_tests/test_ivy/test_frontends/test_<backend>/__init__.py`, 
which will hold pairs of framework-specific classes and corresponding Ivy or 
native Python classes. For example, in `ivy/ivy_tests/test_ivy/test_frontends/test_numpy/__init__.py`, we have: 

.. code-block:: python
    
    numpy_classes_to_ivy_classes = {np._NoValue: None}

Where np._NoValue is a reference to _NoValueType class defined in numpy/numpy/_globals.py, 
which represents a special keyword value and the instance of this class may be used as the
default value assigned to a keyword if no other obvious default (e.g., :code:`None`) is suitable.

When you found that the frontend function of a certain framework that you try to implement 
in our frontend API introduce a new datatype that, like the :code:`numpy._NoValue` example before, 
can not be directly replaced, you may pick an existing Ivy or pure python datatype 
and use them instead in the ivy frontend implementation to mimic the same effect and record 
the pair of framework-specific class’s reference and your replacement class’s reference 
in the corresponding dictionary.

As our frontend test will try to pass all the generated inputs in both our own implementation
and the original function, then you cannot directly pass either the framework-specific class
or your chosen counterpart in our test function. Instead, you should pass a :code:`NativeClass` 
object in. The :code:`NativeClass` is defined in ‘ivy/ivy_tests/test_ivy/test_frontends/__init__.py’ as a placeholder class to represent a 
pair of framework specific class and its counterpart. It has only one attribute, which is 
:code:`_native_class`, that holds the reference to the special class being used by the 
targeted framework.

When writing a test for a frontend function where its original counterpart accepts a 
framework-specific class, you should import the :code:`NativeClass` and initialize an instance
of it with :code:`_native_class` set as the reference to the special class, which you have 
added in the `<backend>_classes_to_ivy_classes` dictionary before. Then just pass the 
:code:`NativeClass` instance in the arguments like other generated input and the 
:code:`helpers.test_frontend_function` will replace it with the actual classes accordingly 
in the background.

Here is an example of :code:`NativeClass` being put to use in test.

ivy.sum()
^^^^^^^^^^

.. code-block:: python
    # sum
    Novalue = NativeClass(numpy._NoValue)
    @handle_cmd_line_args
    @given(
        dtype_x_axis=_dtype_x_axis(
            available_dtypes=ivy_np.valid_float_dtypes),
        dtype=st.sampled_from(
            ivy_np.valid_float_dtypes + (None,)),
        keep_dims= st.one_of (st.booleans(), Novalue),
        initial=st.one_of(st.floats(), Novalue),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.sum"
        ),
    )
    def test_numpy_sum(
        dtype_x_axis,
        dtype,
        keep_dims,
        initial,
        as_variable,
        num_positional_args,
        native_array,
        with_out,
        fw,
    ):
        (input_dtype, x, axis), where = dtype_x_axis
        if where is None:
            where = Novalue
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="numpy",
            fn_tree="sum",
            x=np.asarray(x, dtype=input_dtype[0]),
            axis=axis,
            dtype=dtype,
            keepdims=keep_dims,
            initial=initial,
            where=where,
        )

* NumPy.sum has three optional arguments: :code:`where`, :code:`keep_dims`, :code:`initial`, which all have the default value of numpy._NoValue. So we define a NativeClass object Novalue to help recreate the effect of not passing any value to those arguments by using Novalue instead where None used to be generated for the those arguments. 


**Round Up**

This should hopefully allow you to have a better grasp on the Ivy Frontend APIs
after going through the contents! We have a `YouTube tutorial series`_ on this
as well if you prefer a video explanation!

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy frontends discussion`_,
or reach out on `discord`_ in the `ivy frontends channel`_!
