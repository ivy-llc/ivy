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

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def add(x, y):
        return ivy.add(x, y)

In the original Jax library, :code:`add` is under `jax.lax.add`_. Thus, an
identical module of :code:`lax` is created and the function is placed there. It
is then categorised under :code:`operators` as shown in the `jax.lax`_ package directory.
This is to ensure that :code:`jax.lax.add` is available directly without further
major changes when using :code:`ivy`. It is valid by simply importing
:code:`ivy.functional.frontends.jax`.

For the function arguments, it has to be identical to the original function in
Jax to ensure identical behaviour. In this case, `jax.lax.add`_ has two arguments,
where we will also have the same two arguments in our Jax frontend :code:`add`.
Then, this function will return :code:`ivy.add`, which links to the :code:`add`
function according to the framework set in the backend.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def tan(x):
        return ivy.tan(x)

Looking at a second example, :code:`tan`, it is placed under :code:`operators`
according to the `jax.lax`_ directory. By referring to the `jax.lax.tan`_ documentation,
it has only one argument, and just as our :code:`add` function, we link its return to
:code:`ivy.tan` so that the computation operation depends on the backend framework.

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
`numpy mathematical functions`_ directory. This ensures direct access to
:code:`numpy.add` in :code:`ivy` by simply importing
:code:`ivy.functional.frontends.numpy`.

The function arguments for this function are slightly more complex due to the extra
optional arguments. Additional handling code is added to recover the behaviour
according to the `numpy.add`_ documentation. For example, if :code:`dtype` is specified,
the arguments to be added will be casted to the desired type through
:code:`ivy.astype`. The returned result is then obtained through :code:`ivy.add`
just like the other examples.

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

With :code:`tan` as the second example, it has a sub-category of
:code:`trigonometric_functions` according to the `numpy mathematical functions`_
directory. By referring to the `numpy.tan`_ documentation, it has additional
arguments just like its :code:`add` function, thus needing additional handling code.

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    def add(x, y, name=None):
        return ivy.add(x, y)

In the original TensorFlow library (`tf`_ directory), :code:`add` does not have
a specific category. Therefore, it is categorised under :code:`functions` in Ivy.
This ensures that :code:`tf.add` is available directly without further major
changes when using :code:`ivy`. It is valid by simply importing
:code:`ivy.functional.frontends.tensorflow`.

There are three arguments according to the `tf.add`_ documentation, where we
have written accordingly as shown above. Just like the previous examples, it will
also return :code:`ivy.add` for the linking of backend framework.

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/math.py
    def tan(x, name=None):
        return ivy.tan(x)

Let's look at another example, :code:`tan`, it is placed under :code:`functions` just
like :code:`add`. By referring to the `tf.tan`_ documentation, we code the arguments
accordingly, then link its return to :code:`ivy.tan` so that the computation
operation is decided according to the backend framework.

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def add(input, other, *, alpha=1, out=None):
        return ivy.add(input, other * alpha, out=out)

For PyTorch, :code:`add` is categorised under :code:`pointwise_ops` as shown in
the `torch`_ directory. This ensures direct access to :code:`torch.add` in :code:`ivy`
without further major changes. It is valid by simply importing
:code:`ivy.functional.frontends.torch`.

For the function arguments, it has to be identical to the original function in
PyTorch to ensure identical behaviour. In this case, the native `torch.add`_ has
both positional and keyword arguments, where we will use the same for our PyTorch
frontend :code:`add`. As for its return, we will link it to :code:`ivy.add` as usual.
However, the arguments work slightly different in this example. From understanding
the PyTorch `torch.add`_ documentation, you will notice that :code:`alpha`
acts as a scale for the :code:`other` argument. Thus, we will recover the original
behaviour by passing :code:`other * alpha` into :code:`ivy.add`.

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def tan(input, *, out=None):
        return ivy.tan(input, out=out)

Using :code:`tan` as a second example, it is placed under :code:`pointwise_ops`
according to the `torch`_ directory. By referring to the `torch.tan`_ documentation,
we code its positional and keyword arguments accordingly, then return with
:code:`ivy.tan` to link the operation to the backend framework.

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
the backend :code:`ivy.cumprod()` does not come with this argument, and thus does not
support this behaviour by default.

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

To enable this behaviour, we will need to incorporate functions that resemble the
required behaviour. For example, we can reverse the result by calling
:code:`ivy.flip()` after running :code:`ivy.cumprod()`.

**TensorFlow Frontend**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
        ret = ivy.cumprod(x, axis, exclusive)
        if reverse:
            return ivy.flip(ret, axis)
        return ret

Through compositions, we can easily meet the required input-output behaviour.

Temporary Compositions
----------------------

Sometimes, there is a clear omission of an Ivy function, which would make the frontend
implementation much simpler. For example, implementing :code:`median` for the NumPy
frontend would currently require a very manual and heavily compositional implementation.
However, if the function :code:`ivy.median` was added to Ivy's functional API, then this
frontend implementation would become very simple, with some light wrapping around
:code:`ivy.median`.

Adding :code:`ivy.median` would be a sensible decision, as many frameworks support this
function. However, functions are added to Ivy in an iterative and deliberate manner,
which doesn't always align with the timelines for the frontend implementations.
Sometimes Ivy's API is not ready to have a new function added. In such cases, the
frontend function should be added as a heavy composition, but a :code:`#ToDo` comment
should be added, explaining that this frontend implementation will be updated if/when
:code:`ivy.<func_name>` is implemented.

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
data type support in the native framework (which go beyond the data type limitations of
the framework itself) should be flagged
`as such <https://github.com/unifyai/ivy/blob/6089953297b438c58caa71c058ed1599f40a270c/ivy/functional/frontends/tensorflow/math.py#L88>`_
in a manner like the following:

.. code-block:: python
    
   logical_and.supported_dtypes = ("bool",)

The same logic applies to unsupported devices. Even if the wrapped Ivy function supports
more devices, we should still flag the frontend function devices to be the same as those
supported by the function in the native framework.

Instance Methods
----------------------

To allow for the Frontend API to be used with Instance Methods, we have created frontend 
framework specific classes which behave similiar to the Ivy Array class. These framework 
specific classes wrap any existing Ivy Array or Native Array and allow for them to be used
with framework specific instance methods. The example below highlights the difference between 
the functional method and its relevant instance method.

**Examples**

**tensorflow.add**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math.py
    def add(x, y, name=None):
    return ivy.add(x, y)


**tensorflow.add Instance Method**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/tensor.py
    def add(self, y, name="add"):
        return tf_frontend.add(self.data, y, name)

* As you can see, the instance method is very similar to the functional method, but it 
  takes the first argument as self. This is because the instance method is called 
  on an instance of the framework specific class, which wraps the Ivy Array or Native Array.
* We then return the relevant frontend function, passing in the wrapped Ivy Array or Native Array 
  as :code:`self.data`


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
