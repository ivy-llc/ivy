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
.. _`open task`: https://lets-unify.ai/ivy/contributing/4_open_tasks.html#open-tasks

Introduction
------------

On top of the Ivy functional API and backend functional APIs, Ivy has another set of
framework-specific frontend functional APIs, which play an important role in code
transpilations, as explained `here`_.

Let's start with some examples to have a better idea on Ivy Frontends!

The Basics
----------

**NOTE:** Type hints, docstrings and examples are not required when working on
frontend functions.

There will be some implicit discussion of the locations of frontend functions in these examples, however an explicit
explanation of how to place a frontend function can be found in a sub-section of the Frontend APIs `open_task`_.

**Jax**

JAX has two distinct groups of functions, those in the :code:`jax.lax` namespace and
those in the :code:`jax.numpy` namespace. The former set of functions map very closely
to the API for the Accelerated Linear Algebra (`XLA <https://www.tensorflow.org/xla>`_)
compiler, which is used under the hood to run high performance JAX code. The latter set
of functions map very closely to NumPy's well known API. In general, all functions in
the :code:`jax.numpy` namespace are themselves implemented as a composition of the
lower-level functions in the :code:`jax.lax` namespace.

When transpiling between frameworks, the first step is to compile the computation graph
into low level python functions for the source framework using Ivy's graph
compiler, before then replacing these nodes with the associated functions in Ivy's
frontend API. Given that all jax code can be decomposed into :code:`jax.lax`
function calls, when transpiling :code:`jax` code it should always be possible to
express the computation graph as a composition of only :code:`jax.lax` functions.
Therefore, arguably these are the *only* functions we should need to implement in the
JAX frontend. However, in general we wish to be able to compile a graph in the backend
framework with varying levels of dynamicism. A graph of only :code:`jax.lax` functions
chained together in general is more *static* and less *dynamic* than a graph which
chains :code:`jax.numpy` functions together. We wish to enable varying extents of
dynamicism when compiling a graph with our graph compiler, and therefore we also
implement the functions in the :code:`jax.numpy` namespace in our frontend API for JAX.

Thus, both :code:`lax` and :code:`numpy` modules are created in the JAX frontend API.
We start with the function :code:`lax.add` as an example.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def add(x, y):
        return ivy.add(x, y)

:code:`lax.add` is categorised under :code:`operators` as shown in the `jax.lax`_
package directory. We organize the functions using the same categorizations as the
original framework, and also mimic the importing behaviour regarding modules and
namespaces etc.

For the function arguments, these must be identical to the original function in
Jax. In this case, `jax.lax.add`_ has two arguments,
and so we will also have the same two arguments in our Jax frontend :code:`lax.add`.
In this case, the function will then simply return :code:`ivy.add`,
which in turn will link to the backend-specific implementation :code:`ivy.add`
according to the framework set in the backend.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def tan(x):
        return ivy.tan(x)

Using :code:`lax.tan` as a second example, we can see that this is placed under
:code:`operators`, again in the `jax.lax`_ directory.
By referring to the `jax.lax.tan`_ documentation, we can see that it has only one
argument. In the same manner as our :code:`add` function, we simply link its return
to :code:`ivy.tan`, and again the computation then depends on the backend framework.

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
Similarly, :code:`subok` controls whether or not subclasses of the :code:`numpy.ndarray`
should be permitted as inputs to the function.
Again, this is a very framework-specific argument. All ivy functions by default do
enable subclasses of the :code:`ivy.Array` to be passed, and the frontend function will
be operating with :code:`ivy.Array` instances rather than :code:`numpy.ndarray`
instances, and so we omit this argument. Again, it has no bearing on input-output
behaviour and so this is not a problem when transpiling between frameworks.

See the section "Unused Arguments" below for more details.

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
Thus, we can mimic the original behaviour by simply passing :code:`other * alpha`
into :code:`ivy.add`.

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def tan(input, *, out=None):
        return ivy.tan(input, out=out)

:code:`tan` is also placed under :code:`pointwise_ops` as is the case in the `torch`_
framework. Looking at the `torch.tan`_ documentation, we can mimic the same arguments,
and again simply wrap :code:`ivy.tan`,
also making use of the :code:`out` argument in this case.

Unused Arguments
----------------

As can be seen from the examples above, there are often cases where we do not add
support for particular arguments in the frontend function. Generally, we can omit
support for a particular argument only if: the argument **does not** fundamentally
affect the input-output behaviour of the function in a mathematical sense. The only
two exceptions to this rule are arguments related to either the data type or the device
on which the returned array(s) should reside. Examples of arguments which can be
omitted, on account that they do not change the mathematics of the function are
arguments which relate to:

* the layout of the array in memory, such as :code:`order` in
  `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_.

* the algorithm or approximations used under the hood, such as :code:`precision` and
  :code:`preferred_element_type` in
  `jax.lax.conv_general_dilated <https://github.com/google/jax/blob/1338864c1fcb661cbe4084919d50fb160a03570e/jax/_src/lax/convolution.py#L57>`_.

* the specific array class in the original framework, such as :code:`subok` in
  `numpy.add <https://numpy.org/doc/1.23/reference/generated/numpy.add.html>`_.

* the labelling of functions for organizational purposes, such as :code:`name` in
  `tf.math.add <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/python/ops/math_ops.py#L3926-L4004>`_.

There are likely to be many other examples of arguments which do not fundamentally
affect the input-output behaviour of the function in a mathematical sense, and so can
also be omitted from Ivy's frontend implementation.

The reason we omit these arguments in Ivy is because Ivy is not designed to provide
low-level control to functions that extend beyond the pure mathematics of the function.
This is a requirement because Ivy abstracts the backend framework,
and therefore also abstracts everything below the backend framework's functional API,
including the backend array class, the low-level language compiled to, the device etc.
Most ML frameworks do not offer per-array control of the memory layout, and control for
the finer details of the algorithmic approximations under the hood, and so we cannot
in general offer this level of control at the Ivy API level, nor the frontend API level
as a direct result. As explained above, this is not a problem, as the memory layout has
no bearing at all on the input-output behaviour of the function. In contrast, the
algorithmic approximation may have a marginal bearing on the final results in some
cases, but Ivy is only designed to unify to within a reasonable numeric approximation
in any case, and so omitting these arguments also very much fits within Ivy's design.


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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return current_backend(x).cumprod(x, axis, exclusive, out=out)

To enable this behaviour, we need to incorporate other Ivy functions which are
compositionally able to mimic the required behaviour.
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

Missing Ivy Functions
---------------------

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
in a unique PR. Once merged, you can then resume working on the frontend function,
which will now be a much easier task with the new addition to Ivy.

Temporary Compositions
----------------------

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
`globally flagged <https://github.com/unifyai/ivy/blob/6eb2cadf04f06aace9118804100b0928dc71320c/ivy/functional/backends/numpy/__init__.py#L21>`_.

Classes and Instance Methods
----------------------------

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


Framework-Specific Argument Types
---------------------------------

Some of the frontend functions that we need to implement include framework-specific
classes as the default values for some of the arguments,
which do not have a counterpart in other frameworks.
When re-implementing these functions in Ivy's frontend, we would like to still include
those arguments without directly using these special classes, which do not exist in Ivy.

A good example is the special class :code:`numpy._NoValue`, which is sometimes used
instead of :code:`None` as the default value for arguments in numpy. For example,
the :code:`keepdims`, :code:`initial` and :code:`where` arguments of :code:`numpy.sum`
use :code:`numpy._NoValue` as the default value, while :code:`axis`, :code:`dtype` and
:code:`out` use :code:`None`, as can be seen in the
`source code <https://github.com/numpy/numpy/blob/v1.23.0/numpy/core/fromnumeric.py#L2162-L2299>`_.

We now introduce how to deal with such framework-specific classes. For each backend
framework, there is a dictionary named `<backend>_classes_to_ivy_classes` in
`ivy/ivy_tests/test_ivy/test_frontends/test_<backend>/__init__.py`.
This holds pairs of framework-specific classes and the corresponding Ivy or
native Python classes to map to.
For example, in `ivy/ivy_tests/test_ivy/test_frontends/test_numpy/__init__.py`, we have:

.. code-block:: python

    numpy_classes_to_ivy_classes = {np._NoValue: None}

Where :code:`np._NoValue` is a reference to the :code:`_NoValueType` class defined in
:code:`numpy/numpy/_globals.py`.

Any time a new framework-specific data type is discovered, such as the
:code:`numpy._NoValue` example given, then this should be added as a key to the
dictionary, and the most appropriate pure-python or Ivy class or instance should be
added as the value.

During frontend testing, the helper :code:`test_frontend_function` by default passes
all the generated inputs into both Ivy's frontend implementation and also the original
function. For the framework-specific classes discussed, this is a problem.
Handling the framework-specific class in the Ivy frontend would add a dependency to the
frontend framework being mimicked. This breaks Ivy's design philosophy,
whereby only the specific backend framework being used should be a dependency.
Our solution is to pass the value from the :code:`<framework>_classes_to_ivy_classes`
dict to the Ivy frontend function and the key from the
:code:`<framework>_classes_to_ivy_classes` dict to the original function during testing
in :code:`test_frontend_function`.

The way we do this is to wrap all framework-specific classes inside a
:code:`NativeClass` during frontend testing. The :code:`NativeClass` is defined in
:code:`ivy/ivy_tests/test_ivy/test_frontends/__init__.py`, and this acts as a
placeholder class to represent the framework-specific class and its counterpart.
It has only one attribute, :code:`_native_class`, which holds the reference to the
special class being used by the targeted framework.
Then, in order to pass the key and value to the original and frontend functions
respectively, :code:`test_frontend_function` detects all :code:`NativeClass` instances
in the arguments, makes use of :code:`<framework>_classes_to_ivy_classes` internally
to find the corresponding value to the key wrapped inside the :code:`NativeClass`
instance, and then passes the key and value as inputs to the corresponding functions
correctly.


As an example, we show how :code:`NativeClass` is used in the frontend test for the
:code:`sum` function in the NumPy frontend:

.. code-block:: python
    # sum
    Novalue = NativeClass(numpy._NoValue)
    @handle_cmd_line_args
    @given(
        dtype_x_axis=_dtype_x_axis(available_dtypes=helpers.get_dtypes("float")),
        dtype=helpers.get_dtypes("float", full=False, none=True),
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

The function has three optional arguments which have the default value of
:code:`numpy._NoValue`, being: :code:`where`, :code:`keep_dims` and :code:`initial`.
We therefore define a :code:`NativeClass` object :code:`Novalue`, and pass this as input
to each of these arguments when calling :code:`test_frontend_function`.

**Round Up**

This should hopefully have given you a better grasp on the what the Ivy Frontend APIs
are for, how they should be implemented, and the things to watch out for!
We also have a short `YouTube tutorial series`_ on this
as well if you prefer a video explanation!

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy frontends discussion`_,
or reach out on `discord`_ in the `ivy frontends channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/_9KeK-idaFs" class="video">
    </iframe>
