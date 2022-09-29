Data Types
==========

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`backend setting`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`infer_dtype`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L249
.. _`import time`: https://github.com/unifyai/ivy/blob/9c2eb725387152d721040d8638c8f898541a9da4/ivy/__init__.py#L225
.. _`ivy.Dtype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L1096
.. _`empty class`: https://github.com/unifyai/ivy/blob/9c2eb725387152d721040d8638c8f898541a9da4/ivy/__init__.py#L38
.. _`also specified`: https://github.com/unifyai/ivy/blob/9c2eb725387152d721040d8638c8f898541a9da4/ivy/__init__.py#L241
.. _`tuples`: https://github.com/unifyai/ivy/blob/9c2eb725387152d721040d8638c8f898541a9da4/ivy/__init__.py#L256
.. _`valid tuples`: https://github.com/unifyai/ivy/blob/9c2eb725387152d721040d8638c8f898541a9da4/ivy/__init__.py#L303
.. _`invalid tuples`: https://github.com/unifyai/ivy/blob/9c2eb725387152d721040d8638c8f898541a9da4/ivy/__init__.py#L309
.. _`data type class`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/functional/backends/torch/__init__.py#L14
.. _`true native data types`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/functional/backends/torch/__init__.py#L16
.. _`valid data types`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/functional/backends/torch/__init__.py#L29
.. _`invalid data types`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/functional/backends/torch/__init__.py#L56
.. _`original definition`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/__init__.py#L225
.. _`new definition`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/functional/backends/torch/__init__.py#L16
.. _`handled`: https://github.com/unifyai/ivy/blob/a594075390532d2796a6b649785b93532aee5c9a/ivy/backend_handler.py#L194
.. _`data_type.py`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py
.. _`ivy.can_cast`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L246
.. _`ivy.default_dtype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L879
.. _`ivy.set_default_dtype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L1555
.. _`data types discussion`: https://github.com/unifyai/ivy/discussions/1307
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`data types channel`: https://discord.com/channels/799879767196958751/982738078445760532


The data types supported by Ivy are as follows:

* int8
* int16
* int32
* int64
* uint8
* uint16
* uint32
* uint64
* bfloat16
* float16
* float32
* float64
* bool

These are all defined at `import time`_, with each of these set as an `ivy.Dtype`_ instance.
The :class:`ivy.Dtype` class derives from :class:`str`,
and has simple logic in the constructor to verify that the string formatting is correct.
All data types can be queried as attributes of the :mod:`ivy` namespace, such as ``ivy.float32`` etc.

In addition, *native* data types are `also specified`_ at import time.
Likewise, these are all *initially* set as `ivy.Dtype`_ instances.

There is also an :class:`ivy.NativeDtype` class defined, but this is initially set as an `empty class`_.

The following `tuples`_ are also defined: ``all_dtypes``, ``all_numeric_dtypes``, ``all_int_dtypes``,
``all_float_dtypes``. These each contain all possible data types which fall into the corresponding category.
Each of these tuples is also replicated in a new set of four `valid tuples`_
and a set of four `invalid tuples`_.
When no backend is set, all data types are assumed to be valid, and so the invalid tuples are all empty,
and the valid tuples are set as equal to the original four *"all"* tuples.

However, when a backend is set, then some of these are updated.
Firstly, the :class:`ivy.NativeDtype` is replaced with the backend-specific `data type class`_.
Secondly, each of the native data types are replaced with the `true native data types`_.
Thirdly, the `valid data types`_ are updated.
Finally, the `invalid data types`_ are updated.

This leaves each of the data types unmodified,
for example ``ivy.float32`` will still reference the  `original definition`_ in :mod:`ivy/ivy/__init__.py`,
whereas ``ivy.native_float32`` will now reference the `new definition`_ in
:mod:`/ivy/functional/backends/backend/__init__.py`.

The tuples ``all_dtypes``, ``all_numeric_dtypes``, ``all_int_dtypes`` and ``all_float_dtypes``
are also left unmodified.
Importantly, we must ensure that unsupported data types are removed from the :mod:`ivy` namespace.
For example, torch supports ``uint8``, but does not support ``uint16``, ``uint32`` or ``uint64``.
Therefore, after setting a torch backend via :code:`ivy.set_backend('torch')`,
we should no longer be able to access ``ivy.uint16``.
This is `handled`_ in :func:`ivy.set_backend`.

Data Type Module
----------------

The `data_type.py`_ module provides a variety of functions for working with data types.
A few examples include
:func:`ivy.astype` which copies an array to a specified data type,
:func:`ivy.broadcast_to` which broadcasts an array to a specified shape,
and :func:`ivy.result_type` which returns the dtype that results from applying the type promotion rules to the arguments.

Many functions in the :mod:`data_type.py` module are *convenience* functions,
which means that they do not directly modify arrays, as explained in the :ref:`Function Types` section.

For example, the following are all convenience functions:
`ivy.can_cast`_, which determines if one data type can be cast to another data type according to type-promotion rules,
`ivy.dtype`_, which gets the data type for the input array,
`ivy.set_default_dtype`_, which sets the global default data dtype,
and `ivy.default_dtype`_, which returns the correct data type to use.

`ivy.default_dtype`_ is arguably the most important function.
Any function in the functional API that receives a ``dtype`` argument will make use of this function,
as explained below.


Data Type Promotion
-------------------

In order to ensure that the same data type is always returned when operations are
performed on arrays with different data types, regardless of which backend framework is
set, Ivy has it's own set of data type promotion rules and corresponding  functions.
These rules build directly on top of the
`rules <https://data-apis.org/array-api/latest/API_specification/type_promotion.html>`_
outlined in the `Array API Standard`_.

The rules are simple: all data type promotions in Ivy should adhere to this
`promotion table <https://github.com/unifyai/ivy/blob/db96e50860802b2944ed9dabacd8198608699c7c/ivy/__init__.py#L366>`_,
which is the union of the Array API Standard
`promotion table <https://github.com/unifyai/ivy/blob/db96e50860802b2944ed9dabacd8198608699c7c/ivy/__init__.py#L223>`_
and an extra
`promotion table <https://github.com/unifyai/ivy/blob/db96e50860802b2944ed9dabacd8198608699c7c/ivy/__init__.py#L292>`_.

In order to ensure adherence to this promotion table, many backend functions make use of
the functions
`ivy.promote_types <https://github.com/unifyai/ivy/blob/db96e50860802b2944ed9dabacd8198608699c7c/ivy/functional/ivy/data_type.py#L1804>`_,
`ivy.type_promote_arrays <https://github.com/unifyai/ivy/blob/db96e50860802b2944ed9dabacd8198608699c7c/ivy/functional/ivy/data_type.py#L1940>`_,
or
`ivy.promote_types_of_inputs <https://github.com/unifyai/ivy/blob/db96e50860802b2944ed9dabacd8198608699c7c/ivy/functional/ivy/data_type.py#L2085>`_.
These functions: promote data types in the inputs and return the new data types,
promote the data types of the arrays in the input and return new arrays,
and promote the data types of the numeric or array values inputs and
return new type promoted values, respectively.

For an example of how some of these functions are used,
the implementations for :func:`ivy.add` in each backend framework are as follows:

# JAX

.. code-block:: python

    def add(
        x1: Union[float, JaxArray],
        x2: Union[float, JaxArray],
        /,
        *,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
        return jnp.add(x1, x2)

# NumPy

.. code-block:: python

    @_handle_0_dim_output
    def add(
        x1: Union[float, np.ndarray],
        x2: Union[float, np.ndarray],
        /,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
        return np.add(x1, x2, out=out)

# TensorFlow

.. code-block:: python

    def add(
        x1: Union[float, tf.Tensor, tf.Variable],
        x2: Union[float, tf.Tensor, tf.Variable],
        /,
        *,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
        return tf.experimental.numpy.add(x1, x2)

# PyTorch

.. code-block:: python

    def add(
        x1: Union[float, torch.Tensor],
        x2: Union[float, torch.Tensor],
        /,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
        return torch.add(x1, x2, out=out)

It's important to always make use of the Ivy promotion functions as opposed to
backend-specific promotion functions such as :func:`jax.numpy.promote_types`,
:func:`numpy.promote_types`, :func:`tf.experimental.numpy.promote_types` and
:func:`torch.promote_types`, as these will generally have promotion rules which will
subtly differ from one another and from Ivy's unified promotion rules.

Arguments in other Functions
----------------------------

All ``dtype`` arguments are keyword-only.
All creation functions include the ``dtype`` argument, for specifying the data type of the created array.
Some other non-creation functions also support the ``dtype`` argument,
such as :func:`ivy.prod` and :func:`ivy.sum`, but most functions do not include it.
The non-creation functions which do support it are generally functions that involve a compounding reduction across the
array, which could result in overflows, and so an explicit ``dtype`` argument is useful to handling such cases.

The ``dtype`` argument is handled in the `infer_dtype`_ wrapper, for all functions which have the decorator
:code:`@infer_dtype`.
This function calls `ivy.default_dtype`_ in order to determine the correct data type.
As discussed in the :ref:`Function Wrapping` section,
this is applied to all applicable functions dynamically during `backend setting`_.

Overall, `ivy.default_dtype`_ infers the data type as follows:

#. if the ``dtype`` argument is provided, use this directly
#. otherwise, if an array is present in the arguments, set ``arr`` to this array. \
   This will then be used to infer the data type by calling :func:`ivy.dtype` on the array
#. otherwise, if a *relevant* scalar is present in the arguments, set ``arr`` to this scalar \
   and derive the data type from this by calling either :func:`ivy.default_int_dtype` or \
   :func:`ivy.default_float_dtype` depending on whether the scalar is an int or float. \
   This will either return the globally set default int data type or globally set default float data type \
   (settable via :func:`ivy.set_default_int_dtype` and :func:`ivy.set_default_float_dtype` respectively). \
   An example of a *relevant* scalar is ``start`` in the function :func:`ivy.arange`, \
   which is used to set the starting value of the returned array. \
   Examples of *irrelevant* scalars which should **not** be used for determining the data type are ``axis``, \
   ``axes``, ``dims`` etc. which must be integers, and control other configurations of the function \
   being called, with no bearing at all on the data types used by that function.
#. otherwise, if no arrays or relevant scalars are present in the arguments, \
   then use the global default data type, which can either be an int or float data type. \
   This is settable via :func:`ivy.set_default_dtype`.

For the majority of functions which defer to `infer_dtype`_ for handling the data type,
these steps will have been followed and the ``dtype`` argument will be populated with the correct value
before the backend-specific implementation is even entered into. Therefore, whereas the ``dtype`` argument is
listed as optional in the ivy API at :mod:`ivy/functional/ivy/category_name.py`,
the argument is listed as required in the backend-specific implementations at
:mod:`ivy/functional/backends/backend_name/category_name.py`.

Let's take a look at the function :func:`ivy.zeros` as an example.

The implementation in :mod:`ivy/functional/ivy/creation.py` has the following signature:

.. code-block:: python

    @outputs_to_ivy_arrays
    @handle_out_argument
    @infer_dtype
    @infer_device
    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Array:

Whereas the backend-specific implementations in :mod:`ivy/functional/backends/backend_name/statistical.py`
all list ``dtype`` as required.

Jax:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
    ) -> JaxArray:

NumPy:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: np.dtype,
        device: str,
    ) -> np.ndarray:

TensorFlow:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: tf.DType,
        device: str,
    ) -> Union[tf.Tensor, tf.Variable]:

PyTorch:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:

This makes it clear that these backend-specific functions are only entered into once the correct ``dtype``
has been determined.

However, the ``dtype`` argument for functions which don't have the :code:`@infer_dtype` decorator
are **not** handled by `infer_dtype`_,
and so these defaults must be handled by the backend-specific implementations themselves.

One reason for not adding :code:`@infer_dtype` to a function is because it includes *relevant* scalar arguments
for inferring the data type from. `infer_dtype`_ is not able to correctly handle such cases,
and so the dtype handling is delegated to the backend-specific implementations.

For example :func:`ivy.full` doesn't have the :code:`@infer_dtype` decorator even though it has a ``dtype`` argument
because of the *relevant* ``fill_value`` which cannot be correctly handled by `infer_dtype`_.

The PyTorch-specific implementation is as follows:

.. code-block:: python

    def full(
        shape: Union[int, Sequence[int]],
        fill_value: Union[int, float],
        *,
        dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
        device: torch.device,
    ) -> Tensor:
        return torch.full(
            shape_to_tuple(shape),
            fill_value,
            dtype=ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True),
            device=device,
        )

The implementations for all other backends follow a similar pattern to this PyTorch implementation,
where the ``dtype`` argument is optional and :func:`ivy.default_dtype` is called inside the
backend-specific implementation.

Supported and Unsupported Data Types
------------------------------------

Some backend functions (implemented in :mod:`ivy/functional/backends/<some_backend>`)
have attributes named :attr:`supported_dtypes` or :attr:`unsupported_dtypes`,
which flag the data types which this particular function does and does not support
respectively for the associated backend.
Only one of these attributes can be specified for any given function.
In the case of :attr:`supported_dtypes` it is assumed that all unmentioned data types
are unsupported, and in the case of :attr:`unsupported_dtypes` it is assumed that all
unmentioned data types are supported.

These attributes should always be in tuple form, with each entry in the tuple
being a string, like so:

.. code-block:: python

    def expm1(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.expm1(x, out=out)

    expm1.unsupported_dtypes = ("float16",)

For compositional functions, the supported and unsupported data types can then be
inferred automatically using the helper functions
`function_supported_dtypes <https://github.com/unifyai/ivy/blob/9e71fc2b589bf8f6b7a0762602723ac084bb5d9e/ivy/functional/ivy/data_type.py#L1370>`_
and
`function_unsupported_dtypes <https://github.com/unifyai/ivy/blob/9e71fc2b589bf8f6b7a0762602723ac084bb5d9e/ivy/functional/ivy/data_type.py#L1407>`_
respectively, which traverse the abstract syntax tree of the compositional function and
evaluate the relevant attributes for each primary function in the composition.
The same approach applies for most stateful methods, which are themselves compositional.

It should be noted that the :attr:`unsupported_dtypes` is different from
``ivy.invalid_dtypes`` which consists of all the data types that every function
of that particular backend does not support, and so if a certain ``dtype`` is
already present in the ``ivy.invalid_dtypes`` then we should not add it into the
:attr:`unsupported_dtypes` attribute.

Sometimes, it might be possible to support a natively unsupported data type by either
casting to a supported data type and then casting back, or explicitly handling these
data types without deferring to a backend function at all.

An example of the former is :func:`ivy.logical_not` with a tensorflow backend:

.. code-block:: python

    def logical_not(
        x: Union[tf.Tensor, tf.Variable],
        /,
        *,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        return tf.logical_not(tf.cast(x, tf.bool))

An example of the latter is :func:`ivy.abs` with a tensorflow backend:

.. code-block:: python

    def abs(
        x: Union[float, tf.Tensor, tf.Variable],
        /,
        *,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        if "uint" in ivy.dtype(x):
            return x
        else:
            return tf.abs(x)

In some cases, the lack of support for a particular data type by the backend function
might be more difficult to handle correctly. For example, in many cases casting to
another data type will result in a loss of precision, input range, or both.
In such cases, the best solution is to simply add the data type to the
:attr:`unsupported_dtypes` attribute,
rather than trying to implement a long and complex patch to achieve the desired
behaviour.

Some cases where a data type is not supported are very subtle. For example,
``uint8`` is not supported for :func:`ivy.prod` with a torch backend,
despite :func:`torch.prod` handling ``torch.uint8`` types in the input totally fine.

The reason for this is that the `Array API Standard`_ mandates that :func:`prod` upcasts
the unsigned integer return to have the same number of bits as the
default integer data type. By default, the default integer data type in Ivy is
``int32``, and so we should return an array of type ``uint32`` despite the input
arrays being of type ``uint8``. However, torch does not support ``uint32``,
and so we cannot fully adhere to the requirements of the standard for ``uint8``
inputs. Rather than breaking this rule and returning arrays of type ``uint8`` only
with a torch backend, we instead opt to remove official support entirely for
this combination of data type, function and backend framework.
This will avoid all of the potential confusion that could arise if we were to have
inconsistent and unexpected outputs when using officially supported data types in Ivy.


Backend Data Type Bugs
----------------------

In some cases, the lack of support might just be a bug which will likely be resolved in
a future release of the framework. In these cases, as well as adding to the
:attr:`unsupported_dtypes` attribute, we should also add a :code:`#ToDo` comment
in the implementation, explaining that the support of the data type will be added as
soon as the bug is fixed, with a link to an associated open issue in the framework
repos included in the comment.

For example, the following code throws an error when ``dtype`` is
``torch.int32`` but not when it is ``torch.int64``.
This is tested with torch version ``1.12.1``,
which is the latest stable release at the time of writing. This is a
`known bug <https://github.com/pytorch/pytorch/issues/84530>`_:

.. code-block:: python

    dtype = torch.int32  # or torch.int64
    x = torch.randint(1, 10, ([1, 2, 3]), dtype=dtype)
    torch.tensordot(x, x, dims=([0], [0]))

Despite ``torch.int32`` working correctly with :func:`torch.tensordot` in the vast
majority of cases, our solution is to still add :code:`"int32"` into the
:attr:`unsupported_dtypes` attribute, which will prevent the unit tests from failing in the CI.
We also add the following comment above the :attr:`unsupported_dtypes` attribute:

.. code-block:: python

    # ToDo: re-add int32 support once (https://github.com/pytorch/pytorch/issues/84530)
    #  is fixed.
    tensordot.unsupported_dtypes = ("int32",)

Similarly, the following code throws an error for torch version ``1.11.0``
but not ``1.12.1``.

.. code-block:: python

    x = torch.tensor([0], dtype=torch.float32)
    torch.cumsum(x, axis=0, dtype=torch.bfloat16)

Writing short-lived patches for these temporary issues would add unwarranted complexity
to the backend implementations, and introduce the risk of forgetting about the patch,
needlessly bloating the codebase with redundant code.
In such cases, we can explicitly flag which versions support which data types like so:

[ToDo add a code example]

The slight downside of this approach is that there is less data type coverage for each
version of each backend, but taking responsibility for patching this support for all
versions would substantially inflate the implementational requirements for ivy, and so
we have decided to opt out of this responsibility!

Superset Data Type Support
--------------------------

As explained in the superset section of the Deep Dive, we generally go for the superset
of behaviour for all Ivy functions, and data type support is no exception.
Some backends like tensorflow do not support integer array inputs for certain
functions. For example :func:`tensorflow.cos` only supports non-integer values.
However, backends like torch and JAX support integer arrays as inputs.
To ensure that integer types are supported in Ivy when a tensorflow backend is set,
we simply promote any integer array passed to the function to the default float dtype.
As with all superset design decisions, this behavior makes it much easier to support all
frameworks in our frontends, without the need for lots of extra logic for handling
integer array inputs for the frameworks which support it natively.

**Round Up**

This should have hopefully given you a good feel for data types, and how these are handled in Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `data types discussion`_,
or reach out on `discord`_ in the `data types channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/2qOBzQdLXn4" class="video">
    </iframe>