Function Types 🧮
=================

.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L412
.. _`backend setting`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`handle_nestable`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L370
.. _`at import time`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/__init__.py#L114
.. _`add_ivy_array_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/array/wrapping.py#L26
.. _`add_ivy_container_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L69
.. _`from being added`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L78
.. _`_function_w_arrays_n_out_handled`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L166
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`ivy.set_backend`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/backend_handler.py#L153
.. _`ivy.get_backend`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/backend_handler.py#L211
.. _`ivy.nested_map`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/nest.py#L618
.. _`ivy.index_nest`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/nest.py#L15
.. _`ivy.set_default_dtype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L1555
.. _`ivy.set_default_device`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/device.py#L464
.. _`submodules`: https://github.com/unifyai/ivy/tree/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy
.. _`nest.py`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/nest.py
.. _`ivy.default`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L622
.. _`ivy.cache_fn`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L747
.. _`ivy.stable_divide`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L928
.. _`ivy.can_cast`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L246
.. _`ivy.dtype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L1096
.. _`ivy.dev`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py#L325
.. _`ivy.default_dtype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L879
.. _`ivy.get_all_arrays_on_dev`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py#L131
.. _`function types discussion`: https://github.com/unifyai/ivy/discussions/1312
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`function types channel`: https://discord.com/channels/799879767196958751/982737839861145630

Firstly, we explain the difference between *primary*, *compositional*, *mixed* and *standalone* functions.
These four function categorizations are all **mutually exclusive**,
and combined they constitute the set of **all** functions in Ivy, as outlined in the simple Venn diagram below.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/1_function_types/four_function_types.png?raw=true
   :align: center
   :width: 50%

Primary Functions
-----------------

*Primary* functions are essentially the lowest level building blocks in Ivy. Each primary function has a unique
backend-specific implementation for each backend specified in
:mod:`ivy/functional/backends/backend_name/category_name.py`. These are generally implemented as light wrapping
around an existing function in the backend framework, which serves a near-identical purpose.

Primary functions must both be specified in :mod:`ivy/functional/ivy/category_name.py` and also in each of
the backend files :mod:`ivy/functional/backends/backend_name/category_name.py`

The function in :mod:`ivy/functional/ivy/category_name.py` includes the type hints, docstring and docstring examples
(explained in more detail in the subsequent sections), but does not include an actual implementation.

Instead, in :mod:`ivy/functional/ivy/category_name.py`, primary functions simply defer to the backend-specific
implementation.

For example, the code for :func:`ivy.tan` in :mod:`ivy/functional/ivy/elementwise.py`
(with docstrings removed) is given below:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.current_backend(x).tan(x, out=out)

The backend-specific implementation of :func:`ivy.tan`  for PyTorch in
:mod:`ivy/functional/backends/torch/elementwise.py` is given below:

.. code-block:: python

    def tan(
        x: torch.Tensor,
        /,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.tan(x, out=out)

The reason that the Ivy implementation has type hint :code:`Union[ivy.Array, ivy.NativeArray]` but PyTorch
implementation has :class:`torch.Tensor` is explained in the :ref:`Arrays` section.
Likewise, the reason that the :code:`out` argument in the Ivy implementation has array type hint :class:`ivy.Array`
whereas :code:`x` has :code:`Union[ivy.Array, ivy.NativeArray]` is also explained in the :ref:`Arrays` section.

Compositional Functions
-----------------------

*Compositional* functions on the other hand **do not** have backend-specific implementations. They are implemented as
a *composition* of other Ivy functions,
which themselves can be either compositional, primary or mixed (explained below).

Therefore, compositional functions are only implemented in :mod:`ivy/functional/ivy/category_name.py`, and there are no
implementations in any of the backend files :mod:`ivy/functional/backends/backend_name/category_name.py`

For example, the implementation of :func:`ivy.cross_entropy` in :mod:`ivy/functional/ivy/losses.py`
(with docstrings removed) is given below:

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        epsilon: float = 1e-7,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return ivy.negative(ivy.sum(log_pred * true, axis), out=out)


Mixed Functions
---------------

Some functions have some backend-specific implementations in
:mod:`ivy/functional/backends/backend_name/category_name.py`, but not for all backends.
To support backends that do not have a backend-specific implementation,
a compositional implementation is also provided in :mod:`ivy/functional/ivy/category_name.py`.
Because these functions include both a compositional implementation and also at least one backend-specific
implementation, these functions are referred to as *mixed*.

When using ivy without a backend set explicitly (for example :func:`ivy.set_backend` has not been called),
then the function called is always the one implemented in :mod:`ivy/functional/ivy/category_name.py`.
For *primary* functions, then :code:`ivy.current_backend(array_arg).func_name(...)`
will call the backend-specific implementation in :mod:`ivy/functional/backends/backend_name/category_name.py`
directly. However, as just explained, *mixed* functions implement a compositional approach in
:mod:`ivy/functional/ivy/category_name.py`, without deferring to the backend.
Therefore, when no backend is explicitly set,
then the compositional implementation is always used for *mixed* functions,
even for backends that have a more efficient backend-specific implementation.
Typically the backend should always be set explicitly though (using :func:`ivy.set_backend` for example),
and in this case the efficient backend-specific implementation will always be used if it exists.

Standalone Functions
---------------------

*Standalone* functions are functions which do not reference any other *primary*,
*compositional* or *mixed* functions whatsoever.

By definition, standalone functions can only reference themselves or other standalone functions.
Most commonly, these functions are *convenience* functions (see below).

As a first example, every function in the `nest.py`_ module is a standalone function.
All of these either: (a) reference no other function at all, (b) only reference themselves recursively,
or (c) reference other standalone functions.

A few other examples outside of the :mod:`nest.py` module are:
`ivy.default`_ which simply returns :code:`x` if it exists else the default value,
`ivy.cache_fn`_ which wraps a function such that when :code:`cache=True` is passed,
then a previously cached output is returned,
and `ivy.stable_divide`_ which simply adds a small constant to the denominator of the division.

Nestable Functions
------------------

*Nestable* functions are functions which can accept :class:`ivy.Container` instances in place
of **any** of the arguments. Multiple containers can also be passed in for multiple arguments at the same time,
provided that the containers share a common nested structure.
If an :class:`ivy.Container` is passed, then the function is applied to all of the
leaves of the container, with the container leaf values passed into the function at the corresponding arguments.
In this case, the function will return an :class:`ivy.Container` in the output.
*Primary*, *compositional*, *mixed*, and *standalone* functions can all *also* be nestable.
This categorization is **not** mutually exclusive, as outlined by the Venn diagram below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/1_function_types/nestable.png?raw=true
   :align: center
   :width: 50%

The *nestable* property makes it very easy to write a single piece of code that can deal either with individual
arguments or arbitrary batches of nested arguments. This is very useful in machine learning,
where batches of different training data often need to be processed concurrently.
Another example is when the same operation must be performed on each weight in a network.
This *nestable* property of Ivy functions means that the same function can be used for any of these use cases
without modification.

This added support for handling :class:`ivy.Container` instances is all handled automatically when `_wrap_function`_
is applied to every function in the :code:`ivy` module during `backend setting`_. This will add the `handle_nestable`_
wrapping to the function if it has the :code:`@handle_nestable` decorator.
This function wrapping process is covered in a bit more detail in the :ref:`Function Wrapping` section.

Under the hood, the :class:`ivy.Container` API static methods are called when :class:`ivy.Container` instances are passed
in as inputs to functions in the functional API.

Nestable functions are explained in more detail in the :ref:`Containers` section.

Convenience Functions
---------------------

A final group of functions are the *convenience* functions (briefly mentioned above).
Convenience functions do not form part of the computation graph directly, and they do not directly modify arrays.
However, they can be used to organize and improve the code for other functions which do modify the arrays.
Convenience functions can be *primary*, *compositional*, *mixed* or *standalone* functions. Many are also *nestable*.
This is another categorization which is **not** mutually exclusive, as outlined by the Venn diagram below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/1_function_types/convenience.png?raw=true
   :align: center
   :width: 50%

Primary convenience functions include:
`ivy.can_cast`_ which determines if one data type can be cast to another data type according to type-promotion rules,
`ivy.dtype`_ which gets the data type for the input array,
and `ivy.dev`_ which gets the device for the input array.

Compositional convenience functions include:
`ivy.set_default_dtype`_ which sets the global default data dtype,
`ivy.default_dtype`_ which returns the correct data type to use,
considering both the inputs and also the globally set default,
and `ivy.get_all_arrays_on_dev`_ which gets all arrays which are currently on the specified device.

Standalone convenience functions include:
`ivy.get_backend`_ which returns a local Ivy module with the associated backend framework.
`ivy.nested_map`_ which enables an arbitrary function to be mapped across the leaves of an arbitrary nest,
and `ivy.index_nest`_ which enables an arbitrary nest to be recursively indexed.

There are many other examples. The convenience functions are not grouped by file or folder.
Feel free to have a look through all of the `submodules`_, you should be able to spot quite a few!

**Round Up**

This should have hopefully given you a good feel for the different function types.

If you're ever unsure of how best to proceed,
please feel free to engage with the `function types discussion`_,
or reach out on `discord`_ in the `function types channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/mWYhQRu1Vuk" class="video">
    </iframe>