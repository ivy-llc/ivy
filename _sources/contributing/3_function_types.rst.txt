Function Types
==============

.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L137
.. _`backend setting`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/framework_handler.py#L205
.. _`at import time`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/__init__.py#L114
.. _`add_ivy_array_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/array/wrapping.py#L26
.. _`add_ivy_container_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L69
.. _`from being added`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L78
.. _`_function_w_arrays_n_out_handled`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L166
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`ivy.set_backend`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/backend_handler.py#L153
.. _`ivy.get_backend`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/backend_handler.py#L211
.. _`ivy.nested_map`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/nest.py#L333
.. _`ivy.index_nest`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/nest.py#L15
.. _`ivy.set_default_dtype`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/data_type.py#L536
.. _`ivy.set_default_device`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/device.py#L464
.. _`submodules`: https://github.com/unifyai/ivy/tree/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy
.. _`nest.py`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/nest.py
.. _`ivy.default`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L622
.. _`ivy.cache_fn`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L622
.. _`ivy.stable_divide`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L928


Primary Functions
-----------------

*Primary* functions are essentially the lowest level building blocks in Ivy. Each primary function has a unique
backend-specific implementation for each backend specified in
:code:`ivy/functional/backends/backend_name/category_name.py`. These are generally implemented as light wrapping
around an existing function in the backend framework, which serves a near-identical purpose.

Primary functions must both be specified in :code:`ivy/functional/ivy/category_name.py` and also in each of
the backend files :code:`ivy/functional/backends/backend_name/category_name.py`

The function in :code:`ivy/functional/ivy/category_name.py` includes the type hints, docstring and docstring examples
(explained in more detail in the subsequent sections), but does not include an actual implementation.

Instead, in :code:`ivy/functional/ivy/category_name.py`, primary functions simply defer to the backend-specific
implementation.

For example, the code for :code:`ivy.tan` in :code:`ivy/functional/ivy/elementwise.py`
(with docstrings removed) is given below:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.current_backend(x).tan(x, out)

The backend-specific implementation of :code:`ivy.tan`  for PyTorch in
:code:`ivy/functional/backends/torch/elementwise.py` is given below:

.. code-block:: python

    def tan(
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.tan(x, out=out)

The reason that the Ivy implementation has type hint :code:`Union[ivy.Array, ivy.NativeArray]` but PyTorch
implementation has :code:`torch.Tensor` is explained in the :ref:`Native Arrays` section.
Likewise, the reason that the :code:`out` argument in the Ivy implementation has array type hint :code:`ivy.Array`
whereas :code:`x` has :code:`Union[ivy.Array, ivy.NativeArray]` is also explained in the :ref:`Native Arrays` section.

Compositional Functions
-----------------------

*Compositional* functions on the other hand **do not** have backend-specific implementations. They are implemented as
a *composition* of other Ivy functions, which themselves can be either compositional or primary.

Therefore, compositional functions are only implemented in :code:`ivy/functional/ivy/category_name.py`, and there are no
implementations in any of the backend files :code:`ivy/functional/backends/backend_name/category_name.py`

For example, the implementation of :code:`ivy.cross_entropy` in :code:`ivy/functional/ivy/losses.py`
(with docstrings removed) is given below:

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return ivy.negative(ivy.sum(log_pred * true, axis), out=out)


Mixed Functions
---------------

Some functions have some backend-specific implementations in
:code:`ivy/functional/backends/backend_name/category_name.py`, but not for all backends.
To support backends that do not have a backend-specific implementation,
a compositional implementation is also provided in :code:`ivy/functional/ivy/category_name.py`.
Because these functions include both a compositional implementation and also at least one backend-specific
implementation, these functions are refered to as *mixed*.

When using ivy without a backend set explicitly (for example :code:`ivy.set_framework()` has not been called),
then the function called is always the one implemented in :code:`ivy/functional/ivy/category_name.py`.
For *primary* functions, then :code:`ivy.current_backend(array_arg).func_name(...)`
will call the backend-specific implementation in :code:`ivy/functional/backends/backend_name/category_name.py`
directly. However, as just explained, *mixed* functions implement a compositional approach in
:code:`ivy/functional/ivy/category_name.py`, without deferring to the backend.
Therefore, when no backend is explicitly set,
then the compositional implementation is always used for *mixed* functions,
even for backends that have a more efficient backend-specific implementation.
Typically the backend should always be set explicitly though (using :code:`ivy.set_framework()` for example),
and in this case the efficient backend-specific implementation will always be used if it exists.

Standalone Functions
---------------------

*Standalone* functions are functions which do not reference any other *primary*,
*compositional* or *mixed* functions whatsoever.

By definition, standalone functions can only reference themselves or other standlone functions.
Most commonly, these functions are *convenience* functions (see below).

As a first example, every function in the `nest.py`_ module is a standalone function.
All of these either: (a) reference no other function at all, (b) only reference themselves recursively,
or (c) reference other standalone functions.

A few other examples outside of the :code:`nest.py` module are:
`ivy.default`_ which simply returns :code:`x` if it exists else the default value,
`ivy.cache_fn`_ which wraps a function such that when :code:`cache=True` is passed,
then a previously cached output is returned,
and `ivy.stable_divide`_ which simply adds a small constant to the denominator of the division.

Nestable Functions
------------------

*Nestable* functions are functions (compositional or primary) which can accept :code:`ivy.Container` instances in place
of **any** of the arguments. Multiple containers can also be passed in for multiple arguments at the same time,
provided that the containers share an identical nested structure.
If an :code:`ivy.Container` is passed, then the function is applied to all of the
leaves of the container, with the container leaf values passed into the function at the corresponding arguments.
In this case, the function will return an :code:`ivy.Container` in the output.

This property makes it very easy to write a single piece of code that can deal either with individual arrays or
arbitrary batches of nested arrays. This is very useful in machine learning, where batches of different data often need
to be processed concurrently. Another example is when the same operation must be performed on each weight in a network.
This *nestable* property of Ivy functions means that the same function can be used for any of these use cases
without modification.

This added support for handling :code:`ivy.Container` instances is all handled automatically when `_wrap_function`_
is applied to every function (except those appearing in `NON_WRAPPED_FUNCTIONS`_)
in the :code:`ivy` module during `backend setting`_.
This function wrapping process is covered in more detail in the :ref:`Function Wrapping` section.

Under the hood, the static :code:`ivy.Container` methods are called when :code:`ivy.Container` instances are passed in
as inputs to functions in the functional API. This is explained in more detail in the :ref:`Method Types` section.

Convenience Functions
---------------------

A final group of functions are the *convenience* functions.
Convenience functions do not form part of the computation graph directly, and they do not directly modify arrays.
However, they can be used to organize and improve the code for other functions which do modify the arrays.

Convenience functions do not defer to backend implementations in any way.
This sets them apart from *compositional* functions,
which are still ultimately composed of *primary* functions under the hood.
Convenience functions are **not** composed of primary functions under the hood.

A few examples are:
`ivy.set_backend`_ which sets the global backend framework,
`ivy.get_backend`_ which returns a local Ivy module with the associated backend framework,
`ivy.nested_map`_ which enables an arbitrary function to be mapped across the leaves of an arbitrary nest,
`ivy.index_nest`_ which enables an arbitrary nest to be recursively indexed,
`ivy.set_default_dtype`_ which sets the global default data type,
and `ivy.set_default_device`_ which sets the global default device.

There are many other examples. The convenience functions are not grouped in any particular way.
Feel free to have a look through all of the `submodules`_, you should be able to spot quite a few!