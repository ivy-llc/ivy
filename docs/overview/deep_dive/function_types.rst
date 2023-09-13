Function Types
==============

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
.. _`inside the _wrap_function`: https://github.com/unifyai/ivy/blob/1a00001017ceca11baf0a7b83adcc51234d43fce/ivy/func_wrapper.py#L1115
.. _`FN_DECORATORS`: https://github.com/unifyai/ivy/blob/1a00001017ceca11baf0a7b83adcc51234d43fce/ivy/func_wrapper.py#L15
.. _`handle_partial_mixed_function`: https://github.com/unifyai/ivy/blob/1a00001017ceca11baf0a7b83adcc51234d43fce/ivy/functional/ivy/layers.py#L77
.. _`partial_mixed_handler`: https://github.com/unifyai/ivy/blob/1a00001017ceca11baf0a7b83adcc51234d43fce/ivy/functional/backends/torch/layers.py#L29
.. _`handle`: https://github.com/unifyai/ivy/blob/0ef2888cbabeaa8f61ce8aaea4f1175071f7c396/ivy/func_wrapper.py#L1027-L1030
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`function types channel`: https://discord.com/channels/799879767196958751/982737839861145630

Firstly, we explain the difference between *primary*, *compositional*, *mixed* and *standalone* functions.
These four function categorizations are all **mutually exclusive**, and combined they constitute the set of **all** functions in Ivy, as outlined in the simple Venn diagram below.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/deep_dive/function_types/four_function_types.png?raw=true
   :align: center
   :width: 50%
   :class: dark-light

Primary Functions
-----------------

*Primary* functions are essentially the lowest level building blocks in Ivy.
Each primary function has a unique backend-specific implementation for each backend specified in :mod:`ivy/functional/backends/backend_name/category_name.py`.
These are generally implemented as light wrapping around an existing function in the backend framework, which serves a near-identical purpose.

Primary functions must both be specified in :mod:`ivy/functional/ivy/category_name.py` and also in each of the backend files :mod:`ivy/functional/backends/backend_name/category_name.py`.

The function in :mod:`ivy/functional/ivy/category_name.py` includes the type hints, docstring and docstring examples (explained in more detail in the subsequent sections), but does not include an actual implementation.

Instead, in :mod:`ivy/functional/ivy/category_name.py`, primary functions simply defer to the backend-specific implementation.

For example, the code for :func:`ivy.tan` in :mod:`ivy/functional/ivy/elementwise.py` (with decorators and docstrings removed) is given below:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.current_backend(x).tan(x, out=out)

The backend-specific implementation of :func:`ivy.tan`  for PyTorch in :mod:`ivy/functional/backends/torch/elementwise.py` is given below:

.. code-block:: python

    def tan(
        x: torch.Tensor,
        /,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = _cast_for_unary_op(x)
        return torch.tan(x, out=out)

The reason that the Ivy implementation has type hint :code:`Union[ivy.Array, ivy.NativeArray]` but PyTorch implementation has :class:`torch.Tensor` is explained in the :ref:`Arrays` section.
Likewise, the reason that the :code:`out` argument in the Ivy implementation has array type hint :class:`ivy.Array` whereas :code:`x` has :code:`Union[ivy.Array, ivy.NativeArray]` is also explained in the :ref:`Arrays` section.

Compositional Functions
-----------------------

*Compositional* functions on the other hand **do not** have backend-specific implementations.
They are implemented as a *composition* of other Ivy functions, which themselves can be either compositional, primary or mixed (explained below).

Therefore, compositional functions are only implemented in :mod:`ivy/functional/ivy/category_name.py`, and there are no implementations in any of the backend files :mod:`ivy/functional/backends/backend_name/category_name.py`.

For example, the implementation of :func:`ivy.cross_entropy` in :mod:`ivy/functional/ivy/losses.py` (with docstrings and decorators removed) is given below:

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        epsilon: float = 1e-7,
        reduction: str = "sum",
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return _reduce_loss(reduction, log_pred * true, axis, out)


Mixed Functions
---------------
---------------

Sometimes, a function may only be provided by some of the supported backends. In this case, we have to take a mixed approach. We should always have a backend-specific implementation if there is a similar function provided by a certain backend. This maximises runtime efficiency, as the function in the backend will be implemented directly in C or C++. Such functions have some backend-specific implementations in :mod:`ivy/functional/backends/backend_name/category_name.py`, but not for all backends. To support backends that do not have a backend-specific implementation, a compositional implementation is also provided in :mod:`ivy/functional/ivy/category_name.py`. Compositional functions should only be used when there is no similar function to wrap in the backend. 

Because these functions include both a compositional implementation and also at least one backend-specific implementation, these functions are referred to as *mixed*.

When using ivy without a backend set explicitly (for example :func:`ivy.set_backend` has not been called), then the function called is always the one implemented in :mod:`ivy/functional/ivy/category_name.py`.
For *primary* functions, then :code:`ivy.current_backend(array_arg).func_name(...)` will call the backend-specific implementation in :mod:`ivy/functional/backends/backend_name/category_name.py` directly.
However, as just explained, *mixed* functions implement a compositional approach in :mod:`ivy/functional/ivy/category_name.py`, without deferring to the backend.
Therefore, when no backend is explicitly set, then the compositional implementation is always used for *mixed* functions, even for backends that have a more efficient backend-specific implementation.
Typically the backend should always be set explicitly though (using :func:`ivy.set_backend` for example), and in this case the efficient backend-specific implementation will always be used if it exists.


Partial Mixed Functions
-----------------------

There may be instances wherein the native backend function does not encompass the full range of possible cases that ivy wants to support.
One example of this is :code:`ivy.linear` for which the torch native function :code:`torch.nn.functional.linear` only supports the :code:`weight` argument
to be a 2 dimensional tensor while as ivy also allows the :code:`weight` argument to be 3 dimensional. While achieving the objective of having superset
behaviour across the backends, native functionality of frameworks should be made use of as much as possible. Even if a framework-specific function
doesn't provide complete superset behaviour, we should still make use of the partial behaviour that it provides and then add more logic for the
remaining part. This is explained in detail in the :ref:`Maximizing Usage of Native Functionality` section. Ivy allows this partial support with the help of the `partial_mixed_handler`_
attribute which should be added to the backend implementation with a boolean function that specifies some condition on the inputs to switch between the compositional
and primary implementations. For example, the :code:`torch` backend implementation of :code:`linear`` looks like:

.. code-block:: python

   def linear(
       x: torch.Tensor,
       weight: torch.Tensor,
       /,
       *,
       bias: Optional[torch.Tensor] = None,
       out: Optional[torch.Tensor] = None,
   ) -> torch.Tensor:
       return torch.nn.functional.linear(x, weight, bias)

   linear.partial_mixed_handler = lambda x, weight, **kwargs: weight.ndim == 2

And to the compositional implementation, we must add the `handle_partial_mixed_function`_ decorator. When the backend is set, the :code:`handle_partial_mixed_function`
decorator is added to the primary implementation `inside the _wrap_function`_  according to the order in the `FN_DECORATORS`_ list. When the function is executed,
the :code:`handle_partial_mixed_function` decorator first evaluates the boolean function using the given inputs, and we use the backend-specific implementation if the result
is `True` and the compositional implementation otherwise.


For further information on decorators, please refer to the :ref:`Function Wrapping` section.

For all mixed functions, we must add the :code:`mixed_backend_wrappers` attribute to the compositional implementation of mixed functions to specify which additional wrappers need to be applied to the primary implementation and which ones from the compositional implementation should be skipped.
We do this by creating a dictionary of two keys, :code:`to_add` and :code:`to_skip`, each containing the tuple of wrappers to be added or skipped respectively. In general, :code:`handle_out_argument`, :code:`inputs_to_native_arrays` and :code:`outputs_to_ivy_arrays`
should always be added to the primary implementation and :code:`inputs_to_ivy_arrays` should be skipped. For the :code:`linear` function, :code:`mixed_backend_wrappers` was added in the following manner.


.. code-block:: python

   linear.mixed_backend_wrappers = {
      "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
      ),
      "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
   }

When the backend is set, we `handle`_ these wrappers for the primary implementation inside the :code:`_wrap_function`.


Standalone Functions
---------------------

*Standalone* functions are functions which do not reference any other *primary*, *compositional* or *mixed* functions whatsoever.

By definition, standalone functions can only reference themselves or other standalone functions.
Most commonly, these functions are *convenience* functions (see below).

As a first example, every function in the `nest.py`_ module is a standalone function.
All of these either: (a) reference no other function at all, (b) only reference themselves recursively, or (c) reference other standalone functions.

A few other examples outside of the :mod:`nest.py` module are: `ivy.default`_ which simply returns :code:`x` if it exists else the default value, `ivy.cache_fn`_ which wraps a function such that when :code:`cache=True` is passed, then a previously cached output is returned, and `ivy.stable_divide`_ which simply adds a small constant to the denominator of the division.

Nestable Functions
------------------

*Nestable* functions are functions which can accept :class:`ivy.Container` instances in place of **any** of the arguments.
Multiple containers can also be passed in for multiple arguments at the same time, provided that the containers share a common nested structure.
If an :class:`ivy.Container` is passed, then the function is applied to all of the leaves of the container, with the container leaf values passed into the function at the corresponding arguments.
In this case, the function will return an :class:`ivy.Container` in the output.
*Primary*, *compositional*, *mixed*, and *standalone* functions can all *also* be nestable.
This categorization is **not** mutually exclusive, as outlined by the Venn diagram below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/deep_dive/function_types/nestable.png?raw=true
   :align: center
   :width: 50%
   :class: dark-light

The *nestable* property makes it very easy to write a single piece of code that can deal either with individual arguments or arbitrary batches of nested arguments.
This is very useful in machine learning, where batches of different training data often need to be processed concurrently.
Another example is when the same operation must be performed on each weight in a network.
This *nestable* property of Ivy functions means that the same function can be used for any of these use cases without modification.

This added support for handling :class:`ivy.Container` instances is all handled automatically when `_wrap_function`_ is applied to every function in the :code:`ivy` module during `backend setting`_.
This will add the `handle_nestable`_ wrapping to the function if it has the :code:`@handle_nestable` decorator.
This function wrapping process is covered in a bit more detail in the :ref:`Function Wrapping` section.

Nestable functions are explained in more detail in the :ref:`Containers` section.

Convenience Functions
---------------------

A final group of functions are the *convenience* functions (briefly mentioned above).
Convenience functions do not form part of the computation graph directly, and they do not directly modify arrays.
However, they can be used to organize and improve the code for other functions which do modify the arrays.
Convenience functions can be *primary*, *compositional*, *mixed* or *standalone* functions.
Many are also *nestable*.
This is another categorization which is **not** mutually exclusive, as outlined by the Venn diagram below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/deep_dive/function_types/convenience.png?raw=true
   :align: center
   :width: 50%
   :class: dark-light

Primary convenience functions include: `ivy.can_cast`_ which determines if one data type can be cast to another data type according to type-promotion rules, `ivy.dtype`_ which gets the data type for the input array, and `ivy.dev`_ which gets the device for the input array.

Compositional convenience functions include: `ivy.set_default_dtype`_ which sets the global default data dtype, `ivy.default_dtype`_ which returns the correct data type to use, considering both the inputs and the globally set default data type, and `ivy.get_all_arrays_on_dev`_ which gets all arrays which are currently on the specified device.

Standalone convenience functions include: `ivy.get_backend`_ which returns a local Ivy module with the associated backend framework.
`ivy.nested_map`_ which enables an arbitrary function to be mapped across the leaves of an arbitrary nest, and `ivy.index_nest`_ which enables an arbitrary nest to be recursively indexed.

There are many other examples.
The convenience functions are not grouped by file or folder.
Feel free to have a look through all of the `submodules`_, you should be able to spot quite a few!

**Round Up**

This should have hopefully given you a good feel for the different function types.

If you have any questions, please feel free to reach out on `discord`_ in the `function types channel`_!

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/mWYhQRu1Vuk" class="video">
    </iframe>
