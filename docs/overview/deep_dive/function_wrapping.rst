Function Wrapping
=================

.. _`wrapped`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/utils/backend/handler.py#L259
.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L965
.. _`abs`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/functional/ivy/elementwise.py#L28
.. _`creation submodule`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/functional/ivy/creation.py
.. _`zeros`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/functional/ivy/creation.py#L482
.. _`asarray`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/functional/ivy/creation.py#L383
.. _`inputs_to_native_arrays`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L405
.. _`inputs_to_ivy_arrays`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L445
.. _`outputs_to_ivy_arrays`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L525
.. _`to_native_arrays_and_back`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L595
.. _`infer_dtype`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L725
.. _`infer_device`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L763
.. _`handle_out_argument`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L800
.. _`handle_nestable`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L896
.. _`inputs_to_native_shapes`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L488
.. _`outputs_to_ivy_shapes`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L501
.. _`to_native_shapes_and_back`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L514
.. _`handle_view`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L627
.. _`handle_view_indexing`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L659
.. _`handle_array_function`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L299
.. _`handle_complex_input`: https://github.com/unifyai/ivy/blob/bd9b5b1080d33004e821a48c486b3a879b9d6616/ivy/func_wrapper.py#L1393
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`function wrapping thread`: https://discord.com/channels/799879767196958751/1189906704775794688
.. _`handle_partial_mixed_function`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L944
.. _`stored as an attribute`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/func_wrapper.py#L1054
.. _`ivy.linear`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/functional/ivy/layers.py#L81
.. _`handle_exceptions`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/utils/exceptions.py#L189
.. _`example`: https://github.com/unifyai/ivy/blob/5658401b266352d3bf72c95e4af6ae9233115722/ivy/functional/backends/torch/layers.py#L30
.. _`Arrays`: arrays.rst
.. _`Inplace Updates`: inplace_updates.rst
.. _`Data Types`: data_types.rst
.. _`Devices`: devices.rst
.. _`Backend Setting`: backend_setting.rst

When a backend framework is set by calling :code:`ivy.set_backend(backend_name)`, then all Ivy functions are `wrapped`_.
This is achieved by calling `_wrap_function`_, which will apply the appropriate wrapping to the given function, based on what decorators it has.
For example, `abs`_ has the decorators :code:`@to_native_arrays_and_back` and :code:`@handle_out_argument`, and so the backend implementations will also be wrapped with the `to_native_arrays_and_back`_ and `handle_out_argument`_ wrappers.

The new function returned by :code:`_wrap_function` is a replacement of the original function with extra code added to support requirements common to many functions in the API.
This is the main purpose of the wrapping, to avoid code duplication which would exist if we added identical logic in every single function independently.

Depending on the function being wrapped, the new function might handle `Arrays`_, `Inplace Updates`_, `Data Types`_ and/or `Devices`_.

Our test decorators actually transforms to :code:`@given` decorators at Pytest collecting time, therefore this allows us to use other **Hypothesis** decorators like, :code:`@reproduce_failure`, :code:`@settings`, :code:`@seed`.

Decorator order
^^^^^^^^^^^^^^^

The order in which Ivy decorators are applied is important. It is important to follow this order, as the functionality of many functions depends on it. If the decorators are applied in the wrong order, the test may fail or the function may not behave as expected.
The following is the recommended order to follow :

#.  :code:`@handle_complex_input`
#.  :code:`@infer_device`
#.  :code:`@handle_device_shifting`
#.  :code:`@infer_dtype`
#.  :code:`@handle_array_function`
#.  :code:`@outputs_to_ivy_arrays`
#.  :code:`@outputs_to_ivy_shapes`
#.  :code:`@outputs_to_native_arrays`
#.  :code:`@inputs_to_native_arrays`
#.  :code:`@inputs_to_native_shapes`
#.  :code:`@inputs_to_ivy_arrays`
#.  :code:`@handle_out_argument`
#.  :code:`@handle_view_indexing`
#.  :code:`@handle_view`
#.  :code:`@handle_array_like_without_promotion`
#.  :code:`@handle_partial_mixed_function`
#.  :code:`@handle_nestable`
#.  :code:`@handle_ragged`
#.  :code:`@handle_backend_invalid`
#.  :code:`@handle_exceptions`
#.  :code:`@handle_nans`

This recommended order is followed to ensure that tests are efficient and accurate. It is important to follow this order because the decorators depend on each other. For example, the :code:`@infer_device` decorator needs to be applied before the :code:`@infer_dtype` decorator, because the :code:`@infer_dtype` decorator needs to know the device of the function in order to infer the data type.

Conversion Wrappers
^^^^^^^^^^^^^^^^^^^

#.  `inputs_to_native_arrays`_ : This wrapping function converts all :class:`ivy.Array` instances in the arguments to their :class:`ivy.NativeArray` counterparts, based on the `Backend Setting`_ before calling the function.
#.  `inputs_to_ivy_arrays`_ : This wrapping function converts all :class:`ivy.NativeArray` instances in the arguments to their :class:`ivy.Array` counterparts, based on the `Backend Setting`_ before calling the function.
#.  `outputs_to_ivy_arrays`_ : This wrapping function converts all :class:`ivy.NativeArray` instances in the outputs to their :class:`ivy.Array` counterparts, based on the `Backend Setting`_ before calling the function.
#.  `to_native_arrays_and_back`_ : This wrapping function converts all :class:`ivy.Array` instances in the arguments to their :class:`ivy.NativeArray` counterparts, calls the function with those arguments and then converts the :class:`ivy.NativeArray` instances in the output back to :class:`ivy.Array`.
    This wrapping function is heavily used because it enables achieving the objective of ensuring that every ivy function could accept an :class:`ivy.Array` and return an :class:`ivy.Array`, making it independent of the `Backend Setting`_.

Inference Wrappers
^^^^^^^^^^^^^^^^^^

#.  `infer_dtype`_ : This wrapping function infers the `dtype` argument to be passed to a function based on the array arguments passed to it.
    If :code:`dtype` is explicitly passed to the function, then it is used directly.
    This wrapping function could be found in functions from the `creation submodule`_ such as `zeros`_ where we then allow the user to not enter the :code:`dtype` argument to such functions.
#.  `infer_device`_ : Similar to the `infer_dtype`_ wrapping function, the `infer_device`_ function wrapping infers the :code:`device` argument to be passed to a function based on the first array argument passed to it.
    This wrapping function is also used a lot in functions from the `creation submodule`_ such as `asarray`_, where we want to create the `ivy.Array` on the same device as the input array.

Out Argument Support
^^^^^^^^^^^^^^^^^^^^

#.  `handle_out_argument`_ : This wrapping function is used in nearly all ivy functions.
    It enables appropriate handling of the :code:`out` argument of functions.
    In cases where the backend framework natively supports the :code:`out` argument for a function, we prefer to use it as it's a more efficient implementation of the :code:`out` argument for that particular backend framework.
    But in cases when it isn't supported, we support it anyway with `Inplace Updates`_.

Nestable Support
^^^^^^^^^^^^^^^^

#.  `handle_nestable`_ : This wrapping function enables the use of :class:`ivy.Container` arguments in functions and directly calling them through the :code:`ivy` namespace, just like calling a function with :class:`ivy.Array` arguments instead. Thus, the function can be called by passing an :class:`ivy.Container` to any or all of its arguments.

Partial Mixed Function Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. `handle_partial_mixed_function`_: This wrapping function enables switching between compositional and primary implementations of :ref:`overview/deep_dive/function_types:Mixed Functions` based on some condition on the arguments of the function.
#.  The condition is specified through a lambda function which when evaluates to `True` the primary implementation is run and otherwise the compositional implementation is executed.
#.  For backends that have a primary implementation of a mixed function, the reference to the compositional implementation is `stored as an attribute`_ inside the backend function during backend setting. To make use of this decorator, one must
#.  add the :code:`partial_mixed_handler` attribute containing the lambda function to the backend implementation. Here's an `example`_ from the torch backend implementation of linear.

Shape Conversion
^^^^^^^^^^^^^^^^

#.  `inputs_to_native_shapes`_ : This wrapping function converts all :class:`ivy.Shape` instances in the arguments to their :class:`ivy.NativeShape` counterparts, based on the `Backend Setting`_ before calling the function.
#.  `outputs_to_ivy_shapes`_ : This wrapping function converts all :class:`ivy.NativeShape` instances in the outputs to their :class:`ivy.Shape` counterparts, based on the `Backend Setting`_ before calling the function.
#.  `to_native_shapes_and_back`_ : This wrapping function converts all :class:`ivy.Shape` instances in the arguments to their :class:`ivy.NativeShape` counterparts, calls the function with those arguments and then converts the :class:`ivy.NativeShape` instances in the output back to :class:`ivy.Shape`.

View Handling
^^^^^^^^^^^^^

#.  `handle_view`_ : This wrapping function performs view handling based on our :ref:`overview/deep_dive/inplace_updates:Views` policy.
#.  `handle_view_indexing`_ : This wrapping function is aimed at handling views for indexing.

Exception Handling
^^^^^^^^^^^^^^^^^^

#. `handle_exceptions`_ : This wrapping function helps in catching native exceptions and unifying them into `IvyException` or the relevant subclasses. More information can be found in the :ref:`overview/deep_dive/function_wrapping:Exception Handling` section.

Miscellaneous Wrappers
^^^^^^^^^^^^^^^^^^^^^^

#.  `handle_array_function`_ : This wrapping function enables :ref:`overview/deep_dive/arrays:Integrating custom classes with Ivy`
#.  `handle_complex_input`_ : This wrapping function enables handling of complex numbers. It introduces a keyword argument :code:`complex_mode`, which is used to choose the function's behaviour as per the wrapper's docstring.


When calling `_wrap_function`_ during `Backend Setting`_, firstly the attributes of the functions are checked to get all the wrapping functions for a particular function.
Then all the wrapping functions applicable to a function are used to wrap the function.

Each of these topics and each associated piece of logic added by the various wrapper functions are covered in more detail in the next sections.
For now, suffice it to say that they do quite a lot.

**Round Up**

This should have hopefully given you a good feel for how function wrapping is applied to functions in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `function wrapping thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/-RGXxrP849k" class="video">
    </iframe>
