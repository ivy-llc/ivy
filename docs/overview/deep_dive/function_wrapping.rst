Function Wrapping
=================

.. _`wrapped`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L340
.. _`abs`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/functional/ivy/elementwise.py#L2142
.. _`creation submodule`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/functional/ivy/creation.py
.. _`zeros`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/functional/ivy/creation.py#L158
.. _`asarray`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/functional/ivy/creation.py#L110
.. _`inputs_to_native_arrays`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L62
.. _`inputs_to_ivy_arrays`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L104
.. _`outputs_to_ivy_arrays`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L134
.. _`to_native_arrays_and_back`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L164
.. _`infer_dtype`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L176
.. _`infer_device`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L213
.. _`handle_out_argument`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L250
.. _`handle_nestable`: https://github.com/unifyai/ivy/blob/644412e3e691d2a04c7d3cd36fb492aa9f5d6b2d/ivy/func_wrapper.py#L297
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`function wrapping channel`: https://discord.com/channels/799879767196958751/982737993028755496
.. _`function wrapping forum`: https://discord.com/channels/799879767196958751/1028297461611122809
.. _`integer_array_to_float`: https://github.com/unifyai/ivy/blob/5da858be094a8ddb90ffe8886393c1043f4d8ae7/ivy/func_wrapper.py#L244
.. _`handle_cmd_line_args`: https://github.com/unifyai/ivy/blob/f1cf9cee62d162fbbd2a4afccd3a90e0cedd5d1f/ivy_tests/test_ivy/helpers.py#L3081
.. _`corresponding flags`: https://github.com/unifyai/ivy/blob/f1cf9cee62d162fbbd2a4afccd3a90e0cedd5d1f/ivy_tests/test_ivy/conftest.py#L174
.. _`handle_mixed_function`: https://github.com/unifyai/ivy/blob/6a57477daa87e3b3c6d157f10b935ba4fa21c39f/ivy/func_wrapper.py#L923
.. _`stored as an attribute`: https://github.com/unifyai/ivy/blob/6a57477daa87e3b3c6d157f10b935ba4fa21c39f/ivy/func_wrapper.py#L701
.. _`ivy.linear`: https://github.com/unifyai/ivy/blob/7a8fc1ea4eca6d061ae7a3efd1814518d4a6016f/ivy/functional/ivy/layers.py#L172
.. _`handle_exceptions`: https://github.com/unifyai/ivy/blob/40c3f381043d1c470fb4f04a0a5fd380a8a95130/ivy/utils/exceptions.py#L189
.. _`example`: https://github.com/unifyai/ivy/blob/7354979d7336e5138e8cae660f792ece507405b4/ivy/functional/backends/torch/layers.py#L30

When a backend framework is set by calling :code:`ivy.set_backend(backend_name)`, then all Ivy functions are `wrapped`_.
This is achieved by calling `_wrap_function`_, which will apply the appropriate wrapping to the given function, based on what decorators it has.
For example, `abs`_ has the decorators :code:`@to_native_arrays_and_back` and :code:`@handle_out_argument`, and so the backend implementations will also be wrapped with the `to_native_arrays_and_back`_ and `handle_out_argument`_ wrappers.

The new function returned by :code:`_wrap_function` is a replacement of the original function with extra code added to support requirements common to many functions in the API.
This is the main purpose of the wrapping, to avoid code duplication which would exist if we added identical logic in every single function independently.

Depending on the function being wrapped, the new function might handle :ref:`Arrays`, :ref:`Inplace Updates`, :ref:`Data Types` and/or :ref:`Devices`.

Conversion Wrappers
^^^^^^^^^^^^^^^^^^^

#.  `inputs_to_native_arrays`_ : This wrapping function converts all :class:`ivy.Array` instances in the arguments to their :class:`ivy.NativeArray` counterparts, based on the :ref:`Backend Setting` before calling the function.
#.  `inputs_to_ivy_arrays`_ : This wrapping function converts all :class:`ivy.NativeArray` instances in the arguments to their :class:`ivy.Array` counterparts, based on the :ref:`Backend Setting` before calling the function.
#.  `outputs_to_ivy_arrays`_ : This wrapping function converts all :class:`ivy.NativeArray` instances in the outputs to their :class:`ivy.Array` counterparts, based on the :ref:`Backend Setting` before calling the function.
#.  `to_native_arrays_and_back`_ : This wrapping function converts all :class:`ivy.Array` instances in the arguments to their :class:`ivy.NativeArray` counterparts, calls the function with those arguments and then converts the :class:`ivy.NativeArray` instances in the output back to :class:`ivy.Array`.
    This wrapping function is heavily used because it enables achieving the objective of ensuring that every ivy function could accept an :class:`ivy.Array` and return an :class:`ivy.Array`, making it independent of the :ref:`Backend Setting`.
#.  `integer_array_to_float`_: This wrapping function enables conversion of integer array inputs in the positional and keyword arguments to a function to the default float dtype.
    This is currently used to support integer array arguments to functions for which one or more backend frameworks only non-integer numeric dtypes.

Inference Wrappers
^^^^^^^^^^^^^^^^^^

#.  `infer_dtype`_ : This wrapping function infers the `dtype` argument to be passed to a function based on the  array arguments passed to it.
    If :code:`dtype` is explicitly passed to the function, then it is used directly.
    This wrapping function could be found in functions from the `creation submodule`_ such as `zeros`_ where we then allow the user to not enter the :code:`dtype` argument to such functions.
#.  `infer_device`_ : Similar to the `infer_dtype`_ wrapping function, the `infer_device`_ function wrapping  infers the :code:`device` argument to be passed to a function based on the first array argument passed to it.
    This wrapping function is also used a lot in functions from the `creation submodule`_ such as `asarray`_, where we want to create the `ivy.Array` on the same device as the input array.

Out Argument Support
^^^^^^^^^^^^^^^^^^^^

#.  `handle_out_argument`_ : This wrapping function is used in nearly all ivy functions.
    It enables appropriate handling of the :code:`out` argument of functions.
    In cases where the backend framework natively supports the :code:`out` argument for a function, we prefer to use it as it's a more efficient implementation of the :code:`out` argument for that particular backend framework.
    But in cases when it isn't supported, we support it anyway with :ref:`Inplace Updates`.

Nestable Support
^^^^^^^^^^^^^^^^

#.  `handle_nestable`_ : This wrapping function enables the use of :class:`ivy.Container` arguments in functions and directly calling them through the :code:`ivy` namespace, just like calling a function with :class:`ivy.Array` arguments instead.
    Whenever there's a :class:`ivy.Container` argument, this wrapping function defers to the corresponding :ref:`Containers` static method to facilitate the same.
    As a result, the function can be called by passing an :class:`ivy.Container` to any or all of its arguments.

Partial Mixed Function Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. `handle_mixed_function`_: This wrapping function enables switching between compositional and primary implementations of :ref:`Mixed Functions` based on some condition on the arguments of the function.
#.  The condition is specified through a lambda function which when evaluates to `True` the primary implementation is run and otherwise the compositional implementation is executed.
#.  For backends that have a primary implementation of a mixed function, the reference to the compositional implementation is `stored as an attribute`_ inside the backend function during backend setting. To make use of this decorator, one must
#.  add the :code:`partial_mixed_handler` attribute containing the lambda function to the backend implementation. Here's an `example`_ from the torch backend implementation of linear.

Exception Handling 
^^^^^^^^^^^^^^^^^^

#. `handle_exceptions`_: This wrapping function helps in catching native exceptions and unifying them into `IvyException` or the relevant subclasses. More information can be found in the :ref:`Exception Handling` section.

Miscellaneous Wrappers 
^^^^^^^^^^^^^^^^^^^^^^

#.  `handle_cmd_line_args`_: This wrapping function enables us to arbitrarily sample backend at test time using Hypothesis strategies.
    This enables us to infer the framework and generate appropriate data types directly inside the :code:`@given` decorator.
    With this approach in place, it's no longer necessary to check if the data type is supported and skip the test if it's not.
    Another place wherein this decorator is helpful is when we perform configurable argument testing for the parameters :code:`(as_variable, with_out, native_array, container, instance_method, test_gradients)` through the command line.
    The `corresponding flags`_ are used to set these values.


When calling `_wrap_function`_ during :ref:`Backend Setting`, firstly the attributes of the functions are checked to get all the wrapping functions for a particular functions.
Then all the wrapping functions applicable to a function are used to wrap the function.

Each of these topics and each associated piece of logic added by the various wrapper functions are covered in more detail in the next sections.
For now, suffice it to say that they do quite a lot.

**Round Up**

This should have hopefully given you a good feel for how function wrapping is applied to functions in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `function wrapping channel`_ or in the `function wrapping forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/-RGXxrP849k" class="video">
    </iframe>