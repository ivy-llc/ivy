Function Wrapping
=================

.. _`wrapped`: https://github.com/unifyai/ivy/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/ivy/framework_handler.py#L206
.. _`_wrap_or_unwrap_functions`: https://github.com/unifyai/ivy/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/ivy/func_wrapper.py#L341
.. _`at the leaves`: https://github.com/unifyai/ivy/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/ivy/func_wrapper.py#L408
.. _`does a lot`: https://github.com/unifyai/ivy/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/ivy/func_wrapper.py#L138

When a backend framework is set by calling :code:`ivy.set_framework(backend_name)`,
then all Ivy functions are `wrapped`_. This is achieved by calling `_wrap_or_unwrap_functions`_,
with :code:`_wrap_function` as an argument.
:code:`_wrap_or_unwrap_functions` is then called recursively,
and :code:`_wrap_function` is called on each function `at the leaves`_ of the ivy module,
with the returned function replacing the original function.

The new function returned by :code:`_wrap_function`, :code:`_function_wrapped`,
is a replacement of the original function with extra code added to support requirements common to many functions
in the API. This is the main purpose of the wrapping, to avoid code duplication by re-writing identical logic in every
single function independenly. Depending on the function being wrapped, the new function :code:`_function_wrapped`
might handle :ref:`Native Arrays`, :ref:`Inplace Updates`, :ref:`Data Types` and :ref:`Devices`.
Handling all of these cases once in the function wrapper logic removes a lot of potential code duplication.

Each of these topics and each reason for the wrapping are covered in more detail in the following sections.
For now, suffice it to say that :code:`_wrap_function` `does a lot`_