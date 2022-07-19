Function Wrapping
=================

.. _`wrapped`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L412
.. _`abs`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/functional/ivy/elementwise.py#L2142
.. _`to_native_arrays_and_back`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L237
.. _`handle_out_argument`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L323
.. _`function wrapping discussion`: https://github.com/unifyai/ivy/discussions/1314
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`function wrapping channel`: https://discord.com/channels/799879767196958751/982737993028755496

When a backend framework is set by calling :code:`ivy.set_backend(backend_name)`,
then all Ivy functions are `wrapped`_. This is achieved by calling `_wrap_function`_, which will apply the appropriate
wrapping to the given function, based on what decorators it has. For example, `abs`_ has the decorators
:code:`@to_native_arrays_and_back` and :code:`@handle_out_argument`, and so the backend implementations will also be
wrapped with the `to_native_arrays_and_back`_ and `handle_out_argument`_ wrappers.

The new function returned by :code:`_wrap_function`
is a replacement of the original function with extra code added to support requirements common to many functions
in the API. This is the main purpose of the wrapping, to avoid code duplication which would exist if we added
identical logic in every single function independently.

Depending on the function being wrapped, the new function
might handle :ref:`Arrays`, :ref:`Inplace Updates`, :ref:`Data Types` and/or :ref:`Devices`.

Each of these topics and each associated piece of logic added by the various wrapper functions are covered in more
detail in the next sections. For now, suffice it to say that they do quite a lot.

**Round Up**

This should have hopefully given you a good feel for how function wrapping is applied to functions in Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `function wrapping discussion`_,
or reach out on `discord`_ in the `function wrapping channel`_!
