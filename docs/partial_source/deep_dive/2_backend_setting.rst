Backend Setting
===============

.. _`this function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L154
.. _`import the backend module`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L184
.. _`writing the function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L212
.. _`wrap the functions`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`backend setting discussion`: https://github.com/unifyai/ivy/discussions/1313
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`backend setting channel`: https://discord.com/channels/799879767196958751/982737886963187772

The backend framework is set by calling :code:`ivy.set_backend(backend_name)`. When calling `this function`_,
the following steps are performed:

#. store a global copy of the original :code:`ivy.__dict__` to :code:`ivy_original_dict`, if this is not already stored.
#. `import the backend module`_, for example :code:`ivy.functional.backends.torch`, \
   if the backend has been passed in as a string. \
   All functions in this unmodified backend module are *primary* functions, because only primary functions are stored \
   in :code:`ivy.functional.backends.backend_name`. This backend module does not include any *compositional* functions.
#. loop through the original :code:`ivy_original_dict` (which has all functions, including compositional), and
   (a) add the primary function from the backend if it exists, (b) else add the compositional
   function from :code:`ivy_original_dict`.
#. `wrap the functions`_ where necessary, extending them with shared repeated functionality and
   `writing the function`_ to :code:`ivy.__dict__`. Wrapping is used in order to avoid excessive code duplication in
   every backend function implementation. This is explained in more detail in the next section:
   :ref:`Function Wrapping`.

**Round Up**

This should have hopefully given you a good feel for how the backend framework is set.

If you're ever unsure of how best to proceed,
please feel free to engage with the `backend setting discussion`_,
or reach out on `discord`_ in the `backend setting channel`_!
