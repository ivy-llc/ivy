Backend Setting
===============

.. _`this function`: https://github.com/unifyai/ivy/blob/7fb947f85075aa18e1000461617e067b5a3f56da/ivy/backend_handler.py#L153
.. _`import the backend module`: https://github.com/unifyai/ivy/blob/7fb947f85075aa18e1000461617e067b5a3f56da/ivy/backend_handler.py#L182
.. _`in either case`: https://github.com/unifyai/ivy/blob/7fb947f85075aa18e1000461617e067b5a3f56da/ivy/backend_handler.py#L200
.. _`wrap the functions`: https://github.com/unifyai/ivy/blob/7fb947f85075aa18e1000461617e067b5a3f56da/ivy/backend_handler.py#L206

The backend framework is set by calling :code:`ivy.set_backend(backend_name)`. When calling `this function`_,
the following steps are performed:

#. store a global copy of the original :code:`ivy.__dict__` to :code:`ivy_original_dict`, if this is not already stored.
#. `import the backend module`_, for example :code:`ivy.functional.backends.torch`, \
   if the backend has been passed in as a string. \
   All functions in this unmodified backend module are *primary* functions, because only primary functions are stored \
   in :code:`ivy.functional.backends.backend_name`. This backend module does not include any *compositional* functions.
#. loop through the original :code:`ivy_original_dict` (which has all functions, including compositional),
   for each function, and (a) add the primary function from the backend if it exists, (b) else add the compositional
   function from :code:`ivy_original_dict`, writing the function to :code:`ivy.__dict__` `in either case`_.
#. `wrap the functions`_, extending each of them with shared repeated functionality, which is solved via wrapping in
   order to avoid excessive code duplication in every backend function implementation. \
   This is explained in more detail in the next section: :ref:`Function Wrapping`.