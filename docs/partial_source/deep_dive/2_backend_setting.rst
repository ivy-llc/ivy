Backend Setting
===============

.. _`this function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L154
.. _`implicit_backend`: https://github.com/unifyai/ivy/blob/master/ivy/backend_handler.py#L16
.. _`import the backend module`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L184
.. _`writing the function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L212
.. _`wrap the functions`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`backend setting discussion`: https://github.com/unifyai/ivy/discussions/1313
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`backend setting channel`: https://discord.com/channels/799879767196958751/982737886963187772

The backend framework can either be set by calling :code:`ivy.set_backend(backend_name)` or it can inferred from the \
arguments. For the latter, a global variable `implicit_backend`_ is located in the file which is initialized as :code:`numpy`\
, and is always used to infer the backend in cases where: (a) no backend has been set using the :code:`set_backend` \
function and (b) the backend cannot be inferred from the inputs. If the framework can be inferred from the inputs, then\
this is always used, and the `implicit_backend`_ is overwritten with the framework inferred. :code:`numpy` will always be\
the default backend unless it explicitly set or is inferred.\

When calling `this function`_ for setting the backend, the following steps are performed:

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

It's helpful to look at an example:

.. code-block:: python

   x = ivy.array([[2., 3.]])
   ivy.get_backend()
   <module 'ivy.functional.backends.numpy' from '/opt/project/ivy/functional/backends/numpy/__init__.py'>

.. code-block:: python

   y = ivy.multiply(torch.Tensor([3.]), torch.Tensor([4.]))
   ivy.get_backend()
   <module 'ivy.functional.backends.torch' from '/opt/project/ivy/functional/backends/torch/__init__.py'>

.. code-block:: python

   ivy.set_backend('jax')
   z = ivy.matmul(jax.numpy.array([[2.,3.]]), jax.numpy.array([[5.],[6.]]))
   ivy.get_backend()
   <module 'ivy.functional.backends.jax' from '/opt/project/ivy/functional/backends/jax/__init__.py'>
   ivy.unset_backend()
   ivy.get_backend()
   <module 'ivy.functional.backends.torch' from '/opt/project/ivy/functional/backends/torch/__init__.py'>

In the last example above, the moment any backend is set, it will be used over the `implicit_backend`_. However when the \
backend is unset using the :code:`ivy.unset_backend`, the `implicit_backend`_ will be used as a fallback, which will \
assume the backend from the last run. While the `implicit_backend`_ functionality gives more freedom to the user , the \
recommended way of doing things would be set the backend explicitly.

**Round Up**

This should have hopefully given you a good feel for how the backend framework is set.

If you're ever unsure of how best to proceed,
please feel free to engage with the `backend setting discussion`_,
or reach out on `discord`_ in the `backend setting channel`_!
