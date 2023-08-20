Backend Setting
===============

.. _`this function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L154
.. _`implicit_backend`: https://github.com/unifyai/ivy/blob/3358b5bbadbe4cbc0509cad4ea8f05f178dfd8b8/ivy/utils/backend/handler.py
.. _`import the backend module`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L184
.. _`writing the function`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L212
.. _`wrap the functions`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`backend setting channel`: https://discord.com/channels/799879767196958751/982737886963187772
.. _`backend setting forum`: https://discord.com/channels/799879767196958751/982737886963187772

The backend framework can either be set by calling :code:`ivy.set_backend(backend_name)` or it can inferred from the arguments.
For the latter, a global variable `implicit_backend`_ is located in the file which is initialized as numpy, and is always used to infer the backend in cases where: (a) no backend has been set using the :code:`set_backend` function and (b) the backend cannot be inferred from the inputs.
If the framework can be inferred from the inputs, then this is always used, and the `implicit_backend`_ is overwritten with the framework inferred.
numpy will always be the default backend unless it is explicitly set or is inferred.
When calling `this function`_ for setting the backend, the following steps are performed:

#. store a global copy of the original :attr:`ivy.__dict__` to :code:`ivy_original_dict`, if this is not already stored.
#. `import the backend module`_, for example :mod:`ivy.functional.backends.torch`, if the backend has been passed in as a string.
   All functions in this unmodified backend module are *primary* functions, because only primary functions are stored in :mod:`ivy.functional.backends.backend_name`.
   This backend module does not include any *compositional* functions.
#. loop through the original :code:`ivy_original_dict` (which has all functions, including compositional), and (a) add the primary function from the backend if it exists, (b) else add the compositional function from :code:`ivy_original_dict`.
#. `wrap the functions`_ where necessary, extending them with shared repeated functionality and `writing the function`_ to :attr:`ivy.__dict__`.
   Wrapping is used in order to avoid excessive code duplication in every backend function implementation.
   This is explained in more detail in the next section: :ref:`Function Wrapping`.

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
   ivy.previous_backend()
   ivy.get_backend()
   <module 'ivy.functional.backends.torch' from '/opt/project/ivy/functional/backends/torch/__init__.py'>

In the last example above, the moment any backend is set, it will be used over the `implicit_backend`_.
However when the current backend is set to the previous using the :func:`ivy.previous_backend`, the `implicit_backend`_ will be used as a fallback, which will assume the backend from the last run.
While the `implicit_backend`_ functionality gives more freedom to the user, the recommended way of doing things would be set the backend explicitly.
In addition, all the previously set backends can be cleared by calling :func:`ivy.unset_backend`.

Dynamic Backend Setting
-----------------------

.. _`ivy.set_dynamic_backend`: https://github.com/unifyai/ivy/blob/main/ivy/__init__.py#L1134.
.. _`ivy.unset_dynamic_backend`: https://github.com/unifyai/ivy/blob/main/ivy/__init__.py#L1143.
.. _`ivy.dynamic_backend_as`: https://github.com/unifyai/ivy/blob/main/ivy/__init__.py#L1174.
.. _`ivy.Array`: https://github.com/unifyai/ivy/blob/main/ivy/data_classes/array/array.py#L186.
.. _`ivy.Container`: https://github.com/unifyai/ivy/blob/main/ivy/data_classes/container/base.py#L4166.
.. _`converted to numpy`: https://github.com/unifyai/ivy/blob/main/ivy/utils/backend/handler.py#L283.
.. _`converted from numpy`: https://github.com/unifyai/ivy/blob/main/ivy/utils/backend/handler.py#L363.

Working with different backends in Ivy can be challenging, especially when you need to switch between backends frequently.
To make this easier, users can make use of the dynamic backend attribute of :class:`ivy.Array` and :class:`ivy.Container` classes which allow you to automatically convert ivy arrays to the new backend whenever the backend is changed.
Essentially, when the user calls :code:`ivy.set_backend(<backend>, dynamic=True)`, the following steps are performed:

#. First, all live objects in the current project scope are found and then filtered to only include :class:`ivy.Array`/:class:`ivy.Container` objects.
#. Then, these objects are iterated through and `converted to numpy`_ as an intermediary using the current backend.
#. Next, the global :code:`ivy.__dict__` is updated to the new backend as mentioned in the Backend Setting section above.
#. Finally, the objects are `converted from numpy`_ to the target backend using the newly set backend.

By default, the dynamic backend attribute is set to True when you create an ivy array (e.g., :code:`x = ivy.array([1,2,3])`), but the attribute is mutable and can be changed after the ivy array is created (e.g., :code:`x.dynamic_backend= True`). 
Here's an example to illustrate how this works in practice:

.. code-block:: python

   ivy.set_backend('torch')
   x = ivy.array([1,2,3])
   y = ivy.array([1,2,3])
   y.dynamic_backend=False
   x.dynamic_backend=True
   x.data # torch tensor
   y.data # torch.tensor

   ivy.set_backend('jax')
   x.data # will be a jax array
   y.data # will still be a torch tensor since dynamic_backend=False

In addition to setting the dynamic backend attribute for individual ivy arrays, you can also set or unset the dynamic backend feature globally for all such instances using `ivy.set_dynamic_backend`_ and `ivy.unset_dynamic_backend`_ respectively.

Another useful feature of the dynamic backend is the `ivy.dynamic_backend_as`_ context manager. This allows you to write code like this:

.. code-block:: python

   with ivy.dynamic_backend_as(True):
     a = ivy.array([0., 1.])
     b = ivy.array([2., 3.])

   with ivy.dynamic_backend_as(False):
     c = ivy.array([4., 5.])
     d = ivy.array([6., 7.])

This makes it easy to define different sections of your project with different settings, without having to explicitly call :code:`ivy.set_<something>` and :code:`ivy.unset_<something>` etc.

There is one technical point to keep in mind when using the dynamic backend attribute. Consider the following example:

.. code-block:: python

   ivy.set_backend("tensorflow")
   arr = ivy.array([1,2,3])
   arr.dynamic_backend= False

   ivy.set_backend("torch")

   # arr.data should still be a tf.Tensor

   arr.dynamic_backend = True

   ivy.set_backend("jax")

   # This would cause a problem since the conversion goes from TF -> JAX, whereas the backend stack goes from Torch -> Jax.

To avoid the above issue, we update the :attr:`.data` attribute to be a native array for the current set backend framework in the setter method for dynamic_backend attribute for `ivy.Array`_ and `ivy.Container`_ classes. So after the line :code:`arr.dynamic_backend = True` in the example above, then :attr:`arr.data` would be a torch.Tensor and not a tf.Tensor.

Backend and Frontend Version Support
------------------------------------

Each time a new ivy backend is set, the backend_handler modifies the :attr:`ivy.__dict__` to support the multiple versions of functions that are not forward compatible.
For example, :func:`torch.ones_like` in the latest stable version :code:`1.12` has many new arguments :code:`dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format` compared to the same function at version :code:`0.3.1`.
None of these new arguments will cause any forward compatibility issues: they weren't used in old code, and they can now just be used in new code if desired.
However, the removal of the :code:`out` argument does break forward compatibility.
Old torch code will raise an :exc:`Argument Not Found` error if being run with new torch versions.
However, such forward-breaking changes are in the vast minority.

We currently use a naming convention for such functions and name them as :code:`fn_name_v_1p12_and_above` which means that this particular implementation of the function is valid for versions :code:`1.12` and above.
Similarly, :code:`fn_name_v_1p01_to_1p1` means that the function is valid for versions between :code:`1.01` and :code:`1.1` both inclusive.
Each time a backend is set, we go through the :attr:`backend.__dict__` and for all functions for which multiple versions are detected, we simply import and assign the original :code:`fn_name` to the version specific one.
We do so by detecting the version of the backend framework installed on the user's end.

We follow the same workflow for providing version support to the frontend functions.
Again the version is inferred by importing the corresponding framework on the user's system.
If the user's system doesn't have the backend framework installed, we default to the latest version.


**Round Up**

This should have hopefully given you a good feel for how the backend framework is set.

If you have any questions, please feel free to reach out on `discord`_ in the `backend setting channel`_ or in the `backend setting forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/ROt5E8aHgww" class="video">
    </iframe>
