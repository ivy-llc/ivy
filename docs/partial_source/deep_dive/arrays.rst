Arrays
======

.. _`inputs_to_native_arrays`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L149
.. _`outputs_to_ivy_arrays`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L209
.. _`empty class`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/__init__.py#L8
.. _`overwritten`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/functional/backends/torch/__init__.py#L11
.. _`self._data`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/array/__init__.py#L89
.. _`ArrayWithElementwise`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/array/elementwise.py#L12
.. _`ivy.Array.add`: https://github.com/unifyai/ivy/blob/63d9c26acced9ef40e34f7b4fc1c1a75017f9c69/ivy/array/elementwise.py#L22
.. _`programmatically`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/__init__.py#L148
.. _`backend type hints`: https://github.com/unifyai/ivy/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/ivy/functional/backends/torch/elementwise.py#L219
.. _`Ivy type hints`: https://github.com/unifyai/ivy/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/ivy/functional/ivy/elementwise.py#L1342
.. _`__setitem__`: https://github.com/unifyai/ivy/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/ivy/array/__init__.py#L234
.. _`function wrapping`: https://github.com/unifyai/ivy/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/ivy/func_wrapper.py#L170
.. _`inherits`: https://github.com/unifyai/ivy/blob/8cbffbda9735cf16943f4da362ce350c74978dcb/ivy/array/__init__.py#L44
.. _`is the case`: https://data-apis.org/array-api/latest/API_specification/array_object.html
.. _`__add__`: https://github.com/unifyai/ivy/blob/e4d9247266f5d99faad59543923bb24b88a968d9/ivy/array/__init__.py#L291
.. _`__sub__`: https://github.com/unifyai/ivy/blob/e4d9247266f5d99faad59543923bb24b88a968d9/ivy/array/__init__.py#L299
.. _`__mul__`: https://github.com/unifyai/ivy/blob/e4d9247266f5d99faad59543923bb24b88a968d9/ivy/array/__init__.py#L307
.. _`__truediv__`: https://github.com/unifyai/ivy/blob/e4d9247266f5d99faad59543923bb24b88a968d9/ivy/array/__init__.py#L319
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`arrays channel`: https://discord.com/channels/799879767196958751/933380487353872454
.. _`arrays forum`: https://discord.com/channels/799879767196958751/1028296936203235359
.. _`wrapped logic`: https://github.com/unifyai/ivy/blob/6a729004c5e0db966412b00aa2fce174482da7dd/ivy/func_wrapper.py#L95

There are two types of array in Ivy, there is the :class:`ivy.NativeArray` and also the :class:`ivy.Array`.

Native Array
------------

The :class:`ivy.NativeArray` is simply a placeholder class for a backend-specific array class, such as :class:`np.ndarray`, :class:`tf.Tensor`, :class:`torch.Tensor` or :class:`jaxlib.xla_extension.DeviceArray`.

When no framework is set, this is an `empty class`_.
When a framework is set, this is `overwritten`_ with the backend-specific array class.

Ivy Array
---------

The :class:`ivy.Array` is a simple wrapper class, which wraps around the :class:`ivy.NativeArray`, storing it in `self._data`_.

All functions in the Ivy functional API which accept *at least one array argument* in the input are implemented as instance methods in the :class:`ivy.Array` class.
The only exceptions to this are functions in the `nest <https://github.com/unifyai/ivy/blob/906ddebd9b371e7ae414cdd9b4bf174fd860efc0/ivy/functional/ivy/nest.py>`_ module and the `meta <https://github.com/unifyai/ivy/blob/906ddebd9b371e7ae414cdd9b4bf174fd860efc0/ivy/functional/ivy/meta.py>`_ module, which have no instance method implementations.

The organization of these instance methods follows the same organizational structure as the files in the functional API.
The :class:`ivy.Array` class `inherits`_ from many category-specific array classes, such as `ArrayWithElementwise`_, each of which implement the category-specific instance methods.

Each instance method simply calls the functional API function internally, but passes in :code:`self._data` as the first *array* argument.
`ivy.Array.add`_ is a good example.
However, it's important to bear in mind that this is *not necessarily the first argument*, although in most cases it will be.
We also **do not** set the :code:`out` argument to :code:`self` for instance methods.
If the only array argument is the :code:`out` argument, then we do not implement this instance method.
For example, we do not implement an instance method for `ivy.zeros <https://github.com/unifyai/ivy/blob/1dba30aae5c087cd8b9ffe7c4b42db1904160873/ivy/functional/ivy/creation.py#L116>`_.

Given the simple set of rules which underpin how these instance methods should all be implemented, if a source-code implementation is not found, then this instance method is added `programmatically`_.
This serves as a helpful backup in cases where some methods are accidentally missed out.

The benefit of the source code implementations is that this makes the code much more readable, with important methods not being entirely absent from the code.
It also enables other helpful perks, such as auto-completions in the IDE etc.

Most special methods also simply wrap a corresponding function in the functional API, as `is the case`_ in the Array API Standard.
Examples include `__add__`_, `__sub__`_, `__mul__`_ and `__truediv__`_ which directly call :func:`ivy.add`, :func:`ivy.subtract`, :func:`ivy.multiply` and :func:`ivy.divide` respectively.
However, for some special methods such as `__setitem__`_, there are substantial differences between the backend frameworks which must be addressed in the :class:`ivy.Array` implementation.

Array Handling
--------------

When calling backend-specific functions such as :func:`torch.sin`, we must pass in :class:`ivy.NativeArray` instances.
For example, :func:`torch.sin` will throw an error if we try to pass in an :class:`ivy.Array` instance.
It must be provided with a :class:`torch.Tensor`, and this is reflected in the `backend type hints`_.

However, all Ivy functions must return :class:`ivy.Array` instances, which is reflected in the `Ivy type hints`_.
The reason we always return :class:`ivy.Array` instances from Ivy functions is to ensure that any subsequent Ivy code is fully framework-agnostic, with all operators performed on the returned array being handled by the special methods of the :class:`ivy.Array` class, and not the special methods of the backend :class:`ivy.NativeArray` class.

For example, calling any of (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.) on the array will result in (:meth:`__add__`, :meth:`__sub__`, :meth:`__mul__`, :meth:`__truediv__` etc.) being called on the array class.

For most special methods, calling them on the :class:`ivy.NativeArray` would not be a problem because all backends are generally quite consistent, but as explained above, for some functions such as `__setitem__`_ there are substantial differences which must be addressed in the :class:`ivy.Array` implementation in order to guarantee unified behaviour.

Given that all Ivy functions return :class:`ivy.Array` instances, all Ivy functions must also support :class:`ivy.Array` instances in the input, otherwise it would be impossible to chain functions together!

Therefore, most functions in Ivy must adopt the following pipeline:

#. convert all :class:`ivy.Array` instances in the input arguments to :class:`ivy.NativeArray` instances
#. call the backend-specific function, passing in these :class:`ivy.NativeArray` instances
#. convert all of the :class:`ivy.NativeArray` instances which are returned from the backend function back into :class:`ivy.Array` instances, and return

Given the repeating nature of these steps, this is all entirely handled in the `inputs_to_native_arrays`_ and `outputs_to_ivy_arrays`_ wrappers, as explained in the :ref:`Function Wrapping` section.

All Ivy functions *also* accept :class:`ivy.NativeArray` instances in the input.
This is for a couple of reasons.
Firstly, :class:`ivy.Array` instances must be converted to :class:`ivy.NativeArray` instances anyway, and so supporting them in the input is not a problem.
Secondly, this makes it easier to combine backend-specific code with Ivy code, without needing to explicitly wrap any arrays before calling sections of Ivy code.

Therefore, all input arrays to Ivy functions have type :code:`Union[ivy.Array, ivy.NativeArray]`, whereas the output arrays have type :class:`ivy.Array`.
This is further explained in the :ref:`Function Arguments` section.

However, :class:`ivy.NativeArray` instances are not permitted for the :code:`out` argument, which is used in most functions.
This is because the :code:`out` argument dictates the array to which the result should be written, and so it effectively serves the same purpose as the function return.
This is further explained in the :ref:`Inplace Updates` section.

As a final point, extra attention is required for *compositional* functions, as these do not directly defer to a backend implementation.
If the first line of code in a compositional function performs operations on the input array, then this will call the special methods on an :class:`ivy.NativeArray` and not on an :class:`ivy.Array`.
For the reasons explained above, this would be a problem.

Therefore, all compositional functions have a separate piece of `wrapped logic`_ to ensure that all :class:`ivy.NativeArray` instances are converted to :class:`ivy.Array` instances before entering into the compositional function.

**Round Up**

This should have hopefully given you a good feel for the different types of arrays, and how these are handled in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `arrays channel`_ or in the `arrays forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/tAlDPnWcLDE" class="video">
    </iframe>