Arrays
======

.. _`empty class`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/__init__.py#L8
.. _`overwritten`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/functional/backends/torch/__init__.py#L11
.. _`self._data`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/array/__init__.py#L89
.. _`ArrayWithElementwise`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/array/elementwise.py#L12
.. _`ivy.Array.add`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/array/elementwise.py#L22
.. _`programmatically`: https://github.com/unifyai/ivy/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/ivy/__init__.py#L148

There are two types of array in Ivy, there is the :code:`ivy.NativeArray` and also the :code:`ivy.Array`.

Native Array
------------

The :code:`ivy.NativeArray` is simply a placeholder class for a backend-specific array class,
such as :code:`np.ndarray`, :code:`tf.Tensor` and :code:`torch.Tensor`

When no framework is set, this is an `empty class`_.
When a framework is set, this is `overwritten`_ with the backend-specific array class.

Ivy Array
---------

The :code:`ivy.Array` is a simple wrapper class, which wraps around the :code:`ivy.NativeArray`,
storing it in `self._data`_.

All functions in the Ivy API which accept *at least one array argument* in the input are implemented as instance methods
in the :code:`ivy.Array` class.

The organization of these instance methods follows the same organizational structure as the
files in the functional API.
The :code:`ivy.Array` class inherits from many category-specific array classes, such as `ArrayWithElementwise`_,
each of which implement the category-specific instance methods.

Each instance method simply calls the functional API function internally,
but passes in :code:`self` as the first array argument. `ivy.Array.add`_ is a good example.

Given the simple set of rules which underpin how these instance methods should all be implemented,
if a source-code implementation is not found, then this instance method is added `programmatically`_.
This serves as a helpful backup in cases where some functions are accidentally missed out.

Array Handling
--------------

# ToDo: write