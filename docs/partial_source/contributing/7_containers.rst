Containers
==========

.. _`ivy.Container`: https://github.com/unifyai/ivy/blob/e47a7b18628aa73ba0c064d3d07352a7ab672bd1/ivy/container/container.py#L25
.. _`dict`: https://github.com/unifyai/ivy/blob/e47a7b18628aa73ba0c064d3d07352a7ab672bd1/ivy/container/base.py#L56
.. _`__setitem__`: https://github.com/unifyai/ivy/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/ivy/array/__init__.py#L234

The `ivy.Container`_ inherits from `dict`_, and is useful for storing nested data.
For example, the container is equally suitable for storing batches of training data,
or for storing the weights of a network.

The methods of the :code:`ivy.Container` class are more varied than those of the :code:`ivy.Array`.
All methods of the :code:`ivy.Array` are instance methods,
and almost all of them directly wrap a function in the functional API.
There are a few exceptions, which wrap instance methods on the backend :code:`ivy.NativeArray`, such as `__setitem__`_.

For the :code:`ivy.Container`, there are also methods which are specific to the container itself,
for performing nested operations on the leaves of the container for example.
In addition, there are also static methods, which are not present in the :code:`ivy.Array`.
Overall, this results in the following five mutually exclusive groups of methods.
Each of these are explained in the following sub-sections.

#. Container instance methods
#. Container static methods
#. API instance methods
#. API special methods
#. API static methods

Container Instance Methods
--------------------------

ToDo: write

Container Static Methods
------------------------

ToDo: write

API Instance Methods
--------------------

ToDo: write

API Special Methods
--------------------

ToDo: write

API Static Methods
------------------

ToDo: write