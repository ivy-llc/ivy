Containers
==========

.. _`ivy.Container`: https://github.com/unifyai/ivy/blob/e47a7b18628aa73ba0c064d3d07352a7ab672bd1/ivy/container/container.py#L25
.. _`dict`: https://github.com/unifyai/ivy/blob/e47a7b18628aa73ba0c064d3d07352a7ab672bd1/ivy/container/base.py#L56
.. _`__setitem__`: https://github.com/unifyai/ivy/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/ivy/array/__init__.py#L234
.. _`ivy.Container.map`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L4030
.. _`ivy.Container.all_true`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L1490
.. _`ivy.Container.to_iterator`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L3019
.. _`ContainerBase`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L56
.. _`ivy.Container.multi_map`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L593
.. _`ivy.Container.diff`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L396
.. _`ivy.Container.common_key_chains`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L663
.. _`ivy.Container.multi_map_in_static_method`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/base.py#L167
.. _`ivy.Container.static_add`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/elementwise.py#L71
.. _`ivy.Container.static_tan`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/elementwise.py#L1240
.. _`ivy.Container.static_roll`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/manipulation.py#L135

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

Container instance methods are methods which are specific to the container itself.
A few examples include `ivy.Container.map`_ which is used for mapping a function to all leaves of the container,
`ivy.Container.all_true`_ which determines if all container leaves evaluate to boolean `True`,
and `ivy.Container.to_iterator`_ returns an iterator for traversing the leaves of the container.

There are many more examples, check out the abstract `ContainerBase`_ class to see some more!

Container Static Methods
------------------------

Container static methods are also methods which are specific to containers,
but which generally operate across *multiple* containers rather than a single container.
This underpins the decision to not bind these method to a single container instance,
and instead implement them as *static* methods.

A few examples include `ivy.Container.multi_map`_
which is used for mapping a function to all leaves of *multiple* containers with the same nested structure,
`ivy.Container.diff`_ which displays the difference in nested structure between multiple containers,
and `ivy.Container.common_key_chains`_ which returns the nested structure that is common to all containers.

There are many more examples, check out the abstract `ContainerBase`_ class to see some more!

API Static Methods
------------------

Unlike the :code:`ivy.Array` class, the :code:`ivy.Container` also implements all functions in the functional API as
*static* methods. The main reason for this is to support the *nestable* property of all functions in the API,
which is explained in detail in the :ref:`Function Types` section.

To recap, what this means is that every function can arbitrarily accept :code:`ivy.Container` instances for **any**
of the arguments, and in such cases the function will automatically be mapped to the leaves of this container.
When multiple containers are passed, this mapping is only applied to their shared nested structure,
with the mapping applied to each of these leaves.

In such cases, the function in the functional API defers to the *static* :code:`ivy.Container` implementation.
Under the hood, `ivy.Container.multi_map_in_static_method`_ enables us to pass in arbitrary combinations of containers
and non-containers, and perform the correct mapping across the leaves.
Internally, :code:`ivy.Container.multi_map_in_static_method` calls `ivy.Container.multi_map`_.

A few examples are `ivy.Container.static_add`_, `ivy.Container.static_tan`_ and `ivy.Container.static_roll`_

API Instance Methods
--------------------

The *API* instance methods serve a similar purpose to the instance methods of the :code:`ivy.Array` class.
They enable functions in Ivy's functional API to be called as instance methods.
The difference is that with the :code:`ivy.Container`,
the API function is applied recursively to all the leaves of the container.

Under the hood, either `ivy.Container.map`_ or `ivy.Container.multi_map`_ is used.
For example, X makes use of the former, while Y makes use of the latter.

As is the case for :code:`ivy.Array`, [inheritance explanation].

API Special Methods
--------------------

ToDo: write