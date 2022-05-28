Containers
==========

.. _`ivy.Container`: https://github.com/unifyai/ivy/blob/e47a7b18628aa73ba0c064d3d07352a7ab672bd1/ivy/container/container.py#L25
.. _`dict`: https://github.com/unifyai/ivy/blob/e47a7b18628aa73ba0c064d3d07352a7ab672bd1/ivy/container/base.py#L56
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
.. _`ivy.Container.add`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/elementwise.py#L92
.. _`ivy.Container.tan`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/elementwise.py#L1259
.. _`ivy.Container.roll`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/container/manipulation.py#L158
.. _`static method is added`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/__init__.py#L199
.. _`instance method is added`: https://github.com/unifyai/ivy/blob/8d1eef71522be7f98b601e5f97bb2c54142795b3/ivy/__init__.py#L173
.. _`inherits`: https://github.com/unifyai/ivy/blob/8cbffbda9735cf16943f4da362ce350c74978dcb/ivy/container/container.py#L25
.. _`ContainerWithElementwise`: https://github.com/unifyai/ivy/blob/8cbffbda9735cf16943f4da362ce350c74978dcb/ivy/container/elementwise.py#L12
.. _`__repr__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4588
.. _`__getattr__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4782
.. _`__setattr__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4790
.. _`__getitem__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4842
.. _`__setitem__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4884
.. _`__contains__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4904
.. _`__getstate__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4912
.. _`__setstate__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/base.py#L4927
.. _`implemented`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/container.py#L98
.. _`__add__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/container.py#L115
.. _`__sub__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/container.py#L121
.. _`__mul__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/container.py#L127
.. _`__truediv__`: https://github.com/unifyai/ivy/blob/36e32ca1f17ef1e4c1b986599b45974156c19737/ivy/container/container.py#L133


The `ivy.Container`_ inherits from `dict`_, and is useful for storing nested data.
For example, the container is equally suitable for storing batches of training data,
or for storing the weights of a network.

The methods of the :code:`ivy.Container` class are more varied than those of the :code:`ivy.Array`.
All methods of the :code:`ivy.Array` are instance methods,
and almost all of them directly wrap a function in the functional API.

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
In cases where there are no containers passed,
`ivy.Container.multi_map_in_static_method`_ will simply call the function once on the arguments provided.

A few examples are `ivy.Container.static_add`_, `ivy.Container.static_tan`_ and `ivy.Container.static_roll`_.

As with :code:`ivy.Array`,
given the simple set of rules which underpin how these static methods should all be implemented,
if a source-code implementation is not found,
then this `static method is added`_ programmatically.
This serves as a helpful backup in cases where some static methods are accidentally missed out.

The benefit of the source code implementations is that this makes the code much more readable,
without important methods being entirely absent.
It also enables other helpful perks, such as auto-completions in the IDE etc.

API Instance Methods
--------------------

The *API* instance methods serve a similar purpose to the instance methods of the :code:`ivy.Array` class.
They enable functions in Ivy's functional API to be called as instance methods on the :code:`ivy.Container` class.
The difference is that with the :code:`ivy.Container`,
the API function is applied recursively to all the leaves of the container.

Under the hood, every *instance* method calls the corresponding *static* method.
For example, `ivy.Container.add`_ calls :code:`ivy.Container.static_add`,
`ivy.Container.tan`_ calls :code:`ivy.Container.static_tan`,
and `ivy.Container.roll`_ calls :code:`ivy.Container.static_roll`.

As is the case for :code:`ivy.Array`,
the organization of these instance methods follows the same organizational structure as the
files in the functional API.
The :code:`ivy.Container` class `inherits`_ from many category-specific array classes,
such as `ContainerWithElementwise`_, each of which implement the category-specific instance methods.

Again, as with :code:`ivy.Array`,
given the simple set of rules which underpin how these instance methods should all be implemented,
if a source-code implementation is not found,
then this `instance method is added`_ programmatically.
Again, this serves as a helpful backup in cases where some static methods are accidentally missed out.

Again, the benefit of the source code implementations is that this makes the code much more readable,
without important methods being entirely absent.
It also enables other helpful perks, such as auto-completions in the IDE etc.

API Special Methods
--------------------

As is the case for the :code:`ivy.Array`,
most special methods of the :code:`ivy.Container` simply wrap a corresponding function in the functional API.

The special methods which **do not** wrap functions in the functional API are implemented in `ContainerBase`_,
which is the base abstract class for all containers.
These special methods include
`__repr__`_ which controls how the container is printed in the terminal,
`__getattr__`_ that enables keys in the underlying :code:`dict` to be queried as attributes,
`__setattr__`_ that enables attribute setting to update the underlying :code:`dict`,
`__getitem__`_ that enables the underlying :code:`dict` to be queried via a chain of keys,
`__setitem__`_ that enables the underlying :code:`dict` to be set via a chain of keys,
`__contains__`_ that enables us to check for chains of keys in the underlying :code:`dict`,
and `__getstate__`_ and `__setstate__`_ which combined enable the container to be pickled and unpickled.

As for the special methods which **do** simply wrap corresponding functions in the functional API,
these are `implemented`_ in the main :code:`ivy.Container` class.

These special methods all directly wrap the corresponding API *static* :code:`ivy.Container` method.

Examples include `__add__`_, `__sub__`_, `__mul__`_ and `__truediv__`_ which directly call
:code:`ivy.Container.static_add`, :code:`ivy.Container.static_subtract`,
:code:`ivy.Container.static_multiply` and :code:`ivy.Container.static_divide` respectively.