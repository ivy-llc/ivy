Containers
==========

.. _`ivy.Container`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L52
.. _`dict`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L51
.. _`ivy.Container.cont_map`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3070
.. _`ivy.Container.cont_all_true`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L1592
.. _`ivy.Container.cont_to_iterator`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L2043
.. _`ContainerBase`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L51
.. _`ivy.Container.cont_multi_map`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L623
.. _`ivy.Container.cont_diff`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L427
.. _`ivy.Container.cont_common_key_chains`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L741
.. _`ivy.Container.cont_multi_map_in_function`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L162
.. _`ivy.Container.tan`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/elementwise.py#L7347
.. _`ivy.Container.roll`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/manipulation.py#L927
.. _`instance method is added`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/__init__.py#L683
.. _`inherits`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L52
.. _`ContainerWithElementwise`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/elementwise.py#L9
.. _`__repr__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3629
.. _`__getattr__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3860
.. _`__setattr__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3882
.. _`__getitem__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3934
.. _`__setitem__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3976
.. _`__contains__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L3996
.. _`__getstate__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L4004
.. _`__setstate__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/base.py#L4019
.. _`implemented`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L133
.. _`__add__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L191
.. _`__sub__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L290
.. _`__mul__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L389
.. _`__truediv__`: https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/container/container.py#L399
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`containers thread`: https://discord.com/channels/799879767196958751/1189906066549506048


The `ivy.Container`_ inherits from `dict`_, and is useful for storing nested data.
For example, the container is equally suitable for storing batches of training data, or for storing the weights of a network.

The methods of the :class:`ivy.Container` class are more varied than those of the :class:`ivy.Array`.
All methods of the :class:`ivy.Array` are instance methods, and almost all of them directly wrap a function in the functional API.

For the :class:`ivy.Container`, there are also methods which are specific to the container itself, for performing nested operations on the leaves of the container for example.

Overall, this results in the following three mutually exclusive groups of :class:`ivy.Container` methods.
Each of these are explained in the following sub-sections.

#. Container instance methods
#. API instance methods
#. API special methods

Container Instance Methods
--------------------------

Container instance methods are methods which are specific to the container itself.
A few examples include `ivy.Container.cont_map`_ which is used for mapping a function to all leaves of the container, `ivy.Container.cont_all_true`_ which determines if all container leaves evaluate to boolean `True`, and `ivy.Container.cont_to_iterator`_ which returns an iterator for traversing the leaves of the container.

There are many more examples, check out the abstract `ContainerBase`_ class to see some more!

API Instance Methods
--------------------

The *API* instance methods serve a similar purpose to the instance methods of the :class:`ivy.Array` class.
They enable functions in Ivy's functional API to be called as instance methods on the :class:`ivy.Container` class.
The difference is that with the :class:`ivy.Container`, the API function is applied recursively to all the leaves of the container.
The :class:`ivy.Container` instance methods should **exactly match** the instance methods of the :class:`ivy.Array`, both in terms of the methods implemented and the argument which :code:`self` replaces in the function being called.
This means :code:`self` should always replace the first array argument in the function.
`ivy.Container.add <https://github.com/unifyai/ivy/blob/1dba30aae5c087cd8b9ffe7c4b42db1904160873/ivy/container/elementwise.py#L158>`_ is a good example.

However, as with the :class:`ivy.Array` class, it's important to bear in mind that this is *not necessarily the first argument*, although in most cases it will be.
We also **do not** set the :code:`out` argument to :code:`self` for instance methods.
If the only array argument is the :code:`out` argument, then we do not implement this instance method.
For example, we do not implement an instance method for `ivy.zeros <https://github.com/unifyai/ivy/blob/1dba30aae5c087cd8b9ffe7c4b42db1904160873/ivy/functional/ivy/creation.py#L116>`_.

As is the case for :class:`ivy.Array`, the organization of these instance methods follows the same organizational structure as the files in the functional API.
The :class:`ivy.Container` class `inherits`_ from many category-specific array classes, such as `ContainerWithElementwise`_, each of which implements the category-specific instance methods.

As with :class:`ivy.Array`, given the simple set of rules which underpin how these instance methods should all be implemented, if a source-code implementation is not found, then this `instance method is added`_ programmatically. This serves as a helpful backup in cases where some instance methods are accidentally missed out.

Again, the benefit of the source code implementations is that this makes the code much more readable, with important methods not being entirely absent from the code.
It also enables other helpful perks, such as auto-completions in the IDE etc.

API Special Methods
--------------------

All non-operator special methods are implemented in `ContainerBase`_, which is the abstract base class for all containers.
These special methods include `__repr__`_ which controls how the container is printed in the terminal, `__getattr__`_ that primarily enables keys in the underlying :code:`dict` to be queried as attributes, whereas if no attribute, item or method is found which matches the name provided on the container itself, then the leaves will also be recursively traversed, searching for the attribute.
If it turns out to be a callable function on the leaves, then it will call the function on each leaf and update the leaves with the returned results, for a more detailed explanation with examples, see the code block below.
`__setattr__`_ that enables attribute setting to update the underlying :code:`dict`, `__getitem__`_ that enables the underlying :code:`dict` to be queried via a chain of keys, `__setitem__`_ that enables the underlying :code:`dict` to be set via a chain of keys, `__contains__`_ that enables us to check for chains of keys in the underlying :code:`dict`, and `__getstate__`_ and `__setstate__`_ which combined enable the container to be pickled and unpickled.

.. code-block:: python

    x = ivy.Container(a=ivy.array([0.]), b=ivy.Container(a=ivy.array([[0.]]), b=ivy.array([1., 2., 3.])))
    print(x.shape)
    {
        a: [
            1
        ],
        b: {
            a: [
                1,
                1
            ],
            b: [
                3
            ]
        }
    }

    print(x.ndim)
    {
        a: 1,
        b: {
            a: 2,
            b: 1
        }
    }


    num_dims = x.shape.__len__()
    print(num_dims)
    {
        a: 1,
        b: {
            a: 2,
            b: 1
        }
    }

    print(len(x.shape))
    # doesn't work because Python in low-level C has a restriction on the return type of `len` to be `int`

    print(num_dims.real)
    {
        a: 1,
        b: {
            a: 2,
            b: 1
        }
    }

    print(bin(num_dims))
    # doesn't work because some Python built-in functions have enforcement on input argument types

    # external method flexibility enables positional and keyword arguments to be passed into the attribute
    y = ivy.Container(l1=[1, 2, 3], c1=ivy.Container(l1=[3, 2, 1], l2=[4, 5, 6]))

    print(y.__getattr__("count", 1))
    {
        c1: {
            l1: 1,
            l2: 0
        },
        l1: 1
    }

    print(y.count(1))
    # doesn't work since essentially the argument 1 won't be passed to `__getattr__`

    print(y.__getattr__("__add__", [10]))
    {
        c1: {
            l1: [
                3,
                2,
                1,
                10
            ],
            l2: [
                4,
                5,
                6,
                10
            ]
        },
        l1: [
            1,
            2,
            3,
            10
        ]
    }

As for the special methods which are `implemented`_ in the main :class:`ivy.Container` class, they all make calls to the corresponding standard operator functions.

As a result, the operator functions will make use of the special methods of the lefthand passed input objects if available, otherwise it will make use of the reverse special method of the righthand operand.
For instance, if the lefthand operand at any given leaf of the container in an :class:`ivy.Array`, then the operator function will make calls to the special methods of this array object.
As explained in the `Arrays <arrays.rst>`_ section of the Deep Dive, these special methods will in turn call the corresponding functions from the ivy functional API.

Examples include `__add__`_, `__sub__`_, `__mul__`_ and `__truediv__`_ which will make calls to :func:`ivy.add`, :func:`ivy.subtract`, :func:`ivy.multiply` and :func:`ivy.divide` respectively if the lefthand operand is an :class:`ivy.Array` object.
Otherwise, these special methods will be called on whatever objects are at the leaves of the container, such as int, float, :class:`ivy.NativeArray` etc.

Nestable Functions
------------------

As introduced in the `Function Types <function_types.rst>`_ section, most functions in Ivy are *nestable*, which means that they can accept :class:`ivy.Container` instances in place of **any** of the arguments.

Here, we expand on this explanation.
Please check out the explanation in the `Function Types <function_types.rst>`_ section first.

**Explicitly Nestable Functions**

The *nestable* behaviour is added to any function which is decorated with the `handle_nestable <https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/ivy/func_wrapper.py#L429>`_ wrapper.
This wrapper causes the function to be applied at each leaf of any containers passed in the input.
More information on this can be found in the `Function Wrapping <https://github.com/unifyai/ivy/blob/b725ed10bca15f6f10a0e5154af10231ca842da2/docs/partial_source/deep_dive/function_wrapping.rst>`_ section of the Deep Dive.

Additionally, any nestable function which returns multiple arrays, will return the same number of containers for its container counterpart.
This property makes the function symmetric with regards to the input-output behavior, irrespective of whether :class:`ivy.Array` or :class:`ivy.Container` instances are used.
Any argument in the input can be replaced with a container without changing the number of inputs, and the presence or absence of ivy.Container instances in the input should not change the number of return values of the function.
In other words, if containers are detected in the input, then we should return a separate container for each array that the function would otherwise return.

The current implementation checks if the leaves of the container have a list of arrays.
If they do, this container is then unstacked to multiple containers(as many as the number of arrays), which are then returned inside a list.

**Implicitly Nestable Functions**

*Compositional* functions are composed of other nestable functions, and hence are already **implicitly nestable**.
So, we do not need to explicitly wrap it at all.

Let's take the function :func:`ivy.cross_entropy` as an example.
The internally called functions are: :func:`ivy.clip`, :func:`ivy.log`, :func:`ivy.sum` and :func:`ivy.negative`, each of which are themselves *nestable*.

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = -1,
        epsilon: float =1e-7,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return ivy.negative(ivy.sum(log_pred * true, axis, out=out), out=out)

Therefore, when passing an :class:`ivy.Container` instance in the input, each internal function will, in turn, correctly handle the container, and return a new container with the correct operations having been performed.
This makes it very easy and intuitive to debug the code, as the code is stepped through chronologically.
In effect, all leaves of the input container are being processed concurrently, during the computation steps of the :func:`ivy.cross_entropy` function.

However, what if we had added the `handle_nestable <https://github.com/unifyai/ivy/blob/5f58c087906a797b5cb5603714d5e5a532fc4cd4/ivy/func_wrapper.py#L407>`_ wrapping as a decorator directly to the function :func:`ivy.cross_entropy`?

In this case, the :func:`ivy.cross_entropy` function would itself be called multiple times, on each of the leaves of the container.
The functions :func:`ivy.clip`, :func:`ivy.log`, :func:`ivy.sum` and :func:`ivy.negative` would each only consume and return arrays, and debugging the :func:`ivy.cross_entropy` function would then become less intuitively chronological, with each leaf of the input container now processed sequentially, rather than concurrently.

Therefore, our approach is to **not** wrap any compositional functions which are already *implicitly nestable* as a result of the *nestable* functions called internally.

**Explicitly Nestable Compositional Functions**

There may be some compositional functions which are not implicitly nestable for some reason, and in such cases adding the explicit `handle_nestable <https://github.com/unifyai/ivy/blob/5f58c087906a797b5cb5603714d5e5a532fc4cd4/ivy/func_wrapper.py#L407>`_ wrapping may be necessary.
One such example is the :func:`ivy.linear` function which is not implicitly nestable despite being compositional. This is because of the use of special functions like :func:`__len__` and :func:`__list__` which, among other functions, are not nestable and can't be made nestable.
But we should try to avoid this, in order to make the flow of computation as intuitive to the user as possible.

When tracing the code, the computation graph is **identical** in either case, and there will be no implications on performance whatsoever.
The implicit nestable solution may be slightly less efficient in eager mode, as the leaves of the container are traversed multiple times rather than once, but if performance is of concern then the code should always be traced in any case.
The distinction is only really relevant when stepping through and debugging with eager mode execution, and for the reasons outlined above, the preference is to keep compositional functions implicitly nestable where possible.

**Shared Nested Structure**

When the nested structures of the multiple containers are *shared* but not *identical*, then the behaviour of the nestable function is a bit different.
Containers have *shared* nested structures if all unique leaves in any of the containers are children of a nested structure which is shared by all other containers.

Take the example below, the nested structures of containers :code:`x` and :code:`y` are shared but not identical.

.. code-block:: python

    x = ivy.Container(a={'b': 2, 'c': 4}, d={'e': 6, 'f': 9})
    y = ivy.Container(a=2, d=3)

The shared key chains (chains of keys, used for indexing the container) are :code:`a` and :code:`d`.
The key chains unique to :code:`x` are :code:`a/b`, :code:`a/c`, :code:`d/e` and :code:`d/f`.
The unique key chains all share the same base structure as all other containers (in this case only one other container, :code:`y`).
Therefore, the containers :code:`x` and :code:`y` have a shared nested structure.

When calling *nestable* functions on containers with non-identical structure, then the shared leaves of the shallowest container are broadcast to the leaves of the deepest container.

It's helpful to look at an example:

.. code-block:: python

    print(x / y)
    {
        a: {
          b: 1.0,
          c: 2.0
        },
        d: {
          e: 2.0,
          f: 3.0
        }
    }

In this case, the integer at :code:`y.a` is broadcast to the leaves :code:`x.a.b` and :code:`x.a.c`, and the integer at :code:`y.d` is broadcast to the leaves :code:`x.d.e` and :code:`x.d.f`.

Another example of containers with shared nested structure is given below:

.. code-block:: python

    x = ivy.Container(a={'b': 2, 'c': 4}, d={'e': 6, 'f': 8})
    y = ivy.Container(a=2, d=3)
    z = ivy.Container(a={'b': 10, 'c': {'g': 11, 'h': 12}}, d={'e': 13, 'f': 14})

Adding these containers together would result in the following:

.. code-block:: python

    print(x + y + z)
    {
        a: {
          b: 14,
          c: {
            g: 17,
            h: 18,
          }
        },
        d: {
          e: 22,
          f: 25
        }
    }

An example of containers which **do not** have a shared nested structure is given below:

.. code-block:: python

    x = ivy.Container(a={'b': 2, 'c': 4}, d={'e': 6, 'f': 8})
    y = ivy.Container(a=2, d=3, g=4)
    z = ivy.Container(a={'b': 10, 'c': {'g': 11, 'h': 12}}, d={'e': 13, 'g': 14})

This is for three reasons, (a) the key chain :code:`g` is not shared by any container other than :code:`y`, (b) the key chain :code:`d/f` for :code:`x` is not present in :code:`z` despite :code:`d` not being a non-leaf node in :code:`z`, and likewise the key chain :code:`d/g` for :code:`z` is not present in :code:`x` despite :code:`d` not being a non-leaf node in :code:`x`.

**Round Up**

This should have hopefully given you a good feel for containers, and how these are handled in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `containers thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/oHcoYFi2rvI" class="video">
    </iframe>
