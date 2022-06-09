Function Types
==============

.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L137
.. _`backend setting`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/framework_handler.py#L205
.. _`at import time`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/__init__.py#L114
.. _`add_ivy_array_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/array/wrapping.py#L26
.. _`add_ivy_container_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L69
.. _`from being added`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L78
.. _`_function_w_arrays_n_out_handled`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L166
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`ivy.set_backend`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/backend_handler.py#L153
.. _`ivy.get_backend`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/backend_handler.py#L211
.. _`ivy.nested_map`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/nest.py#L333
.. _`ivy.index_nest`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/nest.py#L15
.. _`ivy.set_default_dtype`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/data_type.py#L536
.. _`ivy.set_default_device`: https://github.com/unifyai/ivy/blob/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy/device.py#L464
.. _`submodules`: https://github.com/unifyai/ivy/tree/30b7ca4f8a50a52f51884738fe7323883ce891bd/ivy/functional/ivy
.. _`nest.py`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/nest.py
.. _`ivy.default`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L622
.. _`ivy.cache_fn`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L747
.. _`ivy.stable_divide`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/general.py#L928
.. _`ivy.can_cast`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/data_type.py#L22
.. _`ivy.dtype`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/data_type.py#L140
.. _`ivy.dev`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/device.py#L132
.. _`ivy.default_dtype`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/data_type.py#L484
.. _`ivy.get_all_arrays_on_dev`: https://github.com/unifyai/ivy/blob/f18df2e19d6a5a56463fa1a15760c555a30cb2b2/ivy/functional/ivy/device.py#L71
.. _`function types discussion`: https://github.com/unifyai/ivy/discussions/1312
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`function types channel`: https://discord.com/channels/799879767196958751/982737839861145630

Firstly, we explain the difference between *primary*, *compositional*, *mixed* and *standalone* functions.
These four function categorizations are all **mutually exclusive**,
and combined they constitute the set of **all** functions in Ivy, as outlined in the simple Venn diagram below.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/four_function_types.png?raw=true
   :align: center
   :width: 50%

Primary Functions
-----------------

*Primary* functions are essentially the lowest level building blocks in Ivy. Each primary function has a unique
backend-specific implementation for each backend specified in
:code:`ivy/functional/backends/backend_name/category_name.py`. These are generally implemented as light wrapping
around an existing function in the backend framework, which serves a near-identical purpose.

Primary functions must both be specified in :code:`ivy/functional/ivy/category_name.py` and also in each of
the backend files :code:`ivy/functional/backends/backend_name/category_name.py`

The function in :code:`ivy/functional/ivy/category_name.py` includes the type hints, docstring and docstring examples
(explained in more detail in the subsequent sections), but does not include an actual implementation.

Instead, in :code:`ivy/functional/ivy/category_name.py`, primary functions simply defer to the backend-specific
implementation.

For example, the code for :code:`ivy.tan` in :code:`ivy/functional/ivy/elementwise.py`
(with docstrings removed) is given below:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.current_backend(x).tan(x, out=out)

The backend-specific implementation of :code:`ivy.tan`  for PyTorch in
:code:`ivy/functional/backends/torch/elementwise.py` is given below:

.. code-block:: python

    def tan(
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.tan(x, out=out)

The reason that the Ivy implementation has type hint :code:`Union[ivy.Array, ivy.NativeArray]` but PyTorch
implementation has :code:`torch.Tensor` is explained in the :ref:`Arrays` section.
Likewise, the reason that the :code:`out` argument in the Ivy implementation has array type hint :code:`ivy.Array`
whereas :code:`x` has :code:`Union[ivy.Array, ivy.NativeArray]` is also explained in the :ref:`Arrays` section.

Compositional Functions
-----------------------

*Compositional* functions on the other hand **do not** have backend-specific implementations. They are implemented as
a *composition* of other Ivy functions,
which themselves can be either compositional, primary or mixed (explained below).

Therefore, compositional functions are only implemented in :code:`ivy/functional/ivy/category_name.py`, and there are no
implementations in any of the backend files :code:`ivy/functional/backends/backend_name/category_name.py`

For example, the implementation of :code:`ivy.cross_entropy` in :code:`ivy/functional/ivy/losses.py`
(with docstrings removed) is given below:

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return ivy.negative(ivy.sum(log_pred * true, axis), out=out)


Mixed Functions
---------------

Some functions have some backend-specific implementations in
:code:`ivy/functional/backends/backend_name/category_name.py`, but not for all backends.
To support backends that do not have a backend-specific implementation,
a compositional implementation is also provided in :code:`ivy/functional/ivy/category_name.py`.
Because these functions include both a compositional implementation and also at least one backend-specific
implementation, these functions are referred to as *mixed*.

When using ivy without a backend set explicitly (for example :code:`ivy.set_backend()` has not been called),
then the function called is always the one implemented in :code:`ivy/functional/ivy/category_name.py`.
For *primary* functions, then :code:`ivy.current_backend(array_arg).func_name(...)`
will call the backend-specific implementation in :code:`ivy/functional/backends/backend_name/category_name.py`
directly. However, as just explained, *mixed* functions implement a compositional approach in
:code:`ivy/functional/ivy/category_name.py`, without deferring to the backend.
Therefore, when no backend is explicitly set,
then the compositional implementation is always used for *mixed* functions,
even for backends that have a more efficient backend-specific implementation.
Typically the backend should always be set explicitly though (using :code:`ivy.set_backend()` for example),
and in this case the efficient backend-specific implementation will always be used if it exists.

Standalone Functions
---------------------

*Standalone* functions are functions which do not reference any other *primary*,
*compositional* or *mixed* functions whatsoever.

By definition, standalone functions can only reference themselves or other standalone functions.
Most commonly, these functions are *convenience* functions (see below).

As a first example, every function in the `nest.py`_ module is a standalone function.
All of these either: (a) reference no other function at all, (b) only reference themselves recursively,
or (c) reference other standalone functions.

A few other examples outside of the :code:`nest.py` module are:
`ivy.default`_ which simply returns :code:`x` if it exists else the default value,
`ivy.cache_fn`_ which wraps a function such that when :code:`cache=True` is passed,
then a previously cached output is returned,
and `ivy.stable_divide`_ which simply adds a small constant to the denominator of the division.

Nestable Functions
------------------

*Nestable* functions are functions which can accept :code:`ivy.Container` instances in place
of **any** of the arguments. Multiple containers can also be passed in for multiple arguments at the same time,
provided that the containers share a common nested structure.
If an :code:`ivy.Container` is passed, then the function is applied to all of the
leaves of the container, with the container leaf values passed into the function at the corresponding arguments.
In this case, the function will return an :code:`ivy.Container` in the output.
*Primary*, *compositional*, *mixed*, and *standalone* functions can all *also* be nestable.
This categorization is **not** mutually exclusive, as outlined by the Venn diagram below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/nestable.png?raw=true
   :align: center
   :width: 50%

The *nestable* property makes it very easy to write a single piece of code that can deal either with individual
arguments or arbitrary batches of nested arguments. This is very useful in machine learning,
where batches of different training data often need to be processed concurrently.
Another example is when the same operation must be performed on each weight in a network.
This *nestable* property of Ivy functions means that the same function can be used for any of these use cases
without modification.

This added support for handling :code:`ivy.Container` instances is all handled automatically when `_wrap_function`_
is applied to every function (except those appearing in `NON_WRAPPED_FUNCTIONS`_)
in the :code:`ivy` module during `backend setting`_.
This function wrapping process is covered in a bit more detail in the :ref:`Function Wrapping` section.

Under the hood, the :code:`ivy.Container` API static methods are called when :code:`ivy.Container` instances are passed
in as inputs to functions in the functional API. This is explained in more detail in the :ref:`Containers` section.

**Shared Nested Structure**

NOTE - implementing the behaviour for shared nested structures is a work in progress,
the master branch will soon support all of the examples given below, but not yet 🚧

When the nested structures of the multiple containers are *shared* but not *identical*,
then the behaviour of the nestable function is a bit different.
Containers have *shared* nested structures if all unique leaves in any of the containers
are children of a nested structure which is shared by all other containers.

Take the example below, the nested structures of containers :code:`x` and :code:`y`
are shared but not identical.

.. code-block:: python

    x = ivy.Container(a={'b': 2, 'c': 4}, d={'e': 6, 'f': 8})
    y = ivy.Container(a=2, d=3)

The shared key chains (chains of keys, used for indexing the container)
are :code:`a` and :code:`d`. The key chains unique to :code:`x` are :code:`a/b`, :code:`a/c`,
:code:`d/e` and :code:`d/f`. The unique key chains all share the same base structure as
all other containers (in this case only one other container, :code:`y`).
Therefore, the containers :code:`x` and :code:`y` have shared nested structure.

When calling *nestable* functions on containers with non-identical structure,
then the shared leaves of the shallowest container are broadcast to the leaves of the
deepest container.

It's helpful to look at an example:

.. code-block:: python

    print(x / y)
    {
        a: {
          b: 1,
          c: 2
        },
        d: {
          e: 3,
          f: 2.67
        }
    }

In this case, the integer at :code:`y.a` is broadcast to the leaves :code:`x.a.b` and
:code:`x.a.c`, and the integer at :code:`y.d` is broadcast to the leaves :code:`x.d.e`
and :code:`x.d.f`.

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

An example of containers which **do not** have shared nested structure is given below:

.. code-block:: python

    x = ivy.Container(a={'b': 2, 'c': 4}, d={'e': 6, 'f': 8})
    y = ivy.Container(a=2, d=3, g=4)
    z = ivy.Container(a={'b': 10, 'c': {'g': 11, 'h': 12}}, d={'e': 13, 'g': 14})

This is for three reasons, (a) the key chain :code:`g` is not shared by any container other
than :code:`y`, (b) the key chain :code:`d/f` for :code:`x` is not present in
:code:`z` despite :code:`d` not being a non-leaf node in :code:`z`,
and likewise the key chain :code:`d/g` for :code:`z` is not present in :code:`x`
despite :code:`d` not being a non-leaf node in :code:`x`.

**Container-dependent Functions**

*Container-dependent* functions are functions containing arguments which, if provided,
**must** be provided as an :code:`ivy.Container`.
*Container-dependent* functions are never *nestable*, as we will explain.
Due to their dependence on containers, *container-dependent* functions all have a natural
many (the containers) to one (all other arguments) correspondence in the arguments,
unlike *nestable* functions which have a one-to-one correspondence between the arguments
by default.

A couple of examples of *Container-dependent* functions are:
:code:`ivy.execute_with_gradients` and :code:`ivy.multi_head_attention`.

We'll go through the signatures and docstring descriptions for both of these in turn.

.. code-block:: python

    def execute_with_gradients(
        func: Callable,
        xs: ivy.Container,
        retain_grads: bool = False,
    ) -> Tuple[ivy.Array, ivy.Container, Any]:
        """
        Call function func with container of input variables xs, and return the
        functions first output y, the gradients dy/dx as a new container, and any other
        function outputs after the returned y value.
        """

Technically, this function *could* be made fully nestable, whereby the function would
be independently applied on each leaf node of the :code:`ivy.Container` of variables,
but this would be much less efficient, with the backend autograd function
(such as :code:`torch.autograd.grad`) being called many times independently for each
variable in the container of variables :code:`xs`. By making this function non-nestable,
we do not map the function across each of the container leaves, and instead pass the
entire container into the backend autograd function directly,
which is much more efficient.

If the function were *nestable*, it would also repeatedly return :code:`y` and all
other function return values at each leaf of the single returned container,
changing the signature of the function, and causing repeated redundancy in the return.

The example :code:`ivy.multi_head_attention` is a bit different.

.. code-block:: python

    def multi_head_attention(
        x: Union[ivy.Array, ivy.NativeArray],
        scale: Number,
        num_heads: int,
        context: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        to_q_fn: Optional[Callable] = None,
        to_kv_fn: Optional[Callable] = None,
        to_out_fn: Optional[Callable] = None,
        to_q_v: Optional[ivy.Container] = None,
        to_kv_v: Optional[ivy.Container] = None,
        to_out_v: Optional[ivy.Container] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Applies multi-head attention, with the array (x) to determine the queries from,
        the scale for the query-key similarity measure, the number of
        attention heads, the context to determine the keys and values from,
        the mask to apply to the query-key values, the function (to_q_fn) to compute
        queries from input x, the function (to_kv_fn) to compute keys and values from
        the context, the function (to_out_fn) to compute the output from the scaled
        dot-product attention, the variables (to_q_v) for function to_q_fn, the
        variables (to_kv_v) for function to_kv_fn, and the variables (to_out_v) for
        function to_out_fn.
        """

This function fundamentally could not be made *nestable*,
as the function takes a many-to-one approach with regards to the optional containers:
:code:`to_q_v`, :code:`to_kv_v` and :code:`to_out_v`.
The containers are optionally used for the purpose of returning a single
:code:`ivy.Array` at the end. Calling this function on each leaf of the containers
passed in the input would not make any sense.

Hopefully, these two examples explain why *Container-dependent* functions
(with arguments which, if provided, **must** be provided as an :code:`ivy.Container`),
are never implemented as *nestable* functions.

Convenience Functions
---------------------

A final group of functions are the *convenience* functions (briefly mentioned above).
Convenience functions do not form part of the computation graph directly, and they do not directly modify arrays.
However, they can be used to organize and improve the code for other functions which do modify the arrays.
Convenience functions can be *primary*, *compositional*, *mixed* or *standalone* functions. Many are also *nestable*.
This is another categorization which is **not** mutually exclusive, as outlined by the Venn diagram below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/convenience.png?raw=true
   :align: center
   :width: 50%

Primary convenience functions include:
`ivy.can_cast`_ which determines if one data type can be cast to another data type according to type-promotion rules,
`ivy.dtype`_ which gets the data type for the input array,
and `ivy.dev`_ which gets the device for the input array.

Compositional convenience functions include:
`ivy.set_default_dtype`_ which sets the global default data dtype,
`ivy.default_dtype`_ which returns the correct data type to use,
considering both the inputs and also the globally set default,
and `ivy.get_all_arrays_on_dev`_ which gets all arrays which are currently on the specified device.

Standalone convenience functions include:
`ivy.get_backend`_ which returns a local Ivy module with the associated backend framework.
`ivy.nested_map`_ which enables an arbitrary function to be mapped across the leaves of an arbitrary nest,
and `ivy.index_nest`_ which enables an arbitrary nest to be recursively indexed.

There are many other examples. The convenience functions are not grouped by file or folder.
Feel free to have a look through all of the `submodules`_, you should be able to spot quite a few!

**Round Up**

This should have hopefully given you a good feel for the different function types.

If you're ever unsure of how best to proceed,
please feel free to engage with the `function types discussion`_,
or reach out on `discord`_ in the `function types channel`_!
