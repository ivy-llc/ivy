Function Types
==============

.. _`_wrap_function`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L137
.. _`framework setting`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/framework_handler.py#L205
.. _`at import time`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/__init__.py#L114
.. _`add_ivy_array_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/array/wrapping.py#L26
.. _`add_ivy_container_instance_methods`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L69
.. _`from being added`: https://github.com/unifyai/ivy/blob/055dcb3b863b70c666890c580a1d6cb9677de854/ivy/container/wrapping.py#L78
.. _`_function_w_arrays_n_out_handled`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L166
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9


Primary Functions
-----------------

*Primary* functions are essentially the lowest level building blocks in Ivy. Each primary function has a unique
backend-specific implementation for each backend specified in
:code:`ivy/functional/backends/backend_name/category_name.py`. These are generally implemented as light wrapping
around an existing backend-specific function, which serves a near-identical purpose.

Primary functions must both be specified in :code:`ivy/functional/ivy/category_name.py` and also in each of
the backend files :code:`ivy/functional/backends/backend_name/category_name.py`

The function in :code:`ivy/functional/ivy/category_name.py` includes the type hints, docstring and docstring examples
(explained in more detail in subsequent sections), but does not include an actual implementation.

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
        return _cur_framework(x).tan(x, out=out)

The backend-specific implementation of :code:`ivy.tan`  for PyTorch in
:code:`ivy/functional/backends/torch/elementwise.py` is given below:

.. code-block:: python

    def tan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.tan(x, out=out)

Compositional Functions
-----------------------

*Compositional* functions on the other hand **do not** have backend-specific implementations. They are implemented as
a *composition* of other Ivy methods, which themselves can be either compositional or primary.

Therefore, compositional functions are only implemented in :code:`ivy/functional/ivy/category_name.py`, and there are no
implementations in any of the backend files :code:`ivy/functional/backends/backend_name/category_name.py`

For example, the implementation of :code:`ivy.cross_entropy` in :code:`ivy/functional/ivy/losses.py`
(with docstrings removed) is given below:

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[int] = -1,
        epsilon: Optional[float] = 1e-7,
        *,
        out: Optional[Union[ivy.Array, ivy.Container]] = None
    ) -> ivy.Array:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return -ivy.sum(log_pred * true, axis)


Partial Primary Functions
-------------------------

*Partial primary* functions have some backend-specific implementations in
:code:`ivy/functional/backends/backend_name/category_name.py`, but not for all backends.
To support backends that do not have a backend-specific implementation,
a compositional implementation is also provided in :code:`ivy/functional/ivy/category_name.py`.

When using ivy without a framework set explicitly (for example :code:`ivy.set_framework()` has not been called),
then the function called is always the one implemented in :code:`ivy/functional/ivy/category_name.py`.
For *primary* functions, then :code:`_cur_framework(x).func_name(...)`
will call the backend-specific implementation in :code:`ivy/functional/backends/backend_name/category_name.py`
directly. However, as just explained, *partial primary* functions implement a compositional approach in
:code:`ivy/functional/ivy/category_name.py`, without deferring to the backend.
Therefore, without any explicit framework setting, then the compositional implementation is always used,
even for backends that have a more efficient backend-specific implementation.
Typically the framework should always be set explicitly (using :code:`ivy.set_framework()` for example),
and in this case the efficient backend-specific implementation will always be used if it exists.

Flexible Functions
------------------

*Flexible* functions are functions (compositional or primary) which can receive either arrays or containers in the
input, as well as arbitrary combinations.
More specifically, array arguments for *flexible* functions have the type hint
:code:`Union[ivy.Array, ivy.NativeArray, ivy.Container]`.

Additionally, all *flexible* functions are also implemented as instance methods on both the :code:`ivy.Array` and
:code:`ivy.Container` classes.

Every function which receives at least one array argument in the input and also returns at least one array
is implemented as a *flexible* function by default.

This added support for handling :code:`ivy.Container` instances is all handled automatically when `_wrap_function`_
is applied to every function (except those appearing in `NON_WRAPPED_FUNCTIONS`_)
in the :code:`ivy` namespace during `framework setting`_.

As part of this wrapping, `_function_w_arrays_n_out_handled`_ also ensures that :code:`ivy.Array` instances in the input
are converted to :code:`ivy.NativeArray` instances before passing to the backend implementation,
and are then converted back to :code:`ivy.Array` instances before returning.

Additionally, the :code:`ivy.Array` and :code:`ivy.Container` instance methods are also all added programmatically
`at import time`_ when `add_ivy_array_instance_methods`_ and `add_ivy_container_instance_methods`_
are called respectively.

However, the :code:`ivy.Array` and :code:`ivy.Container` instance methods should also be implemented explicitly in the
source code. Once the explicit implementation is added in the source code, it will then prevent this specific
programmatic implementation `from being added`_.

For example, the implementation of :code:`ivy.Array.tan` is as follows:

.. code-block:: python

    def tan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.tan(self, out=out)

Likewise, the implementation of :code:`ivy.Container.tan` is as follows:

.. code-block:: python

    def tan(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.tan(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

The :code:`ivy.Container.tan` implementation is a bit more complicated as there are a few arugments which dictate how
the mapping is performed across the leaves of the container, when using :code:`ivy.Container.map`.

Adding the implementation explicitly in source has the benefit that autocompletions and will work in the IDE,
and other IDE checks won't show errors which otherwise appear when calling unfound instance methods or using types in
the arguments which are not supported in the source code implementation.

The purpose of the programmatic instance method setting is then simply as a backup for better robustness,
adding any instance methods which have not yet been added in source code, or were just forgotten.
