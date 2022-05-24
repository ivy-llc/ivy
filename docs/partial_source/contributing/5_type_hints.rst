Type Hints
==========

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`spec/API_specification/signatures`: https://github.com/data-apis/array-api/tree/main/spec/API_specification/signatures

The most basic three rules are (a) all arguments should use full and thorough type hints,
and (b) all arguments should be added on a new line, and (c) the return type hint should also be added on a new line.

For diving deeper into other requirements for the type-hints, it's useful to look at some examples.

Examples
--------

For the purposes of explanation, we will use four functions as examples:
:code:`ivy.tan`, :code:`ivy.roll`, :code:`ivy.add` and :code:`ivy.zeros`.

We present both the Ivy API signature and also a backend-specific signature for each function:

.. code-block:: python

    # Ivy
    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:

    # PyTorch
    def tan(
        x: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

.. code-block:: python

    # Ivy
    def roll(
        x: Union[ivy.Array, ivy.NativeArray],
        shift: Union[int, Sequence[int]],
        axis: Optional[Union[int, Sequence[int]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

    # NumPy
    def roll(
        x: np.ndarray,
        shift: Union[int, Sequence[int]],
        axis: Optional[Union[int, Sequence[int]]] = None,
    ) -> np.ndarray:

.. code-block:: python

    # Ivy
    def add(
        x1: Union[ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

    # TensorFlow
    def add(
        x1: Tensor,
        x2: Tensor
    ) -> Tensor:

.. code-block:: python

    # Ivy
    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:

    # JAX
    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
    ) -> JaxArray:

Arrays
------

In each example, we can see that the input arrays have type :code:`Union[ivy.Array, ivy.NativeArray]`
whereas the output arrays have type :code:`ivy.Array`. This is the case for all functions in the Ivy API.
We always return an :code:`ivy.Array` instance to ensure that any subsequent Ivy code is fully framework-agnostic, with
all operators performed on the returned array now handled by the special methods of the :code:`ivy.Array` class,
and not the special methods of the backend array class (:code:`ivy.NativeArray`). For example,
calling any of (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.) on the array will result in
(:code:`__add__`, :code:`__sub__`, :code:`__mul__`, :code:`__div__` etc.) being called on the array class.

:code:`ivy.NativeArray` instances are also not permitted for the :code:`out` argument, which is used in many functions.
This is because the :code:`out` argument dicates the array to which the result should be written, and so it effectively
serves the same purpose as the function return when no :code:`out` argument is specified.

However, there is no need to prevent native arrays from being permitted in the input.
For Ivy methods which wrap backend-specific implementations, the input arrays needs to be converted to a native arrays
(such as :code:`torch.Tensor`) anyway before calling the wrapped backend function.
This is also not a problem for compositional Ivy functions such as :code:`ivy.lstm_update`
which do not defer to any backend function,
the native array inputs can simply be converted to :code:`ivy.Array` instances before executing the Ivy implementation.

Integer Sequences
-----------------

For sequences of integers, generally the `Array API Standard`_ dictates that these should be of type :code:`Tuple[int]`,
and not :code:`List[int]`. However, in order to make Ivy code less brittle,
we accept arbitrary integer sequences :code:`Sequence[int]` for such arguments
(which includes :code:`list`, :code:`tuple` etc.).
This does not break the standard, as the standard is only intended to define a subset of required function behaviour.
The standard can be freely extended, as we are doing here.
Good examples of this are the :code:`axis` argument of :code:`ivy.roll`
and the :code:`shape` argument of :code:`ivy.zeros`, as shown above.

Keyword-Only Arguments
----------------------

The :code:`dtype`, :code:`device` and :code:`out` arguments should always be provided as keyword-only arguments.
Additionally, the :code:`out` argument should **only** be added if the wrapped backend function directly supports
supports the :code:`out` argument itself. Otherwise, the :code:`out` argument should be omitted.
The reasons for this are explaiend in the :ref:`Adding Functions` section,
but in a nutshell it's because these three arguments are handled by external code which wraps around these functions.
By the time the backend implementation is enterred, the correct :code:`dtype` and :code:`device` to use have both
already been correctly inferred. As for the :code:`out` argument, the inplace update is automatically handled in the
wrapper code if no :code:`out` argument is detected in the backend signature, which is why we should only add it if the
wrapped backend function itself supports the :code:`out` argument, which will result in a more efficient inplace update.

Flexible Functions
------------------

Most functions in the Ivy API can also consume and return :code:`ivy.Container` instances in place of the **any** of
the function arguments. Because of this, we refer to these functions as as *flexible* functions.
These functions can also accept arbitrary combinations of :code:`ivy.Container` instances and non-containers,
broadcasting the other non-container arguments to the leaves of the containers.
All *flexible* functions are also implemented as instance methods on the :code:`ivy.Container` class,
with the first argument replaced with :code:`self` in general, with a few exceptions.