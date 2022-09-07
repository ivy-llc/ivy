Function Arguments
==================

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`spec/API_specification/signatures`: https://github.com/data-apis/array-api/tree/main/spec/API_specification/signatures
.. _`function arguments discussion`: https://github.com/unifyai/ivy/discussions/1320
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`function arguments channel`: https://discord.com/channels/799879767196958751/982738240354254898
.. _`Array API Standard convention`: https://data-apis.org/array-api/2021.12/API_specification/array_object.html#api-specification-array-object--page-root

Here, we explain how the function arguments differ between the placeholder implementation at
:code:`ivy/functional/ivy/category_name.py`, and the backend-specific implementation at
:code:`ivy/functional/backends/backend_name/category_name.py`.

Many of these points are already adressed in the previous sections:
:ref:`Arrays`, :ref:`Data Types`, :ref:`Devices` and :ref:`Inplace Updates`.
However, we thought it would be convenient to revisit all of these considerations in a single section,
dedicated to function arguments.

As for type-hints,
all functions in the Ivy API at :code:`ivy/functional/ivy/category_name.py` should have full and thorough type-hints.
Likewise, all backend implementations at
:code:`ivy/functional/backends/backend_name/category_name.py` should also have full and thorough type-hints.

In order to understand the various requirements for function arguments, it's useful to first look at some examples.

Examples
--------

For the purposes of explanation, we will use four functions as examples:
:code:`ivy.tan`, :code:`ivy.roll`, :code:`ivy.add` and :code:`ivy.zeros`.

We present both the Ivy API signature and also a backend-specific signature for each function:

.. code-block:: python

    # Ivy
    @to_native_arrays_and_back
    @handle_out_argument
    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:

    # PyTorch
    def tan(
        x: torch.Tensor,
        /,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

.. code-block:: python

    # Ivy
    @to_native_arrays_and_back
    @handle_out_argument
    def roll(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

    # NumPy
    def roll(
        x: np.ndarray,
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
    ) -> np.ndarray:

.. code-block:: python

    # Ivy
    @to_native_arrays_and_back
    @handle_out_argument
    def add(
        x1: Union[ivy.Array, ivy.NativeArray, float],
        x2: Union[ivy.Array, ivy.NativeArray, float],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> Union[ivy.Array, float]:

    # TensorFlow
    def add(
        x1: Union[tf.Tensor, tf.Variable, float],
        x2: Union[tf.Tensor, tf.Variable, float],
        /,
    ) -> Union[tf.Tensor, tf.Variable, float]:

.. code-block:: python

    # Ivy
    @outputs_to_ivy_arrays
    @handle_out_argument
    @infer_dtype
    @infer_device
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


Positional and Keyword Arguments
------
In both signatures, we follow the `Array API Standard convention`_ about positional and keyword arguments.

* Positional parameters must be positional-only parameters. Positional-only parameters have no externally-usable name. When a method accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order. This is indicated with an :code:`/` after all the position-only arguments.
* Optional parameters must be keyword-only arguments. A :code:`*` must be added before any of the keyword-only arguments.

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
This is all explained in more detail in the :ref:`Arrays` section.

out Argument
------------

The :code:`out` argument should always be provided as keyword-only arguments.
Additionally, the :code:`out` argument should **only** be added to the backend functions if the wrapped backend function
directly supports supports the :code:`out` argument itself. Otherwise, the :code:`out` argument should be omitted from
the backend implementation. The inplace update is automatically handled in the
wrapper code if no :code:`out` argument is detected in the backend signature, which is why we should only add it if the
wrapped backend function itself supports the :code:`out` argument,
which will result in the most efficient inplace update.
This is all explained in more detail in the :ref:`Inplace Updates` section.

dtype and device arguments
--------------------------

In the Ivy API at :code:`ivy/functional/ivy/category_name.py`,
the :code:`dtype` and :code:`device` arguments should both always be provided as keyword-only arguments,
with default value of :code:`None`.
In contrast, these arguments should both be added as required arguments in the backend implementation
at :code:`ivy/functional/backends/backend_name/category_name.py`.
In a nutshell, by the time the backend implementation is entered,
the correct :code:`dtype` and :code:`device` to use have both already been correctly handled
by code which is wrapped around the backend implementation.
This is futher explained in the :ref:`Data Types` and :ref:`Devices` sections respectively.

Numbers in Operator Functions
-----------------------------

All operator functions (which have a corresponding
such as :code:`+`, :code:`-`, :code:`*`, :code:`/`) must also be fully compatible with
numbers (:code:`float` or :code:`int`) passed into any of the array inputs,
even in the absence of any arrays.
For example, :code:`ivy.add(1, 2)`, :code:`ivy.add(1.5, 2)` and
:code:`ivy.add(1.5, ivy.array([2]))` should all run without error.
Therefore, the type hints for :code:`ivy.add` include :code:`float` as one of the types
in the :code:`Union` for the array inputs,
and also as one of the types in the :code:`Union` for the output.
`PEP 484 Type Hints <https://peps.python.org/pep-0484/#the-numeric-tower>`_
state that "when an argument is annotated as having type float,
an argument of type int is acceptable". Therefore, we only include :code:`float` in the
type hints.

Integer Sequences
-----------------

For sequences of integers, generally the `Array API Standard`_ dictates that these should be of type :code:`Tuple[int]`,
and not :code:`List[int]`. However, in order to make Ivy code less brittle,
we accept arbitrary integer sequences :code:`Sequence[int]` for such arguments
(which includes :code:`list`, :code:`tuple` etc.).
This does not break the standard, as the standard is only intended to define a subset of required behaviour.
The standard can be freely extended, as we are doing here.
Good examples of this are the :code:`axis` argument of :code:`ivy.roll`
and the :code:`shape` argument of :code:`ivy.zeros`, as shown above.

Nestable Functions
------------------

Most functions in the Ivy API can also consume and return :code:`ivy.Container` instances in place of the **any** of
the function arguments. if an :code:`ivy.Container` is passed, then the function is mapped across all of the leaves of
this container. Because of this feature, we refer to these functions as *nestable* functions.
However, because so many functions in the Ivy API are indeed *nestable* functions,
and because this flexibility applies to **every** argument in the function,
every type hint for these functions should technically be extended like so: :code:`Union[original_type, ivy.Container]`.

However, this would be very cumbersome, and would only serve to hinder the readability of the docs.
Therefore, we simply omit these :code:`ivy.Container` type hints from *nestable* functions,
and instead mention in the docstring whether the function is *nestable* or not.

**Round Up**

These examples should hopefully give you a good understanding of what is required when adding function arguments.

If you're ever unsure of how best to proceed,
please feel free to engage with the `function arguments discussion`_,
or reach out on `discord`_ in the `function arguments channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/5cAbryXza18" class="video">
    </iframe>