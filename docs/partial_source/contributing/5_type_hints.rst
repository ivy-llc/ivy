Type Hints
==========

.. _`spec/API_specification/signatures`: https://github.com/data-apis/array-api/tree/main/spec/API_specification/signatures

All arguments should use full and thorough type hints.

For the purposes of explanation, we will use three randomly chosen functions as examples:
:code:`ivy.tan`, :code:`ivy.roll` and :code:`ivy.add`

All functions which consume at least one array and also return at least one array can also consume and return
:code:`ivy.Container` instances in place of the arrays. Because of this, we refer to these functions as as *flexible*
functions. These functions can also accept combinations of arrays and :code:`ivy.Container` instances,
broadcasting the arrays to the leaves of the containers.
All *flexible* functions are also implemented as instance methods on both the :code:`ivy.Array` and
:code:`ivy.Container` classes, with the first array argument replaced with :code:`self` in both cases.

Furthermore, all array arguments can either be passed an :code:`ivy.Array` or an :code:`ivy.NativeArray` instance
(i.e. :code:`torch.Tensor`, :code:`np.ndarray` etc. depending on the backend). However, :code:`ivy.NativeArray`
instances are not permitted in the output. Arrays are always returned as :code:`ivy.Array` instances.
Among other reasons, this is to ensure that operators (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.)
performed on the array are framework-agnostic. We could not guarantee this if returning :code:`ivy.NativeArray`
instances, as we would have no control over the relevant special methods (:code:`__add__`, :code:`__sub__`,
:code:`__mul__`, :code:`__div__` etc.).

:code:`ivy.NativeArray` instances are also not permitted for the :code:`out` argument, which is used in many functions.
This is because the :code:`out` argument dicates the array to which the result should be written, and so it effectively
serves the same purpose as the function return when no :code:`out` argument is specified.

All arguments should be added on a new line, and the return type hint should also be added on a new line.

Taking all of these points into consideration, let's take a look at the type hints for :code:`ivy.tan`:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        *,
        out: Optional[Union[ivy.Array, ivy.Container]] = None
    ) -> Union[ivy.Array, ivy.Container]:

Because :code:`tan` consumes one array and also returns one array, it is a *flexible* function.
Therefore, both inputs :code:`x` and :code:`out` and the return are all permitted to be :code:`ivy.Container` instances.
Additionally, in alignment with the above explanations, :code:`x` permits :code:`ivy.NativeArray` instances, but :code:`out` and the
function return do not.

Similarly, the type hints for :code:`ivy.roll`

.. code-block:: python

    def roll(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        shift: Union[int, Tuple[int, ...]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:

and :code:`ivy.add`

.. code-block:: python

    def add(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        *,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:

both also follow the same pattern.