Function Formatting
===================

For the purposes of explanation, we will use three randomly chosen functions as examples:
:code:`ivy.tan`, :code:`ivy.roll` and :code:`ivy.add`

Type Hints
----------

All arguments should use full and thorough type hints.

We refer to all functions which can consume at least one :code:`ivy.Array` instance and also return at least one
:code:`ivy.Array` instance as *flexible* functions. This is because all such functions can also consume and return
:code:`ivy.Container` instances in place of the arrays. These methods can also accept combinations of :code:`ivy.Array`
and :code:`ivy.Container` instances, broadcasting the arrays to the leaves of the containers in such cases.

Furthermore, all functions can accept either :code:`ivy.Array` or :code:`ivy.NativeArray` instances in the input, but
:code:`ivy.NativeArray` instances are not permitted in the output.

:code:`ivy.NativeArray` instances are also not permitted for the :code:`out` argument, which is used in many functions.
This is because the :code:`out` argument dicates the array to which the result should be written, and so it effectively
serves the same purpose as the function return.

All arguments should be added on a new line, and the return type hint should also be added on a new line.

Let's take a look at the type hints for :code:`ivy.tan`:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None
    ) -> Union[ivy.Array, ivy.Container]:

In alignment with the above explanations, :code:`x` permits :code:`ivy.NativeArray` instances, but :code:`out` and the
function return do not.

Similarly, the type hints for :code:`ivy.roll`

.. code-block:: python

    def roll(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        shift: Union[int, Tuple[int, ...]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:

and :code:`ivy.add`

.. code-block:: python

    def add(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:

both also follow the same pattern.

Writing Docstrings
------------------

Adding Examples
---------------
