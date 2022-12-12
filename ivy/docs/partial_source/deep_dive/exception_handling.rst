Exception Handling
==================

.. _`exception handling channel`: https://discord.com/channels/799879767196958751/1028267924043092068
.. _`exception handling forum`: https://discord.com/channels/799879767196958751/1028297940168626257
.. _`discord`: https://discord.gg/sXyFF8tDtm

As Ivy is unifying multiple backends, various issues are seen during exception handling:

#. each backend throws their own exceptions
#. exceptions thrown are backend-specific, therefore inconsistent

To unify the handling of exceptions and assertions, Ivy includes a custom exception class and decorator, which are explained further in the following sub-sections.


Ivy Exception Class
-------------------

Firstly, Ivy's base exception class is :code:`IvyException` class, which inherits from the Python :code:`Exception` class.

.. code-block:: python

    # in ivy/exceptions.py
    class IvyException(Exception):
        def __init__(self, message):
            super().__init__(message)

In cases where an exception class for a specific purpose is required, we inherit from the :code:`IvyException` class.
For example, the :code:`IvyBackendException` class is created to unify backend exceptions.

.. code-block:: python

    # in ivy/exceptions.py
    class IvyBackendException(IvyException):
        def __init__(self, *messages):
            self._default = [
                "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
            ]
            self._delimiter = ": "
            for message in messages:
                self._default.append(message)
            super().__init__(self._delimiter.join(self._default))

In some Array API tests, :code:`IndexError` and :code:`ValueError` are explicitly tested to ensure that the functions are behaving correctly.
Thus, the :code:`IvyError` class unifies these special cases.
This is to reduce repetition and the creation of similar exception classes.

.. code-block:: python

    # in ivy/exceptions.py
    class IvyError(IndexError, ValueError, IvyException):
        def __init__(self, *messages):
            self._default = [
                "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
            ]
            self._delimiter = ": "
            for message in messages:
                self._default.append(message)
            super().__init__(self._delimiter.join(self._default))

@handle_exceptions Decorator
----------------------------

To ensure that all backend exceptions are caught properly, a decorator is used to handle functions in the :code:`try/except` block.

.. code-block:: python

    # in ivy/exceptions.py
    def handle_exceptions(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def new_fn(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (IndexError, ValueError) as e:
                raise ivy.exceptions.IvyError(fn.__name__, str(e)) from None
            except Exception as e:
                raise ivy.exceptions.IvyBackendException(fn.__name__, str(e)) from None

        new_fn.handle_exceptions = True
        return new_fn

The decorator is then added to each function for wrapping.
Let's look at an example of :func:`ivy.all`.

.. code-block:: python

    # in ivy/functional/ivy/utility.py
    @handle_exceptions
    def all(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.current_backend(x).all(x, axis=axis, keepdims=keepdims, out=out)

When a backend throws an exception, it will be caught in the decorator and an :code:`IvyBackendException` or :code:`IvyError` will be raised.
This ensures that all exceptions are consistent.

Let's look at the comparison of before and after adding the decorator.

**without decorator**

In NumPy,

.. code-block:: none

    >>> x = ivy.array([0,0,1])
    >>> ivy.all(x, axis=2)
    <error_stack>
    numpy.AxisError: axis 2 is out of bounds for array of dimension 1

In PyTorch,

.. code-block:: none

    >>> x = ivy.array([0,0,1])
    >>> ivy.all(x, axis=2)
    <error_stack>
    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 2)

The errors raised are different across backends, therefore confusing and inconsistent.

**with decorator**

In NumPy,

.. code-block:: none

    >>> x = ivy.array([0,0,1])
    >>> ivy.all(x, axis=2)
    <error_stack>
    ivy.exceptions.IvyError: numpy: all: axis 2 is out of bounds for array of dimension 1

In PyTorch,

    >>> x = ivy.array([0,0,1])
    >>> ivy.all(x, axis=2)
    <error_stack>
    ivy.exceptions.IvyError: torch: all: Dimension out of range (expected to be in range of [-1, 0], but got 2)

The errors are unified into an :code:`IvyError`, with the current backend and function stated to provide clearer information.
The message string is inherited from the native exception.

Assertion Function
------------------

There are often conditions or limitations needed to ensure that a function is working correctly.

Inconsistency is observed such as some functions:

#. use :code:`assert` for checks and throw :code:`AssertionError`, or
#. use :code:`if/elif/else` blocks and raise :code:`Exception`, :code:`ValueError`, etc.

To unify the behaviours, our policy is to use conditional blocks and raise :code:`IvyException` whenever a check is required.
Moreover, to reduce code redundancy, conditions which are commonly used are collected as helper functions with custom parameters in :mod:`ivy/assertions.py`.
This allows them to be reused and promotes cleaner code.

Let's look at an example!

**Helper: check_less**

.. code-block:: python

    # in ivy/assertions.py
    def check_less(x1, x2, allow_equal=False, message=""):
    # less_equal
    if allow_equal and ivy.any(x1 > x2):
        raise ivy.exceptions.IvyException(
            "{} must be lesser than or equal to {}".format(x1, x2)
            if message == ""
            else message
        )
    # less
    elif not allow_equal and ivy.any(x1 >= x2):
        raise ivy.exceptions.IvyException(
            "{} must be lesser than {}".format(x1, x2) if message == "" else message
        )

**ivy.set_split_factor**

.. code-block:: python

    # in ivy/functional/ivy/device.py
    @handle_exceptions
    def set_split_factor(
        factor: float,
        device: Union[ivy.Device, ivy.NativeDevice] = None,
        /,
    ) -> None:
        ivy.assertions.check_less(0, factor, allow_equal=True)
        global split_factors
        device = ivy.default(device, default_device())
        split_factors[device] = factor

Instead of coding a conditional block and raising an exception if the conditions are not met, a helper function is used to simplify the logic and increase code readability.

**Round Up**

This should have hopefully given you a good feel for how function wrapping is applied to functions in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `exception handling channel`_ or in the `exception handling forum`_!