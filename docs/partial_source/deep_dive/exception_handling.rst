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

Configurable Mode for Stack Trace
---------------------------------

Due to the transpilation nature of Ivy, user can code in the frontend framework
which they are familiar with, then run their code with another backend
framework. For instance, someone who is familiar with NumPy can run their code
with a PyTorch backend via Ivy's NumPy frontend. Given their total lack of
familiarity of certain backend frameworks (which is PyTorch in the above scenario),
they might not want to see stack traces which go right down through Ivy
functions and through PyTorch functions.

Therefore, options are made available for the stack traces to either truncate
at the frontend or ivy level, or in other cases, no truncation at all.

Let's look at the 3 different modes with example of :code:`ivy.all` below!

1. Full

This is the default mode and keeps the complete stack traces. All frontend and
ivy specific stack traces are displayed.

.. code-block:: none

    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
    ...
    File ".../ivy/ivy/func_wrapper.py", line 124, in new_fn
      return fn(*new_args, **new_kwargs)
    File ".../ivy/ivy/func_wrapper.py", line 186, in new_fn
      ret = fn(*args, **kwargs)
    File ".../ivy/ivy/functional/backends/torch/utility.py", line 19, in all
      return torch.all(x, dim=axis, keepdim=keepdims, out=out)

    During the handling of the above exception, another exception occurred:

    ...
    File ".../ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 15, in all
      ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
    File ".../ivy/ivy/func_wrapper.py", line 82, in new_fn
      return fn(*args, **kwargs)
    File ".../ivy/ivy/exceptions.py", line 120, in new_fn
      raise ivy.exceptions.IvyError(fn.__name__, str(e))

    IvyError: torch: all: Dimension out of range (expected to be in range of [-1, 0], but got 2)

2. Frontend-only

This option displays only frontend-related stack traces. If compared with the
stack traces in the :code:`full` mode above, the :code:`torch` related traces
are pruned. Only the :code:`numpy` frontend related errors are shown.
A message is also displayed to inform that the traces are truncated and
the instructions to switch it back to the :code:`full` mode is included.

.. code-block:: none

    >>> ivy.set_exception_trace_mode('frontend')
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)

    <stack trace is truncated to frontend specific files, call
    `ivy.set_exception_trace_mode('full')` to view the full trace>

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to frontend specific files, call
    `ivy.set_exception_trace_mode('full')` to view the full trace>
      ...
      File ".../ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 15, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)

    IvyError: torch: all: Dimension out of range (expected to be in range of [-1, 0], but got 2)

3. Ivy specific

This option displays only ivy-related stack traces. If compared with the
stack traces in the :code:`frontend` mode above, the :code:`torch` related traces
are available again. A message is also displayed to inform that the traces are
truncated and the instructions to switch it back to the :code:`full` mode
is included.

.. code-block:: none

    >>> ivy.set_exception_trace_mode('ivy')
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)

    <stack trace is truncated to ivy specific files, call
    `ivy.set_exception_trace_mode('full')` to view the full trace>
      ...
      File ".../ivy/ivy/func_wrapper.py", line 186, in new_fn
        ret = fn(*args, **kwargs)
      File ".../ivy/ivy/functional/backends/torch/utility.py", line 19, in all
        return torch.all(x, dim=axis, keepdim=keepdims, out=out)

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to ivy specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
      ...
      File ".../ivy/ivy/exceptions.py", line 121, in new_fn
        raise ivy.exceptions.IvyError(fn.__name__, str(e))

    IvyError: torch: all: Dimension out of range (expected to be in range of [-1, 0], but got 2)

Ivy :code:`func_wrapper` Pruning
--------------------------------

Due to the wrapping operations in Ivy, a long list of less informative
:code:`func_wrapper` traces is often seen in the stack. An example is shown
below:

.. code-block:: none

    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
      ...
      File ".../ivy/ivy/func_wrapper.py", line 462, in new_fn
        return fn(*args, **kwargs)
      File ".../ivy/ivy/func_wrapper.py", line 402, in new_fn
        return fn(*args, **kwargs)
      File ".../ivy/ivy/func_wrapper.py", line 124, in new_fn
        return fn(*new_args, **new_kwargs)
      File ".../ivy/ivy/func_wrapper.py", line 186, in new_fn
        ret = fn(*args, **kwargs)
      File ".../ivy/ivy/functional/backends/torch/utility.py", line 19, in all
        return torch.all(x, dim=axis, keepdim=keepdims, out=out)

    During the handling of the above exception, another exception occurred:

      File "<stdin>", line 1, in <module>
      File ".../ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 285, in new_fn
        ret = fn(*args, **kwargs)
      File ".../ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 263, in new_fn
        return fn(*ivy_args, **ivy_kwargs)
      ...

    IvyError: torch: all: Dimension out of range (expected to be in range of [-1, 0], but got 2)

Including all of these wrapper functions in the stack trace can be very
unwieldy, thus they can be prevented entirely by setting
:code:`ivy.set_show_func_wrapper_trace_mode(False)`.

.. code-block:: none

    >>> ivy.set_show_func_wrapper_trace_mode(False)
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)

    <func_wrapper.py stack trace is squashed, call
    `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File ".../ivy/ivy/exceptions.py", line 118, in new_fn
        return fn(*args, **kwargs)
      File ".../ivy/ivy/functional/backends/torch/utility.py", line 19, in all
        return torch.all(x, dim=axis, keepdim=keepdims, out=out)

    During the handling of the above exception, another exception occurred:

    <func_wrapper.py stack trace is squashed, call
    `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "<stdin>", line 1, in <module>
      File ".../ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 15, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
      File ".../ivy/ivy/exceptions.py", line 121, in new_fn
        raise ivy.exceptions.IvyError(fn.__name__, str(e))

    IvyError: torch: all: Dimension out of range (expected to be in range of [-1, 0], but got 2)

From the above example, it can be seen that the :code:`func_wrapper` related
traces have been hidden. A message is displayed as well to the user so that
they are aware of the pruning. The instructions to recover the
:code:`func_wrapper` traces are shown too.


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
            except (IndexError, ValueError, AttributeError) as e:
                _print_traceback_history()
                raise ivy.exceptions.IvyError(fn.__name__, str(e))
            except Exception as e:
                _print_traceback_history()
                raise ivy.exceptions.IvyBackendException(fn.__name__, str(e))

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
