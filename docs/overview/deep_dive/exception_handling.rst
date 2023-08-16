Exception Handling
==================

.. _`exception handling channel`: https://discord.com/channels/799879767196958751/1028267924043092068
.. _`discord`: https://discord.gg/sXyFF8tDtm

As Ivy is unifying multiple backends, various issues are seen during exception handling:

#. each backend throws their own exceptions
#. exceptions thrown are backend-specific, therefore inconsistent

To unify the handling of exceptions and assertions, Ivy includes a custom exception class and decorator, which are explained further in the following sub-sections.


Ivy Exception Class
-------------------

Firstly, Ivy's base exception class is :code:`IvyException` class, which inherits from the Python :code:`Exception` class.

.. code-block:: python

    # in ivy/utils/exceptions.py
    class IvyException(Exception):
        def __init__(self, *messages, include_backend=False):
            self.native_error = (
                messages[0]
                if len(messages) == 1
                and isinstance(messages[0], Exception)
                and not include_backend
                else None
            )
            if self.native_error is None:
                super().__init__(
                    _combine_messages(*messages, include_backend=include_backend)
                )
            else:
                super().__init__(str(messages[0]))

In cases where an exception class for a specific purpose is required, we inherit from the :code:`IvyException` class.
For example, the :code:`IvyBackendException` class is created to unify backend exceptions.

.. code-block:: python

    # in ivy/utils/exceptions.py
    class IvyBackendException(IvyException):
        def __init__(self, *messages, include_backend=False):
            super().__init__(*messages, include_backend=include_backend)

In some Array API tests, :code:`IndexError` and :code:`ValueError` are explicitly tested to ensure that the functions are behaving correctly.
Thus, the :code:`IvyIndexError` and :code:`IvyValueError` classes unifies these special cases.
For a more general case, the :code:`IvyError` class can be used.

.. code-block:: python

    # in ivy/utils/exceptions.py
    class IvyError(IvyException):
        def __init__(self, *messages, include_backend=False):
            super().__init__(*messages, include_backend=include_backend)

More Custom Exception classes were created to unify sub-categories of errors. We try our best to ensure that the same type of 
Exception is raised for the same type of Error regardless of the backend.
This will ensure that the exceptions are truly unified for all the different types of errors.
The implementations of these custom classes are exactly the same as :code:`IvyError` class.
Currently there are 5 custom exception classes in ivy.

1. :code:`IvyIndexError`: This Error is raised for anything Indexing related. For Instance, providing out of bound axis in any function.
2. :code:`IvyValueError`: This is for anything related to providing wrong values. For instance, passing :code:`high` value 
                          smaller than :code:`low` value in :code:`ivy.random_uniform`.
3. :code:`IvyAttributeError`: This is raised when an undefined attribute is referenced.
4. :code:`IvyBroadcastShapeError`: This is raised whenever 2 shapes are expected to be broadcastable but are not.
5. :code:`IvyDtypePromotionError`: Similar to :code:`IvyBroadcastShapeError`, this is raised when 2 dtypes are expected to be promotable but are not.

The correct type of Exception class should be used for the corresponding type of error across the backends. This will truly unify all the exceptions raised in Ivy.

Configurable Mode for Stack Trace
---------------------------------

Ivy's transpilation nature allows users to write code in their preferred frontend 
framework and then execute it with a different backend framework. For example, a 
user who is comfortable with NumPy can use Ivy's NumPy frontend to run their code 
with a JAX backend. However, since they may have no prior experience with JAX or 
other backend frameworks, they may not want to encounter stack traces that traverse 
Ivy and JAX functions. In such cases, it may be preferable for the user to avoid 
encountering stack traces that extend through Ivy and JAX functions.

Therefore, options are made available for the stack traces to either truncate
at the frontend or ivy level, or in other cases, no truncation at all.

Let's look at the 3 different modes with example of :code:`ivy.all` below!

1. Full

This is the default mode and keeps the complete stack traces. All :code:`numpy`
frontend, ivy specific and native :code:`jax` stack traces are displayed.
The format of the error displayed in this mode is :code:`Ivy error: backend name: backend function name: native error: error message`

.. code-block:: none

    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
      File "/ivy/ivy/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 911, in _handle_nestable
        return fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 392, in _handle_array_like_without_promotion
        return fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 805, in _handle_out_argument
        return fn(*args, out=out, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 432, in _inputs_to_native_arrays
        return fn(*new_args, **new_kwargs)
      File "/ivy/ivy/func_wrapper.py", line 535, in _outputs_to_ivy_arrays
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 349, in _handle_array_function
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/backends/jax/utility.py", line 22, in all
        raise ivy.utils.exceptions.IvyIndexError(error)

    During the handling of the above exception, another exception occurred:

      File "/ivy/other_test.py", line 22, in <module>
        ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 523, in _handle_numpy_out
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 396, in _outputs_to_numpy_arrays
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 352, in _inputs_to_ivy_arrays_np
        return fn(*ivy_args, **ivy_kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 453, in _from_zero_dim_arrays_to_scalar
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/ivy/ivy/utils/exceptions.py", line 217, in _handle_exceptions
        raise ivy.utils.exceptions.IvyIndexError(

    IvyIndexError: jax: all: ValueError: axis 2 is out of bounds for array of dimension 1


2. Frontend-only

This option displays only frontend-related stack traces. If compared with the
stack traces in the :code:`full` mode above, the :code:`jax` related traces
are pruned. Only the :code:`numpy` frontend related errors are shown.
A message is also displayed to inform that the traces are truncated and
the instructions to switch it back to the :code:`full` mode is included.
In this case, the format of the error is :code:`Ivy error: backend name: backend function name: error message`

.. code-block:: none

    >>> ivy.set_exception_trace_mode('frontend')
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
    <stack trace is truncated to frontend specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to frontend specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 523, in _handle_numpy_out
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 396, in _outputs_to_numpy_arrays
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 352, in _inputs_to_ivy_arrays_np
        return fn(*ivy_args, **ivy_kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 453, in _from_zero_dim_arrays_to_scalar
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)

    IvyIndexError: jax: all: axis 2 is out of bounds for array of dimension 1


3. Ivy specific

This option displays only ivy-related stack traces. If compared to the different
stack traces modes above, the ivy backend :code:`jax` related
traces (which were hidden in the :code:`frontend` mode) are available again
and the ivy frontend :code:`numpy` related traces remain visible.
However, the native :code:`jax` traces remain hidden because they are not
ivy-specific.
A message is also displayed to inform that the traces are truncated and the
instructions to switch it back to the :code:`full` mode is included.
The format of the error displayed is the same as the :code:`frontend` mode above.

.. code-block:: none

    >>> ivy.set_exception_trace_mode('ivy')
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
    <stack trace is truncated to ivy specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
      File "/ivy/ivy/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 911, in _handle_nestable
        return fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 392, in _handle_array_like_without_promotion
        return fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 805, in _handle_out_argument
        return fn(*args, out=out, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 432, in _inputs_to_native_arrays
        return fn(*new_args, **new_kwargs)
      File "/ivy/ivy/func_wrapper.py", line 535, in _outputs_to_ivy_arrays
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/func_wrapper.py", line 349, in _handle_array_function
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/backends/jax/utility.py", line 22, in all
        raise ivy.utils.exceptions.IvyIndexError(error)

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to ivy specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
      File "/ivy/other_test.py", line 21, in <module>
        ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 523, in _handle_numpy_out
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 396, in _outputs_to_numpy_arrays
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 352, in _inputs_to_ivy_arrays_np
        return fn(*ivy_args, **ivy_kwargs)
      File "/ivy/ivy/functional/frontends/numpy/func_wrapper.py", line 453, in _from_zero_dim_arrays_to_scalar
        ret = fn(*args, **kwargs)
      File "/ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/ivy/ivy/utils/exceptions.py", line 217, in _handle_exceptions
        raise ivy.utils.exceptions.IvyIndexError(

    IvyIndexError: jax: all: axis 2 is out of bounds for array of dimension 1


Ivy :code:`func_wrapper` Pruning
--------------------------------

Due to the wrapping operations in Ivy, a long list of less informative
:code:`func_wrapper` traces is often seen in the stack.
Including all of these wrapper functions in the stack trace can be very
unwieldy, thus they can be prevented entirely by setting
:code:`ivy.set_show_func_wrapper_trace_mode(False)`.
Examples are shown below to demonstrate the combination of this mode and the
3 different stack traces mode explained above.

1. Full

The :code:`func_wrapper` related traces have been hidden. All other traces
such as ivy-specific, frontend-related and the native traces remain visible.
A message is displayed as well to the user so that they are aware of the
pruning. The instructions to recover the :code:`func_wrapper` traces are
shown too.

.. code-block:: none

    >>> ivy.set_show_func_wrapper_trace_mode(False)
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
    <func_wrapper.py stack trace is squashed, call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/ivy/ivy/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/backends/jax/utility.py", line 22, in all
        raise ivy.utils.exceptions.IvyIndexError(error)

    During the handling of the above exception, another exception occurred:

    <func_wrapper.py stack trace is squashed, call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/ivy/other_test.py", line 22, in <module>
        ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
      File "/ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/ivy/ivy/utils/exceptions.py", line 217, in _handle_exceptions
        raise ivy.utils.exceptions.IvyIndexError(

    IvyIndexError: jax: all: ValueError: axis 2 is out of bounds for array of dimension 1


2. Frontend-only

In the frontend-only stack trace mode, the ivy backend wrapping traces were
hidden but the frontend wrappers were still visible. By configuring the func
wrapper trace mode, the frontend wrappers will also be hidden. This can be
observed from the example below.

.. code-block:: none

    >>> ivy.set_exception_trace_mode('frontend')
    >>> ivy.set_show_func_wrapper_trace_mode(False)
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
    <stack trace is truncated to frontend specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to frontend specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)

    IvyIndexError: jax: all: axis 2 is out of bounds for array of dimension 1


3. Ivy specific

As the wrappers occur in :code:`ivy` itself, all backend and frontend wrappers
remain visible in the ivy-specific mode. By hidding the func wrapper traces,
the stack becomes cleaner and displays the ivy backend and frontend
exception messages only.

.. code-block:: none

    >>> ivy.set_exception_trace_mode('frontend')
    >>> ivy.set_show_func_wrapper_trace_mode(False)
    >>> ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
    <stack trace is truncated to ivy specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/ivy/ivy/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/ivy/ivy/functional/backends/jax/utility.py", line 22, in all
        raise ivy.utils.exceptions.IvyIndexError(error)

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to ivy specific files, call `ivy.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/ivy/other_test.py", line 22, in <module>
        ivy.functional.frontends.numpy.all(ivy.array([1,2,3]), axis=2)
      File "/ivy/ivy/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/ivy/ivy/utils/exceptions.py", line 217, in _handle_exceptions
        raise ivy.utils.exceptions.IvyIndexError(

    IvyIndexError: jax: all: axis 2 is out of bounds for array of dimension 1


@handle_exceptions Decorator
----------------------------

To ensure that all backend exceptions are caught properly, a decorator is used to handle functions in the :code:`try/except` block.

.. code-block:: python

    # in ivy/utils/exceptions.py
    def handle_exceptions(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _handle_exceptions(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            # Not to rethrow as IvyBackendException
            except IvyNotImplementedException as e:
                raise e
            except IvyError as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyError(fn.__name__, e, include_backend=True)
            except IvyBroadcastShapeError as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyBroadcastShapeError(
                    fn.__name__, e, include_backend=True
                )
            except IvyDtypePromotionError as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyDtypePromotionError(
                    fn.__name__, e, include_backend=True
                )
            except (IndexError, IvyIndexError) as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyIndexError(
                    fn.__name__, e, include_backend=True
                )
            except (AttributeError, IvyAttributeError) as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyAttributeError(
                    fn.__name__, e, include_backend=True
                )
            except (ValueError, IvyValueError) as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyValueError(
                    fn.__name__, e, include_backend=True
                )
            except (Exception, IvyBackendException) as e:
                _print_traceback_history()
                raise ivy.utils.exceptions.IvyBackendException(
                    fn.__name__, e, include_backend=True
                )

        _handle_exceptions.handle_exceptions = True
        return _handle_exceptions

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

When a backend throws an exception, it will be caught in the decorator and then the appropriate Error will be raised.
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
    IvyIndexError: numpy: all: AxisError: axis 2 is out of bounds for array of dimension 1

In PyTorch,

    >>> x = ivy.array([0,0,1])
    >>> ivy.all(x, axis=2)
    <error_stack>
    IvyIndexError: torch: all: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 2)

The errors are unified into an :code:`IvyIndexError`, with the current backend and function stated to provide clearer information.
The message string is inherited from the native exception.


Consistency in Errors
---------------------

For consistency, we make sure that the same type of Exception is raised for the same type of error regardless of the backend set.
Lets take an example of :func:`ivy.all` again. In Jax, :code:`ValueError` is raised when the axis is out of bounds,
and for Numpy, :code:`AxisError` is raised. To unify the behaviour, we raise :code:`IvyIndexError` for both cases.

In Numpy,

.. code-block:: python
    
    # in ivy/functional/backends/numpy/utility.py
    def all(
        x: np.ndarray,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        try:
            return np.asarray(np.all(x, axis=axis, keepdims=keepdims, out=out))
        except np.AxisError as e:
            raise ivy.utils.exceptions.IvyIndexError(error)

In Jax,

.. code-block:: python

    # in ivy/functional/backends/jax/utility.py
    def all(
        x: JaxArray,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:
        x = jnp.array(x, dtype="bool")
        try:
            return jnp.all(x, axis, keepdims=keepdims)
        except ValueError as error:
            raise ivy.utils.exceptions.IvyIndexError(error)

In both cases, :code:`IvyIndexError` is raised, to make sure the same type of Exception is raised for this specific error.


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

    # in ivy/utils/assertions.py
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

If you have any questions, please feel free to reach out on `discord`_ in the `exception handling channel`_!

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/eTc24eG9P_s" class="video">
    </iframe>
