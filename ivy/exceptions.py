import ivy
import functools
from typing import Callable
import sys
import traceback as tb


# Helpers #
# ------- #


def _print_new_stack_trace(old_stack_trace):
    print(
        "<func_wrapper.py stack trace is squashed,",
        "call `ivy.set_show_func_wrapper_traces(True)` in order to view this>",
    )
    new_stack_trace = []
    for st in old_stack_trace:
        if "func_wrapper.py" not in repr(st):
            new_stack_trace.append(st)
    print("".join(tb.format_list(new_stack_trace)))


def _custom_exception_handle(type, value, tb_history):
    if ivy.get_show_func_wrapper_trace_mode():
        print("".join(tb.format_tb(tb_history)))
    else:
        _print_new_stack_trace(tb.extract_tb(tb_history))
    print(type.__name__ + ":", value)


def _print_traceback_history():
    if ivy.get_show_func_wrapper_trace_mode():
        print("".join(tb.format_tb(sys.exc_info()[2])))
    else:
        _print_new_stack_trace(tb.extract_tb(sys.exc_info()[2]))
    print("During the handling of the above exception, another exception occurred:\n")


sys.excepthook = _custom_exception_handle


# Classes and Methods #
# ------------------- #


class IvyException(Exception):
    def __init__(self, message):
        super().__init__(message)


class IvyBackendException(IvyException):
    def __init__(self, *messages):
        self._default = [
            "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
        ]
        self._delimiter = ": "
        for message in messages:
            self._default.append(message)
        super().__init__(self._delimiter.join(self._default))


class IvyNotImplementedException(NotImplementedError):
    def __init__(self, message=""):
        super().__init__(message)


class IvyError(IndexError, ValueError, AttributeError, IvyException):
    def __init__(self, *messages):
        self._default = [
            "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
        ]
        self._delimiter = ": "
        for message in messages:
            self._default.append(message)
        super().__init__(self._delimiter.join(self._default))


def handle_exceptions(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Catch all exceptions and raise them in IvyException

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, or raise IvyException if error is thrown.
        """
        try:
            return fn(*args, **kwargs)
        except (IndexError, ValueError, AttributeError) as e:
            if ivy.get_exception_trace_mode():
                _print_traceback_history()
            raise ivy.exceptions.IvyError(fn.__name__, str(e))
        except Exception as e:
            if ivy.get_exception_trace_mode():
                _print_traceback_history()
            raise ivy.exceptions.IvyBackendException(fn.__name__, str(e))

    new_fn.handle_exceptions = True
    return new_fn
