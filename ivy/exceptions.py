import ivy
import functools
from typing import Callable

import os
import types
import traceback


_frontend_truncated_stack_traces = "ivy/functional/frontends"
_ivy_truncated_stack_traces = "ivy/"


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
                msg = filter_traceback(e.__traceback__, "ivy/functional/frontends")
                raise ivy.exceptions.IvyError(fn.__name__, str(msg))
            else:
                raise ivy.exceptions.IvyError(fn.__name__, str(e)) from None
        except Exception as e:
            if ivy.get_exception_trace_mode():
                tb = filter_traceback(e.__traceback__, "ivy/functional/backends")
                exp = ivy.exceptions.IvyBackendException(fn.__name__, str(e))
                exp.__traceback__ = tb
                raise exp
            else:
                tb = filter_traceback(e.__traceback__, "ivy/functional/backends")
                exp = ivy.exceptions.IvyBackendException(fn.__name__, str(e))
                exp.__traceback__ = tb
                raise exp from None

    new_fn.handle_exceptions = True
    return new_fn


def filter_traceback(tb, with_filter):
    out = None
    # Scan the traceback and collect relevant frames.
    frames = list(traceback.walk_tb(tb))
    for f, lineno in reversed(frames):
        if path_contains(f.f_code.co_filename, with_filter):
            out = types.TracebackType(
                out, f, f.f_lasti, lineno
            )  # pytype: disable=wrong-arg-count
    return out


def path_contains(path, string_to_match):
    path = os.path.abspath(path)
    return string_to_match in path
