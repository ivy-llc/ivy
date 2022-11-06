import ivy
import functools
from typing import Callable

import os
import types
import traceback


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
            if (
                ivy.get_exception_trace_mode() == "frontend"
                or ivy.get_exception_trace_mode() == "ivy"
            ):
                tb = _process_traceback_frames(
                    e.__traceback__, ivy.trace_mode_dict[ivy.get_exception_trace_mode()]
                )
                exp = ivy.exceptions.IvyError(fn.__name__, str(e))
                raise exp.with_traceback(tb) from None
            else:
                raise ivy.exceptions.IvyError(fn.__name__, str(e))
        except Exception as e:
            if (
                ivy.get_exception_trace_mode() == "frontend"
                or ivy.get_exception_trace_mode() == "ivy"
            ):
                tb = _process_traceback_frames(
                    e.__traceback__, ivy.trace_mode_dict[ivy.get_exception_trace_mode()]
                )
                exp = ivy.exceptions.IvyBackendException(fn.__name__, str(e))
                raise exp.with_traceback(tb) from None
            else:
                raise ivy.exceptions.IvyBackendException(fn.__name__, str(e))

    new_fn.handle_exceptions = True
    return new_fn


def _process_traceback_frames(tb, with_filter):
    new_tb = None
    tb_list = list(traceback.walk_tb(tb))
    for f, line_no in reversed(tb_list):
        if _path_contains(f.f_code.co_filename, with_filter):
            new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
    return new_tb


def _path_contains(path, string_to_match):
    path = os.path.abspath(path)
    return string_to_match in path
