import ivy
import functools
from typing import Callable
import sys
import traceback as tb
import io

global buffer

# Helpers #
# ------- #


def _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode, buffer=None):
    if trace_mode in ["frontend", "ivy"]:
        if buffer is not None:
            buffer.write(
                "<stack trace is truncated to {} specific files,".format(trace_mode),
                "call `ivy.set_exception_trace_mode('full')` to view the full trace>",
            )
        else:
            print(
                "<stack trace is truncated to {} specific files,".format(trace_mode),
                "call `ivy.set_exception_trace_mode('full')` to view the full trace>",
            )
    if not func_wrapper_trace_mode:
        if buffer is not None:
            buffer.write(
                "<func_wrapper.py stack trace is squashed,",
                "call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>",
            )
        else:
            print(
                "<func_wrapper.py stack trace is squashed,",
                "call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>",
            )


def _print_or_buffered_new_stack_trace(old_stack_trace, trace_mode, func_wrapper_trace_mode, buffer=None):
    _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode, buffer)
    new_stack_trace = []
    for st in old_stack_trace:
        if trace_mode == "full" and not func_wrapper_trace_mode:
            if "func_wrapper.py" not in repr(st):
                new_stack_trace.append(st)
        else:
            if ivy.trace_mode_dict[trace_mode] in repr(st):
                if not func_wrapper_trace_mode and "func_wrapper.py" in repr(st):
                    continue
                new_stack_trace.append(st)
    if buffer is not None:
        buffer.write("".join(tb.format_list(new_stack_trace)))
    else:
        print("".join(tb.format_list(new_stack_trace)))


def _custom_exception_handle(type, value, tb_history, buffer=None):
    trace_mode = ivy.get_exception_trace_mode()
    func_wrapper_trace_mode = ivy.get_show_func_wrapper_trace_mode()
    if trace_mode == "none":
        return
    if trace_mode == "full" and func_wrapper_trace_mode:
        print("".join(tb.format_tb(tb_history)))
    else:
        _print_or_buffered_new_stack_trace(
            tb.extract_tb(tb_history), trace_mode, func_wrapper_trace_mode, buffer
        )
    print(type.__name__ + ":", value)


def _buffered_traceback_history(buffer):
    trace_mode = ivy.get_exception_trace_mode()
    func_wrapper_trace_mode = ivy.get_show_func_wrapper_trace_mode()
    if trace_mode == "none":
        return
    if trace_mode == "full" and func_wrapper_trace_mode:
        buffer.write("".join(tb.format_tb(sys.exc_info()[2])))
    else:
        _print_or_buffered_new_stack_trace(
            tb.extract_tb(sys.exc_info()[2]), trace_mode, func_wrapper_trace_mode, buffer
        )
    buffer.write("During the handling of the above exception, another exception occurred:\n")


sys.excepthook = _custom_exception_handle


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

    global buffer
    buffer = io.StringIO()

    @functools.wraps(fn)
    def _handle_exceptions(*args, **kwargs):
        """
        Catch all exceptions and raise them in IvyException.

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
        # Not to rethrow as IvyBackendException
        except IvyNotImplementedException as e:
            raise e
        except (IndexError, ValueError, AttributeError) as e:
            # cleaning the buffer
            buffer.truncate(0)
            buffer.seek(0)
            
            # storing the configured stack to buffer
            _buffered_traceback_history(buffer)
            raise ivy.utils.exceptions.IvyError(fn.__name__, buffer.getvalue() + " "+ str(e))
        except Exception as e:
            # cleaning the buffer
            buffer.truncate(0)
            buffer.seek(0)
            
            # storing the configured stack to buffer
            _buffered_traceback_history(buffer)
            raise ivy.utils.exceptions.IvyBackendException(fn.__name__, buffer.getvalue() + " "+ str(e))

    _handle_exceptions.handle_exceptions = True
    return _handle_exceptions
