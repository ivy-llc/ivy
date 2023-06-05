import ivy
import functools
from typing import Callable
import sys
import traceback as tb


# Helpers #
# ------- #


def _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode):
    if trace_mode in ["frontend", "ivy"]:
        print(
            "<stack trace is truncated to {} specific files,".format(trace_mode),
            "call `ivy.set_exception_trace_mode('full')` to view the full trace>",
        )
    if not func_wrapper_trace_mode:
        print(
            "<func_wrapper.py stack trace is squashed,",
            "call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>",
        )


def _print_new_stack_trace(old_stack_trace, trace_mode, func_wrapper_trace_mode):
    _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode)
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
    print("".join(tb.format_list(new_stack_trace)))


def _custom_exception_handle(type, value, tb_history):
    trace_mode = ivy.get_exception_trace_mode()
    func_wrapper_trace_mode = ivy.get_show_func_wrapper_trace_mode()
    if trace_mode == "none":
        return
    if trace_mode == "full" and func_wrapper_trace_mode:
        print("".join(tb.format_tb(tb_history)))
    else:
        _print_new_stack_trace(
            tb.extract_tb(tb_history), trace_mode, func_wrapper_trace_mode
        )
    print(type.__name__ + ":", value)


def _print_traceback_history():
    trace_mode = ivy.get_exception_trace_mode()
    func_wrapper_trace_mode = ivy.get_show_func_wrapper_trace_mode()
    if trace_mode == "none":
        return
    if trace_mode == "full" and func_wrapper_trace_mode:
        print("".join(tb.format_tb(sys.exc_info()[2])))
    else:
        _print_new_stack_trace(
            tb.extract_tb(sys.exc_info()[2]), trace_mode, func_wrapper_trace_mode
        )
    print("During the handling of the above exception, another exception occurred:\n")


sys.excepthook = _custom_exception_handle


def _add_native_error(default):
    """
    Append the native error to the message if it exists.

    Parameters
    ----------
    default
        list containing all the messages

    Returns
    -------
    ret
        list containing all the messages, with the native error appended if it exists
    """
    trace_mode = ivy.get_exception_trace_mode()
    if isinstance(default[-1], Exception):
        if isinstance(default[-1], IvyException):
            if default[-1].native_error is not None:
                # native error was passed in the message
                native_error = default[-1].native_error
            else:
                # a string was passed in the message
                # hence the last element is an IvyException
                default[-1] = str(default[-1])
                return default
        else:
            # exception was raised by the backend natively
            native_error = default[-1]
        if trace_mode == "full":
            default[-1] = native_error.__class__.__name__
            default.append(str(native_error))
        else:
            default[-1] = str(native_error)
    return default


def _combine_messages(*messages, include_backend=True):
    if not include_backend:
        return " ".join(messages)
    default = [
        "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
    ]
    delimiter = ": "
    for message in messages:
        default.append(message)

    # adding the native error as well if it exists and the trace mode is set to "full"
    default = _add_native_error(default)
    return delimiter.join(default)


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


class IvyBackendException(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyNotImplementedException(NotImplementedError):
    def __init__(self, message=""):
        super().__init__(message)


class IvyError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyIndexError(IvyException, IndexError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyAttributeError(IvyException, AttributeError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyValueError(IvyException, ValueError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyBroadcastShapeError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyDtypePromotionError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


def handle_exceptions(fn: Callable) -> Callable:
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
