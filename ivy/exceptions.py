import ivy
import functools
from typing import Callable


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


class IvyError(IndexError, ValueError, IvyException):
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
        except (IndexError, ValueError) as e:
            raise ivy.exceptions.IvyError(fn.__name__, str(e)) from None
        except Exception as e:
            raise ivy.exceptions.IvyBackendException(fn.__name__, str(e)) from None

    new_fn.handle_exceptions = True
    return new_fn
