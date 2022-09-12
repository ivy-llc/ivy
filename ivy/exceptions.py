import ivy
import functools
from typing import Callable


class IvyException(Exception):
    def __init__(self, fn_name, message):
        self._default = (
            "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
        )
        self._delimiter = ": "
        super().__init__(self._delimiter.join([fn_name, self._default, message]))


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
        except Exception as e:
            raise ivy.exceptions.IvyException(fn.__name__, str(e)) from None

    new_fn.handle_exceptions = True
    return new_fn
