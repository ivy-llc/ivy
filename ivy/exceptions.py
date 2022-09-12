import ivy
import functools
from typing import Callable


class IvyException(Exception):
    def __init__(self, message):
        self._default_msg = ivy.current_backend_str() + ": "
        super().__init__(self._default_msg + message)


def handle_exceptions(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Catch all backend exceptions and throw them in ivy.IvyException

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, or raise ivy.IvyException if error is thrown.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise ivy.IvyException(str(e))

    new_fn.handle_exceptions = True
    return new_fn
