import ivy
import functools
from typing import Callable


def handle_numpy_casting(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, casting=None, **kwargs):
        """
        Check numpy casting type.

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
        # TODO: implement!
        ivy.array(casting)
        return fn(*args, **kwargs)

    new_fn.handle_exceptions = True
    return new_fn
