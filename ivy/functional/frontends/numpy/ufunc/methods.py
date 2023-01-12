# global
import inspect

# local
import ivy.functional.frontends.numpy as np_frontend


# Class #
# ----- #


class ufunc:
    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, *args, **kwargs):
        return getattr(np_frontend, self.name)(*args, **kwargs)

    # properties #
    # ------------#
    @property
    def nargs(self):
        """
        Total number of arguments of the given ufunc.
        """
        return len(inspect.signature(getattr(np_frontend, self.name)).parameters)
