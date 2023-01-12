# global
import inspect
import re

# local
import ivy.functional.frontends.numpy as np_frontend


# Class #
# ----- #


class ufunc:
    def __init__(self, name) -> None:
        self.__name__ = name
        self.func = getattr(np_frontend, self.__name__)

    # properties #
    # ------------#

    @property
    def nargs(self):
        sig = inspect.signature(self.func)
        return len(
            [
                param
                for param in sig.parameters.values()
                if param.kind == param.POSITIONAL_ONLY
            ]
        )

    @property
    def nin(self):
        return self.args

    @property
    def nout(self):
        ret = inspect.getsourcelines(self.func)
        ret_pattern = r"return\s*(.*)\n*$"
        returns = re.findall(ret_pattern, ret[0][-1])
        return len(returns[0].split(","))

    @property
    def ntypes(self):
        pass

    @property
    def signature(self):
        pass

    @property
    def types(self):
        pass

    # Methods #
    # ---------#

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        pass
