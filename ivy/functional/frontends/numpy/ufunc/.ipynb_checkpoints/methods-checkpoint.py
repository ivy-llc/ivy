# global
import inspect
from math import inf

# local
import ivy.functional.frontends.numpy as np_frontend

identities = {
    "abs": None,
    "absolute": None,
    "add": 0,
    "arccos": None,
    "arccosh": None,
    "arcsin": None,
    "arcsinh": None,
    "arctan": None,
    "arctan2": None,
    "arctanh": None,
    "bitwise_and": -1,
    "bitwise_not": None,
    "bitwise_or": 0,
    "bitwise_xor": 0,
    "cbrt": None,
    "ceil": None,
    "conj": None,
    "conjugate": None,
    "copysign": None,
    "cos": None,
    "cosh": None,
    "deg2rad": None,
    "degrees": None,
    "divide": None,
    "divmod": None,
    "equal": None,
    "exp": None,
    "exp2": None,
    "expm1": None,
    "fabs": None,
    "float_power": None,
    "floor": None,
    "floor_divide": None,
    "fmax": None,
    "fmin": None,
    "fmod": None,
    "frexp": None,
    "gcd": 0,
    "greater": None,
    "greater_equal": None,
    "heaviside": None,
    "hypot": 0,
    "invert": None,
    "isfinite": None,
    "isinf": None,
    "isnan": None,
    "isnat": None,
    "lcm": None,
    "ldexp": None,
    "left_shift": None,
    "less": None,
    "less_equal": None,
    "log": None,
    "log10": None,
    "log1p": None,
    "log2": None,
    "logaddexp": -inf,
    "logaddexp2": -inf,
    "logical_and": True,
    "logical_not": None,
    "logical_or": False,
    "logical_xor": False,
    "matmul": None,
    "maximum": None,
    "minimum": None,
    "mod": None,
    "modf": None,
    "multiply": 1,
    "negative": None,
    "nextafter": None,
    "not_equal": None,
    "positive": None,
    "power": None,
    "rad2deg": None,
    "radians": None,
    "reciprocal": None,
    "remainder": None,
    "right_shift": None,
    "rint": None,
    "sign": None,
    "signbit": None,
    "sin": None,
    "sinh": None,
    "spacing": None,
    "sqrt": None,
    "square": None,
    "subtract": None,
    "tan": None,
    "tanh": None,
    "true_divide": None,
    "trunc": None,
}
# constants #
# --------#
ufuncs = [
    "abs",
    "absolute",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "divmod",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "invert",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
]


# Class #
# ----- #


class ufunc:
    def __init__(self, name) -> None:
        self.__frontend_name__ = name
        # removing first underscore to get original ufunc name
        self.__name__ = name[1:]
        # getting the function from the frontend
        self.func = getattr(np_frontend, self.__frontend_name__)

    # properties #
    # ------------#

    @property
    def nargs(self):
        sig = inspect.signature(self.func)
        return len(
            [
                param
                for param in sig.parameters.values()
                if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]
            ]
        )

    @property
    def nin(self):
        sig = inspect.signature(self.func)
        return len(
            [
                param
                for param in sig.parameters.values()
                if param.kind == param.POSITIONAL_ONLY
            ]
        )

    @property
    def nout(self):
        return self.nargs - self.nin

    @property
    def ntypes(self):
        pass

    @property
    def signature(self):
        pass

    @property
    def types(self):
        pass

    @property
    def identity(self):
        return identities[self.__name__]

    # Methods #
    # ---------#

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def reduce(
        array, axis=0, dtype=None, out=None, keepdims=False, initial=None, where=True
    ):
        pass

    def accumulate(array, axis=0, dtype=None, out=None):
        pass

    def reduceat(array, indices, axis=0, dtype=None, out=None):
        pass

    def outer(A, B, /, **kwargs):
        pass

    def at(a, indices, b=None, /):
        pass
