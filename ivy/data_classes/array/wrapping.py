# local
import ivy

# global
from typing import Callable, Type, List, Iterable
from types import ModuleType

TO_IGNORE = ["shape"]


def _wrap_function(function_name: str) -> Callable:
    """
    Wrap the function called `function_name`.

    Parameters
    ----------
    function_name
        the name of the function e.g. "abs", "mean" etc.

    Returns
    -------
    new_function
        the wrapped function.

    Examples
    --------
    >>> ivy.set_backend("torch")
    >>> from ivy.array.wrapping import _wrap_function
    >>> absolute = _wrap_function("abs")
    >>> x = ivy.array([-1])
    >>> print(absolute(x))
    ivy.array([1])
    """

    def new_function(self, *args, **kwargs):
        """
        Add the data of the current array from which the instance function is invoked as
        the first arg parameter or kwarg parameter.

        Return the new function with the name function_name and the new
        args variable or kwargs as the new inputs.
        """
        function = ivy.__dict__[function_name]
        # gives us the position and name of the array argument
        data_idx = function.array_spec[0]
        if len(args) >= data_idx[0][0]:
            args = ivy.copy_nest(args, to_mutable=True)
            data_idx = [data_idx[0][0]] + [
                0 if idx is int else idx for idx in data_idx[1:]
            ]
            ivy.insert_into_nest_at_index(args, data_idx, self._data)
        else:
            kwargs = ivy.copy_nest(kwargs, to_mutable=True)
            data_idx = [data_idx[0][1]] + [
                0 if idx is int else idx for idx in data_idx[1:]
            ]
            ivy.insert_into_nest_at_index(kwargs, data_idx, self._data)
        return function(*args, **kwargs)

    return new_function


def add_ivy_array_instance_methods(
    cls: Type[ivy.Array], modules: List[ModuleType], to_ignore: Iterable = ()
):
    """
    Loop over all ivy modules such as activations, general, etc. and add the module
    functions to ivy arrays as instance methods using _wrap_function.

    Parameters
    ----------
    cls
        the class we want to add the instance methods to.
    modules
        the modules to loop over: activations, general etc.
    to_ignore
        any items we don't want to add an instance method for.

    Examples
    --------
    As shown, `add_ivy_array_instance_methods` adds all the appropriate functions from
    the activations module as instance methods to our toy `ArrayExample` class:

    >>> from ivy.functional.ivy import activations
    >>> class ArrayExample:
    ...     pass
    >>> ivy.add_ivy_array_instance_methods(ArrayExample, [activations])
    >>> print(hasattr(ArrayExample, "relu"), hasattr(ArrayExample, "softmax"))
    True True
    """
    to_ignore = TO_IGNORE + list(to_ignore)
    for module in modules:
        for key, value in module.__dict__.items():
            # we skip the cases where the function is protected, the instance
            # method has already been added manually and a few other cases
            if (
                key.startswith("_")
                or key[0].isupper()
                or not callable(value)
                or key in cls.__dict__
                or hasattr(cls, key)
                or key in to_ignore
                or key not in ivy.__dict__
            ):
                continue
            try:
                setattr(cls, key, _wrap_function(key))
            except AttributeError:
                pass
