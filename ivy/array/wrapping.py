# local
import ivy


def _wrap_function(function_name):
    """Wraps the function called `function_name`.

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
    >>> ivy.set_framework('torch')
    >>> from ivy.array.wrapping import _wrap_function
    >>> absolute = _wrap_function("abs")
    >>> x = ivy.array([-1])
    >>> print(absolute(x))
    ivy.array([1])

    """

    def new_function(self, *args, **kwargs):
        """Add the data of the current array from which the instance function is invoked
        as the first arg parameter or kwarg parameter. Return the new function with
        the name function_name and the new args variable or kwargs as the new inputs.
        """
        function = ivy.__dict__[function_name]
        # gives us the position and name of the array argument
        data_idx = function.array_spec[0]
        if len(args) > data_idx[0][0]:
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


def add_ivy_array_instance_methods(cls, modules, to_ignore=()):
    """
    Loop over all ivy modules such as activations, general, etc. and add
    the module functions to ivy arrays as instance methods using _wrap_fn.
    """
    for module in modules:
        for key, val in module.__dict__.items():
            if (
                key.startswith("_")
                or key[0].isupper()
                or not callable(val)
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
