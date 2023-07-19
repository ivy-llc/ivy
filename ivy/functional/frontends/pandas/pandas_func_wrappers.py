# function wrappers for pandas frontend to handle commmon operations
from functools import wraps


def outputs_to_self_class(func):
    @wraps(func)
    def _outputs_to_self_class(*args, **kwargs):
        self_arg = args[0]
        return self_arg.__class__(
            func(*args, **kwargs),
            index=self_arg.index,
            columns=self_arg.columns,
            dtype=self_arg.dtype,
            name=self_arg.name,
            copy=self_arg.copy,
        )

    return _outputs_to_self_class
