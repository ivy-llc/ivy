import ivy

# import functools
# from typing import Callable


class IvyException(Exception):
    def __init__(self, message):
        self._default_msg = ivy.current_backend_str() + ": "
        super().__init__(self._default_msg + message)
