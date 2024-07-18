import copy
import functools


def tensorflow_store_config_info(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if all(
            [
                hasattr(self, "_args"),
                hasattr(self, "_kwargs"),
                hasattr(self, "_self_tracked_trackables"),
            ]
        ):
            orig_trackables = copy.copy(self._self_tracked_trackables)
            self._args = (self,) + args
            self._kwargs = kwargs
            self._self_tracked_trackables = orig_trackables

    return wrapper
