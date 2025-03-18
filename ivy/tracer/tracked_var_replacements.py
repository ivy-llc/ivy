# this file makes these functions available through the gc namespace.
# Probably needs to be revisited depending on the traced binary requirements.
from . import tracked_var_proxy as tvp


def len(x):
    return tvp.TrackedVarProxy.tvp__len__(x)


def int(x):
    return tvp.TrackedVarProxy.tvp__int__(x)


def float(x):
    return tvp.TrackedVarProxy.tvp__float__(x)
