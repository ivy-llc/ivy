import sys


import ivy.functional.frontends.torch as torch
import ivy
from ivy.functional.frontends import set_frontend_to_specific_version


from . import ops


tensor = _frontend_array = torch.tensor


# setting to specific version #
# --------------------------- #

if ivy.is_local():
    module = ivy.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

set_frontend_to_specific_version(module)
