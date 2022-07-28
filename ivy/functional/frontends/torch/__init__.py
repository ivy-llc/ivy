# flake8: noqa
version = ['latest_version']
faulty_version = {'1.11.0+cpu': [ones_like]}
try:
    import torch
version.append(torch.__version__)
except:
print("Torch not installed, assuming latest version: ", version[-1])

if version[-1] in faulty_version:
    unusual_imports(1)  # we import with faulty functions replaced with version specific ones
else:
    unusual_imports(0)

import importlib
from . import indexing_slicing_joining_mutating_ops
from .indexing_slicing_joining_mutating_ops import *
from . import pointwise_ops
from .pointwise_ops import *
from . import creation_ops
from .creation_ops import *

faulty_func={'1.11.0+cpu':[ones_like]}
if torch.__version__ in faulty_func:
    for i in faulty_func[torch.__version__]:
        importlib.import_module('.creation_ops.ones_like_110',i.__name__)
