import sys
import ivy

from .core import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])
