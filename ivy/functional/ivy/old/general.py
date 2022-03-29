"""
Collection of general Ivy functions.
"""

# global
import gc
import math
import einops
import inspect
import numpy as np
from numbers import Number
from typing import Callable, Any, Union, List, Tuple, Dict, Iterable, Optional

# local
import ivy
from ivy.functional.ivy.device import dev
from ivy.framework_handler import current_framework as _cur_framework

FN_CACHE = dict()
INF = float('inf')
TIMEOUT = 15.0
TMP_DIR = '/tmp'























































