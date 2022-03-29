"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
import numpy as np
torch_scatter = None
import math as _math
import torch as torch
from operator import mul
from torch.types import Number
from functools import reduce as _reduce
from typing import List, Dict, Optional, Union


# local
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.torch.device import dev_from_str, _callable_dev


# API #
# ----#








