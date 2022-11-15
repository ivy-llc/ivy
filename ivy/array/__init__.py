# flake8: noqa
# global
import copy
import functools
import numpy as np
from operator import mul

# local
from .wrapping import add_ivy_array_instance_methods
from .wrapping import add_ivy_array_special_methods
from .wrapping import add_ivy_array_reverse_special_methods
from .array import Array
