# Copyright 2022 The ml_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = '0.0.4'  # Keep in sync with pyproject.toml:version
__all__ = [
    '__version__',
    'bfloat16',
    'finfo',
    'float8_e4m3b11',
    'float8_e4m3fn',
    'float8_e5m2',
]

from typing import Type

from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e5m2
from ml_dtypes._finfo import finfo

import numpy as np

bfloat16: Type[np.generic]
float8_e4m3b11: Type[np.generic]
float8_e4m3fn: Type[np.generic]
float8_e5m2: Type[np.generic]

del np, Type
