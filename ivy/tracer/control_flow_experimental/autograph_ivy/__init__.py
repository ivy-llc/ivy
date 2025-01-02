# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from . import core
from .core import *
from . import converters
from .converters import *
from . import operators
from .operators import *
from . import pyct
from .pyct import *
from . import test_suite
from .test_suite import *

"""Conversion of eager-style Python into TensorFlow graph code.

NOTE: In TensorFlow 2.0, AutoGraph is automatically applied when using
`tf.function`. This module contains lower-level APIs for advanced use.

AutoGraph transforms a subset of Python which operates on TensorFlow objects
into equivalent TensorFlow graph code. When executing the graph, it has the same
effect as if you ran the original code in eager mode.
Python code which doesn't operate on TensorFlow objects remains functionally
unchanged, but keep in mind that `tf.function` only executes such code at trace
time, and generally will not be consistent with eager execution.

For more information, see the
[AutoGraph reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md),
and the [tf.function guide](https://www.tensorflow.org/guide/function#autograph_transformations).
"""
