# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Support for wrapping converted functions bodies with auxiliary logic."""

from ..core import converter, ops
from ..operators import variables


class FunctionScope(object):
    """Context manager that wraps the body of a converted function.

    This context manager handles various operations related to the scope of a
    function:
    """

    def __init__(self, function_name, scope_name, options):
        self.name = scope_name
        self.options = options
        self.callopts = options.call_options()

        use_name_scope = options.uses(converter.Feature.NAME_SCOPES)
        self.use_name_scope = use_name_scope
        if use_name_scope:
            self.name_scope = ops.name_scope(self._sanitize(function_name))

    def _sanitize(self, name):
        """See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope."""
        # TensorFlow doesn't like leading underscores at the top level.
        if name and name.startswith("_"):
            name = "fn" + name
        return name

    def __enter__(self):
        if self.use_name_scope:
            self.name_scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_name_scope:
            self.name_scope.__exit__(exc_type, exc_val, exc_tb)

    def ret(self, value, did_return):
        """Marks a value as returned from the function guarded by the scope."""
        del did_return

        if isinstance(value, variables.UndefinedReturnValue):
            return None

        return value


def with_function_scope(thunk, scope_name, options):
    """Inline version of the FunctionScope context manager."""
    with FunctionScope("lambda_", scope_name, options) as scope:
        return thunk(scope)
