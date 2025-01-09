# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts the ternary conditional operator."""

import gast

from ..core import converter
from ..pyct import parser
from ..pyct import templates


class ConditionalExpressionTransformer(converter.Base):
    """Converts conditional expressions to functional form."""

    def visit_IfExp(self, node):
        node = self.generic_visit(node)

        template = """
                ivy.if_else(
                        test,
                        lambda: true_expr,
                        lambda: false_expr,
                        ())
        """
        return templates.replace_as_expression(
            template, test=node.test, true_expr=node.body, false_expr=node.orelse
        )


def transform(node, ctx):
    node = ConditionalExpressionTransformer(ctx).visit(node)
    return node
