# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Handles control flow statements: while, for, if."""

import gast

from ..core import converter
from ..pyct import anno
from ..pyct import cfg
from ..pyct import origin_info
from ..pyct import qual_names
from ..pyct import templates
from ..pyct.static_analysis import activity
from ..pyct.static_analysis import annos
from ..pyct.static_analysis import liveness
from ..pyct.static_analysis import (
    reaching_definitions,
)
from ..pyct.static_analysis import reaching_fndefs


class _Function(object):
    scope = None


class ControlFlowTransformer(converter.Base):
    """Transforms control flow structures like loops an conditionals."""

    def visit_Lambda(self, node):
        with self.state[_Function] as fn:
            fn.scope = anno.getanno(node, anno.Static.SCOPE)
            return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        with self.state[_Function] as fn:
            fn.scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
            return self.generic_visit(node)

    def _create_nonlocal_declarations(self, vars_):
        vars_ = set(vars_)
        results = []
        global_vars = self.state[_Function].scope.globals & vars_

        if global_vars:
            results.append(gast.Global([str(v) for v in global_vars]))

        nonlocal_vars = [
            v for v in vars_ if not v.is_composite() and v not in global_vars
        ]
        if nonlocal_vars:
            results.append(gast.Nonlocal([str(v) for v in nonlocal_vars]))

        return results

    def _create_undefined_assigns(self, undefined_symbols):
        assignments = []
        for s in undefined_symbols:
            template = """
                var = None
            """

            assignments += templates.replace(
                template, var=s, symbol_name=gast.Constant(s.ssf(), kind=None)
            )
        return assignments

    def _get_block_basic_vars(self, modified, live_in, live_out):
        nonlocals = self.state[_Function].scope.nonlocals
        basic_scope_vars = []
        for s in modified:
            if s.is_composite():
                # TODO(mdan): Raise an error when this happens for a TF scope.
                continue
            # Variables not live into or out of the scope are considered local to the
            # scope.
            if s in live_in or s in live_out or s in nonlocals:
                basic_scope_vars.append(s)
            continue
        return frozenset(basic_scope_vars)

    def _get_block_composite_vars(self, modified, live_in):
        # The scope variables corresponding to composite symbols (e.g. `self.x`).
        composite_scope_vars = []
        for s in modified:
            if not s.is_composite():
                continue
            # Mutations made to objects created inside the scope will appear as writes
            # to composite symbols. Because these mutations appear as modifications
            # made to composite symbols, we check whether the composite's parent is
            # actually live into the scope.
            # Example:
            #     while cond:
            #         x = Foo()
            #         x.foo = 2 * x.foo    # x.foo is live into the scope, but x is not.
            #
            # Note that some parents might not be symbols - for example, in x['foo'],
            # 'foo' is a parent, but it's a literal, not a symbol. We don't check the
            # liveness of literals.
            support_set_symbols = tuple(sss for sss in s.support_set if sss.is_symbol())
            if not all(sss in live_in for sss in support_set_symbols):
                continue
            composite_scope_vars.append(s)
        return frozenset(composite_scope_vars)

    def extract_variable_name(self, expr):
        node = gast.parse(expr, mode="eval")
        if isinstance(node.body, (gast.Subscript, gast.Attribute)):
            return node.body.value.id
        else:
            return node.body.id

    def process_variables(self, var_list):
        processed_vars = []
        for var in var_list:
            processed_vars.append(self.extract_variable_name(str(var)))
        return processed_vars

    def _get_block_vars(self, node, modified):
        """Determines the variables affected inside a control flow statement."""
        defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
        live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
        live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
        fn_scope = self.state[_Function].scope

        basic_scope_vars = self._get_block_basic_vars(modified, live_in, live_out)
        composite_scope_vars = self._get_block_composite_vars(modified, live_in)
        scope_vars = tuple(basic_scope_vars | composite_scope_vars)
        scope_vars = (v for v in scope_vars if v not in fn_scope.globals)

        # Variables that are modified inside the scope, but not defined
        # before entering it. Only simple variables must be defined. The
        # composite ones will be implicitly checked at runtime.
        possibly_undefined = (
            modified - defined_in - fn_scope.globals - fn_scope.nonlocals
        )
        undefined = tuple(v for v in possibly_undefined if not v.is_composite())

        # Variables that are modified inside the scope, and depend on values outside
        # it.
        input_only = basic_scope_vars & live_in - live_out

        # Place the outputs first, then sort lexicographically.
        scope_vars = sorted(scope_vars, key=lambda v: (v in input_only, v))
        scope_vars = self.process_variables(scope_vars)
        nouts = len(scope_vars) - len(input_only)

        scope_vars = list(set(scope_vars))

        return scope_vars, undefined, nouts

    def _create_variables(self, var_names):
        if not var_names:
            return gast.Tuple(elts=[], ctx=gast.Load())

        var_nodes = []
        for var_name in var_names:
            var_node = gast.Name(
                id=str(var_name), ctx=gast.Load(), annotation=None, type_comment=None
            )
            var_nodes.append(var_node)

        tuple_node = gast.Tuple(elts=var_nodes, ctx=gast.Load())

        return tuple_node

    def visit_If(self, node):
        node = self.generic_visit(node)
        body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
        orelse_scope = anno.getanno(node, annos.NodeAnno.ORELSE_SCOPE)

        cond_vars, undefined, _ = self._get_block_vars(
            node, body_scope.bound | orelse_scope.bound
        )

        undefined_assigns = self._create_undefined_assigns(undefined)

        # nonlocal_declarations = self._create_nonlocal_declarations(cond_vars)

        tuple_vars = self._create_variables(cond_vars)
        return_nodes = gast.Return(value=tuple_vars)

        # cond_params = self.process_variables(cond_vars)

        reserved = body_scope.referenced | orelse_scope.referenced

        orelse_body = node.orelse
        # if not orelse_body:
        #         orelse_body = [gast.Pass()]

        template = """
                def body_name(cond_vars):
                    body
                    return_nodes
                def orelse_name(cond_vars):
                    orelse
                    return_nodes
                undefined_assigns
                
                tuple_vars = ivy.if_else(
                    test,
                    body_name,
                    orelse_name,
                    tuple_vars,
                    )
            """
        new_nodes = templates.replace(
            template,
            body=node.body,
            body_name=self.ctx.namer.new_symbol("if_body", reserved),
            orelse=orelse_body,
            orelse_name=self.ctx.namer.new_symbol("else_body", reserved),
            cond_vars=cond_vars,
            return_nodes=return_nodes,
            tuple_vars=tuple_vars,
            test=node.test,
            undefined_assigns=undefined_assigns,
        )
        origin_info.copy_origin(node, new_nodes[-1])
        return new_nodes

    def visit_While(self, node):
        node = self.generic_visit(node)
        body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

        loop_vars, undefined, _ = self._get_block_vars(node, body_scope.bound)

        undefined_assigns = self._create_undefined_assigns(undefined)

        # nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)

        tuple_vars = self._create_variables(loop_vars)
        return_nodes = gast.Return(value=tuple_vars)

        # loop_params = self.process_variables(loop_vars)

        reserved = body_scope.referenced

        template = """ 
                    def body_name(loop_vars):
                        body
                        return_nodes
                    def test_name(loop_vars):
                        return test
                    undefined_assigns
                    
                    tuple_vars = ivy.while_loop(
                            test_name,
                            body_name,
                            tuple_vars,
                            )
                """
        new_nodes = templates.replace(
            template,
            body=node.body,
            body_name=self.ctx.namer.new_symbol("loop_body", reserved),
            loop_vars=loop_vars,
            return_nodes=return_nodes,
            tuple_vars=tuple_vars,
            test=node.test,
            test_name=self.ctx.namer.new_symbol("loop_test", reserved),
            undefined_assigns=undefined_assigns,
        )
        origin_info.copy_origin(node, new_nodes[-1])
        return new_nodes

    def visit_For(self, node):
        node = self.generic_visit(node)
        body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
        iter_scope = anno.getanno(node, annos.NodeAnno.ITERATE_SCOPE)

        loop_vars, undefined, _ = self._get_block_vars(
            node, body_scope.bound | iter_scope.bound
        )

        undefined_assigns = self._create_undefined_assigns(undefined)

        reserved = body_scope.referenced | iter_scope.referenced

        tuple_vars = self._create_variables(loop_vars)
        return_nodes = gast.Return(value=tuple_vars)

        # itr holds a single arg with the iterates, which may be a
        # tuple.
        itr = self.ctx.namer.new_symbol("itr", reserved)
        template = """
            node_target = itr
        """
        iterate_expansion = templates.replace(
            template, itr=itr, node_target=node.target
        )
        origin_info.copy_origin(node, iterate_expansion)

        template = """
            def body_name(itr, _v):
                tuple_vars = _v
                iterate_expansion
                body
                return_nodes
            undefined_assigns
            tuple_vars = ivy.for_loop(
                    node_itr,
                    body_name,
                    tuple_vars
                    )
        """
        new_nodes = templates.replace(
            template,
            body=node.body,
            body_name=self.ctx.namer.new_symbol("loop_body", reserved),
            iterate_arg_name=itr,
            iterate_expansion=iterate_expansion,
            node_itr=node.iter,
            undefined_assigns=undefined_assigns,
            tuple_vars=tuple_vars,
            return_nodes=return_nodes,
        )
        origin_info.copy_origin(node, new_nodes[-1])
        return new_nodes


class AnnotatedDef(reaching_definitions.Definition):
    def __init__(self):
        super(AnnotatedDef, self).__init__()
        self.directives = {}


def transform(node, ctx):
    graphs = cfg.build(node)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)
    node = reaching_definitions.resolve(node, ctx, graphs)
    node = reaching_fndefs.resolve(node, ctx, graphs)
    node = liveness.resolve(node, ctx, graphs)

    node = ControlFlowTransformer(ctx).visit(node)
    return node
