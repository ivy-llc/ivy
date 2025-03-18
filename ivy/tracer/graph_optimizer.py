from _ast import AST, Assign, Call, Delete
from typing import Any, Callable, Dict, Union, Tuple, List
from abc import ABC, abstractmethod
import ast
import re
from tracer.graph import Graph
from tracer import globals as glob


class Check(ast.NodeVisitor, ABC):
    @abstractmethod
    def is_possible(self) -> bool:
        pass

    @abstractmethod
    def get_contents(self) -> Dict[str, Any]:
        pass


class Apply(ast.NodeTransformer):
    pass


# Check Classes #
# --------------#


class DataFormatCheck(Check):
    conv_blocks: List[Assign]
    conv_block_variables: Dict[int, List[str]]
    activations: List[str]
    inside_block: bool
    backend: str

    def __init__(self, backend: str = ""):
        self.backend = backend
        self.conv_blocks = []
        self.conv_block_variables = {}
        self.activations = ["relu", "sigmoid"]
        self.inside_block = False
        super().__init__()

    def _check_call_method(self, node: Call):
        keyword_args = [keyword.arg for keyword in node.keywords]
        if self.backend == "tensorflow" and "data_format" in keyword_args:
            return keyword_args
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.activations:
            return True
        return None

    def is_possible(self, node):
        self.visit(node)
        return bool(self.conv_blocks)

    def get_contents(self):
        return {
            "conv_blocks": self.conv_blocks,
            "conv_block_variables": self.conv_block_variables,
            "backend": self.backend,
            "check_call_method": self._check_call_method,
        }

    def visit_Delete(self, node: Delete) -> Any:
        if self.inside_block:
            for target in node.targets:
                block_variables = self.conv_block_variables[len(self.conv_blocks) - 1]
                if isinstance(target, ast.Name) and target.id in block_variables:
                    block_variables.remove(target.id)
        return super().generic_visit(node)

    def visit_Assign(self, node: Assign) -> Any:
        if isinstance(node.value, ast.Call):
            if keyword_args := self._check_call_method(node.value):
                if not isinstance(keyword_args, bool) and not self.inside_block:
                    self.conv_blocks.append([node, None])
                    self.inside_block = True
                if self.inside_block:
                    self.conv_blocks[-1][1] = node
            elif self.conv_blocks:
                self.conv_blocks[-1][1] = (
                    self.conv_blocks[-1][1]
                    if self.conv_blocks[-1][1]
                    else self.conv_blocks[-1][0]
                )
                self.inside_block = self.inside_block and False
        if self.inside_block:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if len(self.conv_blocks) - 1 not in self.conv_block_variables:
                        self.conv_block_variables[len(self.conv_blocks) - 1] = [
                            target.id
                        ]
                    else:
                        self.conv_block_variables[len(self.conv_blocks) - 1].append(
                            target.id
                        )
        return node


# Apply Classes #
# --------------#


class DataFormatApply(Apply):
    current_block: int = 0
    conv_blocks: List[Assign] = None
    conv_block_variables: Dict[int, List[str]]
    backend: str

    def __init__(
        self,
        conv_blocks: List[Assign] = None,
        conv_block_variables: Dict[int, List[str]] = None,
        backend: str = None,
        check_call_method: Callable = None,
    ):
        self.conv_blocks = conv_blocks
        self.conv_block_variables = conv_block_variables
        self.backend = backend
        self.check_call_method = check_call_method
        self.input_var_name = None
        super().__init__()

    def _get_start_transpose(self, node: Call):
        self.input_var_name = node.args[0].id
        return ast.parse(
            # we create a new variable prepended with `optimization_` to avoid
            # modifiying a variable which can be used by other parts of the graph
            f"optimization_{node.args[0].id} = ivy.permute_dims(\n"
            f"    {node.args[0].id},\n"
            f"    axes=(0, *range(2, {node.args[0].id}.ndim), 1)\n"
            ")"
        ).body[0]

    def _get_end_transpose(self, variables: List[str]):
        return [
            ast.parse(
                f"permute_func_{variables[0]} = lambda x: ivy.permute_dims(\n"
                f"    x, axes=(0, x.ndim-1, *range(1, x.ndim-1))\n"
                f") if ivy.is_array(x) else x\n"
            ).body[0]
        ] + [
            ast.parse(
                f"{variable} = (\n"
                f"    permute_func_{variables[0]}({variable})\n"
                f"    if ivy.is_array({variable})\n"
                f"    else permute_func_{variables[0]}({variable}[0])\n"
                f"    if isinstance({variable}, (list, tuple))\n"
                f"    else {variable}\n"
                ")"
            ).body[0]
            for variable in variables
        ]

    def _get_node(self, node):
        # if we are in the middle of optimizing a node, we need to replace the argument variables
        # which had been changed by `_get_start_transpose` to avoid side effects
        if self.input_var_name:

            # replace modified variable names in the args
            new_args = []
            for arg in node.value.args:
                if isinstance(arg, ast.Name) and arg.id == self.input_var_name:
                    new_args.append(ast.Name(id="optimization_" + arg.id, ctx=ast.Store()))
                else:
                    new_args.append(arg)
            node.value.args = new_args

            # replace modified variable names in the kwargs
            new_keywords = []
            for kw in node.value.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id == self.input_var_name:
                    new_keywords.append(ast.Name(id="optimization_" + kw.value.id, ctx=ast.Store()))
                else:
                    new_keywords.append(kw)
            node.value.keywords = new_keywords

        self.input_var_name = None  # reset the var name now we are outside the conv block
        return [node]

    def visit_Assign(self, node: Assign) -> Union[Call, Tuple[Call, Call]]:
        super().generic_visit(node)
        if isinstance(node.value, ast.Call) and (
            keyword_args := self.check_call_method(node.value)
        ):
            if not isinstance(keyword_args, bool):
                keyword_arg = node.value.keywords[keyword_args.index("data_format")]
                value_mapping = {
                    "NCHW": "NHWC",
                    "NHWC": "NHWC",
                    "channel_first": "channel_last",
                    "channel_last": "channel_last",
                    "NCS": "NSC",
                    "NSC": "NSC",
                }

                # do not optimize any unsupported data formats
                if keyword_arg.value.value not in value_mapping:
                    return node

                keyword_arg.value = ast.Constant(
                    value=value_mapping[keyword_arg.value.value]
                )
        if self.conv_blocks and self.current_block < len(self.conv_blocks):
            start = self.conv_blocks[self.current_block][0]
            end = self.conv_blocks[self.current_block][1]
            if node in [start, end]:
                before, after = [], []
                if node is start:
                    before = [self._get_start_transpose(node.value)]
                if node is end:
                    after = self._get_end_transpose(
                        self.conv_block_variables[self.current_block]
                    )
                    self.current_block += 1
                return before + self._get_node(node) + after
        return node


def check_and_optimize(
    check_class: Check, apply_class: Apply, root: AST = None, **kwargs
):
    check_obj = check_class(**kwargs)
    if check_obj.is_possible(root):
        root = apply_class(**check_obj.get_contents()).visit(root)
        return root


def apply_possible_optimizations(graph: Graph, backend: str = "tensorflow"):
    source_code = graph.obtain_sourcecode()[0]
    source_code_tree = ast.parse(source_code)
    source_code_tree = check_and_optimize(
        DataFormatCheck, DataFormatApply, root=source_code_tree, backend=backend
    )
    if source_code_tree:
        optimized_source_code = ast.unparse(source_code_tree)
        locals_dict = dict()
        exec(optimized_source_code, locals_dict)
        traced_fn = locals_dict["traced_fn"]
        graph._scripted_call = traced_fn


def is_optimization_feasible(graph: Graph):
    # the intermediate step of lowering to ivy breaks memory-related or
    # setattr operations
    negative_pattern = "stride|setattr"
    source_code = graph.obtain_sourcecode()[0]
    return any(
        f"torch.nn.functional.{func}" in source_code
        for func in glob.TRANSPOSE_OPTIMIZATION_FUNCTIONS
    ) and not re.match(negative_pattern, source_code)
