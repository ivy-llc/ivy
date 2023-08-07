import inspect
import ivy_tests.test_ivy.helpers.decorators.base.import_helpers as import_helpers

from typing import Dict
from ivy_tests.test_ivy.helpers.structs import ParametersInfo
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.decorators.strategies import (
    num_positional_args_from_dict,
)


class ParameterInfoStrategyBuilder:
    def __init__(self, routine_tree: str, routines: Dict):
        self.routine_tree = routine_tree
        self.routines = routines

    @classmethod
    def from_function(cls, fn_tree: str):
        functions = {}
        module_tree, fn_name = import_helpers.partition_function_tree(fn_tree)

        for framework in available_frameworks:
            fn = import_helpers.import_function_from_ivy(
                module_tree, fn_name, framework
            )
            functions[framework] = fn

        return cls(fn_name, functions)

    @classmethod
    def from_method(cls, method_tree: str):
        methods = {}
        module_tree, class_name, method_name = import_helpers.partition_method_tree(
            method_tree
        )

        for framework in available_frameworks:
            fn = import_helpers.import_method_from_ivy(
                module_tree, class_name, method_name, framework
            )
            methods[framework] = fn

        return cls(method_tree, methods)

    def _build_parameters_info_struct(self, framework: str):
        total = num_positional_only = num_keyword_only = 0
        routine = self.routines[framework]
        for param in inspect.signature(routine).parameters.values():
            if param.name == "self":
                continue
            total += 1
            if param.kind == param.POSITIONAL_ONLY:
                num_positional_only += 1
            elif param.kind == param.KEYWORD_ONLY:
                num_keyword_only += 1
            elif param.kind == param.VAR_KEYWORD:
                num_keyword_only += 1
        return ParametersInfo(
            total=total,
            positional_only=num_positional_only,
            keyword_only=num_keyword_only,
        )

    def _build_parameters_info_dict(self):
        ret = {}

        for framework in available_frameworks:
            parameter_info = self._build_parameters_info_struct(framework)
            ret[framework] = parameter_info

        return ret

    def build(self):
        dict_for_num_pos_strategy = self._build_parameters_info_dict()
        return num_positional_args_from_dict(dict_for_num_pos_strategy)
