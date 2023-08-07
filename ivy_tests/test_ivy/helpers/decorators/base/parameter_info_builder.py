import inspect

from typing import Any, Callable
from ivy_tests.test_ivy.helpers.structs import ParametersInfo
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.decorators.strategies import (
    num_positional_args_from_dict,
)


class ParameterInfoStrategyBuilder:
    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    @classmethod
    def from_function(cls, fn_tree: str):
        pass

    @classmethod
    def from_method(cls, method_tree: str):
        pass

    def _build_parameters_info_struct(self):
        total = num_positional_only = num_keyword_only = 0
        for param in inspect.signature(self._fn).parameters.values():
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
            parameter_info = self(self._fn)
            ret[framework] = parameter_info

        return ret

    def build(self):
        dict_for_num_pos_strategy = self._build_parameters_info_dict()
        return num_positional_args_from_dict(dict_for_num_pos_strategy)
