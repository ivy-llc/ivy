from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionData:
    module_tree: str
    fn_name: str
    supported_device_dtypes: dict = None


@dataclass(frozen=True)
class MethodData:
    class_module_tree: str
    class_name: str
    method_name: str
    init_module_tree: str
    init_name: str
    supported_device_dtypes: dict = None


@dataclass(frozen=True, kw_only=True)
class ParametersInfo:
    total: int
    positional_only: int
    keyword_only: int
