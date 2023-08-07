from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionData:
    module_tree: str
    fn_name: str


@dataclass(frozen=True)
class MethodData:
    prefix_to_tree = "ivy.functional.frontends."
    class_module_tree: str
    class_name: str
    method_name: str
    init_module_tree: str
    init_name: str


@dataclass(frozen=True, kw_only=True)
class ParametersInfo:
    total: int
    positional_only: int
    keyword_only: int


@dataclass(frozen=True)
class SupportedDevicesDtypes:
    supported_device_dtypes: dict = None
