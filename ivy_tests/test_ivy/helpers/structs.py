from typing import Callable, List
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendMethodData:
    ivy_init_module: str
    framework_init_module: str
    init_name: str
    method_name: str


@dataclass(kw_only=True, frozen=True)
class FrontendTestData:
    fw_function: Callable
    ivy_function: Callable
    ivy_function_args: List[str]
    fn_tree: str
    fn_name: str
    supported_device_dtypes: dict = None
