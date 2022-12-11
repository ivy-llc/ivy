from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendMethodData:
    ivy_init_module: str
    framework_init_module: str
    init_name: str
    method_name: str
