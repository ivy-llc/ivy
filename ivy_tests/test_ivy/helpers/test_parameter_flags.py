from dataclasses import dataclass
from hypothesis import strategies as st  # NOQA


class ContainerFlags:
    pass


class NumPositionalArg:  # TODO for backward compatibility only
    pass


class NumPositionalArgMethod:
    pass


class NumPositionalArgFn:
    pass


class NativeArrayFlags:
    pass


class AsVariableFlags:
    pass


@dataclass(frozen=True)
class _FlagBuilder:
    strategy: st.SearchStrategy


_BuiltNativeArray = _FlagBuilder(st.lists(st.booleans(), min_size=1, max_size=1))
_BuiltAsVariable = _FlagBuilder(st.lists(st.booleans(), min_size=1, max_size=1))
_BuiltContainer = _FlagBuilder(st.booleans())
_BuiltInstance = _FlagBuilder(st.booleans())
_BuiltWithOut = _FlagBuilder(st.booleans())
_BuiltGradient = _FlagBuilder(st.booleans())


flags_mapping = {
    "as_variable": "_BuiltAsVariable",
    "native_array": "_BuiltNativeArray",
    "container": "_BuiltContainer",
    "with_out": "_BuiltWithOut",
    "instance_method": "_BuiltInstance",
    "test_gradients": "_BuiltGradient",
}


def build_flag(key: str, value: bool):
    if value is not None:
        value = st.just(value)
    globals()[flags_mapping[key]] = _FlagBuilder(strategy=value)
