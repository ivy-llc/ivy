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


BuiltNativeArrayStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltAsVariableStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltContainerStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltInstanceStrategy = st.booleans()
BuiltWithOutStrategy = st.booleans()
BuiltGradientStrategy = st.booleans()


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
    globals()[flags_mapping[key]] = value
