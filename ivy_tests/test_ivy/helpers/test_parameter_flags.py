from hypothesis import strategies as st  # NOQA


class ContainerFlags:  # TODO remove
    pass


class NumPositionalArg:  # TODO for backward compatibility only
    pass


class NumPositionalArgMethod:  # TODO remove
    pass


class NumPositionalArgFn:  # TODO remove
    pass


class NativeArrayFlags:  # TODO remove
    pass


class AsVariableFlags:  # TODO remove
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


# Strategy Helpers #


class BackendTestFlags:
    def __init__(
        self,
        with_out,
        instance_method,
        as_variable,
        native_array,
        container_flags,
        gradient,
    ):
        self.with_out = with_out
        self.instance_method = instance_method
        self.native_arrays = native_array
        self.container = container_flags
        self.as_variable = as_variable
        self.gradient = gradient


@st.composite
def backend_flags(
    draw,
    *,
    instance_method=BuiltInstanceStrategy,
    with_out=BuiltWithOutStrategy,
    gradient=BuiltGradientStrategy,
    as_variable=BuiltAsVariableStrategy,
    native_arrays=BuiltNativeArrayStrategy,
    container_flags=BuiltContainerStrategy
):
    return draw(
        st.builds(
            BackendTestFlags,
            with_out=with_out,
            instance_method=instance_method,
            gradient=gradient,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container=container_flags,
        )
    )
