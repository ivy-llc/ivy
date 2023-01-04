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


class FunctionTestFlags:
    def __init__(
        self,
        num_positional_args,
        with_out,
        instance_method,
        as_variable,
        native_arrays,
        container,
        test_gradients,
    ):
        self.num_positional_args = num_positional_args
        self.with_out = with_out
        self.instance_method = instance_method
        self.native_arrays = native_arrays
        self.container = container
        self.as_variable = as_variable
        self.test_gradients = test_gradients

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"instance_method={self.instance_method}. "
            f"native_arrays={self.native_arrays}. "
            f"container={self.container}. "
            f"as_variable={self.as_variable}. "
            f"test_gradients={self.test_gradients}."
        )


@st.composite
def function_flags(
    draw,
    *,
    num_positional_args,
    instance_method,
    with_out,
    test_gradients,
    as_variable,
    native_arrays,
    container_flags,
):
    return draw(
        st.builds(
            FunctionTestFlags,
            num_positional_args=num_positional_args,
            with_out=with_out,
            instance_method=instance_method,
            test_gradients=test_gradients,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container=container_flags,
        )
    )
