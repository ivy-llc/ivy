from hypothesis import strategies as st  # NOQA


BuiltNativeArrayStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltAsVariableStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltContainerStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltInstanceStrategy = st.booleans()
BuiltInplaceStrategy = st.just(False)
BuiltGradientStrategy = st.booleans()
BuiltWithOutStrategy = st.booleans()
BuiltCompileStrategy = st.booleans()


flags_mapping = {
    "native_array": "BuiltNativeArrayStrategy",
    "as_variable": "BuiltAsVariableStrategy",
    "container": "BuiltContainerStrategy",
    "instance_method": "BuiltInstanceStrategy",
    "test_gradients": "BuiltGradientStrategy",
    "with_out": "BuiltWithOutStrategy",
    "inplace": "BuiltInplace",
    "test_compile": "BuiltCompileStrategy",
}


def build_flag(key: str, value: bool):
    if value is not None:
        value = st.just(value)
    # Prevent silently passing if variables names were changed
    assert flags_mapping[key] in globals().keys(), (
        f"{flags_mapping[key]} is not " f"a valid flag variable."
    )
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
        test_compile,
    ):
        self.num_positional_args = num_positional_args
        self.with_out = with_out
        self.instance_method = instance_method
        self.native_arrays = native_arrays
        self.container = container
        self.as_variable = as_variable
        self.test_gradients = test_gradients
        self.test_compile = test_compile

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"instance_method={self.instance_method}. "
            f"native_arrays={self.native_arrays}. "
            f"container={self.container}. "
            f"as_variable={self.as_variable}. "
            f"test_gradients={self.test_gradients}. "
            f"test_compile={self.test_compile}. "
        )

    def __repr__(self):
        return self.__str__()


@st.composite
def function_flags(
    draw,
    *,
    num_positional_args,
    instance_method,
    with_out,
    test_gradients,
    test_compile,
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
            test_compile=test_compile,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container=container_flags,
        )
    )


class FrontendFunctionTestFlags:
    def __init__(
        self,
        num_positional_args,
        with_out,
        inplace,
        as_variable,
        native_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.with_out = with_out
        self.inplace = inplace
        self.native_arrays = native_arrays
        self.as_variable = as_variable

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"inplace={self.inplace}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
        )

    def __repr__(self):
        return self.__str__()


@st.composite
def frontend_function_flags(
    draw,
    *,
    num_positional_args,
    with_out,
    inplace,
    as_variable,
    native_arrays,
):
    return draw(
        st.builds(
            FrontendFunctionTestFlags,
            num_positional_args=num_positional_args,
            with_out=with_out,
            inplace=inplace,
            as_variable=as_variable,
            native_arrays=native_arrays,
        )
    )


class MethodTestFlags:
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
        container_flags,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable
        self.container_flags = container_flags

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"container_flags={self.container_flags}. "
        )

    def __repr__(self):
        return self.__str__()


@st.composite
def method_flags(
    draw,
    *,
    num_positional_args,
    as_variable,
    native_arrays,
    container_flags,
):
    return draw(
        st.builds(
            MethodTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container_flags=container_flags,
        )
    )


class FrontendMethodTestFlags:
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
        )

    def __repr__(self):
        return self.__str__()


@st.composite
def frontend_method_flags(
    draw,
    *,
    num_positional_args,
    as_variable,
    native_arrays,
):
    return draw(
        st.builds(
            FrontendMethodTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
        )
    )
