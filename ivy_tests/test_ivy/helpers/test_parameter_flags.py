import abc
from hypothesis import strategies as st
from . import globals as test_globals

import ivy
from ivy.functional.ivy.gradients import _variable


@st.composite
def _gradient_strategy(draw):
    if test_globals.CURRENT_BACKEND().backend == "numpy":
        draw(st.just(False))
    draw(st.booleans())


@st.composite
def _as_varaible_strategy(draw):
    if (
        test_globals.CURRENT_BACKEND is not test_globals._Notsetval
        and test_globals.CURRENT_BACKEND().backend == "numpy"
    ):
        return draw(st.just([False]))
    if not test_globals.CURRENT_FRONTEND_STR:
        if (
            test_globals.CURRENT_FRONTEND is not test_globals._Notsetval
            and test_globals.CURRENT_FRONTEND().backend == "numpy"
        ):
            return draw(st.just([False]))
    return draw(st.lists(st.booleans(), min_size=1, max_size=1))


@st.composite
def _compile_strategy(draw):  # TODO remove later when paddle is supported
    if test_globals.CURRENT_BACKEND().backend == "paddle":
        draw(st.just(False))
    draw(st.booleans())


BuiltNativeArrayStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltAsVariableStrategy = _as_varaible_strategy()
BuiltContainerStrategy = st.lists(st.booleans(), min_size=1, max_size=1)
BuiltInstanceStrategy = st.booleans()
BuiltInplaceStrategy = st.just(False)
BuiltGradientStrategy = _gradient_strategy()
BuiltWithOutStrategy = st.booleans()
BuiltWithCopyStrategy = st.just(False)
BuiltCompileStrategy = _compile_strategy()
BuiltFrontendArrayStrategy = st.booleans()


flags_mapping = {
    "native_array": "BuiltNativeArrayStrategy",
    "as_variable": "BuiltAsVariableStrategy",
    "container": "BuiltContainerStrategy",
    "instance_method": "BuiltInstanceStrategy",
    "test_gradients": "BuiltGradientStrategy",
    "with_out": "BuiltWithOutStrategy",
    "with_copy": "BuiltWithCopyStrategy",
    "inplace": "BuiltInplace",
    "test_compile": "BuiltCompileStrategy",
}


def build_flag(key: str, value: bool):
    if value is not None:
        value = st.just(value)
    # Prevent silently passing if variables names were changed
    assert (
        flags_mapping[key] in globals().keys()
    ), f"{flags_mapping[key]} is not a valid flag variable."
    globals()[flags_mapping[key]] = value


# Strategy Helpers #


def as_cont(*, x):
    """Return x as an Ivy Container, containing x at all its leaves."""
    return ivy.Container({"a": x, "b": {"c": x, "d": x}})


class TestFlags(metaclass=abc.ABCMeta):
    def apply_flags(self, args_to_iterate, input_dtypes, on_device, offset):
        pass


class FunctionTestFlags(TestFlags):
    def __init__(
        self,
        ground_truth_backend,
        num_positional_args,
        with_out,
        with_copy,
        instance_method,
        as_variable,
        native_arrays,
        container,
        test_gradients,
        test_compile,
    ):
        self.ground_truth_backend = ground_truth_backend
        self.num_positional_args = num_positional_args
        self.with_out = with_out
        self.with_copy = with_copy
        self.instance_method = instance_method
        self.native_arrays = native_arrays
        self.container = container
        self.as_variable = as_variable
        self.test_gradients = test_gradients
        self.test_compile = test_compile

    def apply_flags(self, args_to_iterate, input_dtypes, on_device, offset):
        ret = []
        for i, entry in enumerate(args_to_iterate, start=offset):
            x = ivy.array(entry, dtype=input_dtypes[i], device=on_device)
            if self.as_variable[i]:
                x = _variable(x)
            if self.native_arrays[i]:
                x = ivy.to_native(x)
            if self.container[i]:
                x = as_cont(x=x)
            ret.append(x)
        return ret

    def __str__(self):
        return (
            f"ground_truth_backend={self.ground_truth_backend}"
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"with_copy={self.with_copy}"
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
    ground_truth_backend,
    num_positional_args,
    instance_method,
    with_out,
    with_copy,
    test_gradients,
    test_compile,
    as_variable,
    native_arrays,
    container_flags,
):
    return draw(
        st.builds(
            FunctionTestFlags,
            ground_truth_backend=ground_truth_backend,
            num_positional_args=num_positional_args,
            with_out=with_out,
            with_copy=with_copy,
            instance_method=instance_method,
            test_gradients=test_gradients,
            test_compile=test_compile,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container=container_flags,
        )
    )


class FrontendFunctionTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        with_out,
        with_copy,
        inplace,
        as_variable,
        native_arrays,
        generate_frontend_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.with_out = with_out
        self.with_copy = with_copy
        self.inplace = inplace
        self.native_arrays = native_arrays
        self.as_variable = as_variable
        self.generate_frontend_arrays = generate_frontend_arrays

    def apply_flags(self, args_to_iterate, input_dtypes, on_device, offset):
        ret = []
        for i, entry in enumerate(args_to_iterate, start=offset):
            x = ivy.array(entry, dtype=input_dtypes[i], device=on_device)
            if self.as_variable[i]:
                x = _variable(x)
            if self.native_arrays[i]:
                x = ivy.to_native(x)
            ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"with_copy={self.with_copy}"
            f"inplace={self.inplace}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"generate_frontend_arrays={self.generate_frontend_arrays}. "
        )

    def __repr__(self):
        return self.__str__()


@st.composite
def frontend_function_flags(
    draw,
    *,
    num_positional_args,
    with_out,
    with_copy,
    inplace,
    as_variable,
    native_arrays,
    generate_frontend_arrays,
):
    return draw(
        st.builds(
            FrontendFunctionTestFlags,
            num_positional_args=num_positional_args,
            with_out=with_out,
            with_copy=with_copy,
            inplace=inplace,
            as_variable=as_variable,
            native_arrays=native_arrays,
            generate_frontend_arrays=generate_frontend_arrays,
        )
    )


class InitMethodTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable

    def apply_flags(self, args_to_iterate, input_dtypes, on_device, offset):
        ret = []
        for i, entry in enumerate(args_to_iterate, start=offset):
            x = ivy.array(entry, dtype=input_dtypes[i], device=on_device)
            if self.as_variable[i]:
                x = _variable(x)
            if self.native_arrays[i]:
                x = ivy.to_native(x)
            ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
        )

    def __repr__(self):
        return self.__str__()


@st.composite
def init_method_flags(
    draw,
    *,
    num_positional_args,
    as_variable,
    native_arrays,
):
    return draw(
        st.builds(
            InitMethodTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
        )
    )


class MethodTestFlags(TestFlags):
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
        self.container = container_flags

    def apply_flags(self, args_to_iterate, input_dtypes, on_device, offset):
        ret = []
        for i, entry in enumerate(args_to_iterate, start=offset):
            x = ivy.array(entry, dtype=input_dtypes[i], device=on_device)
            if self.as_variable[i]:
                x = _variable(x)
            if self.native_arrays[i]:
                x = ivy.to_native(x)
            if self.container[i]:
                x = as_cont(x=x)
            ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"container_flags={self.container}. "
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


class FrontendMethodTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable

    def apply_flags(self, args_to_iterate, input_dtypes, on_device, offset):
        ret = []
        for i, entry in enumerate(args_to_iterate, start=offset):
            x = ivy.array(entry, dtype=input_dtypes[i], device=on_device)
            if self.as_variable[i]:
                x = _variable(x)
            if self.native_arrays[i]:
                x = ivy.to_native(x)
            ret.append(x)
        return ret

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
