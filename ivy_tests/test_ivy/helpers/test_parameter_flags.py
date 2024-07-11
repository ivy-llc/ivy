import abc
from hypothesis import strategies as st
from . import globals as test_globals
from .pipeline_helper import BackendHandler

from dataclasses import dataclass
from hypothesis.strategies import SearchStrategy


@dataclass
class DynamicFlag:
    strategy: SearchStrategy


@st.composite
def _gradient_strategy(draw):
    if test_globals.CURRENT_BACKEND == "numpy":
        return draw(st.just(False))
    return draw(st.booleans())


@st.composite
def _as_varaible_strategy(draw):
    if (
        test_globals.CURRENT_BACKEND is not test_globals._Notsetval
        and test_globals.CURRENT_BACKEND == "numpy"
    ):
        return draw(st.just([False]))
    if not test_globals.CURRENT_FRONTEND_STR:
        if (
            test_globals.CURRENT_FRONTEND is not test_globals._Notsetval
            and test_globals.CURRENT_FRONTEND == "numpy"
        ):
            return draw(st.just([False]))
    return draw(st.lists(st.booleans(), min_size=1, max_size=1))


BuiltNativeArrayStrategy = DynamicFlag(st.lists(st.booleans(), min_size=1, max_size=1))
BuiltAsVariableStrategy = DynamicFlag(_as_varaible_strategy())
BuiltContainerStrategy = DynamicFlag(st.lists(st.booleans(), min_size=1, max_size=1))
BuiltInstanceStrategy = DynamicFlag(st.booleans())
BuiltInplaceStrategy = DynamicFlag(st.just(False))
BuiltGradientStrategy = DynamicFlag(_gradient_strategy())
BuiltWithOutStrategy = DynamicFlag(st.booleans())
BuiltWithCopyStrategy = DynamicFlag(st.just(False))
BuiltCompileStrategy = DynamicFlag(st.just(False))
BuiltTraceStrategy = DynamicFlag(st.just(False))
BuiltTraceEachStrategy = DynamicFlag(st.just(False))
BuiltFrontendArrayStrategy = DynamicFlag(st.booleans())
BuiltTranspileStrategy = DynamicFlag(st.just(False))
BuiltPrecisionModeStrategy = DynamicFlag(st.booleans())
BuiltCythonWrapperStrategy = DynamicFlag(st.just(False))


flags_mapping = {
    "native_array": "BuiltNativeArrayStrategy",
    "as_variable": "BuiltAsVariableStrategy",
    "container": "BuiltContainerStrategy",
    "instance_method": "BuiltInstanceStrategy",
    "test_gradients": "BuiltGradientStrategy",
    "with_out": "BuiltWithOutStrategy",
    "with_copy": "BuiltWithCopyStrategy",
    "inplace": "BuiltInplace",
    "test_trace": "BuiltTraceStrategy",
    "test_trace_each": "BuiltTraceEachStrategy",
    "transpile": "BuiltTranspileStrategy",
    "precision_mode": "BuiltPrecisionModeStrategy",
    "test_cython_wrapper": "BuiltCythonWrapperStrategy",
}


def build_flag(key: str, value: bool):
    if value is not None:
        value = st.just(value)
    # Prevent silently passing if variables names were changed
    assert (
        flags_mapping[key] in globals()
    ), f"{flags_mapping[key]} is not a valid flag variable."
    globals()[flags_mapping[key]].strategy = value


# Strategy Helpers #


class TestFlags(metaclass=abc.ABCMeta):
    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
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
        test_trace,
        test_trace_each,
        transpile,
        precision_mode,
        test_cython_wrapper,
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
        self.test_trace = test_trace
        self.test_trace_each = test_trace_each
        self.transpile = transpile
        self.precision_mode = precision_mode
        self.test_cython_wrapper = test_cython_wrapper

    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
        ret = []
        with BackendHandler.update_backend(backend) as backend:
            for i, entry in enumerate(args_to_iterate, start=offset):
                x = backend.array(entry, dtype=input_dtypes[i], device=on_device)
                if self.as_variable[i]:
                    x = backend.gradients._variable(x)
                if self.native_arrays[i]:
                    x = backend.to_native(x)
                if self.container[i]:
                    x = backend.Container({"a": x, "b": {"c": x, "d": x}})
                ret.append(x)
        return ret

    def __str__(self):
        return (
            f"ground_truth_backend={self.ground_truth_backend}"
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"with_copy={self.with_copy}. "
            f"instance_method={self.instance_method}. "
            f"native_arrays={self.native_arrays}. "
            f"container={self.container}. "
            f"as_variable={self.as_variable}. "
            f"test_gradients={self.test_gradients}. "
            f"test_trace={self.test_trace}. "
            f"test_trace_each={self.test_trace_each}. "
            f"transpile={self.transpile}. "
            f"precision_mode={self.precision_mode}. "
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
    test_trace,
    test_trace_each,
    transpile,
    as_variable,
    native_arrays,
    container_flags,
    precision_mode,
    test_cython_wrapper,
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
            test_trace=test_trace,
            test_trace_each=test_trace_each,
            transpile=transpile,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container=container_flags,
            precision_mode=precision_mode,
            test_cython_wrapper=test_cython_wrapper,
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
        test_trace,
        test_trace_each,
        generate_frontend_arrays,
        transpile,
        precision_mode,
    ):
        self.num_positional_args = num_positional_args
        self.with_out = with_out
        self.with_copy = with_copy
        self.inplace = inplace
        self.native_arrays = native_arrays
        self.as_variable = as_variable
        self.test_trace = test_trace
        self.test_trace_each = test_trace_each
        self.generate_frontend_arrays = generate_frontend_arrays
        self.transpile = transpile
        self.precision_mode = precision_mode

    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
        ret = []
        with BackendHandler.update_backend(backend) as backend:
            for i, entry in enumerate(args_to_iterate, start=offset):
                x = backend.array(entry, dtype=input_dtypes[i], device=on_device)
                if self.as_variable[i]:
                    x = backend.gradients._variable(x)
                if self.native_arrays[i]:
                    x = backend.to_native(x)
                ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"with_out={self.with_out}. "
            f"with_copy={self.with_copy}. "
            f"inplace={self.inplace}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"test_trace={self.test_trace}. "
            f"test_trace_each={self.test_trace_each}"
            f"generate_frontend_arrays={self.generate_frontend_arrays}. "
            f"transpile={self.transpile}."
            f"precision_mode={self.precision_mode}. "
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
    test_trace,
    test_trace_each,
    generate_frontend_arrays,
    transpile,
    precision_mode,
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
            test_trace=test_trace,
            test_trace_each=test_trace_each,
            generate_frontend_arrays=generate_frontend_arrays,
            transpile=transpile,
            precision_mode=precision_mode,
        )
    )


class InitMethodTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
        precision_mode,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable
        self.precision_mode = precision_mode

    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
        ret = []
        with BackendHandler.update_backend(backend) as backend:
            for i, entry in enumerate(args_to_iterate, start=offset):
                x = backend.array(entry, dtype=input_dtypes[i], device=on_device)
                if self.as_variable[i]:
                    x = backend.gradients._variable(x)
                if self.native_arrays[i]:
                    x = backend.to_native(x)
                ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"precision_mode={self.precision_mode}. "
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
    precision_mode,
):
    return draw(
        st.builds(
            InitMethodTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
            precision_mode=precision_mode,
        )
    )


class MethodTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
        container_flags,
        precision_mode,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable
        self.container = container_flags
        self.precision_mode = precision_mode

    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
        ret = []
        with BackendHandler.update_backend(backend) as backend:
            for i, entry in enumerate(args_to_iterate, start=offset):
                x = backend.array(entry, dtype=input_dtypes[i], device=on_device)
                if self.as_variable[i]:
                    x = backend.gradients._variable(x)
                if self.native_arrays[i]:
                    x = backend.to_native(x)
                if self.container[i]:
                    x = backend.Container({"a": x, "b": {"c": x, "d": x}})
                ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"container_flags={self.container}. "
            f"precision_mode={self.precision_mode}. "
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
    precision_mode,
):
    return draw(
        st.builds(
            MethodTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container_flags=container_flags,
            precision_mode=precision_mode,
        )
    )


class FrontendInitTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable

    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
        ret = []
        with BackendHandler.update_backend(backend) as backend:
            for i, entry in enumerate(args_to_iterate, start=offset):
                x = backend.array(entry, dtype=input_dtypes[i], device=on_device)
                if self.as_variable[i]:
                    x = backend.gradients._variable(x)
                if self.native_arrays[i]:
                    x = backend.to_native(x)
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
def frontend_init_flags(
    draw,
    *,
    num_positional_args,
    as_variable,
    native_arrays,
):
    return draw(
        st.builds(
            FrontendInitTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
        )
    )


class FrontendMethodTestFlags(TestFlags):
    def __init__(
        self,
        num_positional_args,
        as_variable,
        native_arrays,
        precision_mode,
        inplace,
        test_trace,
        test_trace_each,
        generate_frontend_arrays,
    ):
        self.num_positional_args = num_positional_args
        self.native_arrays = native_arrays
        self.as_variable = as_variable
        self.precision_mode = precision_mode
        self.inplace = inplace
        self.test_trace = test_trace
        self.test_trace_each = test_trace_each
        self.generate_frontend_arrays = generate_frontend_arrays

    def apply_flags(self, args_to_iterate, input_dtypes, offset, *, backend, on_device):
        ret = []
        with BackendHandler.update_backend(backend) as backend:
            for i, entry in enumerate(args_to_iterate, start=offset):
                x = backend.array(entry, dtype=input_dtypes[i], device=on_device)
                if self.as_variable[i]:
                    x = backend.gradients._variable(x)
                if self.native_arrays[i]:
                    x = backend.to_native(x)
                ret.append(x)
        return ret

    def __str__(self):
        return (
            f"num_positional_args={self.num_positional_args}. "
            f"native_arrays={self.native_arrays}. "
            f"as_variable={self.as_variable}. "
            f"precision_mode={self.precision_mode}. "
            f"inplace={self.inplace}. "
            f"test_trace={self.test_trace}."
            f"test_trace_each={self.test_trace_each}"
            f"generate_frontend_arrays={self.generate_frontend_arrays}."
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
    precision_mode,
    inplace,
    test_trace,
    test_trace_each,
    generate_frontend_arrays,
):
    return draw(
        st.builds(
            FrontendMethodTestFlags,
            num_positional_args=num_positional_args,
            as_variable=as_variable,
            native_arrays=native_arrays,
            precision_mode=precision_mode,
            inplace=inplace,
            test_trace=test_trace,
            test_trace_each=test_trace_each,
            generate_frontend_arrays=generate_frontend_arrays,
        )
    )
