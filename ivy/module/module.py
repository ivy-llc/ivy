# global
from typing import Dict, Mapping, Optional, Sequence

# local
import ivy
import ivy.tracer.globals as glob
import ivy.tracer.tracer as tracer
import ivy.transpiler as transpiler


# Helper functions


def _check_train_mode(module, kwargs):
    if hasattr(module, "training"):
        return module.training
    if kwargs is None:
        return False

    # search through the kwargs for any which set
    # the train mode, and return the value found
    for train_kwarg in glob.TRAIN_KWARGS:
        if train_kwarg in kwargs:
            return kwargs[train_kwarg]

    return False


def _set_module_modes(
    ivy_module: ivy.Module,
    source_module: object,
    target_module: object,
    source: Optional[str],
    to: Optional[str],
    original_mode: str,
    lazy: bool,
):
    """
    Sets the mode of the passed traced `ivy_module`, `source_module`
    and `target_module` to be the same as the `original_mode`

    Parameters
    ----------
    ivy_module : ivy.Module
        A traced/transpiled ivy module which needs it's graph
        resetting to the original mode
    source_module : Any
        The source module (native framework module) which needs
        it's graph resetting to the original mode
    target_module : Any
        The target module from transpilation (native framework module)
        which needs it's graph resetting to the original mode.
        Expected as `None` if we are tracing rather than transpiling.
    source : str
        The source framework for transpilation.
        Expected as `None` if we are tracing rather than transpiling.
    to : str
        The framework we are tracing/transpiling to.
    original_mode : str
        The original mode of the module which we are
        resetting these modules back to.
    lazy : bool
        Whether the current tracing/transpilation process
        is happening lazily or eagerly.
    """
    if (
        not lazy
        and ivy_module._module_graph._is_trainable_module
        and original_mode is not None
    ):
        # set mode of target graph to be the same as the original
        # and reset original module mode (for torch and paddle)
        if original_mode == "train":
            ivy_module._module_graph._train(update_glob=False)
            if source in ["torch", "paddle"]:
                source_module.train()
            if to in ["torch", "paddle"]:
                target_module.train()
        else:
            ivy_module._module_graph._eval(update_glob=False)
            if source in ["torch", "paddle"]:
                source_module.eval()
            if to in ["torch", "paddle"]:
                target_module.eval()


# Transpilation functions


def to_torch_module(module, lazy):
    from .torch_module import __TorchModule

    torch_module = __TorchModule(module, lazy=lazy)
    # set tracing flags
    torch_module._ivy_module._lazy_traced = lazy
    torch_module._ivy_module._target = "torch"
    return torch_module


def to_haiku_module(module, lazy):
    ivy_module = module
    from .haiku_module import __HaikuModule

    # set tracing flags
    module._lazy_traced = lazy
    module._target = "jax"
    ivy_module._lazy_traced = lazy
    ivy_module._target = "jax"
    __HaikuModule.lazy = lazy
    __HaikuModule.ivy_module = module
    return __HaikuModule


def to_flax_module(module, lazy):
    from .flax_module import __FlaxModule

    flax_module = __FlaxModule(module, lazy=lazy)
    # set tracing flags
    flax_module._lazy_traced = lazy
    flax_module._target = "jax"
    return flax_module


def to_keras_module(module, lazy):
    from .keras_module import KerasModel

    keras_module = KerasModel(module, lazy=lazy)
    # set tracing flags
    keras_module._ivy_module._lazy_traced = lazy
    keras_module._ivy_module._target = "tensorflow"
    return keras_module


def to_paddle_module(module, lazy):
    from .paddle_module import __PaddleLayer

    paddle_module = __PaddleLayer(module, lazy=lazy)
    # set tracing flags
    paddle_module._ivy_module._lazy_traced = lazy
    paddle_module._ivy_module._target = "paddle"
    return paddle_module


def _transpile_trainable_module(
    source_module,
    source,
    to,
    source_mod: Optional[str] = None,
    to_mod: Optional[str] = None,
    args: Optional[Sequence] = None,
    kwargs: Optional[Mapping] = None,
    with_numpy: bool = False,
    graph_caching: bool = False,
    graph_optimizations: bool = True,
    modes_to_trace: str = "all",
    backend_compile: bool = False,
    params_v=None,
):
    """Converts module in source backend to the target backend. Returns a
    lazily traceable module (in target backend) if no instance args and kwargs
    are provided.

    params_v : Required for creation of ivy.Module from some source modules (e.g. Haiku)
    """

    if to == "numpy":
        raise ValueError(
            "A module can not be fully transpiled to NumPy. To get an equivalent NumPy"
            " function, transpile the forward pass instead."
        )

    BACKEND_TO_MODULE_FROM_BACKEND = {
        "torch": ivy.ModuleConverters.from_torch_module,
        "jax": {
            "haiku": ivy.ModuleConverters.from_haiku_module,
            "flax": ivy.ModuleConverters.from_flax_module,
        },
        "tensorflow": ivy.ModuleConverters.from_keras_module,
        "paddle": ivy.ModuleConverters.from_paddle_module,
    }

    original_mode = "train" if _check_train_mode(source_module, kwargs) else "eval"
    lazy_transpile = args is None and kwargs is None

    if source == "jax" and source_mod is None:
        import flax

        source_mod = "flax" if isinstance(source_module, flax.linen.Module) else "haiku"

    if source != "ivy":  # Probably there is a cleaner way of doing this
        # first let's construct a ivy.Module from the source module
        fw_kwargs = {}
        if params_v is not None:
            params_key = "params_hk" if source_mod == "haiku" else "params_fx"
            fw_kwargs[params_key] = params_v
        module_converter = BACKEND_TO_MODULE_FROM_BACKEND[source]
        if source == "jax":
            module_converter = module_converter[source_mod]
        ivy_module = module_converter(
            source_module,
            instance_args=args,
            instance_kwargs=kwargs,
            **fw_kwargs,
        )
    else:
        ivy_module = source_module
        source = ivy.current_backend_str()

    _add_running_stats_to_params(ivy_module, source)
    # transpile the inner graph
    ivy_module._module_graph = transpiler.graph_transpile(
        ivy_module._call,
        source=source,
        to=to,
        args=args,
        with_numpy=with_numpy,
        graph_caching=graph_caching,
        graph_optimizations=graph_optimizations,
        modes_to_trace=modes_to_trace,
        kwargs=kwargs,
        backend_compile=backend_compile,
        v=ivy_module.v,
    )

    # if the target is ivy, return an ivy.Module, otherwise convert into corresponding module
    if to == "ivy":
        from tracer.conversion import to_native

        ivy_module.v = ivy_module.v.cont_map(lambda x, _: to_native(x))
        _set_module_modes(
            ivy_module, source_module, None, source, to, original_mode, lazy_transpile
        )
        return ivy_module

    TO_NATIVE_MODULE = {
        "torch": to_torch_module,
        "jax": {
            "haiku": to_haiku_module,
            "flax": to_flax_module,
        },
        "tensorflow": to_keras_module,
        "paddle": to_paddle_module,
    }

    to_converter = TO_NATIVE_MODULE[to]
    if to == "jax":
        to_converter = to_converter[to_mod]
    target_module = to_converter(ivy_module, lazy=lazy_transpile)

    if modes_to_trace != "all":
        # ensures that the mode is set correctly if only a single branch has been traced
        original_mode = modes_to_trace

    _set_module_modes(
        ivy_module,
        source_module,
        target_module,
        source,
        to,
        original_mode,
        lazy_transpile,
    )

    return target_module


def _add_running_stats_to_params(ivy_module, to):
    if to != "torch" or not hasattr(ivy_module, "_native_module"):
        return
    for name, mod in ivy_module._native_module.named_modules():
        if hasattr(mod, "running_mean"):
            ivy_module.v[f"{name}/running_mean"] = mod.running_mean
            ivy_module.v[f"{name}/running_var"] = mod.running_var


def _trace_trainable_module(
    source_module,
    to,
    to_mod: Optional[str] = None,
    args: Optional[Sequence] = None,
    kwargs: Optional[Dict] = None,
    with_numpy: bool = False,
    graph_caching: bool = False,
    modes_to_trace: str = "all",
    backend_compile: bool = False,
    params_v=None,
):
    """Traces module in the target backend. Returns a
    lazily traceable module (in target backend) if no instance args and kwargs
    are provided.

    params_v : Required for creation of ivy.Module from some source modules (e.g. Haiku)
    """
    BACKEND_TO_MODULE_FROM_BACKEND = {
        "torch": ivy.ModuleConverters.from_torch_module,
        "jax": {
            "haiku": ivy.ModuleConverters.from_haiku_module,
            "flax": ivy.ModuleConverters.from_flax_module,
        },
        "tensorflow": ivy.ModuleConverters.from_keras_module,
        "paddle": ivy.ModuleConverters.from_paddle_module,
    }

    original_mode = "train" if _check_train_mode(source_module, kwargs) else "eval"
    lazy_trace = args is None and kwargs is None

    if to != "ivy":  # Probably there is a cleaner way of doing this
        # first let's construct a ivy.Module from the source module
        fw_kwargs = {}

        if params_v is not None:
            params_key = "params_hk" if to_mod == "haiku" else "params_fx"
            fw_kwargs[params_key] = params_v

        module_converter = BACKEND_TO_MODULE_FROM_BACKEND[to]

        if to == "jax":
            module_converter = module_converter[to_mod]

        ivy_module = module_converter(
            source_module,
            instance_args=args,
            instance_kwargs=kwargs,
            **fw_kwargs,
        )
    else:
        ivy_module = source_module

    _add_running_stats_to_params(ivy_module, to)
    # trace the inner graph
    ivy_module._module_graph = tracer.trace_graph(
        ivy_module._call,
        to=to,
        args=args,
        with_numpy=with_numpy,
        graph_caching=graph_caching,
        kwargs=kwargs,
        backend_compile=backend_compile,
        modes_to_trace=modes_to_trace,
        v=ivy_module.v,
    )

    # if the target is ivy, return an ivy.Module, otherwise convert into corresponding module
    if to == "ivy":
        from tracer.conversion import to_native

        ivy_module.v = ivy_module.v.cont_map(lambda x, _: to_native(x))
        _set_module_modes(
            ivy_module, source_module, None, None, to, original_mode, lazy_trace
        )
        return ivy_module

    TO_NATIVE_MODULE = {
        "torch": to_torch_module,
        "jax": {
            "haiku": to_haiku_module,
            "flax": to_flax_module,
        },
        "tensorflow": to_keras_module,
        "paddle": to_paddle_module,
    }

    to_converter = TO_NATIVE_MODULE[to]

    if to == "jax":
        to_converter = to_converter[to_mod]

    target_module = to_converter(ivy_module, lazy=lazy_trace)

    if modes_to_trace != "all":
        # ensures that the mode is set correctly if only a single branch has been traced
        original_mode = modes_to_trace

    _set_module_modes(
        ivy_module, source_module, target_module, None, to, original_mode, lazy_trace
    )

    return target_module
