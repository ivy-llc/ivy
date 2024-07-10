import cProfile
import pstats
import subprocess
import logging
from tempfile import NamedTemporaryFile
from importlib.util import find_spec

is_snakeviz = find_spec("snakeviz")


class Profiler(cProfile.Profile):
    """A Profiler class that allows code profiling.

    Attributes
    ----------
        print_stats (bool, optional): prints profiling statistics.
        viz (bool, optional): visualizes the results using `snakeviz`.

        Bonus args and kwargs are passed to cProfile.Profile __init__

    Example
    -------
        with Profiler(print_stats=False, viz=True):
            fn(x, y)
    """

    def __init__(self, *args, **kwargs):
        self.print_stats = kwargs.pop("print_stats", True)
        self.viz = kwargs.pop("viz", False)

        super().__init__(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        self.pr = super().__enter__(*args, **kwargs)

        return self.pr

    def __exit__(self, *exc):
        super().__exit__(*exc)
        if exc == (None, None, None):
            stats = pstats.Stats(self.pr)
            stats.sort_stats(pstats.SortKey.TIME)

            if self.viz:
                if is_snakeviz:
                    # creates a temp file that gets automatically
                    # deleted when everything is done
                    with NamedTemporaryFile(suffix=".prof") as f:
                        stats.dump_stats(filename=f.name)
                        subprocess.run(["snakeviz", f"{f.name}"])
                else:
                    logging.warning("snakeviz must be installed for visualization")

            if self.print_stats:
                stats.print_stats()


def tensorflow_profile_start(
    logdir: str,
    host_tracer_level: int = 2,
    python_tracer_level: int = 0,
    device_tracer_level: int = 1,
    delay_ms: int = None,
):
    """Initialize and start the profiler.

    Parameters
    ----------
    logdir: str
    Directory where the profile data will be saved to.
    host_tracer_level: int
        Adjust CPU tracing level. Values are: 1 - critical info only, 2 - info, 3 - verbose. [default value is 2]
    python_tracer_level: int
        Toggle tracing of Python function calls. Values are: 1 - enabled, 0 - disabled [default value is 0]
    device_tracer_level: int
        Adjust device (TPU/GPU) tracing level. Values are: 1 - enabled, 0 - disabled [default value is 1]
    delay_ms: int
        Requests for all hosts to start profiling at a timestamp that is delay_ms away from the current time. delay_ms is in milliseconds. If zero, each host will start profiling immediately upon receiving the request. Default value is None, allowing the profiler guess the best value.
        Save the weights on the Module.

    Returns
    -------
    None
    """  # noqa: E501
    from tensorflow.profiler.experimental import ProfilerOptions, start

    options = ProfilerOptions(
        host_tracer_level=host_tracer_level,
        python_tracer_level=python_tracer_level,
        device_tracer_level=device_tracer_level,
        delay_ms=delay_ms,
    )
    start(logdir, options=options)


def tensorflow_profile_stop():
    """Stop the profiler."""
    from tensorflow.profiler.experimental import stop

    stop()


def torch_profiler_init(
    logdir=None,
    activities=None,
    schedule=None,
    on_trace_ready=None,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_flops=False,
    with_modules=False,
    experimental_config=None,
):
    """Initialize and returns a Torch profiler instance.

    Parameters
    ----------
        logdir : str
            Directory where the profile data will be saved to.

        activities : iterable
            list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.

        schedule : Callable
            callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.

        on_trace_ready : Callable
            callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.

        record_shapes : bool
            save information about operator's input shapes.

        profile_memory : bool
            track tensor memory allocation/deallocation.

        with_stack : bool
            record source information (file and line number) for the ops.

        with_flops : bool
            use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).

        with_modules : bool
            record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op, then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.

        experimental_config _ExperimentalConfig : _ExperimentalConfig
            A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.

    Returns
    -------
    Torch profiler instance.
    """  # noqa: E501
    from torch.profiler import profile, tensorboard_trace_handler

    profiler = profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=(
            tensorboard_trace_handler(logdir)
            if on_trace_ready is None and logdir is not None
            else on_trace_ready
        ),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
        with_modules=with_modules,
        experimental_config=experimental_config,
    )
    return profiler


def torch_profiler_start(profiler):
    """Start the profiler.

    Parameters
    ----------
    profiler : torch.profiler.profile
        Torch profiler instance.

    Returns
    -------
    None
    """
    profiler.start()


def torch_profiler_stop(profiler):
    """Start the profiler.

    Parameters
    ----------
    profiler : torch.profiler.profile
        Torch profiler instance.

    Returns
    -------
    None
    """
    from torch.autograd.profiler import KinetoStepTracker
    from torch.profiler.profiler import PROFILER_STEP_NAME

    profiler.stop()
    KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
