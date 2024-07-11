from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import functools
import time
import os
import copy
import importlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ivy


sns.set()
LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"


COLUMNS = [
    "exp no.",
    "label",
    "backend",
    "device",
    "eager time",
    "graph time",
    "percent_speed_up",
]


class _AvoidGPUPreallocation:
    def __init__(self, backend):
        self._backend = backend
        if backend == "tensorflow":
            self.tf = importlib.import_module("tensorflow")

    def __enter__(self):
        if self._backend == "tensorflow":
            gpus = self.tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                self.tf.config.experimental.set_memory_growth(gpu, True)
        elif self._backend == "jax":
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._backend == "tensorflow":
            gpus = self.tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                self.tf.config.experimental.set_memory_growth(gpu, True)
        elif self._backend == "jax":
            del os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]
        if self and (exc_type is not None):
            raise exc_val
        return self


def _move_to_device(args=None, kwargs=None, device="cpu"):
    args_idxs = ivy.nested_argwhere(args, ivy.is_array)
    kwargs_idxs = ivy.nested_argwhere(kwargs, ivy.is_array)

    def func(x):
        return ivy.to_device(x, device, out=x)

    if args is not None:
        args = ivy.map_nest_at_indices(args, args_idxs, func)
    if kwargs is not None:
        kwargs = ivy.map_nest_at_indices(kwargs, kwargs_idxs, func)
    return args, kwargs


def _compute_time(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        start = time.time()
        fn(*args, **kwargs)
        end = time.time()
        return round(end - start, 6)

    return new_fn


def _read_or_create_csv(output_path="./report.csv"):
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write(",".join(COLUMNS) + "\n")
    return pd.read_csv(output_path)


def _write_to_csv(df, row_list, output_path="./report.csv"):
    row = dict(zip(COLUMNS, row_list))
    df = df.append(row, ignore_index=True)
    df.to_csv(output_path, index=False)


def eager_benchmark(
    obj: Union[Callable, str],
    functional_api: bool = False,
    num_experiments: int = 1,
    label: Optional[str] = None,
    backends: Optional[List[str]] = None,
    devices: Optional[List[str]] = None,
    args: Optional[Tuple[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    output_path="./report.csv",
):
    """Benchmark the function or module passed in input on the required
    backends and devices.

    Parameters
    ----------
    obj
        The function or module to be benchmarked with and without graph compilation.
        In case of a function from ivy's functional API, this parameter would receive
        a string which is the function name, along with functional_api set to True.
    functional_api
        Should only be set to ``True`` if the obj being passed is a part of ivy's
        functional API. (Default value = ``False``).
    num_experiments
        Option to run benchmarking multiple times to account for subtle variations.
        (Default value = 1).
    label
        The preferred name for the experiment as would be added to the csv. If no
        name is provided, then the __name__ of the obj would be picked by default
        (Default value = ``None``).
    backends
        A list of strings for backends to benchmark with. Should be among the backends
        that ivy supports (Default value = ``None``).
    devices
        A list of target devices that ivy supports with the backends. The devices that
        are invalid for a particular backend would be ignored
        (Default value  = ``None``).
    args
        The positional arguments to be passed to the obj.
    kwargs
        The keyword arguments to be passed to obj.
    output_path
        The path to the csv file to write to. By default results are written to
        reports.csv in the folder from where the script it run
        (Default value = ``None``).

    Examples
    --------
    With an :code:`ivy` function:

    >>> import ivy
    >>> from benchmark import eager_benchmark
    >>> ivy.set_backend("torch")
    >>> fn = "conv1d"
    >>> args = (
    ...     ivy.array([[[0.0], [3.0], [0.0]]], device="cpu"),
    ...     ivy.array([[[0.0]], [[1.0]], [[0.0]]], device="cpu"),
    ...     (1,),
    ...     "SAME",
    ... )
    >>> kwargs = {"data_format": "NWC", "dilations": (1,)}
    >>> eager_benchmark(
    ...     fn,
    ...     label="conv1d",
    ...     backends=["jax", "numpy", "tensorflow", "torch"],
    ...     devices=["cpu", "gpu:0"],
    ...     args=args,
    ...     kwargs=kwargs,
    ...     functional_api=True,
    ...     output_path="./ivy/report.csv"
    ... )

    With a compositional function:

    >>> import ivy
    >>> from benchmark import eager_benchmark
    >>> ivy.set_backend("torch")
    >>> def fn(*args, **kwargs):
    ...     return ivy.conv1d(*args, **kwargs) + 1
    >>> args = (
    ...     ivy.array([[[0.0], [3.0], [0.0]]], device="cpu"),
    ...     ivy.array([[[0.0]], [[1.0]], [[0.0]]], device="cpu"),
    ...     (1,),
    ...     "SAME",
    ... )
    >>> kwargs = {"data_format": "NWC", "dilations": (1,)}
    >>> eager_benchmark(
    ...     fn,
    ...     label="compos",
    ...     backends=["jax", "numpy", "tensorflow", "torch"],
    ...     devices=["cpu", "gpu:0"],
    ...     args=args,
    ...     kwargs=kwargs,
    ...     output_path="./ivy/report.csv"
    ... )

    With a module:

    >>> import ivy
    >>> from benchmark import eager_benchmark
    >>> ivy.set_backend("torch")
    >>> module = ivy.GELU(approximate=False)
    >>> args = (ivy.random_uniform(shape=(4, 32)),)
    >>> eager_benchmark(
    ...     module,
    ...     label="GELU",
    ...     backends=["jax", "numpy", "tensorflow", "torch"],
    ...     devices=["cpu", "gpu:0"],
    ...     args=args,
    ...     output_path="./ivy/report.csv"
    ... )
    """
    backends = ivy.default(backends, [])
    devices = ivy.default(devices, [])
    output_path = ivy.default(output_path, "./report.csv")
    print("\nBenchmarking backends : " + " ".join(backends))
    print(f"Number of experiments : {num_experiments}" + "\n")
    for i in range(num_experiments):
        if num_experiments > 1:
            print("====================")
            print(f"Experiment {i + 1}")
            print("====================\n")
        for backend in backends:
            with _AvoidGPUPreallocation(backend) as _:
                print("------------------------------------------------\n")
                print(f"backend : {backend}")
                ivy.set_backend(backend, dynamic=True)
                valid_devices = [
                    device
                    for device in devices
                    if device.split(":")[0] not in ivy.invalid_devices
                ]
                for device in valid_devices:
                    print(f"device : {device}")
                obj_call = obj
                if functional_api:
                    obj_call = ivy.__dict__[obj]
                for i, device in enumerate(valid_devices):
                    args, kwargs = _move_to_device(
                        args=args, kwargs=kwargs, device=device
                    )
                    if isinstance(obj_call, ivy.Module):
                        obj_call_copy = copy.deepcopy(obj_call)
                        obj_call_copy.trace(args=args, kwargs=kwargs)
                        traced_fn = obj_call_copy
                    else:
                        traced_fn = ivy.trace(obj_call, args=args, kwargs=kwargs)
                    kwargs = ivy.default(kwargs, {})
                    args = ivy.default(args, ())
                    untraced_time = _compute_time(obj_call)(*args, **kwargs)
                    traced_time = _compute_time(traced_fn)(*args, **kwargs)
                    label = obj_call.__name__ if label is None else label
                    percent_speed_up = round(
                        abs(untraced_time - traced_time) / untraced_time * 100, 6
                    )
                    df = _read_or_create_csv(output_path)
                    _write_to_csv(
                        df,
                        [
                            len(df.index),
                            label,
                            backend,
                            device,
                            untraced_time,
                            traced_time,
                            percent_speed_up,
                        ],
                        output_path,
                    )
                    args, kwargs = _move_to_device(
                        args=args, kwargs=kwargs, device="cpu"
                    )
                    ivy.clear_cached_mem_on_dev(device)
                    print(LINE_UP * (len(valid_devices) - i), end=LINE_CLEAR)
                    print(f"device : {device}\t --> done\n")
                ivy.unset_backend()
    print(f"Results written to {output_path} ...")


def visualize_speed_up(
    file_path: Optional[str] = None,
    output_path: Optional[str] = None,
    devices: Union[List[str], str] = "all",
    backends: Union[List[str], str] = "all",
    labels: Optional[Union[List[str], str]] = None,
):
    """Visualize the speed up results stored in the csv.

    Parameters
    ----------
    file_path
        The path of the csv file where the results are stored.
    output_path
        The path to the png file to store the graphs in.
    devices
        A filter for the devices for which graphs should be generated.
    backends
        A filter for the backends for which graphs should be generated.
    labels
        A filter for the labels for which graphs should be generated.

    Examples
    --------
    Visualize for given set of devices and backends:

    >>> from benchmark import visualize_speed_up
    >>> visualize_speed_up(
    ...     file_path="./ivy/report.csv",
    ...     output_path="./ivy/save_fig.png",
    ...     backends=["torch", "jax"],
    ...     devices=["cpu", "gpu:0"],
    ... )

    Visualize for a specific experiment label:

    >>> from benchmark import visualize_speed_up
    >>> visualize_speed_up(
    ...     file_path="./ivy/report.csv",
    ...     output_path="./ivy/save_fig.png",
    ...     backends=["jax"],
    ...     devices=["gpu:0"],
    ...     labels=["GELU"],
    ... )
    """
    file_path = ivy.default(file_path, "./report.csv")
    output_path = ivy.default(output_path, "./saved_fig.png")
    df = pd.read_csv(file_path)
    df = df.query("label in @labels") if labels is not None else df
    backends = list(df["backend"].unique()) if backends == "all" else backends
    devices = list(df["device"].unique()) if devices == "all" else devices
    fig, axes = plt.subplots(len(devices), len(backends))
    fig.set_figwidth(30)
    fig.set_figheight(12)
    fig.tight_layout(pad=10.0)
    axes = axes if isinstance(axes, np.ndarray) else np.asarray([axes])
    while len(axes.shape) < 2:
        if len(devices) > len(backends):
            axes = np.expand_dims(axes, len(axes.shape))
        else:
            axes = np.expand_dims(axes, 0)
    for device, axis in zip(devices, axes):
        for backend, ax in zip(backends, axis):
            ax.set_title(f"{backend} : {device}", {"fontsize": 18})
            ax.set_ylabel("Percent Speed up on compiling", {"fontsize": 18})
            ax.tick_params(axis="both", labelsize=15)
            query = df.query("backend == @backend and device == @device")
            if not query.empty:
                ax.violinplot(query["percent_speed_up"])
            else:
                warnings.warn(
                    f"No records matching the filters passedbackend={backend} and"
                    f" device={device}"
                )
    plt.savefig(output_path)
    print(f"plot saved to {output_path} ...")
