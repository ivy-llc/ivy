# global
from typing import Callable, Union
import functools
import time
import os
import copy
import importlib
import pandas as pd
import ivy


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


def move_to_device(args=None, kwargs=None, device="cpu"):
    args_idxs = ivy.nested_argwhere(args, ivy.is_array)
    kwargs_idxs = ivy.nested_argwhere(kwargs, ivy.is_array)
    func = lambda x: ivy.to_device(x, device, out=x)
    if args is not None:
        args = ivy.map_nest_at_indices(args, args_idxs, func)
    if kwargs is not None:
        kwargs = ivy.map_nest_at_indices(kwargs, kwargs_idxs, func)
    return args, kwargs


def compute_time(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        start = time.time()
        fn(*args, **kwargs)
        end = time.time()
        return round(end - start, 6)

    return new_fn


def read_or_create_csv(output_path="./ivy/report.csv"):
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write(",".join(COLUMNS) + "\n")
    return pd.read_csv(output_path)


def write_to_csv(df, row_list, output_path="./ivy/report.csv"):
    row = {k: v for k, v in zip(COLUMNS, row_list)}
    df = df.append(row, ignore_index=True)
    df.to_csv(output_path, index=False)


def eager_benchmark(
    obj: Union[Callable, str],
    label=None,
    backends=["jax"],
    devices=["cpu"],
    functional_api=False,
    args=None,
    kwargs=None,
    output_path="./ivy/report.csv",
):
    print("\nBenchmarking backends : " + " ".join(backends) + "\n")
    for backend in backends:
        with _AvoidGPUPreallocation(backend) as _:
            print("------------------------------------------------\n")
            print("backend : {}".format(backend))
            ivy.set_backend(backend, dynamic=True)
            valid_devices = [
                device
                for device in devices
                if device.split(":")[0] not in ivy.invalid_devices
            ]
            for device in valid_devices:
                print("device : {}".format(device))
            obj_call = obj
            if functional_api:
                obj_call = ivy.__dict__[obj]
            for i, device in enumerate(valid_devices):
                # move the arrays to dev
                args, kwargs = move_to_device(args=args, kwargs=kwargs, device=device)

                # compilation
                if isinstance(obj_call, ivy.Module):
                    obj_call_copy = copy.deepcopy(obj_call)
                    obj_call_copy.compile(args=args, kwargs=kwargs)
                    compiled_fn = obj_call_copy
                else:
                    compiled_fn = ivy.compile(obj_call, args=args, kwargs=kwargs)

                # wrapping both function to compute time
                kwargs = ivy.default(kwargs, {})
                args = ivy.default(args, ())
                uncompiled_time = compute_time(obj_call)(*args, **kwargs)
                compiled_time = compute_time(compiled_fn)(*args, **kwargs)

                label = obj_call.__name__ if label is None else label
                percent_speed_up = round(
                    abs(uncompiled_time - compiled_time) / uncompiled_time * 100, 6
                )

                # write results to the csv
                df = read_or_create_csv(output_path)
                write_to_csv(
                    df,
                    [
                        len(df.index),
                        label,
                        backend,
                        device,
                        uncompiled_time,
                        compiled_time,
                        percent_speed_up,
                    ],
                    output_path,
                )

                # move the arrays back to the cpu
                args, kwargs = move_to_device(args=args, kwargs=kwargs, device="cpu")

                # clear device memory
                ivy.clear_cached_mem_on_dev(device)
                print(LINE_UP * (len(valid_devices) - i), end=LINE_CLEAR)
                print("device : {}\t --> done\n".format(device))
            ivy.unset_backend()
