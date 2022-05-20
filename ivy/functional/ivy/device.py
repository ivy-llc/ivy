"""Collection of device Ivy functions."""

# global
import os
import gc
import abc
import math
import time
import queue
import psutil
import inspect
import logging
import nvidia_smi
from typing import Optional

# noinspection PyUnresolvedReferences
try:
    nvidia_smi.nvmlInit()
except nvidia_smi.NVMLError_LibraryNotFound:
    pass
from typing import Union, Type, Callable, Iterable, Dict, Any

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

default_device_stack = list()
dev_handles = dict()
split_factors = dict()
max_chunk_sizes = dict()


# Extra #
# ------#


class DefaultDevice:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, device):
        self._dev = device

    def __enter__(self):
        set_default_device(self._dev)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_device()
        return self


# Helpers #

# noinspection PyShadowingNames
def _get_nvml_gpu_handle(device):
    global dev_handles
    if device in dev_handles:
        return dev_handles[device]
    gpu_idx = int(device.split(":")[-1])
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)
    dev_handles[device] = handle
    return handle


# Device Queries #

# Array Printing

# noinspection PyShadowingNames
def get_all_arrays_on_dev(device):
    """Gets all arrays which are currently alive on the specified device.

    Parameters
    ----------
    device

    """
    all_arrays = list()
    for obj in gc.get_objects():
        # noinspection PyBroadException
        try:
            if ivy.is_array(obj) and ivy.dev(obj) == device:
                all_arrays.append(obj)
        except Exception:
            pass
    return ivy.Container(dict(zip([str(id(a)) for a in all_arrays], all_arrays)))


# noinspection PyShadowingNames
def num_arrays_on_dev(device):
    """Returns the number of arrays which are currently alive on the specified device.

    Parameters
    ----------
    device

    """
    return len(get_all_arrays_on_dev(device))


# noinspection PyShadowingNames
def print_all_arrays_on_dev(device):
    """Prints all arrays which are currently alive on the specified device.

    Parameters
    ----------
    device

    """
    for arr in get_all_arrays_on_dev(device):
        print(type(arr), arr.shape)


# Retrieval


def dev(
    x: Union[ivy.Array, ivy.NativeArray], as_str: bool = False
) -> Union[ivy.Device, str]:
    """
    Get the native device handle for input array x.

    Parameters
    ----------
    x
          array for which to get the device handle.

    as_str
          Whether or not to return the dev in string format. Default is False.

    Returns
    -------
    ret
          Device handle for the array, in native framework format.

    Examples
    --------
          >>> x = ivy.array([1,0,2])
          >>> y = ivy.dev(x)
          >>> print(y)
          "cpu"
    """
    return _cur_framework(x).dev(x, as_str)


# Conversions

# noinspection PyShadowingNames
def dev_to_str(device: Union[ivy.Device, str]) -> str:
    """Convert native data type to string representation.

    Parameters
    ----------
    device
        The device handle to convert to string.

    Returns
    -------
    ret
        Device string e.g. 'cuda:0'.

    """
    return _cur_framework().dev_to_str(device)


# noinspection PyShadowingNames
def dev_from_str(device: Union[ivy.Device, str]) -> ivy.Device:
    """Convert device string representation to native device type.

    Parameters
    ----------
    device
        The device string to conver to native device handle.

    Returns
    -------
    ret
        Native device handle.

    """
    return _cur_framework().dev_from_str(device)


# Memory

# noinspection PyShadowingNames
def clear_mem_on_dev(device: ivy.Device) -> None:
    """Clear memory cache on target device.

    Parameters
    ----------
    device
        The device string to conver to native device handle.

    """
    return _cur_framework(None).clear_mem_on_dev(device)


# noinspection PyShadowingNames
def total_mem_on_dev(device: ivy.Device) -> float:
    """Get the total amount of memory (in GB) for a given device string. In case of CPU,
    the total RAM is returned.

    Parameters
    ----------
    device
        The device string to conver to native device handle.

    Returns
    -------
    ret
        The total memory on the device in GB.

    """
    if "gpu" in device:
        handle = _get_nvml_gpu_handle(device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.total / 1e9
    elif device == "cpu":
        return psutil.virtual_memory().total / 1e9
    else:
        raise Exception(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


# noinspection PyShadowingNames
def used_mem_on_dev(device: ivy.Device, process_specific=False) -> float:
    """Get the used memory (in GB) for a given device string. In case of CPU, the used
    RAM is returned.

    Parameters
    ----------
    device
        The device string to conver to native device handle.
    process_specific
        Whether the check the memory used by this python process alone. Default is
        False.

    Returns
    -------
    ret
        The used memory on the device in GB.

    """
    ivy.clear_mem_on_dev(device)
    if "gpu" in device:
        if process_specific:
            raise Exception("process-specific GPU queries are currently not supported")
        handle = _get_nvml_gpu_handle(device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1e9
    elif device == "cpu":
        if process_specific:
            return psutil.Process(os.getpid()).memory_info().rss
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 1e9
    else:
        raise Exception(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


# noinspection PyShadowingNames
def percent_used_mem_on_dev(device: ivy.Device, process_specific=False) -> float:
    """Get the percentage used memory for a given device string. In case of CPU, the
    used RAM is returned.

    Parameters
    ----------
    device
        The device string to conver to native device handle.
    process_specific
        Whether the check the memory used by this python process alone. Default is
        False.

    Returns
    -------
    ret
        The percentage used memory on the device.

    """
    ivy.clear_mem_on_dev(device)
    if "gpu" in device:
        if process_specific:
            raise Exception("process-specific GPU queries are currently not supported")
        handle = _get_nvml_gpu_handle(device)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return (info.used / info.total) * 100
    elif device == "cpu":
        vm = psutil.virtual_memory()
        if process_specific:
            return (psutil.Process(os.getpid()).memory_info().rss / vm.total) * 100
        return (1 - (vm.available / vm.total)) * 100
    else:
        raise Exception(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


# Utilization

# noinspection PyShadowingNames
def dev_util(device: ivy.Device) -> float:
    """Get the current utilization (%) for a given device.

    Parameters
    ----------
    device
        The device string of the device to query utilization for.

    Returns
    -------
    ret
        The device utilization (%)

    """
    if device == "cpu":
        return psutil.cpu_percent()
    elif "gpu" in device:
        handle = _get_nvml_gpu_handle(device)
        return nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        raise Exception(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


# Availability


def gpu_is_available() -> bool:
    """Determine whether a GPU is available to use, with the backend framework.

    Returns
    -------
    ret
        Boolean, as to whether a gpu is available.

    Examples
    --------
    >>> print(ivy.gpu_is_available())
    True

    """
    return _cur_framework().gpu_is_available()


def num_cpu_cores() -> int:
    """Determine the number of cores available in the cpu.

    Returns
    -------
    ret
        Number of cores available in CPU

    Examples
    --------
    >>> print(ivy.num_cpu_cores())
    2

    """
    return psutil.cpu_count()


def num_gpus() -> int:
    """Determine the number of available GPUs, with the backend framework.

    Returns
    -------
    ret
        Number of available GPUs.

    Examples
    --------
    >>> print(ivy.num_gpus())
    1

    """
    return _cur_framework().num_gpus()


def tpu_is_available() -> bool:
    """Determine whether a TPU is available to use, with the backend framework.

    Returns
    -------
    ret
        Boolean, as to whether a tpu is available.

    Examples
    --------
    >>> print(ivy.tpu_is_available())
    True

    """
    return _cur_framework().tpu_is_available()


# Default Device #

# noinspection PyShadowingNames
def _assert_dev_correct_formatting(device):
    assert device[0:3] in ["gpu", "tpu", "cpu"]
    if device != "cpu":
        assert device[3] == ":"
        assert device[4:].isnumeric()


# noinspection PyShadowingNames
def default_device(device=None, item=None, as_str: bool = False):
    """Summary.

    Parameters
    ----------
    device
         (Default value = None)
    item
         (Default value = None)
    as_str
         (Default value = False)

    Returns
    -------
    ret

    """
    if ivy.exists(device):
        _assert_dev_correct_formatting(ivy.dev_to_str(device))
        return device
    elif ivy.exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_array(item):
            return ivy.dev(item, as_str=as_str)
    global default_device_stack
    if not default_device_stack:
        ret = "gpu:0" if ivy.gpu_is_available() else "cpu"
    else:
        ret = default_device_stack[-1]
    if as_str:
        return ivy.dev_to_str(ret)
    return ivy.dev_from_str(ret)


# noinspection PyShadowingNames
def set_default_device(device):
    """Summary.

    Parameters
    ----------
    device

    """
    _assert_dev_correct_formatting(device)
    global default_device_stack
    default_device_stack.append(device)


def unset_default_device():
    """"""
    global default_device_stack
    if default_device_stack:
        default_device_stack.pop(-1)


# Device Allocation #

# noinspection PyShadowingNames
def to_dev(
    x: Union[ivy.Array, ivy.NativeArray],
    device: ivy.Device = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Move the input array x to the desired device, specified by device string.

    Parameters
    ----------
    x
       input array to be moved to the desired device
    device
        device to move the input array `x` to
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        input array x placed on the desired device

    Examples
    --------
    >>> x = ivy.array([1., 2., 3.])
    >>> x = ivy.to_dev(x, 'cpu')

    """
    return _cur_framework(x).to_dev(x, device, out)


# Function Splitting #

# noinspection PyShadowingNames
def split_factor(device=None):
    """Get the global split factor for a given device, which can be used to scale batch
    splitting chunk sizes for the device across the codebase. Default global value for
    each device is 1.

    Parameters
    ----------
    device
        The device to query the split factor for. Sets the default device by default.

    Returns
    -------
    ret
        The split factor for the specified device.

    """
    global split_factors
    device = ivy.default(device, default_device())
    if device in split_factors:
        return split_factors[device]
    split_factors[device] = 0.0
    return split_factors[device]


# noinspection PyShadowingNames
def set_split_factor(factor, device=None):
    """Set the global split factor for a given device, which can be used to scale batch
    splitting chunk sizes for the device across the codebase.

    Parameters
    ----------
    factor
        The factor to set the device-specific split factor to.
    device
        The device to set the split factor for. Sets the default device by default.

    """
    assert 0 <= factor
    global split_factors
    device = ivy.default(device, default_device())
    split_factors[device] = factor


# noinspection PyShadowingNames
def split_func_call(
    func: Callable,
    inputs: Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]],
    mode: str,
    max_chunk_size: int = None,
    chunk_size: int = None,
    input_axes: Union[int, Iterable[int]] = 0,
    output_axes: Union[int, Iterable[int]] = None,
    stop_gradients: bool = False,
    device=None,
) -> Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]]:
    """Call a function by splitting its inputs along a given axis, and calling the
    function in chunks, rather than feeding the entire input array at once. This can be
    useful to reduce memory usage of the device the arrays are on.

    Parameters
    ----------
    func
        The function to be called.
    inputs
        A list of inputs to pass into the function.
    mode
        The mode by which to unify the return values, must be one of
        [ concat | mean | sum ]
    max_chunk_size
        The maximum size of each of the chunks to be fed into the function.
    chunk_size
        The size of each of the chunks to be fed into the function. Specifying this arg
        overwrites the global split factor. Default is None.
    input_axes
        The axes along which to split each of the inputs, before passing to the
        function. Default is 0.
    output_axes
        The axes along which to concat each of the returned outputs. Default is same as
        fist input axis.
    stop_gradients
        Whether to stop the gradients for each computed return. Default is False.
    device
        The device to set the split factor for. Sets the default device by default.

    Returns
    -------
    ret
        The return from the function, following input splitting and re-concattenation.

    """
    if isinstance(input_axes, int):
        input_axes = [input_axes] * len(inputs)
    if not ivy.exists(max_chunk_size) and not ivy.exists(chunk_size):
        shape_key = "_".join([str(inp.shape) for inp in inputs])
        if shape_key in max_chunk_sizes:
            max_chunk_size = max_chunk_sizes[shape_key]
        else:
            max_chunk_size = 0
        max_dim = max([inp.shape[inp_ax] for inp, inp_ax in zip(inputs, input_axes)])
        if max_dim > max_chunk_size:
            max_chunk_sizes[shape_key] = max_dim
            max_chunk_size = max_dim
    chunk_size = ivy.default(
        chunk_size,
        lambda: 1
        + int(
            round((max_chunk_size - 1) * ivy.split_factor(ivy.default_device(device)))
        ),
        True,
    )
    dim_size = inputs[0].shape[input_axes[0]]
    if chunk_size >= dim_size:
        return func(*inputs)
    num_chunks = dim_size / chunk_size
    num_chunks_floored = math.floor(num_chunks)
    num_chunks_ceiled = math.ceil(num_chunks)
    chunk_sizes = [chunk_size] * num_chunks_floored
    if num_chunks != num_chunks_floored:
        chunk_sizes.append(dim_size - chunk_size * num_chunks_floored)
    inputs_split = [
        ivy.split(inp, chunk_sizes, input_axes[i], True)
        if ivy.is_array(inp)
        else inp.split(chunk_sizes, input_axes[i], True)
        for i, inp in enumerate(inputs)
    ]
    is_mean = mode == "mean"
    is_sum = mode == "sum"
    post_fn = ivy.stop_gradient if stop_gradients else lambda x: x
    if is_mean or is_sum:
        sums = None
        for inps in zip(*inputs_split):
            if not sums:
                sums = func(*inps)
                sums = (
                    [post_fn(s) for s in sums]
                    if isinstance(sums, tuple)
                    else [post_fn(sums)]
                )
            else:
                ret = func(*inps)
                if isinstance(ret, tuple):
                    for i, r in enumerate(ret):
                        sums[i] = sums[i] + post_fn(r)
                else:
                    sums[0] = sums[0] + post_fn(ret)
        sums_or_means = [s / num_chunks_ceiled for s in sums] if is_mean else sums
        return sums_or_means[0] if len(sums_or_means) == 1 else tuple(sums_or_means)
    rets = [func(*i) for i in zip(*inputs_split)]
    rets = [
        tuple([post_fn(r) for r in ret]) if isinstance(ret, tuple) else (post_fn(ret),)
        for ret in rets
    ]
    num_outputs = len(rets[0])
    if output_axes is None:
        output_axes = [input_axes[0]] * num_outputs
    elif isinstance(output_axes, int):
        output_axes = [output_axes] * num_outputs
    ret = [ivy.concat([r[i] for r in rets], output_axes[i]) for i in range(num_outputs)]
    return ret[0] if len(ret) == 1 else ret


# Multi-Device #


class MultiDev:
    def __init__(self, data: Iterable, axis=0):
        if isinstance(data, MultiDev):
            # noinspection PyUnresolvedReferences,PyProtectedMember
            data = data._dict
        self._axis = axis
        self._data = data
        self._length = len(data)
        self._counter = 0

    def __len__(self):
        return self._length

    def __repr__(self):
        return "MultiDev(" + self._data.__repr__() + ")"


class MultiDevItem(MultiDev):
    def __init__(self, data: Dict[ivy.Device, Any], axis=0):
        super().__init__(data, axis)

    @property
    def shape(self):
        shapes = [
            list(x.shape) if hasattr(x, "shape") else None for x in self._data.values()
        ]
        if not shapes or None in shapes:
            return None
        shape0 = shapes[0]
        for shp in shapes[1:]:
            assert shp == shape0
        shape0[self._axis] = shape0[self._axis] * len(self)
        return tuple(shape0)

    def _slice(self, slice_obj: slice):
        stacked_dim_size = 0
        ret_dict = dict()
        for ds, sub_item in self._data.items():
            if not hasattr(sub_item, "shape"):
                continue
            shp = sub_item.shape
            rel_slice_obj = slice(
                slice_obj.start - stacked_dim_size, slice_obj.stop - stacked_dim_size, 1
            )
            stacked_dim_size += shp[self._axis]
            if slice_obj.start < stacked_dim_size:
                if slice_obj.stop < stacked_dim_size:
                    ret_dict[ds] = sub_item[rel_slice_obj]
                    return MultiDevItem(ret_dict)
                else:
                    ret_dict[ds] = sub_item[rel_slice_obj.start :]
        return MultiDevItem(ret_dict)

    def __getitem__(self, query):
        if isinstance(query, str):
            return self._data[query]
        elif isinstance(query, int):
            return self._slice(slice(query, query + 1, 1))
        elif isinstance(query, slice):
            return self._slice(query)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return "MultiDevItem(" + self._data.__repr__() + ")"


class MultiDevIter(MultiDev):
    def __init__(self, data: Iterable, devices):
        self._devs = devices
        super().__init__(data)

    # noinspection PyShadowingNames
    def at_dev(self, device):
        """Summary.

        Parameters
        ----------
        device

        """
        return [x[device] if isinstance(x, MultiDevItem) else x for x in self._data]

    def at_devs(self):
        """"""
        return {ds: self.at_dev(ds) for ds in self._devs}

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter == self._length:
            raise StopIteration
        ret = self.__getitem__(self._counter)
        self._counter += 1
        return ret

    def __repr__(self):
        return "MultiDevIter(" + self._data.__repr__() + ")"


class MultiDevNest(MultiDevIter):
    def __init__(self, data: Iterable, devices, max_depth=1):
        self._max_depth = max_depth
        super().__init__(data, devices)

    # noinspection PyShadowingNames
    def at_dev(self, device):
        """Summary.

        Parameters
        ----------
        device

        """
        return ivy.nested_map(
            self._data,
            lambda x: x[device] if isinstance(x, MultiDevItem) else x,
            max_depth=self._max_depth,
        )

    def __repr__(self):
        return "MultiDevNest(" + self._data.__repr__() + ")"


# Device Distribution #


class DevDistItem(MultiDevItem):
    def __repr__(self):
        return "DevDistItem(" + self._data.__repr__() + ")"


class DevDistIter(MultiDevIter):
    def __repr__(self):
        return "DevDistIter(" + self._data.__repr__() + ")"


class DevDistNest(MultiDevNest):
    def __repr__(self):
        return "DevDistNest(" + self._data.__repr__() + ")"


def dev_dist_array(x, devices: Union[Iterable[str], Dict[str, int]], axis=0):
    """Distribute an array across the specified devices, returning a list of sub-arrays,
    each on a different device.

    Parameters
    ----------
    x
        The array to distribute across devices.
    devices
        The devices to distribute the array across.
    axis
        The axis along which to split the array. Default is 0.
    devices

    Dict

    Returns
    -------
    ret
        array distributed across the target devices

    """
    split_arg = list(devices.values()) if isinstance(devices, dict) else len(devices)
    return DevDistItem(
        {
            ds: ivy.to_dev(x_sub, ds)
            for x_sub, ds in zip(
                ivy.split(x, split_arg, axis, with_remainder=True), devices
            )
        }
    )


def dev_dist(x, devices: Union[Iterable[str], Dict[str, int]], axis=0):
    """Distribute the input item across the specified devices, returning a list of sub-
    items, each on a different device.

    Parameters
    ----------
    x
        The input array or container to distribute across devices.
    devices
        The devices to distribute the input across.
    axis
        The axis along which to split the input. Default is 0.
    devices

    Dict

    Returns
    -------
    ret
        array or container distributed across the target devices

    """
    if ivy.is_array(x):
        return dev_dist_array(x, devices, axis)
    elif isinstance(x, ivy.Container):
        return x.dev_dist(devices, axis)
    return x


def dev_dist_iter(xs, devices: Union[Iterable[str], Dict[str, int]], axis=0):
    """Distribute elements of the iterbale xs across the specified devices.

    Parameters
    ----------
    xs
        The iterable of items to distribute.
    devices
        The devices to distribute the iterable elements across.
    axis
        The axis along which to split the arrays in the iterable xs. Default is 0.
    devices

    Dict

    Returns
    -------
    ret
        iterable with each element distributed to the target devices

    """
    if isinstance(devices, str):
        devices = [devices]
    return DevDistIter([dev_dist(x, devices, axis) for x in xs], devices)


def dev_dist_nest(
    args, kwargs, devices: Union[Iterable[str], Dict[str, int]], axis=0, max_depth=1
):
    """Distribute the nested input arguments across the specified devices.

    Parameters
    ----------
    args
        The positional nested arguments to distribute.
    kwargs
        The keyword nested arguments to distribute.
    devices
        The devices to distribute the nested arguments across.
    axis
        The axis along which to split the arrays in the arguments. Default is 0.
    max_depth
        The maximum nested depth to reach. Default is 1. Increase this if the nest is
        deeper.
    devices

    Dict


    Returns
    -------
    ret
        nested arguments distributed to the target devices

    """
    if isinstance(devices, str):
        devices = [devices]
    args_dist = ivy.nested_map(
        args, lambda x: dev_dist(x, devices, axis), max_depth=max_depth
    )
    kwargs_dist = ivy.nested_map(
        kwargs, lambda x: dev_dist(x, devices, axis), max_depth=max_depth
    )
    return DevDistNest(args_dist, devices), DevDistNest(kwargs_dist, devices)


# Device Cloning #


class DevClonedItem(MultiDevItem):
    def __repr__(self):
        return "DevClonedItem(" + self._data.__repr__() + ")"


class DevClonedIter(MultiDevIter):
    def __repr__(self):
        return "DevClonedIter(" + self._data.__repr__() + ")"


class DevClonedNest(MultiDevNest):
    def __repr__(self):
        return "DevClonedNest(" + self._data.__repr__() + ")"


def dev_clone_array(x, devices):
    """Clone an array across the specified devices, returning a list of cloned arrays,
    each on a different device.

    Parameters
    ----------
    x
        The array to clone across devices.
    devices
        The devices to clone the array to.

    Returns
    -------
    ret
        array cloned to each of the target devices

    """
    return DevClonedItem({ds: ivy.stop_gradient(ivy.to_dev(x, ds)) for ds in devices})


def dev_clone(x, devices):
    """Clone the input item to each of the specified devices, returning a list of cloned
    items, each on a different device.

    Parameters
    ----------
    x
        The input array or container to clone to each device.
    devices
        The deviceices to clone the input to.

    Returns
    -------
    ret
        array or container distributed across the target devices

    """
    if ivy.is_array(x):
        return dev_clone_array(x, devices)
    elif isinstance(x, ivy.Container):
        return x.dev_clone(devices)
    return x


def dev_clone_iter(xs, devices):
    """Clone elements of the iterable xs to each of the specified devices.

    Parameters
    ----------
    xs
        The iterable of items to clone.
    devices
        The devices to clone each of the iterable elements to.

    Returns
    -------
    ret
        iterable with each element cloned to each of the target devices

    """
    if isinstance(devices, str):
        devices = [devices]
    return DevClonedIter([dev_clone(x, devices) for x in xs], devices)


def dev_clone_nest(args, kwargs, devices, max_depth=1):
    """Clone the input arguments across the specified devices.

    Parameters
    ----------
    args
        The positional arguments to clone.
    kwargs
        The keyword arguments to clone.
    devices
        The devices to clone the arguments to.
    max_depth
        The maximum nested depth to reach. Default is 1. Increase this if the nest is
        deeper.

    Returns
    -------
    ret
        arguments cloned to each of the target devices

    """
    if isinstance(devices, str):
        devices = [devices]
    args_cloned = ivy.nested_map(
        args, lambda x: dev_clone(x, devices), max_depth=max_depth
    )
    kwargs_cloned = ivy.nested_map(
        kwargs, lambda x: dev_clone(x, devices), max_depth=max_depth
    )
    return DevClonedNest(args_cloned, devices), DevClonedNest(kwargs_cloned, devices)


# Device Unification #

# noinspection PyShadowingNames
def _concat_unify_array(xs, device, axis):
    return ivy.concat([ivy.to_dev(x_sub, device) for x_sub in xs.values()], axis)


# noinspection PyShadowingNames
def _sum_unify_array(xs, device, _=None):
    return sum([ivy.to_dev(x_sub, device) for x_sub in xs.values()])


# noinspection PyShadowingNames
def _mean_unify_array(xs, device, _=None):
    return _sum_unify_array(xs, device) / len(xs)


# noinspection PyShadowingNames
def dev_unify_array(xs, device, mode, axis=0):
    """Unify a list of sub-arrays, on arbitrary devices, to a single array on the
    specified device.

    Parameters
    ----------
    xs
        The list of arrays to unify onto the specified device.
    device
        The device to unify the arrays to.
    mode
        The mode by which to unify, must be one of [ concat | mean | sum ]
    axis
        The axis along which to concattenate the array, if concat mode is set. Default
        is 0.

    Returns
    -------
    ret
        array unified to the target device

    """
    return {
        "concat": _concat_unify_array,
        "sum": _sum_unify_array,
        "mean": _mean_unify_array,
    }[mode](xs, device, axis)


# noinspection PyShadowingNames
def dev_unify(xs, device, mode, axis=0):
    """Unify a list of sub-arrays, on arbitrary devices, to a single concattenated array
    on the specified device.

    Parameters
    ----------
    xs
        The list of sub-arrays to unify onto the specified device.
    device
        The device to unify the sub-arrays to.
    mode
        The mode by which to unify, must be one of [ concat | mean | sum ]
    axis
        The axis along which to concattenate the array, if concat mode is set. Default
        is 0.

    Returns
    -------
    ret
        array unified to the target device

    """
    if isinstance(xs, ivy.MultiDevContainer):
        xs = MultiDevItem(xs.at_devs())
    elif not isinstance(xs, MultiDevItem):
        return xs
    # noinspection PyProtectedMember
    xs0 = next(iter(xs.items()))[1]
    if ivy.is_array(xs0):
        return dev_unify_array(xs, device, mode, axis)
    elif isinstance(xs0, ivy.Container):
        return ivy.Container.unify(xs, device, mode, axis)
    return xs


# noinspection PyShadowingNames
def dev_unify_iter(xs, device, mode, axis=0, transpose=False):
    """Unify elements of the iterbale xs to a single target device.

    Parameters
    ----------
    xs
        The iterable of items to unify.
    device
        The device to unify the elements of the iterable to.
    mode
        The mode by which to unify, must be one of [ concat | mean | sum ]
    axis
        The axis along which to concattenate the sub-arrays. Default is 0.
    transpose
        Whether to transpose the first and second dimensions of the iterator. Default is
        False.

    Returns
    -------
    ret
        iterable with each element unified to a single target devices

    """
    # noinspection PyProtectedMember
    xs = xs._data if isinstance(xs, MultiDevIter) else xs
    if transpose:
        # ToDo: make this more elegant, this method should not be
        #  responsible for transposing iterators
        xs_t = [
            MultiDevItem({ivy.dev(i) if ivy.is_array(i) else i.dev: i for i in mdi})
            for mdi in list(map(list, zip(*xs)))
        ]
        return [dev_unify(x, device, mode, axis) for x in xs_t]
    return dev_unify(xs, device, mode, axis)


# noinspection PyShadowingNames,PyProtectedMember
def dev_unify_nest(
    args: Type[MultiDev], kwargs: Type[MultiDev], device, mode, axis=0, max_depth=1
):
    """Unify the input nested arguments, which consist of sub-arrays spread across
    arbitrary devices, to unified arrays on the single target device.

    Parameters
    ----------
    args
        The nested positional arguments to unify.
    kwargs
        The nested keyword arguments to unify.
    device
        The device to unify the nested arguments to.
    mode
        The mode by which to unify, must be one of [ concat | mean | sum ]
    axis
        The axis along which to concattenate the sub-arrays. Default is 0.
    max_depth
        The maximum nested depth to reach. Default is 1. Increase this if the nest is
        deeper.
    args
    kwargs

    Returns
    -------
    ret
        nested arguments unified to the target device

    """
    args = args._data if isinstance(args, MultiDevIter) else args
    kwargs = kwargs._data if isinstance(kwargs, MultiDevIter) else kwargs
    args_uni = ivy.nested_map(
        args, lambda x: dev_unify(x, device, mode, axis), max_depth=max_depth
    )
    kwargs_uni = ivy.nested_map(
        kwargs, lambda x: dev_unify(x, device, mode, axis), max_depth=max_depth
    )
    return args_uni, kwargs_uni


# Device Mappers #


class DevMapper(abc.ABC):
    def __init__(
        self,
        fn,
        ret_fn,
        queue_class,
        worker_class,
        devices,
        timeout=None,
        constant=None,
        unique=None,
    ):
        """Device Mapper base class.

        Parameters
        ----------
        fn
            The function which the device mapper parallelises across devices.
        ret_fn
            The function which receives the ivy.MultiDevIter as input, and produces a
            single device output.
        queue_class
            The class to use for creating queues.
        worker_class
            The class to use for creating parallel workers.
        devices
            A list of devices on which to parallelise the function.
        timeout
            The timeout for getting items from the queues. Default is global.
        constant
            A dict of keyword arguments which are the same for each process. Default is
            None.
        unique
            A dict of keyword argument sequences which are unique for each process.
            Default is None.

        """
        constant_kwargs = ivy.default(constant, {})
        unique_kwargs = ivy.default(unique, {})
        self._fn = fn
        self._ret_fn = ret_fn
        self._devs = devices
        self._num_workers = len(devices)
        self._timeout = ivy.default(timeout, ivy.queue_timeout())
        self._workers = dict()
        self._input_queues = dict()
        self._output_queues = dict()
        self._worker_class = worker_class
        for i, ds in enumerate(self._devs):
            input_queue = queue_class()
            output_queue = queue_class()
            worker_kwargs = dict(
                **constant_kwargs, **{k: v[i] for k, v in unique_kwargs.items()}
            )
            worker = self._worker_class(
                target=self._worker_fn,
                args=(
                    input_queue,
                    output_queue,
                    devices[i],
                    worker_kwargs,
                    ivy.current_framework_str(),
                ),
            )
            worker.start()
            self._input_queues[ds] = input_queue
            self._output_queues[ds] = output_queue
            self._workers[ds] = worker

    def __getstate__(self):
        # prevent already running processes from being pickled as sent to new processes
        state = self.__dict__.copy()
        state["_workers"] = None
        state["_ret_fn"] = None
        return state

    # noinspection PyShadowingNames
    def _worker_fn(self, input_queue, output_queue, device, kwargs, framework_str):
        ivy.set_framework(framework_str)
        ivy.set_default_device(device)
        for k, v in kwargs.items():
            if isinstance(v, ivy.Module) and not v.built:
                v.build(device=device)
        if "device" in inspect.getfullargspec(self._fn).args:
            kwargs["device"] = device
        while True:
            try:
                loaded_kwargs = input_queue.get(timeout=self._timeout)
            except queue.Empty:
                continue
            if loaded_kwargs is None:
                return
            if "split_factor" in loaded_kwargs:
                ivy.set_split_factor(loaded_kwargs["split_factor"], device)
                del loaded_kwargs["split_factor"]
            ret = self._fn(**loaded_kwargs, **kwargs)
            output_queue.put(ret)

    def map(self, used_devs=None, split_factors=None, **kwargs):
        """Map the function fn to each of the MultiDevice args and kwargs, running each
        function in parallel with CUDA-safe multiprocessing.

        Parameters
        ----------
        used_devs
            The devices used in the current mapping pass. Default is all devs.
        split_factors
            The updated split factors 0 < sf < 1 for each device. Default is None.
        kwargs
            The MutliDevice keyword arguments to map the function to.

        Returns
        -------
        ret
            The results of the function, returned as a MultiDevice instance.

        """
        if ivy.exists(split_factors):
            kwargs["split_factor"] = split_factors
        used_devs = ivy.default(used_devs, self._devs)
        [
            self._input_queues[ds].put({k: v[ds] for k, v in kwargs.items()})
            for ds in used_devs
        ]
        return self._ret_fn(
            ivy.MultiDevIter(
                [
                    self._output_queues[ds].get(timeout=self._timeout)
                    for ds in used_devs
                ],
                self._num_workers,
            )
        )

    @abc.abstractmethod
    def __del__(self):
        raise NotImplementedError


class DevMapperMultiProc(DevMapper):
    """"""

    def __init__(self, fn, ret_fn, devices, timeout=None, constant=None, unique=None):
        multiprocessing = ivy.multiprocessing("forkserver")
        super().__init__(
            fn,
            ret_fn,
            multiprocessing.Queue,
            multiprocessing.Process,
            devices,
            timeout,
            constant,
            unique,
        )

    def __del__(self):
        # noinspection PyBroadException
        try:
            for i, w in enumerate(self._workers.values()):
                self._input_queues[i].put(None)
                w.join(timeout=0.25)
            for q in self._input_queues.values():
                q.cancel_join_thread()
                q.close()
            for q in self._output_queues.values():
                q.cancel_join_thread()
                q.close()
        except Exception:
            pass
        finally:
            for w in self._workers.values():
                if w.is_alive():
                    w.terminate()


# Device Manager #


class DevManager:
    """"""

    def __init__(
        self,
        dev_mapper=None,
        devices: Union[Iterable[str], Dict[str, int]] = None,
        da_dim_size=None,
        safety_factor=1.1,
        min_dev_dim_size=0,
        max_dev_dim_step_ratio=0.1,
        min_unit_dev_tune_steps=10,
        min_sf_tune_steps=10,
        starting_split_factor=0.0,
        max_split_factor_step_size=0.05,
        tune_dev_alloc=True,
        tune_dev_splits=True,
    ):
        """Create device manager, which unlike the device mapper, handles all argument
        cloning and distributing internally. The device manager only receivess a
        specification regarding the ratio of the batch each device should consume.

        Parameters
        ----------
        dev_mapper
            The pre-built device mapper used by the manager internally.
            (Default value = None)
        devices
            The devices to distribute and clone the arguments across.
        da_dim_size
            The size of the dimension along which the device allocation splitting is
            performed. (Default value = None)
        safety_factor
            The factor by which to be safe in the avoidance of OOM GPU errors.
            Default is 1.1.
        min_dev_dim_size
            The minimum dimension size to pass to a device. Default is 0.
        max_dev_dim_step_ratio
            The maximum step ratio for changing the dimension for a device.
            Default is 0.1.
        min_unit_dev_tune_steps
            The minimum number of tune steps to make when optimizing with unit step
            size. Default is 10.
        min_sf_tune_steps
            Minimum number of split factor tune steps. Default is 10.
        starting_split_factor
            The initial device-specific split factor. Default is 0.
        max_split_factor_step_size
            The maximum step size for changing the split factor for a device.
            Default is 0.05.
        tune_dev_alloc
            Whether to tune the device split sizes internally based on device
            utilization tracking, and use the provided values for initialization.
            Default is True.
        tune_dev_splits
            Whether to tune the per-device split sizes internally. Default is True.

        """
        with_dev_mapping = True if ivy.exists(dev_mapper) else False
        tune_dev_alloc = False if not with_dev_mapping else tune_dev_alloc
        self._dev_mapper = dev_mapper
        devices = ivy.default(devices, [ivy.default_device()])
        self._num_devs = len(devices)
        self._dim_size = da_dim_size
        assert 1 <= safety_factor
        self._safety_factor = safety_factor
        self._min_dev_dim_size = min_dev_dim_size
        self._max_dev_dim_step_ratio = max_dev_dim_step_ratio
        if self._dim_size:
            self._max_dev_dim_step_size = max(
                int(round(self._max_dev_dim_step_ratio * self._dim_size)), 1
            )
        self._min_unit_dev_tune_steps = min_unit_dev_tune_steps
        self._min_sf_tune_steps = min_sf_tune_steps
        self._max_split_factor_step_size = max_split_factor_step_size
        self._with_dev_mappig = with_dev_mapping
        self._tune_da = tune_dev_alloc
        self._tune_ds = tune_dev_splits
        self._tuned = (
            not tune_dev_alloc or self._num_devs == 1
        ) and not tune_dev_splits
        self._first_da_tune_step = True
        self._first_ds_tune_step = True
        self._da_tune_count = 0
        self._unit_da_tune_count = 0
        self._ds_tune_count = 0
        if tune_dev_alloc:
            self._tune_step = self.da_tune_step
        elif tune_dev_splits:
            self._tune_step = self.ds_tune_step
        else:
            self._tune_step = None
        self._observed_configs = set()
        self._da_directions = dict()
        self._da_directions_flipped = dict()
        if isinstance(devices, dict):
            self._dev_da_ratios = devices
        else:
            self._dev_da_ratios = dict(
                zip(devices, [1 / self._num_devs] * self._num_devs)
            )
        self._devs_keys = self._dev_da_ratios.keys()
        self._percent_mem_inc_per_unit_da_dim = dict(
            zip(self._devs_keys, [0] * self._num_devs)
        )
        self._percent_mem_inc_per_sf = dict(zip(self._devs_keys, [0] * self._num_devs))
        self._percent_util_inc_per_unit_da_dim = dict(
            zip(self._devs_keys, [1] * self._num_devs)
        )
        self._delta_da_dim_sizes = dict(zip(self._devs_keys, [0] * self._num_devs))
        self._delta_sfs = dict(zip(self._devs_keys, [0] * self._num_devs))
        self._dev_percent_mems = None
        self._dev_utils = None
        if with_dev_mapping and ivy.exists(self._dim_size):
            self._compute_devs_da()
        self._devs_ds = {ds: starting_split_factor for ds in self._devs_keys}
        if self._tune_ds and not with_dev_mapping:
            [ivy.set_split_factor(starting_split_factor, ds) for ds in self._devs_keys]
        self._da_time = time.perf_counter()
        self._da_step_time = 0
        self._ds_time = time.perf_counter()
        self._ds_step_time = 0

    # Device Allocation #

    def _shift_da_splits(self, ordered_dev_util_keys, deltas):
        for i in range(math.floor(self._num_devs / 2)):

            # less and more utilized keys
            less_util_dev = ordered_dev_util_keys[i]
            more_util_dev = ordered_dev_util_keys[-i - 1]

            # less utilized
            delta = max(
                min(
                    deltas[less_util_dev],
                    self._devs_da[more_util_dev] - self._min_dev_dim_size,
                ),
                1,
            )
            if ivy.exists(self._max_dev_dim_step_size):
                delta = min(delta, self._max_dev_dim_step_size)
            self._devs_da[less_util_dev] += delta
            self._delta_da_dim_sizes[less_util_dev] = delta

            # more utilized
            self._devs_da[more_util_dev] -= delta
            self._delta_da_dim_sizes[more_util_dev] = -delta

    def _compute_devs_da(self):
        split_sizes = [
            int(round(r * self._dim_size)) for r in self._dev_da_ratios.values()
        ]
        combined_batch_size = sum(split_sizes)
        excess_size = combined_batch_size - self._dim_size
        if excess_size > 0:
            for i in range(abs(excess_size)):
                split_sizes[i] -= 1
        elif excess_size < 0:
            for i in range(abs(excess_size)):
                split_sizes[i] += 1
        self._devs_da = {k: v for k, v in zip(self._devs_keys, split_sizes)}

    def _compute_dev_da_ratios(self):
        self._dev_da_ratios = {k: v / self._dim_size for k, v in self._devs_da.items()}

    def da_tune_step(self, oom=False):
        """Summary.

        Parameters
        ----------
        oom
             (Default value = False)

        """
        if self._tuned:
            return
        new_dev_utils = dict(
            sorted(
                {k: dev_util(k) for k in self._devs_keys}.items(),
                key=lambda item: item[1],
            )
        )
        new_dev_utils_keys = list(new_dev_utils.keys())
        highest_util_dev = new_dev_utils_keys[-1]
        highest_util = new_dev_utils[highest_util_dev]
        if oom:
            new_dev_percent_mems = {k: 100 for k in self._devs_keys}
        else:
            new_dev_percent_mems = dict(
                sorted(
                    {k: percent_used_mem_on_dev(k) for k in self._devs_keys}.items(),
                    key=lambda item: item[1],
                )
            )

        # first step
        if self._first_da_tune_step:

            # log
            logging.info("tuning device allocation...")

            # shift the device splits by 1
            self._shift_da_splits(new_dev_utils_keys, {k: 1 for k in self._devs_keys})

            # update device percentage memory usages and utilizations
            self._dev_percent_mems = new_dev_percent_mems
            self._dev_utils = new_dev_utils

            # increment count, update ratios and tune step, and return
            self._da_tune_count += 1
            self._first_da_tune_step = False
            self._compute_dev_da_ratios()
            if self._tune_ds:
                self._tune_step = self.ds_tune_step
            self._da_time = time.perf_counter()
            return

        # otherwise

        # check if all directions have changed, and if so,
        # half the max dev dim step size
        if self._max_dev_dim_step_size > 1:
            da_directions = {
                k: 1 if i < math.floor(self._num_devs / 2) else -1
                for i, (k, v) in enumerate(new_dev_utils.items())
            }
            if len(self._da_directions) == 0:
                self._da_directions = da_directions
                self._da_directions_flipped = {k: False for k in self._devs_keys}
            else:
                self._da_directions_flipped = {
                    k: da_directions[k] * v < 0 for k, v in self._da_directions.items()
                }
            if sum(self._da_directions_flipped.values()) == self._num_devs:
                self._da_directions.clear()
                self._max_dev_dim_step_size = max(
                    int(round(self._max_dev_dim_step_size / 2)), 1
                )

        # percentage memory increase per unit dim
        delta_percent_mems = {
            k: new_dev_percent_mems[k] - self._dev_percent_mems[k]
            for k in self._devs_keys
        }
        self._percent_mem_inc_per_unit_da_dim = {
            k: (
                (
                    (self._da_tune_count - 1) * self._percent_mem_inc_per_unit_da_dim[k]
                    + (delta_percent_mems[k] / delta_dim_size)
                )
                / self._da_tune_count
            )
            if delta_dim_size != 0
            else self._percent_mem_inc_per_unit_da_dim[k]
            for k, delta_dim_size in self._delta_da_dim_sizes.items()
        }

        # percentage utility increase per unit dim
        delta_utils = {
            k: new_dev_utils[k] - self._dev_utils[k] for k in self._devs_keys
        }
        self._percent_util_inc_per_unit_da_dim = {
            k: max(
                (
                    (
                        (self._da_tune_count - 1)
                        * self._percent_util_inc_per_unit_da_dim[k]
                        + (delta_utils[k] / delta_dim_size)
                    )
                    / self._da_tune_count
                ),
                0.1,
            )
            if delta_dim_size != 0
            else self._percent_util_inc_per_unit_da_dim[k]
            for k, delta_dim_size in self._delta_da_dim_sizes.items()
        }

        # shift the device splits
        desired_percent_increases = {
            k: highest_util - new_dev_utils[k] for k in self._devs_keys
        }
        raw_deltas = {
            k: int(
                round(
                    desired_percent_increases[k]
                    / self._percent_util_inc_per_unit_da_dim[k]
                )
            )
            for k in self._devs_keys
        }
        permissable_steps = {
            k: min(
                math.floor(
                    (
                        (100 - new_dev_percent_mems[k])
                        / max(self._percent_mem_inc_per_unit_da_dim[k], 0.1)
                    )
                    / self._safety_factor
                ),
                self._dim_size,
            )
            for k in self._devs_keys
        }
        deltas = {
            k: min(v, pm)
            for (k, v), pm in zip(raw_deltas.items(), permissable_steps.values())
        }
        self._shift_da_splits(new_dev_utils_keys, deltas)

        # update device utilizations and percentage memory usages
        self._dev_utils = new_dev_utils
        self._dev_percent_mems = new_dev_percent_mems

        # increment count, update ratios and tune step
        self._compute_dev_da_ratios()
        self._da_tune_count += 1
        if self._tune_ds:
            self._tune_step = self.ds_tune_step

        # if step size is 1, check if tuning is complete, and return if so
        if self._max_dev_dim_step_size == 1:

            # check if da tuning is complete
            if (
                self.repeated_config_check()
                and self._unit_da_tune_count >= self._min_unit_dev_tune_steps
                and not self._tune_ds
                or (self._ds_tune_count >= self._min_sf_tune_steps)
            ):
                self._observed_configs.clear()
                self._percent_mem_inc_per_unit_da_dim.clear()
                self._delta_da_dim_sizes.clear()
                self._dev_percent_mems.clear()
                logging.info("device allocation tuning complete!")
                self._tuned = True

            self._unit_da_tune_count += 1

        # log time
        now = time.perf_counter()
        self._da_step_time = now - self._da_time
        self._da_time = now
        if self._tuned:
            return
        logging.info(
            "new allocation sizes {}, still tuning...".format(
                str(["{:.2f}".format(v) for v in self._devs_da.values()])
            )
        )

    # Device Splitting #

    def _shift_ds(self, deltas):
        for ds, delta in deltas.items():
            clipped_delta = min(delta, self._max_split_factor_step_size)
            self._devs_ds[ds] = min(self._devs_ds[ds] + clipped_delta, 1)
            self._delta_sfs[ds] = clipped_delta
            if not self._with_dev_mappig:
                ivy.set_split_factor(min(self._devs_ds[ds] + clipped_delta, 1), ds)

    def ds_tune_step(self, oom=False):
        """Summary.

        Parameters
        ----------
        oom
             (Default value = False)

        """
        if self._tuned:
            return
        if oom:
            new_dev_percent_mems = {k: 100 for k in self._devs_keys}
        else:
            new_dev_percent_mems = dict(
                sorted(
                    {k: percent_used_mem_on_dev(k) for k in self._devs_keys}.items(),
                    key=lambda item: item[1],
                )
            )

        # first step
        if self._first_ds_tune_step:

            # log
            logging.info("tuning device splitting...")

            # shift the device splits by 1%
            self._shift_ds({k: 0.01 for k in self._devs_keys})

            # update device percentage memory usages and utilizations
            self._dev_percent_mems = new_dev_percent_mems

            # increment count, update ratios and tune step, and return
            self._ds_tune_count += 1
            self._first_ds_tune_step = False
            if self._tune_da:
                self._tune_step = self.da_tune_step
            self._ds_time = time.perf_counter()
            return

        # otherwise

        # percentage memory increase per unit dim
        delta_percent_mems = {
            k: new_dev_percent_mems[k] - self._dev_percent_mems[k]
            for k in self._devs_keys
        }
        self._percent_mem_inc_per_sf = {
            k: (
                (
                    (self._ds_tune_count - 1) * self._percent_mem_inc_per_sf[k]
                    + (delta_percent_mems[k] / delta_sf)
                )
                / self._ds_tune_count
            )
            if delta_sf != 0
            else self._percent_mem_inc_per_sf[k]
            for k, delta_sf in self._delta_sfs.items()
        }

        # shift the device splits
        deltas = {
            k: min(
                (max(100 / self._safety_factor - new_dev_percent_mems[k], 0))
                / max(self._percent_mem_inc_per_sf[k], 1),
                self._max_split_factor_step_size,
            )
            for k in self._devs_keys
        }
        self._shift_ds(deltas)

        # update device percentage memory usages
        self._dev_percent_mems = new_dev_percent_mems

        # increment count, update ratios and tune step
        self._ds_tune_count += 1
        if self._tune_da:
            self._tune_step = self.da_tune_step

        # check whether device allocation tuning is ready to terminate
        da_can_terminate = not self._tune_da or self._max_dev_dim_step_size == 1

        # check if ds tuning is complete
        if (
            da_can_terminate
            and self.repeated_config_check()
            and self._ds_tune_count >= self._min_sf_tune_steps
            and not self._tune_da
            or (self._unit_da_tune_count >= self._min_unit_dev_tune_steps)
        ):
            self._observed_configs.clear()
            self._percent_mem_inc_per_sf.clear()
            self._dev_percent_mems.clear()
            logging.info("device splitting tuning complete!")
            self._tuned = True

        # log time
        now = time.perf_counter()
        self._ds_step_time = now - self._ds_time
        self._ds_time = now
        if self._tuned:
            return
        logging.info(
            "new split factors {}, still tuning...".format(
                str(["{:.2f}".format(ivy.split_factor(k)) for k in self._devs_keys])
            )
        )

    # Repeated Config Checking #

    def repeated_config_check(self):
        """"""

        # check if ds tuning is complete, and return if so
        config_list = list()
        if self._tune_da:
            config_list += list(self._devs_da.values())
        if self._tune_ds:
            config_list += [self._devs_ds[ds] for ds in self._devs_keys]
        config = tuple(config_list)
        if config in self._observed_configs:
            return True

        # otherwise add the current config to those observed
        self._observed_configs.add(config)

        return False

    # Mapping #

    def map(self, cloned=None, to_clone=None, distributed=None, to_distribute=None):
        """Map the function fn to each of the MultiDevice args and kwargs, running each
        function in parallel with CUDA-safe multiprocessing.

        Parameters
        ----------
        cloned
            The MutliDevice keyword arguments which are already cloned. Default is None.
        to_clone
            The MutliDevice keyword arguments to clone and map to the function.
            Default is None.
        distributed
            The MutliDevice keyword arguments which already distributed.
            Default is None.
        to_distribute
            The MutliDevice keyword arguments to distribute and map to the function.
            Default is None.

        Returns
        -------
        ret
            The results of the function, returned as a MultiDevice instance.

        """
        used_devs_dict = {k: v for k, v in self._devs_da.items() if v > 0}
        used_devs = list(used_devs_dict.keys())
        cloned = ivy.default(cloned, {})
        if ivy.exists(to_clone):
            to_clone = {k: ivy.dev_clone(v, used_devs) for k, v in to_clone.items()}
        else:
            to_clone = {}
        distributed = ivy.default(distributed, {})
        if ivy.exists(to_distribute):
            to_distribute = {
                k: ivy.dev_dist(v, used_devs_dict) for k, v in to_distribute.items()
            }
        else:
            to_distribute = {}
        if self._tune_ds:
            ret = self._dev_mapper.map(
                **cloned,
                **to_clone,
                **distributed,
                **to_distribute,
                used_devs=used_devs,
                split_factors=self._devs_ds
            )
        else:
            ret = self._dev_mapper.map(
                **cloned,
                **to_clone,
                **distributed,
                **to_distribute,
                used_devs=used_devs
            )
        if self._tuned:
            return ret
        self._tune_step()
        return ret

    def __del__(self):
        if ivy.exists(self._dev_mapper):
            self._dev_mapper.__del__()
            del self._dev_mapper

    @property
    def dim_size(self):
        """"""
        return self._dim_size

    @dim_size.setter
    def dim_size(self, batch_size):
        """Summary.

        Parameters
        ----------
        batch_size

        """
        self._dim_size = batch_size
        if self._tune_da:
            self._max_dev_dim_step_size = max(
                int(round(self._max_dev_dim_step_ratio * self._dim_size)), 1
            )
            self._compute_devs_da()

    @property
    def tune_step(self):
        """"""
        return self._tune_step

    @property
    def tuned(self):
        """"""
        return self._tuned


# Profiler #


class Profiler(abc.ABC):
    """"""

    def __init__(self, save_dir):
        self._save_dir = save_dir

    @abc.abstractmethod
    def start(self):
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


# Function Helper #
# ----------------#

# noinspection PyShadowingNames
def _handle_device(dtype: Optional[ivy.Dtype] = None, arr=None):
    return ivy.dev_from_str(ivy.default_device(dtype, item=arr))
