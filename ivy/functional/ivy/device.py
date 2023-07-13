"""Collection of device Ivy functions."""

# global
import os
import gc
import abc
import math
import psutil
import warnings
import types
from typing import Type, Optional, Tuple

# noinspection PyUnresolvedReferences
try:
    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        pass
except ImportError:
    warnings.warn(
        "pynvml installation was not found in the environment, functionalities"
        " of the Ivy's device module will be limited. Please install pynvml if"
        " you wish to use GPUs with Ivy."
    )
    # nvidia-ml-py (pynvml) is not installed in CPU Dockerfile.

from typing import Union, Callable, Iterable, Any

# local
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions

default_device_stack = list()
soft_device_mode_stack = list()
dev_handles = dict()
split_factors = dict()
max_chunk_sizes = dict()


# Extra #
# ------#


class DefaultDevice:
    """Ivy Device Class."""

    def __init__(
        self,
        device: Union[ivy.Device, ivy.NativeDevice],
        /,
    ) -> None:
        """
        Initialize the DefaultDevice class.

        Parameters
        ----------
        device
            The device string - as an ivy device or nativedevice class

        Examples
        --------
        A "tpu" as device:

        >>> x = ivy.DefaultDevice("tpu")
        """
        self._dev = device

    def __enter__(self):
        """
        Enter the runtime context related to the specified device.

        Returns
        -------
        ret
            Self, an instance of the same class.

        Examples
        --------
        A "cpu" as device:

        >>> with ivy.DefaultDevice("cpu") as device:
        >>>     # with block calls device.__enter__()
        >>>     print(device._dev)
        "cpu"
        """
        ivy.set_default_device(self._dev)
        ivy.set_soft_device_mode(True)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[Type[BaseException]],
        exc_tb: Optional[types.TracebackType],
    ) -> Union[ivy.Device, str]:
        """
        Exit the runtime context related to the specified device.

        Parameters
        ----------
        exc_type
            The type of the exception that was raised.
        exc_val
            The exception that was raised.
        exc_tb
            The traceback of the exception that was raised.

        Returns
        -------
        ret
            If no exception was raised, returns an instance of the same class.

        Examples
        --------
        A "gpu" as device:
        >>> with ivy.DefaultDevice("gpu") as device:
        >>>     pass
        >>> # after with block device.__exit__() is called
        >>> print(device._dev)
        "cpu"
        """
        ivy.unset_default_device()
        ivy.unset_soft_device_mode()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


def handle_soft_device_variable(*args, fn, **kwargs):
    ivy.set_array_mode(False)
    default_device = ivy.default_device()
    args, kwargs = ivy.nested_map(
        [args, kwargs],
        lambda x: (
            ivy.to_device(x, default_device)
            if (ivy.is_native_array(x) and ivy.dev(x) != default_device)
            else x
        ),
    )
    ivy.unset_array_mode()
    return fn(*args, **kwargs)


# Helpers #


def _get_nvml_gpu_handle(device: Union[ivy.Device, ivy.NativeDevice], /) -> int:
    global dev_handles
    if device in dev_handles:
        return dev_handles[device]
    gpu_idx = int(device.split(":")[-1])
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    dev_handles[device] = handle
    return handle


# Device Queries #

# Array Printing


@handle_exceptions
def get_all_ivy_arrays_on_dev(
    device: Union[ivy.Device, ivy.NativeDevice],
    /,
) -> ivy.Container:
    """
    Get all ivy arrays which are currently alive on the specified device.

    Parameters
    ----------
    device
        The device handle from which to get the arrays

    Returns
    -------
    ret
        Container with the arrays found for the specified device [identity, array]

    Examples
    --------
    >>> x = ivy.array([1,0,2])
    >>> y = ivy.dev(x)
    >>> z = ivy.get_all_ivy_arrays_on_dev(y)
    >>> print(z)
    {139740789224448:ivy.array([1,0,2])},
    """
    device = ivy.as_ivy_dev(device)
    all_arrays = list()
    for obj in gc.get_objects():
        if (
            type(obj) == ivy.data_classes.array.array.Array
            and ivy.is_ivy_array(obj)
            and ivy.dev(obj) == device
        ):
            all_arrays.append(obj)

    return ivy.Container(dict(zip([str(id(a)) for a in all_arrays], all_arrays)))


@handle_exceptions
def num_ivy_arrays_on_dev(device: Union[ivy.Device, ivy.NativeDevice], /) -> int:
    """
    Return the number of arrays which are currently alive on the specified device.

    Parameters
    ----------
    device
        The device handle from which to count the arrays

    Returns
    -------
    ret
        Number of arrays on the specified device

    Examples
    --------
    >>> x1 = ivy.array([-1, 0, 5.2])
    >>> x2 = ivy.array([-1, 0, 5.2, 4, 5])
    >>> y = ivy.num_ivy_arrays_on_dev(ivy.default_device())
    >>> print(y)
    2

    >>> x1 = ivy.native_array([-1, 0, 5.2])
    >>> y = ivy.num_ivy_arrays_on_dev(ivy.default_device())
    >>> print(y)
    0

    >>> x = ivy.Container(x1=ivy.array([-1]),
    ...                   x2=ivy.native_array([-1]))
    >>> y = ivy.num_ivy_arrays_on_dev(ivy.default_device())
    >>> print(y)
    1
    """
    return len(ivy.get_all_ivy_arrays_on_dev(device))


@handle_exceptions
@handle_nestable
def print_all_ivy_arrays_on_dev(
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    attr_only: bool = True,
) -> None:
    """
    Print the shape and dtype for all ivy arrays which are currently alive on the
    specified device.

    Parameters
    ----------
    device
        The device on which to print the arrays

    attr_only
        Whether or not to only print the `shape` and `dtype` attributes of the array

    Examples
    --------
    >>> x = ivy.array([[1,0,2], [3,2,1]])
    >>> y = ivy.dev(x)
    >>> ivy.print_all_ivy_arrays_on_dev(y)
    ((3,), 'int32')
    ((3,), 'int32')


    >>> x = ivy.array([[1,0,2], [3,2,1]])
    >>> y = ivy.dev(x)
    >>> ivy.print_all_ivy_arrays_on_dev(y, attr_only = False)
    [1,0,2]
    [3,2,1]
    """
    arrs = ivy.get_all_ivy_arrays_on_dev(device).values()
    if attr_only:
        [print((arr.shape, arr.dtype)) for arr in arrs]
    else:
        [print(arr) for arr in arrs]


ivy.soft_device_mode = False


@handle_exceptions
def set_soft_device_mode(mode: bool) -> None:
    """
    Set the mode of whether to move input arrays to `ivy.default_device()` before
    performing an operation.

    Parameter
    ---------
    mode
        boolean whether to move input arrays
    Examples
    --------
    >>> ivy.set_soft_device_mode(False)
    >>> ivy.soft_device_mode
    False
    >>> ivy.set_soft_device_mode(True)
    >>> ivy.soft_device_mode
    True
    """
    global soft_device_mode_stack
    ivy.utils.assertions.check_isinstance(mode, bool)
    soft_device_mode_stack.append(mode)
    ivy.__setattr__("soft_device_mode", mode, True)


@handle_exceptions
def unset_soft_device_mode() -> None:
    """
    Reset the mode of moving input arrays to `ivy.default_device()` before performing an
    operation.

    Examples
    --------
    >>> ivy.set_soft_device_mode(False)
    >>> ivy.soft_device_mode
    False
    >>> ivy.unset_soft_device_mode()
    >>> ivy.soft_device_mode
    True
    """
    global soft_device_mode_stack
    if soft_device_mode_stack:
        soft_device_mode_stack.pop(-1)
        mode = soft_device_mode_stack[-1] if soft_device_mode_stack else False
        ivy.__setattr__("soft_device_mode", mode, True)


# Retrieval


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
def dev(
    x: Union[ivy.Array, ivy.NativeArray], /, *, as_native: bool = False
) -> Union[ivy.Device, ivy.NativeDevice]:
    """
    Get the native device handle for input array x.

    Parameters
    ----------
        x
            array for which to get the device handle.
        as_native
            Whether or not to return the dev in native format. Default is ``False``.

    Returns
    -------
        ret
            Device handle for the array.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([3, 1, 4, 5])
    >>> y = ivy.dev(x)
    >>> print(y)
    cpu

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[2, 5, 4], [3, 1, 5]])
    >>> y = ivy.dev(x, as_native=True)
    >>> print(y)
    cpu
    """
    return ivy.current_backend(x).dev(x, as_native=as_native)


# Conversions


@handle_exceptions
def as_ivy_dev(device: Union[ivy.Device, str], /) -> ivy.Device:
    """
    Convert device to string representation.

    Parameters
    ----------
    device
        The device handle to convert to string.

    Returns
    -------
    ret
        Device string e.g. 'cuda:0'.

    Examples
    --------
    >>> y = ivy.as_ivy_dev('cuda:0')
    >>> print(y)
    cuda:0
    """
    return ivy.current_backend().as_ivy_dev(device)


@handle_exceptions
def as_native_dev(device: Union[ivy.Device, ivy.NativeDevice], /) -> ivy.NativeDevice:
    """
    Convert device string representation to native device type.

    Parameters
    ----------
    device
        The device string to convert to native device handle.
        A native device handle can be passed in instead - in this case
        the unmodified parameter is returned.

    Returns
    -------
    ret
        Native device handle.

    Examples
    --------
    With :class:`ivy.Device` input:

    >>> ivy.set_backend("numpy")
    >>> ivy.as_native_dev("cpu")
    'cpu'

    >>> ivy.set_backend("tensorflow")
    >>> ivy.as_native_dev("tpu:3")
    '/TPU:3'

    With :class:`ivy.NativeDevice` input:

    >>> import torch
    >>> device = torch.device("cuda")
    >>> device
    device(type='cuda')

    >>> ivy.as_native_dev(device)
    device(type='cuda')
    """
    return ivy.current_backend().as_native_dev(device)


# Memory


@handle_exceptions
def clear_cached_mem_on_dev(device: Union[ivy.Device, ivy.NativeDevice], /) -> None:
    """
    Clear memory cache on target device.

    Parameters
    ----------
    device
        The device string to convert to native device handle or native device handle.

    Examples
    --------
    >>> import torch
    >>> ivy.set_backend("torch")
    >>> device = torch.device("cuda")
    >>> ivy.clear_cached_mem_on_dev(device)
    """
    ivy.current_backend().clear_cached_mem_on_dev(device)


@handle_exceptions
def total_mem_on_dev(device: Union[ivy.Device, ivy.NativeDevice], /) -> float:
    """
    Get the total amount of memory (in GB) for a given device string. In case of CPU,
    the total RAM is returned.

    Parameters
    ----------
    device
        The device string to convert to native device handle.

    Returns
    -------
    ret
        The total memory on the device in GB.

    Examples
    --------
    >>> x = ivy.total_mem_on_dev("cpu")
    >>> print(x)
    53.66700032

    >>> x = ivy.total_mem_on_dev("gpu:0")
    >>> print(x)
    8.589934592
    """
    if "gpu" in device:
        handle = _get_nvml_gpu_handle(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total / 1e9
    elif device == "cpu":
        return psutil.virtual_memory().total / 1e9
    else:
        raise ivy.utils.exceptions.IvyException(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


@handle_exceptions
def used_mem_on_dev(
    device: Union[ivy.Device, ivy.NativeDevice],
    /,
    *,
    process_specific: bool = False,
) -> float:
    """
    Get the used memory (in GB) for a given device string. In case of CPU, the used RAM
    is returned.

    Parameters
    ----------
    device
        The device string to convert to native device handle.
    process_specific
        Whether to check the memory used by this python process alone. Default is
        False.

    Returns
    -------
    ret
        The used memory on the device in GB.

    Examples
    --------
    >>> x = ivy.used_mem_on_dev("cpu", process_specific = False)
    >>> print(x)
    6.219563008

    >>> x = ivy.used_mem_on_dev("cpu", process_specific = True)
    >>> print(x)
    0.902400346

    >>> y = ivy.used_mem_on_dev("gpu:0", process_specific = False)
    >>> print(y)
    0.525205504
    """
    ivy.clear_cached_mem_on_dev(device)
    if "gpu" in device:
        handle = _get_nvml_gpu_handle(device)
        if process_specific:
            pid = os.getpid()
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == pid:
                    return process.usedGpuMemory / 1e9
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1e9
    elif device == "cpu":
        if process_specific:
            return psutil.Process(os.getpid()).memory_info().rss / 1e9
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 1e9
    else:
        raise ivy.utils.exceptions.IvyException(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


@handle_exceptions
def percent_used_mem_on_dev(
    device: Union[ivy.Device, ivy.NativeDevice],
    /,
    *,
    process_specific: bool = False,
) -> float:
    """
    Get the percentage used memory for a given device string. In case of CPU, the used
    RAM is returned.

    Parameters
    ----------
    device
        The device string to convert to native device handle.
    process_specific
        Whether the check the memory used by this python process alone. Default is
        False.

    Returns
    -------
    ret
        The percentage used memory on the device.

    Examples
    --------
    >>> x = ivy.percent_used_mem_on_dev("cpu", process_specific = False)
    >>> print(x)
    94.036902561555

    >>> x = ivy.percent_used_mem_on_dev("cpu", process_specific = True)
    >>> print(x)
    0.7024003467681645

    >>> x = ivy.as_native_dev("gpu:0")
    >>> y = ivy.percent_used_mem_on_dev(x, process_specific = False)
    >>> print(y)
    0.7095597456708771
    """
    ivy.clear_cached_mem_on_dev(device)
    if "gpu" in device:
        handle = _get_nvml_gpu_handle(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if process_specific:
            pid = os.getpid()
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == pid:
                    return (process.usedGpuMemory / info.total) * 100
        return (info.used / info.total) * 100
    elif device == "cpu":
        vm = psutil.virtual_memory()
        if process_specific:
            return (psutil.Process(os.getpid()).memory_info().rss / vm.total) * 100
        return (1 - (vm.available / vm.total)) * 100
    else:
        raise ivy.utils.exceptions.IvyException(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


# Utilization


@handle_exceptions
def dev_util(device: Union[ivy.Device, ivy.NativeDevice], /) -> float:
    """
    Get the current utilization (%) for a given device.

    Parameters
    ----------
    device
        The device string of the device to query utilization for.

    Returns
    -------
    ret
        The device utilization (%)

    Example
    -------
    >>> ivy.dev_util('cpu')
    13.4
    >>> ivy.dev_util('gpu:0')
    7.8
    >>> ivy.dev_util('cpu')
    93.4
    >>> ivy.dev_util('gpu:2')
    57.4
    >>> ivy.dev_util('cpu')
    84.2
    """
    if device == "cpu":
        return psutil.cpu_percent()
    elif "gpu" in device:
        handle = _get_nvml_gpu_handle(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        raise ivy.utils.exceptions.IvyException(
            'Invalid device string input, must be on the form "gpu:idx" or "cpu", '
            "but found {}".format(device)
        )


# Availability


@handle_exceptions
def gpu_is_available() -> bool:
    """
    Determine whether a GPU is available to use, with the backend framework.

    Returns
    -------
    ret
        Boolean, as to whether a gpu is available.

    Examples
    --------
    >>> print(ivy.gpu_is_available())
    False
    """
    return ivy.current_backend().gpu_is_available()


@handle_exceptions
def num_cpu_cores(*, logical: bool = True) -> int:
    """
    Determine the number of cores available in the cpu.

    Parameters
    ----------
    logical
        Whether request is for number of physical or logical cores available in CPU

    Returns
    -------
    ret
        Number of cores available in CPU

    Examples
    --------
    >>> print(ivy.num_cpu_cores(logical=False))
    2
    """
    if logical:
        return psutil.cpu_count(logical=logical)
    else:
        return psutil.cpu_count(logical=False)


@handle_exceptions
def num_gpus() -> int:
    """
    Determine the number of available GPUs, with the backend framework.

    Returns
    -------
    ret
        Number of available GPUs.

    Examples
    --------
    >>> print(ivy.num_gpus())
    1
    """
    return ivy.current_backend().num_gpus()


@handle_exceptions
def tpu_is_available() -> bool:
    """
    Determine whether a TPU is available to use, with the backend framework.

    Returns
    -------
    ret
        Boolean, as to whether a tpu is available.

    Examples
    --------
    >>> print(ivy.tpu_is_available())
    False
    """
    return ivy.current_backend().tpu_is_available()


# Default Device #


# noinspection PyShadowingNames
@handle_exceptions
def default_device(
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    /,
    *,
    item: Optional[Union[list, tuple, dict, ivy.Array, ivy.NativeArray]] = None,
    as_native: bool = None,
) -> Union[ivy.Device, ivy.NativeDevice]:
    """
    Return the input device or the default device. If the as_native flag is set, the
    device will be converted to a native device. If the item is provided, the item's
    device is returned. If the device is not provided, the last default device is
    returned. If a default device has not been set, the first gpu is returned if
    available, otherwise the cpu is returned.

    Parameters
    ----------
    device
        The device to be returned or converted.
    item
        The item to get the device from.
    as_native
        Whether to convert the device to a native device.

    Returns
    -------
    ret
        Device handle or string.

    Examples
    --------
    >>> ivy.default_device()
    device(type='cpu')

    >>> ivy.default_device("gpu:0")
    'gpu:0'

    >>> ivy.default_device(item=[], as_native=False)
    'cpu'

    >>> ivy.default_device(item=(), as_native=True)
    device(type='cpu')

    >>> ivy.default_device(item={"a": 1}, as_native=True)
    device(type='cpu')

    >>> x = ivy.array([1., 2., 3.])
    >>> x = ivy.to_device(x, 'gpu:0')
    >>> ivy.default_device(item=x, as_native=True)
    device(type='gpu', id=0)
    """
    if ivy.exists(device):
        if as_native is True:
            return ivy.as_native_dev(device)
        elif as_native is False:
            return ivy.as_ivy_dev(device)
        return device
    as_native = ivy.default(as_native, False)
    if ivy.exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_array(item):
            return ivy.dev(item, as_native=as_native)
    global default_device_stack
    if not default_device_stack:
        ret = "gpu:0" if ivy.gpu_is_available() else "cpu"
    else:
        ret = default_device_stack[-1]
    if as_native:
        return ivy.as_native_dev(ret)
    return ivy.as_ivy_dev(ret)


@handle_exceptions
def set_default_device(device: Union[ivy.Device, ivy.NativeDevice], /) -> None:
    """
    Set the default device to given device instance.

    Parameters
    ----------
    device
        The device to set as the default device

    Examples
    --------
    >>> ivy.set_default_device("cpu")
    >>> ivy.default_device()
    'cpu'

    >>> ivy.set_backend("torch")
    >>> ivy.set_default_device("gpu:0")
    >>> ivy.default_device(as_native=True)
    device(type='cuda', index=0)

    >>> import torch
    >>> ivy.set_backend("torch")
    >>> device = torch.device("cuda")
    >>> ivy.set_default_device(device)
    >>> ivy.default_device(as_native=True)
    device(type='cuda')
    """
    global default_device_stack
    default_device_stack.append(device)


@handle_exceptions
def unset_default_device() -> None:
    """
    Reset the default device to "cpu".

    Examples
    --------
    >>> ivy.set_default_device("gpu:0")
    >>> ivy.default_device()
    "gpu:0"
    >>> ivy.unset_default_device()
    >>> ivy.default_device()
    "cpu"
    """
    global default_device_stack
    if default_device_stack:
        default_device_stack.pop(-1)


# Device Allocation #


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def to_device(
    x: Union[ivy.Array, ivy.NativeArray],
    device: Union[ivy.Device, ivy.NativeDevice],
    /,
    *,
    stream: Optional[Union[int, Any]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Move the input array x to the desired device, specified by device string.

    Parameters
    ----------
    x
        input array to be moved to the desired device
    device
        device to move the input array `x` to
    stream
        stream object to use during copy. In addition to the types supported in
        array.__dlpack__(), implementations may choose to support any library-specific
        stream object with the caveat that any code using such an object would not be
        portable.
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
    >>> x = ivy.to_device(x, 'cpu')
    >>> print(x.device)
    cpu
    """
    return ivy.current_backend(x).to_device(x, device, stream=stream, out=out)


# Function Splitting #


@handle_exceptions
def split_factor(
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    /,
) -> float:
    """
    Get a device's global split factor, which can be used to scale the device's batch
    splitting chunk sizes across the codebase.

    If the global split factor is set for a given device,
        returns the split factor value for the device from the split factors dictionary
    If the global split factor for a device is not configured,
        returns the default value which is 0.0

    Parameters
    ----------
    device
        The device to query the split factor for. Sets the default device by default.

    Returns
    -------
    ret
        The split factor for the specified device.

    Examples
    --------
    >>> x = ivy.split_factor()
    >>> print(x)
    0.0

    >>> y = ivy.split_factor("gpu:0")
    >>> print(y)
    0.0
    """
    global split_factors
    device = ivy.default(device, default_device())
    return split_factors.setdefault(device, 0.0)


@handle_exceptions
def set_split_factor(
    factor: float, /, *, device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None
) -> None:
    """
    Set the global split factor for a given device, which can be used to scale batch
    splitting chunk sizes for the device across the codebase.

    Parameters
    ----------
    factor
        The factor to set the device-specific split factor to.
    device
        The device to set the split factor for. Sets the default device by default.

    Examples
    --------
    >>> print(ivy.default_device())
    cpu
    >>> ivy.set_split_factor(0.5)
    >>> print(ivy.split_factors)
    {'cpu': 0.5}

    >>> import torch
    >>> ivy.set_backend("torch")
    >>> device = torch.device("cuda")
    >>> ivy.set_split_factor(0.3,device)
    >>> print(ivy.split_factors)
    {device(type='cuda'): 0.3}

    >>> ivy.set_split_factor(0.4,"tpu")
    >>> print(ivy.split_factors)
    {'tpu': 0.4}

    >>> import torch
    >>> ivy.set_backend("torch")
    >>> device = torch.device("cuda")
    >>> ivy.set_split_factor(0.2)
    >>> ivy.set_split_factor(0.3, device='gpu')
    >>> print(ivy.split_factors)
    {'cpu': 0.2, 'gpu': 0.3}
    """
    ivy.utils.assertions.check_less(0, factor, allow_equal=True, as_array=False)
    global split_factors
    device = ivy.default(device, default_device())
    split_factors[device] = factor


@handle_exceptions
def split_func_call(
    func: Callable,
    inputs: Union[ivy.Array, ivy.NativeArray],
    mode: str,
    /,
    *,
    max_chunk_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    input_axes: Union[int, Iterable[int]] = 0,
    output_axes: Optional[Union[int, Iterable[int]]] = None,
    stop_gradients: bool = False,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Call a function by splitting its inputs along a given axis, and calling the function
    in chunks, rather than feeding the entire input array at once. This can be useful to
    reduce memory usage of the device the arrays are on.

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
        overwrites the global split factor. Default is ``None``.
    input_axes
        The axes along which to split each of the inputs, before passing to the
        function. Default is ``0``.
    output_axes
        The axes along which to concat each of the returned outputs. Default is same as
        fist input axis.
    stop_gradients
        Whether to stop the gradients for each computed return. Default is ``False``.
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
        max_dim = max(
            [inp.cont_shape[inp_ax] for inp, inp_ax in zip(inputs, input_axes)]
        )
        if max_dim > max_chunk_size:
            max_chunk_sizes[shape_key] = max_dim
            max_chunk_size = max_dim
    chunk_size = ivy.default(
        chunk_size,
        default_val=lambda: 1
        + int(
            round((max_chunk_size - 1) * ivy.split_factor(ivy.default_device(device)))
        ),
        with_callable=True,
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
        (
            ivy.split(
                inp,
                num_or_size_splits=chunk_sizes,
                axis=input_axes[i],
                with_remainder=True,
            )
            if ivy.is_array(inp)
            else inp.split(
                num_or_size_splits=chunk_sizes, axis=input_axes[i], with_remainder=True
            )
        )
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
    ret = [
        ivy.concat([r[i] for r in rets], axis=output_axes[i])
        for i in range(num_outputs)
    ]
    return ret[0] if len(ret) == 1 else ret


def _is_valid_devices_attributes(fn: Callable) -> bool:
    if hasattr(fn, "supported_devices") and hasattr(fn, "unsupported_devices"):
        fn_supported_devices = fn.supported_devices
        fn_unsupported_devices = fn.unsupported_devices
        if isinstance(fn_supported_devices, dict):
            if isinstance(fn_unsupported_devices, dict):
                backend_str = ivy.current_backend_str()
                if (
                    backend_str in fn_supported_devices
                    and backend_str in fn_unsupported_devices
                ):
                    return False
        else:
            if isinstance(fn_unsupported_devices, tuple):
                return False
    return True


def _get_devices(fn: Callable, complement: bool = True) -> Tuple:
    valid_devices = ivy.valid_devices
    invalid_devices = ivy.invalid_devices
    all_devices = ivy.all_devices

    supported = set(ivy.valid_devices)

    is_backend_fn = "backend" in fn.__module__
    is_frontend_fn = "frontend" in fn.__module__
    is_einops_fn = "einops" in fn.__name__
    if not is_backend_fn and not is_frontend_fn and not is_einops_fn:
        if complement:
            supported = set(all_devices).difference(supported)
        return supported

    # Their values are formated like either
    # 1. fn.supported_devices = ("cpu",)
    # Could also have the "all" value for the framework
    basic = [
        ("supported_devices", set.intersection, valid_devices),
        ("unsupported_devices", set.difference, invalid_devices),
    ]
    for key, merge_fn, base in basic:
        if hasattr(fn, key):
            v = getattr(fn, key)
            if "einops" in fn.__name__ and isinstance(v, dict):
                v = v.get(ivy.current_backend_str(), base)
            ivy.utils.assertions.check_isinstance(v, tuple)
            supported = merge_fn(supported, set(v))

    if complement:
        supported = set(all_devices).difference(supported)

    return tuple(supported)


@handle_exceptions
@handle_nestable
def function_supported_devices(
    fn: Callable, recurse: bool = True
) -> Union[Tuple, dict]:
    """
    Return the supported devices of the current backend's function. The function returns
    a dict containing the supported devices for the compositional and primary
    implementations in case of partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the supported device attribute
    recurse
        Whether to recurse into used ivy functions. Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the supported devices of the function

    Examples
    --------
    >>> import ivy
    >>> print(ivy.function_supported_devices(ivy.ones))
    ('cpu', 'gpu')
    """
    ivy.utils.assertions.check_true(
        _is_valid_devices_attributes(fn),
        (
            "supported_devices and unsupported_devices attributes cannot both "
            "exist in a particular backend"
        ),
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_supported_devices(fn.compos, recurse=recurse),
            "primary": _get_devices(fn, complement=False),
        }
    else:
        supported_devices = set(_get_devices(fn, complement=False))
        if recurse:
            supported_devices = ivy.functional.data_type._nested_get(
                fn, supported_devices, set.intersection, function_supported_devices
            )

    return (
        supported_devices
        if isinstance(supported_devices, dict)
        else tuple(supported_devices)
    )


@handle_exceptions
@handle_nestable
def function_unsupported_devices(
    fn: Callable, recurse: bool = True
) -> Union[Tuple, dict]:
    """
    Return the unsupported devices of the current backend's function. The function
    returns a dict containing the unsupported devices for the compositional and primary
    implementations in case of partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the unsupported device attribute
    recurse
        Whether to recurse into used ivy functions. Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the unsupported devices of the function

    Examples
    --------
    >>> import ivy
    >>> print(ivy.function_unsupported_devices(ivy.ones))
    ()
    """
    ivy.utils.assertions.check_true(
        _is_valid_devices_attributes(fn),
        (
            "supported_devices and unsupported_devices attributes cannot both "
            "exist in a particular backend"
        ),
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_unsupported_devices(fn.compos, recurse=recurse),
            "primary": _get_devices(fn, complement=True),
        }
    else:
        unsupported_devices = set(_get_devices(fn, complement=True))
        if recurse:
            unsupported_devices = ivy.functional.data_type._nested_get(
                fn, unsupported_devices, set.union, function_unsupported_devices
            )
    return (
        unsupported_devices
        if isinstance(unsupported_devices, dict)
        else tuple(unsupported_devices)
    )


# Profiler #


class Profiler(abc.ABC):
    """
    The profiler class is used to profile the execution of some code.

    Parameters
    ----------
    save_dir
        The directory to save the profile data to.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir

    @abc.abstractmethod
    def start(self):
        """
        Start the profiler.

        This should be called before the code to be profiled.
        """
        raise ivy.utils.exceptions.IvyNotImplementedException

    @abc.abstractmethod
    def stop(self):
        """
        Stop the profiler.

        This should be called after the code to be profiled.
        """
        raise ivy.utils.exceptions.IvyNotImplementedException

    @abc.abstractmethod
    def __enter__(self):
        raise ivy.utils.exceptions.IvyNotImplementedException

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise ivy.utils.exceptions.IvyNotImplementedException
