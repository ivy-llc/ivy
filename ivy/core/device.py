"""
Collection of device Ivy functions.
"""

# global
import os
import abc
import math
import queue
import psutil
import inspect
import nvidia_smi

# noinspection PyUnresolvedReferences
try:
    nvidia_smi.nvmlInit()
except nvidia_smi.NVMLError_LibraryNotFound:
    pass
from typing import Union, Type, Callable, Iterable, Dict

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

DEFAULT_DEVICE = None
DEV_HANDLES = dict()
SPLIT_FACTORS = dict()


# Helpers #
# --------#

# noinspection PyShadowingNames
def _get_nvml_gpu_handle(dev_str):
    global DEV_HANDLES
    if dev_str in DEV_HANDLES:
        return DEV_HANDLES[dev_str]
    gpu_idx = int(dev_str.split(':')[-1])
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)
    DEV_HANDLES[dev_str] = handle
    return handle


# Device Queries #
# ---------------#

# Retreival #

def dev(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> ivy.Device:
    """
    Get the native device handle for input array x.

    :param x: Tensor for which to get the device handle.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device handle for the array, in native framework format.
    """
    return _cur_framework(x, f=f).dev(x)


def dev_str(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> str:
    """
    Get the device string for input array x.

    :param x: Tensor for which to get the device string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string for the array, e.g. 'cuda:0', 'cuda:1', 'cpu' etc..
    """
    return _cur_framework(x, f=f).dev_str(x)


# Conversions #

def dev_to_str(dev_in: ivy.Device, f: ivy.Framework = None)\
        -> str:
    """
    Convert native data type to string representation.

    :param dev_in: The device handle to convert to string.
    :type dev_in: device handle
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'cuda:0'.
    """
    return _cur_framework(None, f=f).dev_to_str(dev_in)


# noinspection PyShadowingNames
def str_to_dev(dev_str: str, f: ivy.Framework = None)\
        -> ivy.Device:
    """
    Convert device string representation to native device type.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Native device handle.
    """
    return _cur_framework(None, f=f).str_to_dev(dev_str)


# Memory #

# noinspection PyShadowingNames
def total_mem_on_dev(dev_str: str)\
        -> float:
    """
    Get the total amount of memory (in GB) for a given device string. In case of CPU, the total RAM is returned.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :return: The total memory on the device in GB.
    """
    if 'gpu' in dev_str:
        handle = _get_nvml_gpu_handle(dev_str)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.total/1e9
    elif dev_str == 'cpu':
        return psutil.virtual_memory().total/1e9
    else:
        raise Exception('Invalid device string input, must be on the form "gpu:idx" or "cpu",'
                        'but found {}'.format(dev_str))


# noinspection PyShadowingNames
def used_mem_on_dev(dev_str: str, process_specific=False)\
        -> float:
    """
    Get the used memory (in GB) for a given device string. In case of CPU, the used RAM is returned.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :param process_specific: Whether the check the memory used by this python process alone. Default is False.
    :type process_specific: bool, optional
    :return: The used memory on the device in GB.
    """
    if 'gpu' in dev_str:
        if process_specific:
            raise Exception('process-specific GPU queries are currently not supported')
        handle = _get_nvml_gpu_handle(dev_str)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.used/1e9
    elif dev_str == 'cpu':
        if process_specific:
            return psutil.Process(os.getpid()).memory_info().rss
        vm = psutil.virtual_memory()
        return (vm.total - vm.available)/1e9
    else:
        raise Exception('Invalid device string input, must be on the form "gpu:idx" or "cpu",'
                        'but found {}'.format(dev_str))


# noinspection PyShadowingNames
def percent_used_mem_on_dev(dev_str: str, process_specific=False)\
        -> float:
    """
    Get the percentage used memory for a given device string. In case of CPU, the used RAM is returned.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :param process_specific: Whether the check the memory used by this python process alone. Default is False.
    :type process_specific: bool, optional
    :return: The percentage used memory on the device.
    """
    if 'gpu' in dev_str:
        if process_specific:
            raise Exception('process-specific GPU queries are currently not supported')
        handle = _get_nvml_gpu_handle(dev_str)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return (info.used/info.total)*100
    elif dev_str == 'cpu':
        vm = psutil.virtual_memory()
        if process_specific:
            return (psutil.Process(os.getpid()).memory_info().rss/vm.total)*100
        return (1-(vm.available/vm.total))*100
    else:
        raise Exception('Invalid device string input, must be on the form "gpu:idx" or "cpu",'
                        'but found {}'.format(dev_str))


# Utilization #

# noinspection PyShadowingNames
def dev_util(dev_str: str)\
        -> float:
    """
    Get the current utilization (%) for a given device.

    :param dev_str: The device string of the device to query utilization for.
    :type dev_str: str
    :return: The device utilization (%)
    """
    if dev_str == 'cpu':
        return psutil.cpu_percent()
    elif 'gpu' in dev_str:
        handle = _get_nvml_gpu_handle(dev_str)
        return nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        raise Exception('Invalid device string input, must be on the form "gpu:idx" or "cpu",'
                        'but found {}'.format(dev_str))


# Availability #

def gpu_is_available(f: ivy.Framework = None)\
        -> bool:
    """
    Determine whether a GPU is available to use, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, as to whether a gpu is available.
    """
    return _cur_framework(f=f).gpu_is_available()


def num_gpus(f: ivy.Framework = None)\
        -> int:
    """
    Determine the number of available GPUs, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Number of available GPUs.
    """
    return _cur_framework(f=f).num_gpus()


def tpu_is_available(f: ivy.Framework = None)\
        -> bool:
    """
    Determine whether a TPU is available to use, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, as to whether a tpu is available.
    """
    return _cur_framework(f=f).tpu_is_available()


# Default Device #
# ---------------#

def default_device():
    """
    Return the default device.
    """
    global DEFAULT_DEVICE
    if not ivy.exists(DEFAULT_DEVICE):
        DEFAULT_DEVICE = 'gpu:0' if ivy.gpu_is_available() else 'cpu'
    return DEFAULT_DEVICE


def set_default_device(device):
    assert device[0:3] in ['gpu', 'tpu', 'cpu']
    if device != 'cpu':
        assert device[3] == ':'
        assert device[4:].isnumeric()
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device


# Device Allocation #
# ------------------#

# noinspection PyShadowingNames
def to_dev(x: Union[ivy.Array, ivy.NativeArray], dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Move the input array x to the desired device, specified by device string.

    :param x: Array to move onto the device.
    :type x: array
    :param dev_str: device to move the array to 'cuda:0', 'cuda:1', 'cpu' etc. Keep same device if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The array x, but now placed on the target device.
    """
    return _cur_framework(x, f=f).to_dev(x, dev_str)


# Function Splitting #
# -------------------#

# noinspection PyShadowingNames
def split_factor(dev_str=None):
    """
    Get the global split factor for a given device, which can be used to scale batch splitting chunk sizes for the
    device across the codebase. Default global value for each device is 1.

    :param dev_str: The device to query the split factor for. Sets the default device by default.
    :type dev_str: str, optional
    :return: The split factor for the specified device.
    """
    global SPLIT_FACTORS
    dev_str = ivy.default(dev_str, default_device())
    if dev_str in SPLIT_FACTORS:
        return SPLIT_FACTORS[dev_str]
    SPLIT_FACTORS[dev_str] = 1.
    return SPLIT_FACTORS[dev_str]


# noinspection PyShadowingNames
def set_split_factor(factor, dev_str=None):
    """
    Set the global split factor for a given device, which can be used to scale batch splitting chunk sizes for the
    device across the codebase.

    :param factor: The factor to set the device-specific split factor to.
    :type factor: float
    :param dev_str: The device to set the split factor for. Sets the default device by default.
    :type dev_str: str, optional
    """
    assert 0 <= factor
    global SPLIT_FACTORS
    dev_str = ivy.default(dev_str, default_device())
    SPLIT_FACTORS[dev_str] = factor


def split_func_call(func: Callable, inputs: Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]],
                    mode: str, chunk_size: int, input_axes: Union[int, Iterable[int]] = 0,
                    output_axes: Union[int, Iterable[int]] = None)\
        -> Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]]:
    """
    Call a function by splitting its inputs along a given axis, and calling the function in chunks, rather than feeding
    the entire input array at once. This can be useful to reduce memory usage of the device the arrays are on.
    :param func: The function to be called.
    :type func: callable
    :param inputs: A list of inputs to pass into the function.
    :type inputs: sequence of arrays
    :param mode: The mode by which to unify the return values, must be one of [ concat | mean | sum ]
    :type mode: str
    :param chunk_size: The size of each of the chunks to be fed into the function.
    :type chunk_size: int
    :param input_axes: The axes along which to split each of the inputs, before passing to the function. Default is 0.
    :type input_axes: int or sequence of ints, optional
    :param output_axes: The axes along which to concat each of the returned outputs. Default is same as fist input axis.
    :type output_axes: int or sequence of ints, optional
    :return: The return from the function, following input splitting and re-concattenation.
    """
    if isinstance(input_axes, int):
        input_axes = [input_axes]*len(inputs)
    dim_size = inputs[0].shape[input_axes[0]]
    num_chunks = dim_size / chunk_size
    num_chunks_floored = math.floor(dim_size / chunk_size)
    chunk_sizes = [chunk_size]*num_chunks_floored
    if num_chunks != num_chunks_floored:
        chunk_sizes.append(dim_size - chunk_size * num_chunks_floored)
    inputs_split = [ivy.split(inp, chunk_sizes, input_axes[i], True) if ivy.is_array(inp)
                    else inp.split(chunk_sizes, input_axes[i], True) for i, inp in enumerate(inputs)]
    rets = [func(*i) for i in zip(*inputs_split)]
    rets = [ret if isinstance(ret, tuple) else (ret,) for ret in rets]
    num_outputs = len(rets[0])
    if output_axes is None:
        output_axes = [input_axes[0]] * num_outputs
    elif isinstance(output_axes, int):
        output_axes = [output_axes] * num_outputs
    is_mean = mode == 'mean'
    is_sum = mode == 'sum'
    if is_mean or is_sum:
        rets = [[(r.expand_dims(output_axis) if isinstance(r, ivy.Container) else ivy.expand_dims(r, output_axis)) * cs
                 for output_axis, r in zip(output_axes, ret)] for ret, cs in zip(rets, chunk_sizes)]
    concatted = [ivy.concatenate([r[i] for r in rets], output_axes[i]) if ivy.is_array(rets[0][i])
                 else ivy.Container.concat([r[i] for r in rets], output_axes[i])
                 for i in range(num_outputs)]
    if is_mean:
        ret = [(item.reduce_sum(output_axis) if isinstance(item, ivy.Container)
                else ivy.reduce_sum(item, output_axis))/sum(chunk_sizes)
               for item, output_axis in zip(concatted, output_axes)]
    elif is_sum:
        ret = [(item.reduce_sum(output_axis) if isinstance(item, ivy.Container)
                else ivy.reduce_sum(item, output_axis)) for item, output_axis in zip(concatted, output_axes)]
    else:
        ret = concatted
    return ret[0] if len(ret) == 1 else ret


# Multi-Device #
# -------------#

class MultiDev:

    def __init__(self, iterable, axis=0):
        if isinstance(iterable, MultiDev):
            # noinspection PyUnresolvedReferences,PyProtectedMember
            iterable = iterable._iterable
        self._axis = axis
        self._iterable = iterable
        self._length = len(iterable)
        self._counter = 0

    def __len__(self):
        return self._length

    def __repr__(self):
        return 'MultiDev(' + self._iterable.__repr__() + ')'


class MultiDevItem(MultiDev):

    def __init__(self, lst, axis=0):
        super().__init__(lst, axis)

    def at_dev(self, idx):
        return self._iterable[idx]

    def at_devs(self):
        return self._iterable

    @property
    def shape(self):
        shapes = [list(x.shape) if hasattr(x, 'shape') else None for x in self._iterable]
        if not shapes or None in shapes:
            return None
        shape0 = shapes[0]
        for shp in shapes[1:]:
            assert shp == shape0
        shape0[self._axis] = shape0[self._axis]*len(self)
        return tuple(shape0)

    def __getitem__(self, slice_obj):
        is_int = isinstance(slice_obj, int)
        stacked_dim_size = 0
        if is_int:
            slice_obj = slice(slice_obj, slice_obj+1, 1)
        ret_list = list()
        for sub_item in self._iterable:
            if not hasattr(sub_item, 'shape'):
                continue
            shp = sub_item.shape
            rel_slice_obj = slice(slice_obj.start-stacked_dim_size, slice_obj.stop-stacked_dim_size, 1)
            stacked_dim_size += shp[self._axis]
            if slice_obj.start < stacked_dim_size:
                if slice_obj.stop < stacked_dim_size:
                    ret_list.append(sub_item[rel_slice_obj])
                    return MultiDevItem(ret_list)
                else:
                    ret_list.append(sub_item[rel_slice_obj.start:])
        return MultiDevItem(ret_list)

    def __repr__(self):
        return 'MultiDevItem(' + self._iterable.__repr__() + ')'


class MultiDevIter(MultiDev):

    def __init__(self, iterable, num_devs):
        self._num_devs = num_devs
        super().__init__(iterable)

    def at_dev(self, dev_idx):
        return [x.at_dev(dev_idx) if isinstance(x, MultiDevItem) else x for x in self._iterable]

    def at_devs(self):
        return [self.at_dev(i) for i in range(self._num_devs)]

    def __getitem__(self, item):
        return self._iterable[item]

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
        return 'MultiDevIter(' + self._iterable.__repr__() + ')'


class MultiDevNest(MultiDevIter):

    def __init__(self, iterable, num_devs, max_depth):
        self._max_depth = max_depth
        super().__init__(iterable, num_devs)

    def at_dev(self, dev_idx):
        return ivy.nested_map(self._iterable, lambda x: x.at_dev(dev_idx) if isinstance(x, MultiDevItem) else x,
                              max_depth=self._max_depth)

    def __repr__(self):
        return 'MultiDevNest(' + self._iterable.__repr__() + ')'


# Device Distribution #
# --------------------#

class DistributedItem(MultiDevItem):

    def __repr__(self):
        return 'DistributedItem(' + self._iterable.__repr__() + ')'


class DistributedIter(MultiDevIter):

    def __init__(self, iterable, num_devs):
        super().__init__(iterable, num_devs)

    def __repr__(self):
        return 'DistributedIter(' + self._iterable.__repr__() + ')'


class DistributedNest(MultiDevNest):

    def __init__(self, iterable, num_devs, max_depth=1):
        super().__init__(iterable, num_devs, max_depth)

    def __repr__(self):
        return 'DistributedNest(' + self._iterable.__repr__() + ')'


def distribute_array(x, dev_strs: Union[Iterable[str], Dict[str, int]], axis=0):
    """
    Distribute an array across the specified devices, returning a list of sub-arrays, each on a different device.

    :param x: The array to distribute across devices.
    :type x: array
    :param dev_strs: The devices to distribute the array across.
    :type dev_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the array. Default is 0.
    :type axis: int, optional
    :return: array distributed across the target devices
    """
    split_arg = list(dev_strs.values()) if isinstance(dev_strs, dict) else len(dev_strs)
    return DistributedItem(
        [ivy.to_dev(x_sub, d) for x_sub, d in zip(ivy.split(x, split_arg, axis, with_remainder=True), dev_strs)])


def distribute(x, dev_strs: Union[Iterable[str], Dict[str, int]], axis=0):
    """
    Distribute the input item across the specified devices, returning a list of sub-items, each on a different device.

    :param x: The input array or container to distribute across devices.
    :type x: array or container
    :param dev_strs: The devices to distribute the input across.
    :type dev_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the input. Default is 0.
    :type axis: int, optional
    :return: array or container distributed across the target devices
    """
    if ivy.is_array(x):
        return distribute_array(x, dev_strs, axis)
    elif isinstance(x, ivy.Container):
        return x.distribute(dev_strs, axis)
    return x


def distribute_iter(xs, dev_strs: Union[Iterable[str], Dict[str, int]], axis=0):
    """
    Distribute elements of the iterbale xs across the specified devices.

    :param xs: The iterable of items to distribute.
    :type xs: iterable of any
    :param dev_strs: The devices to distribute the iterable elements across.
    :type dev_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the arrays in the iterable xs. Default is 0.
    :type axis: int, optional
    :return: iterable with each element distributed to the target devices
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return xs
    return DistributedIter([distribute(x, dev_strs, axis) for x in xs], len(dev_strs))


def distribute_nest(args, kwargs, dev_strs: Union[Iterable[str], Dict[str, int]], axis=0, max_depth=1):
    """
    Distribute the nested input arguments across the specified devices.

    :param args: The positional nested arguments to distribute.
    :type args: list of any
    :param kwargs: The keyword nested arguments to distribute.
    :type kwargs: dict of any
    :param dev_strs: The devices to distribute the nested arguments across.
    :type dev_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the arrays in the arguments. Default is 0.
    :type axis: int, optional
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :return: nested arguments distributed to the target devices
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return args, kwargs
    args_dist = ivy.nested_map(args, lambda x: distribute(x, dev_strs, axis), max_depth=max_depth)
    kwargs_dist = ivy.nested_map(kwargs, lambda x: distribute(x, dev_strs, axis), max_depth=max_depth)
    args_lengths = len(dev_strs)
    return DistributedNest(args_dist, args_lengths), DistributedNest(kwargs_dist, args_lengths)


# Device Cloning #
# ---------------#

class ClonedItem(MultiDevItem):

    def __repr__(self):
        return 'ClonedItem(' + self._iterable.__repr__() + ')'


class ClonedIter(MultiDevIter):

    def __init__(self, iterable, num_devs):
        super().__init__(iterable, num_devs)

    def __repr__(self):
        return 'ClonedIter(' + self._iterable.__repr__() + ')'


class ClonedNest(MultiDevNest):

    def __init__(self, iterable, num_devs, max_depth=1):
        super().__init__(iterable, num_devs, max_depth)

    def __repr__(self):
        return 'ClonedNest(' + self._iterable.__repr__() + ')'


def clone_array(x, dev_strs):
    """
    Clone an array across the specified devices, returning a list of cloned arrays, each on a different device.

    :param x: The array to clone across devices.
    :type x: array
    :param dev_strs: The devices to clone the array to.
    :type dev_strs: sequence of strs
    :return: array cloned to each of the target devices
    """
    return ClonedItem([ivy.stop_gradient(ivy.to_dev(x, d)) for d in dev_strs])


def clone(x, dev_strs):
    """
    Clone the input item to each of the specified devices, returning a list of cloned items, each on a different device.

    :param x: The input array or container to clone to each device.
    :type x: array or container
    :param dev_strs: The devices to clone the input to.
    :type dev_strs: sequence of strs
    :return: array or container distributed across the target devices
    """
    if ivy.is_array(x):
        return clone_array(x, dev_strs)
    elif isinstance(x, ivy.Container):
        return x.clone(dev_strs)
    return x


def clone_iter(xs, dev_strs):
    """
    Clone elements of the iterbale xs to each of the specified devices.

    :param xs: The iterable of items to clone.
    :type xs: iterable of any
    :param dev_strs: The devices to clone each of the iterable elements to.
    :type dev_strs: sequence of strs
    :return: iterable with each element cloned to each of the target devices
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return xs
    return ClonedIter([clone(x, dev_strs) for x in xs], len(dev_strs))


def clone_nest(args, kwargs, dev_strs, max_depth=1):
    """
    Clone the input arguments across the specified devices.

    :param args: The positional arguments to clone.
    :type args: list of any
    :param kwargs: The keyword arguments to clone.
    :type kwargs: dict of any
    :param dev_strs: The devices to clone the arguments to.
    :type dev_strs: sequence of strs
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :return: arguments cloned to each of the target devices
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return args, kwargs
    args_cloned = ivy.nested_map(args, lambda x: clone(x, dev_strs), max_depth=max_depth)
    kwargs_cloned = ivy.nested_map(kwargs, lambda x: clone(x, dev_strs), max_depth=max_depth)
    args_lengths = len(dev_strs)
    return ClonedNest(args_cloned, args_lengths), ClonedNest(kwargs_cloned, args_lengths)


# Device Unification #
# -------------------#

# noinspection PyShadowingNames
def _concat_unify_array(xs, dev_str, axis):
    return ivy.concatenate([ivy.to_dev(x_sub, dev_str) for x_sub in xs.at_devs()], axis)


# noinspection PyShadowingNames
def _sum_unify_array(xs, dev_str, _=None):
    return sum([ivy.to_dev(x_sub, dev_str) for x_sub in xs.at_devs()])


# noinspection PyShadowingNames
def _mean_unify_array(xs, dev_str, _=None):
    return _sum_unify_array(xs, dev_str) / len(xs)


# noinspection PyShadowingNames
def unify_array(xs, dev_str, mode, axis=0):
    """
    Unify a list of sub-arrays, on arbitrary devices, to a single array on the specified device.

    :param xs: The list of arrays to unify onto the specified device.
    :type xs: sequence of arrays
    :param dev_str: The device to unify the arrays to.
    :type dev_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the array, if concat mode is set. Default is 0.
    :type axis: int, optional
    :return: array unified to the target device
    """
    return {'concat': _concat_unify_array,
            'sum': _sum_unify_array,
            'mean': _mean_unify_array}[mode](xs, dev_str, axis)


# noinspection PyShadowingNames
def unify(xs, dev_str, mode, axis=0):
    """
    Unify a list of sub-arrays, on arbitrary devices, to a single concattenated array on the specified device.

    :param xs: The list of sub-arrays to unify onto the specified device.
    :type xs: sequence of arrays
    :param dev_str: The device to unify the sub-arrays to.
    :type dev_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the array, if concat mode is set. Default is 0.
    :type axis: int, optional
    :return: array unified to the target device
    """
    if isinstance(xs, ivy.MultiDevContainer):
        xs = MultiDevItem(xs.at_devs())
    elif not isinstance(xs, MultiDevItem):
        return xs
    xs0 = xs.at_dev(0)
    if ivy.is_array(xs0):
        return unify_array(xs, dev_str, mode, axis)
    elif isinstance(xs0, ivy.Container):
        return ivy.Container.unify(xs, dev_str, mode, axis)
    return xs


# noinspection PyShadowingNames
def unify_iter(xs, dev_str, mode, axis=0):
    """
    Unify elements of the iterbale xs to a single target device.

    :param xs: The iterable of items to clone.
    :type xs: iterable of any
    :param dev_str: The device to unify the elements of the iterable to.
    :type dev_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the sub-arrays. Default is 0.
    :type axis: int, optional
    :return: iterable with each element cloned to each of the target devices
    """
    # noinspection PyProtectedMember
    xs = xs._iterable if isinstance(xs, MultiDevIter) else xs
    if isinstance(xs[0], (list, tuple)):
        xs_t = [MultiDevItem(i) for i in list(map(list, zip(*xs)))]
        return [unify(x, dev_str, mode, axis) for x in xs_t]
    return unify(xs, dev_str, mode, axis)


# noinspection PyShadowingNames,PyProtectedMember
def unify_nest(args: Type[MultiDev], kwargs: Type[MultiDev], dev_str, mode, axis=0, max_depth=1):
    """
    Unify the input nested arguments, which consist of sub-arrays spread across arbitrary devices, to unified arrays
    on the single target device.

    :param args: The nested positional arguments to unify.
    :type args: MultiDev
    :param kwargs: The nested keyword arguments to unify.
    :type kwargs: MultiDev
    :param dev_str: The device to unify the nested arguments to.
    :type dev_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the sub-arrays. Default is 0.
    :type axis: int, optional
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :return: nested arguments unified to the target device
    """
    args = args._iterable if isinstance(args, MultiDevIter) else args
    kwargs = kwargs._iterable if isinstance(kwargs, MultiDevIter) else kwargs
    args_uni = ivy.nested_map(args, lambda x: unify(x, dev_str, mode, axis), max_depth=max_depth)
    kwargs_uni = ivy.nested_map(kwargs, lambda x: unify(x, dev_str, mode, axis), max_depth=max_depth)
    return args_uni, kwargs_uni


# Device Mappers #
# ---------------#

class DevMapper(abc.ABC):

    def __init__(self, fn, ret_fn, queue_class, worker_class, dev_strs, timeout=None, constant=None, unique=None):
        """
        Device Mapper base class.

        :param fn: The function which the device mapper parallelises across devices.
        :type fn: callable
        :param ret_fn: The function which receives the ivy.MultiDevIter as input, and produces a single device output.
        :type ret_fn: callable
        :param queue_class: The class to use for creating queues.
        :type queue_class: class
        :param worker_class: The class to use for creating parallel workers.
        :type worker_class: class
        :param dev_strs: A list of devices on which to parallelise the function.
        :type dev_strs: sequence of str
        :param timeout: The timeout for getting items from the queues. Default is global.
        :type timeout: float, optional
        :param constant: A dict of keyword arguments which are the same for each process. Default is None.
        :type constant: dict of any, optional
        :param unique: A dict of keyword argument sequences which are unique for each process. Default is None.
        :type unique: dict of iterables of any, optional
        """
        constant_kwargs = ivy.default(constant, {})
        unique_kwargs = ivy.default(unique, {})
        self._fn = fn
        self._ret_fn = ret_fn
        self._dev_strs = dev_strs
        self._num_workers = len(dev_strs)
        self._timeout = ivy.default(timeout, ivy.queue_timeout())
        self._workers = dict()
        self._input_queues = dict()
        self._output_queues = dict()
        self._worker_class = worker_class
        for i, ds in enumerate(self._dev_strs):
            input_queue = queue_class()
            output_queue = queue_class()
            worker_kwargs = dict(**constant_kwargs, **{k: v[i] for k, v in unique_kwargs.items()})
            worker = self._worker_class(target=self._worker_fn, args=(input_queue, output_queue, dev_strs[i],
                                                                      worker_kwargs))
            worker.start()
            self._input_queues[ds] = input_queue
            self._output_queues[ds] = output_queue
            self._workers[ds] = worker

    def __getstate__(self):
        # prevent already running processes from being pickled as sent to new processes
        state = self.__dict__.copy()
        state['_workers'] = None
        state['_ret_fn'] = None
        return state

    # noinspection PyShadowingNames
    def _worker_fn(self, input_queue, output_queue, dev_str, kwargs):
        ivy.set_framework('torch')
        for k, v in kwargs.items():
            if isinstance(v, ivy.Module) and not v.built:
                v.build(dev_str=dev_str)
        if 'dev_str' in inspect.getfullargspec(self._fn).args:
            kwargs['dev_str'] = dev_str
        while True:
            try:
                loaded_kwargs = input_queue.get(timeout=self._timeout)
            except queue.Empty:
                continue
            if loaded_kwargs is None:
                return
            ret = self._fn(**loaded_kwargs, **kwargs)
            output_queue.put(ret)

    def map(self, used_dev_strs=None, **kwargs):
        """
        Map the function fn to each of the MultiDevice args and kwargs, running each function in parallel with CUDA-safe
        multiprocessing.

        :param used_dev_strs: The devices used in the current mapping pass. Default is all dev_strs.
        :type used_dev_strs: sequence of str, optional
        :param kwargs: The MutliDevice keyword arguments to map the function to.
        :type kwargs: dict of any
        :return: The results of the function, returned as a MultiDevice instance.
        """
        used_dev_strs = ivy.default(used_dev_strs, self._dev_strs)
        [self._input_queues[ds].put({k: v[i] for k, v in kwargs.items()}) for i, ds in enumerate(used_dev_strs)]
        return self._ret_fn(
            ivy.MultiDevIter([self._output_queues[ds].get(timeout=self._timeout) for ds in used_dev_strs],
                             self._num_workers))

    @abc.abstractmethod
    def __del__(self):
        raise NotImplementedError


class DevMapperMultiProc(DevMapper):

    def __init__(self, fn, ret_fn, dev_strs, timeout=None, constant=None, unique=None):
        multiprocessing = ivy.multiprocessing('forkserver')
        super().__init__(fn, ret_fn, multiprocessing.Queue, multiprocessing.Process, dev_strs, timeout,
                         constant, unique)

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
# ---------------#

class DevManager:

    def __init__(self, dev_mapper=None, dev_strs: Union[Iterable[str], Dict[str, int]] = None, da_dim_size=None,
                 safety_factor=1.1, min_dev_dim_size=0, max_dev_dim_step_size=1, starting_split_factor=0.,
                 max_split_factor_step_size=0.05, tune_dev_alloc=True, tune_dev_splits=True):
        """
        Create device manager, which unlike the device mapper, handles all argument cloning and distributing internally.
        The device manager only receivess a specification regarding the ratio of the batch each device should consume.

        :param dev_mapper: The pre-built device mapper used by the manager internally.
        :type dev_mapper: DevMapper
        :param dev_strs: The devices to distribute and clone the arguments across.
        :type dev_strs: sequence of strs or dict of split sizes
        :param da_dim_size: The size of the dimension along which the device allocation splitting is performed.
        :type da_dim_size: int
        :param safety_factor: The factor by which to be safe in the avoidance of OOM GPU errors. Default is 1.1.
        :type safety_factor: float, optional
        :param min_dev_dim_size: The minimum dimension size to pass to a device. Default is 0.
        :type min_dev_dim_size: int, optional
        :param max_dev_dim_step_size: The maximum step size for changing the dimension for a device. Default is 1.
        :type max_dev_dim_step_size: int, optional
        :param starting_split_factor: The initial device-specific split factor. Default is 0.
        :type starting_split_factor: float, optional
        :param max_split_factor_step_size: The maximum step size for changing the split factor for a device.
                                           Default is 0.05.
        :type max_split_factor_step_size: float, optional
        :param tune_dev_alloc: Whether to tune the devices split sizes internally based on device utilization tracking,
                               and use the provided values for initialization. Default is True.
        :type tune_dev_alloc: bool, optional
        """
        num_dev_args = sum([ivy.exists(dev_mapper), ivy.exists(dev_strs), ivy.exists(da_dim_size)])
        if num_dev_args not in [0, 3]:
            raise Exception('either all or None of dev_mapper, dev_strs and da_dim_size must be specified, but found '
                            'dev_mapper={}, dev_strs={}, da_dim_size={}'.format(dev_mapper, dev_strs, da_dim_size))
        tune_dev_alloc = False if num_dev_args == 0 else tune_dev_alloc
        self._dev_mapper = dev_mapper
        self._num_devs = len(dev_strs)
        self._dim_size = da_dim_size
        assert 1 <= safety_factor
        self._safety_factor = safety_factor
        self._min_dev_dim_size = min_dev_dim_size
        self._max_dev_dim_step_size = max_dev_dim_step_size
        self._max_split_factor_step_size = max_split_factor_step_size
        self._tune_da = tune_dev_alloc
        self._tune_ds = tune_dev_splits
        self._tuned = (not tune_dev_alloc and not tune_dev_splits) or self._num_devs == 1
        self._first_tune_step = True
        self._da_tune_count = 0
        self._ds_tune_count = 0
        self._tune_step = self.da_tune_step
        self._observed_configs = set()
        if isinstance(dev_strs, dict):
            self._dev_str_da_ratios = dev_strs
        else:
            self._dev_str_da_ratios = dict(zip(dev_strs, [1 / self._num_devs] * self._num_devs))
        self._dev_strs_keys = self._dev_str_da_ratios.keys()
        if self._tune_ds:
            [ivy.set_split_factor(starting_split_factor, ds) for ds in self._dev_strs_keys]
        self._percent_mem_inc_per_unit_dim = dict(zip(self._dev_strs_keys, [0] * self._num_devs))
        self._percent_util_inc_per_unit_dim = dict(zip(self._dev_strs_keys, [1] * self._num_devs))
        self._delta_dim_sizes = dict(zip(self._dev_strs_keys, [0]*self._num_devs))
        self._dev_percent_mems = None
        self._dev_utils = None
        self._compute_dev_strs_da()

    # Device Allocation #

    def _shift_da_splits(self, ordered_dev_util_keys, deltas):
        for i in range(math.floor(self._num_devs / 2)):

            # less and more utilized keys
            less_util_dev_str = ordered_dev_util_keys[i]
            more_util_dev_str = ordered_dev_util_keys[-i - 1]

            # less utilized
            delta = min(deltas[less_util_dev_str], self._dev_strs_da[more_util_dev_str] - self._min_dev_dim_size)
            if ivy.exists(self._max_dev_dim_step_size):
                delta = min(delta, self._max_dev_dim_step_size)
            self._dev_strs_da[less_util_dev_str] += delta
            self._delta_dim_sizes[less_util_dev_str] = delta

            # more utilized
            self._dev_strs_da[more_util_dev_str] -= delta
            self._delta_dim_sizes[more_util_dev_str] = -delta

    def _compute_dev_strs_da(self):
        split_sizes = [int(round(r * self._dim_size)) for r in self._dev_str_da_ratios.values()]
        combined_batch_size = sum(split_sizes)
        excess_size = combined_batch_size - self._dim_size
        if excess_size > 0:
            for i in range(abs(excess_size)):
                split_sizes[i] -= 1
        elif excess_size < 0:
            for i in range(abs(excess_size)):
                split_sizes[i] += 1
        self._dev_strs_da = {k: v for k, v in zip(self._dev_strs_keys, split_sizes)}

    def _compute_dev_da_ratios(self):
        self._dev_str_da_ratios = {k: v / self._dim_size for k, v in self._dev_strs_da.items()}

    def da_tune_step(self):
        new_dev_utils = dict(sorted({k: dev_util(k) for k in self._dev_strs_keys}.items(), key=lambda item: item[1]))
        new_dev_utils_keys = list(new_dev_utils.keys())
        highest_util_dev = new_dev_utils_keys[-1]
        highest_util = new_dev_utils[highest_util_dev]
        new_dev_percent_mems = dict(sorted({k: percent_used_mem_on_dev(k) for k in self._dev_strs_keys}.items(),
                                           key=lambda item: item[1]))

        # first step
        if self._first_tune_step:

            # shift the device splits by 1
            self._shift_da_splits(new_dev_utils_keys, {k: 1 for k in self._dev_strs_keys})

            # update device percentage memory usages and utilizations
            self._dev_percent_mems = new_dev_percent_mems
            self._dev_utils = new_dev_utils

            # increment count, update ratios and tune step, and return
            self._da_tune_count += 1
            self._first_tune_step = False
            self._compute_dev_da_ratios()
            if self._tune_ds:
                self._tune_step = self.ds_tune_step
            return

        # otherwise

        # percentage memory increase per unit dim
        delta_percent_mems = {k: new_dev_percent_mems[k] - self._dev_percent_mems[k] for k in self._dev_strs_keys}
        self._percent_mem_inc_per_unit_dim = \
            {k: (((self._da_tune_count - 1) * self._percent_mem_inc_per_unit_dim[k] +
                  (delta_percent_mems[k]/delta_dim_size)) / self._da_tune_count)
            if delta_dim_size != 0 else self._percent_mem_inc_per_unit_dim[k]
             for k, delta_dim_size in self._delta_dim_sizes.items()}

        # percentage utility increase per unit dim
        delta_utils = {k: new_dev_utils[k] - self._dev_utils[k] for k in self._dev_strs_keys}
        self._percent_util_inc_per_unit_dim = \
            {k: max((((self._da_tune_count - 1) * self._percent_util_inc_per_unit_dim[k] +
                      (delta_utils[k]/delta_dim_size)) / self._da_tune_count), 0.1)
            if delta_dim_size != 0 else self._percent_util_inc_per_unit_dim[k]
             for k, delta_dim_size in self._delta_dim_sizes.items()}

        # shift the device splits
        desired_percent_increases = {k: highest_util - new_dev_utils[k] for k in self._dev_strs_keys}
        raw_deltas = {k: desired_percent_increases[k] / self._percent_util_inc_per_unit_dim[k]
                      for k in self._dev_strs_keys}
        permissable_steps = \
            {k: min(math.floor(((100-new_dev_percent_mems[k]) / max(self._percent_mem_inc_per_unit_dim[k], 0.1))
                               / self._safety_factor), self._dim_size) for k in self._dev_strs_keys}
        deltas = {k: ivy.minimum(v, pm) for (k, v), pm in zip(raw_deltas.items(), permissable_steps.values())}
        self._shift_da_splits(new_dev_utils_keys, deltas)

        # update device utilizations and percentage memory usages
        self._dev_utils = new_dev_utils
        self._dev_percent_mems = new_dev_percent_mems

        # increment count, update ratios and tune step
        self._compute_dev_da_ratios()
        self._da_tune_count += 1
        if self._tune_ds:
            self._tune_step = self.ds_tune_step

        # check if tuning is complete, and return if so
        config = tuple(self._dev_strs_da.values())
        if config in self._observed_configs:
            self._observed_configs.clear()
            self._percent_mem_inc_per_unit_dim.clear()
            self._delta_dim_sizes.clear()
            self._dev_percent_mems.clear()
            self._tuned = True
            return

        # otherwise add the current config to those observed
        self._observed_configs.add(config)

    # Device Splitting #

    def ds_tune_step(self):
        self._ds_tune_count += 1
        self._compute_dev_da_ratios()
        if self._tune_da:
            self._tune_step = self.da_tune_step

    def map(self, to_clone=None, to_distribute=None):
        """
        Map the function fn to each of the MultiDevice args and kwargs, running each function in parallel with CUDA-safe
        multiprocessing.

        :param to_clone: The MutliDevice keyword arguments to clone and map to the function. Default is None.
        :type to_clone: dict of any, optional
        :param to_distribute: The MutliDevice keyword arguments to distribute and map to the function. Default is None.
        :type to_distribute: dict of any, optional
        :return: The results of the function, returned as a MultiDevice instance.
        """
        used_dev_strs_dict = {k: v for k, v in self._dev_strs_da.items() if v > 0}
        used_dev_strs = list(used_dev_strs_dict.keys())
        if ivy.exists(to_clone):
            to_clone = {k: ivy.clone(v, used_dev_strs).at_devs() for k, v in to_clone.items()}
        else:
            to_clone = {}
        if ivy.exists(to_distribute):
            to_distribute = {k: ivy.distribute(v, used_dev_strs_dict).at_devs() for k, v in to_distribute.items()}
        else:
            to_distribute = {}
        ret = self._dev_mapper.map(**to_clone, **to_distribute, used_dev_strs=used_dev_strs)
        if self._tuned:
            return ret
        self._tune_step()
        return ret

    def __del__(self):
        self._dev_mapper.__del__()
        del self._dev_mapper

    @property
    def dim_size(self):
        return self._dim_size

    @dim_size.setter
    def dim_size(self, batch_size):
        self._dim_size = batch_size
        self._compute_dev_strs_da()

    @property
    def tune_step(self):
        return self._tune_step


# Profiler #
# ---------#

class Profiler(abc.ABC):

    def __init__(self, save_dir):
        self._save_dir = save_dir

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError
