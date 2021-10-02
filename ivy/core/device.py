"""
Collection of device Ivy functions.
"""

# global
import abc
import nvidia_smi
from typing import Union, Type
from psutil import virtual_memory

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Devices Queries #
# ----------------#

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


# noinspection PyShadowingNames
def memory_on_dev(dev_str: str)\
        -> float:
    """
    Get the total amount of memory for a given device string. In case of CPU, the total RAM is returned.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :return: The total memory on the device in GB.
    """
    if 'gpu' in dev_str or 'cuda' in dev_str:
        gpu_idx = int(dev_str.split(':')[-1])
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.total/1e9
    elif 'cpu' in dev_str:
        return virtual_memory().total/1e9
    else:
        raise Exception('Invalid device string input, must be on the form "gpu:idx" or "cpu:idx",'
                        'but found {}'.format(dev_str))


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


# Multi-Device #
# -------------#

class MultiDevice(list):
    pass


class MultiDeviceArgs(MultiDevice):

    def __init__(self, args, length):
        self._counter = 0
        self._args = args
        self._length = length
        super().__init__()

    def __getitem__(self, item):
        return ivy.nested_map(self._args, lambda x: x[item] if isinstance(x, MultiDevice) else x)

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter == self._length:
            raise StopIteration
        ret = self.__getitem__(self._counter)
        self._counter += 1
        return ret

    def __len__(self):
        return self._length


class MultiDeviceKWArgs(MultiDeviceArgs):

    def __init__(self, args, length):
        super().__init__(args, length)


# Device Distribution #
# --------------------#

class Distributed(MultiDevice):

    def __repr__(self):
        return 'Distributed(' + super().__repr__() + ')'


class DistributedArgs(MultiDeviceArgs):

    def __init__(self, args, length):
        super().__init__(args, length)

    def __repr__(self):
        return 'DistributedArgs(' + self._args.__repr__() + ')'


class DistributedKWArgs(MultiDeviceKWArgs):

    def __init__(self, args, length):
        super().__init__(args, length)

    def __repr__(self):
        return 'DistributedKWArgs(' + self._args.__repr__() + ')'


def distribute_array(x, dev_strs, axis=0, check_for_array=True):
    """
    Distribute an array across the specified devices, returning a list of sub-arrays, each on a different device.

    :param x: The array to distribute across devices.
    :type x: array
    :param dev_strs: The devices to distribute the array across.
    :type dev_strs: sequence of strs
    :param axis: The axis along which to split the array. Default is 0.
    :type axis: int, optional
    :param check_for_array: Whether to check if the input is an array, and only split if so. Default is True.
    :type check_for_array: bool, optional
    :return: array distributed across the target devices
    """
    if check_for_array and not ivy.is_array(x):
        return x
    return Distributed(
        [ivy.to_dev(x_sub, d) for x_sub, d in zip(ivy.split(x, len(dev_strs), axis, with_remainder=True), dev_strs)])


def distribute_args(dev_strs, *args, axis=0, **kwargs):
    """
    Distribute the input arguments across the specified devices.

    :param dev_strs: The devices to distribute the arguments across.
    :type dev_strs: sequence of strs
    :param args: The positional arguments to distribute.
    :type args: list of any
    :param axis: The axis along which to split the arrays in the arguments. Default is 0.
    :type axis: int, optional
    :param kwargs: The keyword arguments to distribute.
    :type kwargs: dict of any
    :return: arguments distributed to the target devices
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return args, kwargs
    args_dist = ivy.nested_map(args, lambda x: distribute_array(x, dev_strs, axis))
    kwargs_dist = ivy.nested_map(kwargs, lambda x: distribute_array(x, dev_strs, axis))
    args_lengths = len(dev_strs)
    return DistributedArgs(args_dist, args_lengths), DistributedKWArgs(kwargs_dist, args_lengths)


# Device Cloning #
# ---------------#

class Cloned(MultiDevice):

    def __repr__(self):
        return 'Cloned(' + super().__repr__() + ')'


class ClonedArgs(MultiDeviceArgs):

    def __init__(self, args, length):
        super().__init__(args, length)

    def __repr__(self):
        return 'ClonedArgs(' + self._args.__repr__() + ')'


class ClonedKWArgs(MultiDeviceKWArgs):

    def __init__(self, args, length):
        super().__init__(args, length)

    def __repr__(self):
        return 'Cloned(' + self._args.__repr__() + ')'


def clone_array(x, dev_strs, check_for_array=True):
    """
    Clone an array across the specified devices, returning a list of cloned arrays, each on a different device.

    :param x: The array to clone across devices.
    :type x: array
    :param dev_strs: The devices to clone the array to.
    :type dev_strs: sequence of strs
    :param check_for_array: Whether to check if the input is an array, and only clone if so. Default is True.
    :type check_for_array: bool, optional
    :return: array cloned to each of the target devices
    """
    if check_for_array and not ivy.is_array(x):
        return x
    return Cloned([ivy.to_dev(x, d) for d in dev_strs])


def clone_args(dev_strs, *args, **kwargs):
    """
    Clone the input arguments across the specified devices.

    :param dev_strs: The devices to clone the arguments to.
    :type dev_strs: sequence of strs
    :param args: The positional arguments to clone.
    :type args: list of any
    :param kwargs: The keyword arguments to clone.
    :type kwargs: dict of any
    :return: arguments cloned to each of the target devices
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return args, kwargs
    args_cloned = ivy.nested_map(args, lambda x: clone_array(x, dev_strs))
    kwargs_cloned = ivy.nested_map(kwargs, lambda x: clone_array(x, dev_strs))
    args_lengths = len(dev_strs)
    return ClonedArgs(args_cloned, args_lengths), ClonedKWArgs(kwargs_cloned, args_lengths)


# Device Unification #
# -------------------#

# noinspection PyShadowingNames
def unify_array(x, dev_str, axis=0, check_for_array=True):
    """
    Unify a list of sub-arrays, on arbitrary devices, to a single concattenated array on the specified device.

    :param x: The list of sub-arrays to unify onto the specified device.
    :type x: sequence of arrays
    :param dev_str: The device to unify the sub-arrays to.
    :type dev_str: str
    :param axis: The axis along which to concattenate the array. Default is 0.
    :type axis: int, optional
    :param check_for_array: Whether to check if the input is a list of arrays, and only unify if so. Default is True.
    :type check_for_array: bool, optional
    :return: array unified to the target device
    """
    if check_for_array and not isinstance(x, MultiDevice):
        return x
    return ivy.concatenate([ivy.to_dev(x_sub, dev_str) for x_sub in x], axis)


# noinspection PyShadowingNames,PyProtectedMember
def unify_args(dev_str, args: Type[MultiDevice], kwargs: Type[MultiDevice], axis=0):
    """
    Unify the input arguments, which consist of sub-arrays distributed across arbitrary devices, to a unified arrays
    on a single target device.

    :param dev_str: The device to unify the arguments to.
    :type dev_str: str
    :param args: The positional arguments to unify.
    :type args: MultiDevice
    :param axis: The axis along which to concattenate the sub-arrays. Default is 0.
    :type axis: int, optional
    :param kwargs: The keyword arguments to unify.
    :type kwargs: MultiDevice
    :return: arguments unified to the target device
    """
    args_uni = ivy.nested_map(args._args, lambda x: unify_array(x, dev_str, axis))
    kwargs_uni = ivy.nested_map(kwargs._args, lambda x: unify_array(x, dev_str, axis))
    return args_uni, kwargs_uni


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
