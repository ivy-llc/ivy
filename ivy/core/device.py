"""
Collection of device Ivy functions.
"""

# global
import abc
import math
import nvidia_smi
import numpy as np
from psutil import virtual_memory
from typing import Callable, Union, Iterable

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


# Device Distribution #
# --------------------#

def split_func_call(func: Callable, inputs: Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]],
                    chunk_size: int, input_axes: Union[int, Iterable[int]] = 0,
                    output_axes: Union[int, Iterable[int]] = None, mean: bool = False)\
        -> Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]]:
    """
    Call a function by splitting its inputs along a given axis, and calling the function in chunks, rather than feeding
    the entire input array at once. This can be useful to reduce memory usage of the device the arrays are on.

    :param func: The function to be called.
    :type func: callable
    :param inputs: A list of inputs to pass into the function.
    :type inputs: sequence of arrays
    :param chunk_size: The size of each of the chunks to be fed into the function.
    :type chunk_size: int
    :param input_axes: The axes along which to split each of the inputs, before passing to the function. Default is 0.
    :type input_axes: int or sequence of ints, optional
    :param output_axes: The axes along which to concat each of the returned outputs. Default is same as fist input axis.
    :type output_axes: int or sequence of ints, optional
    :param mean: Whether to compute a weighted mean based on the return from each chunk. Default is False.
    :type mean: bool, optional
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
    if mean:
        rets = [[(r.expand_dims(output_axis) if isinstance(r, ivy.Container) else ivy.expand_dims(r, output_axis)) * cs
                 for output_axis, r in zip(output_axes, ret)] for ret, cs in zip(rets, chunk_sizes)]
    concatted = [ivy.concatenate([r[i] for r in rets], output_axes[i]) if ivy.is_array(rets[0][i])
                 else ivy.Container.concat([r[i] for r in rets], output_axes[i])
                 for i in range(num_outputs)]
    if mean:
        return [(item.reduce_sum(output_axis) if isinstance(item, ivy.Container)
                 else ivy.reduce_sum(item, output_axis))/sum(chunk_sizes)
                for item, output_axis in zip(concatted, output_axes)]
    return concatted


def split_func_call_across_devices(func: Callable,
                                   inputs: Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]],
                                   dev_strs: Union[int, Iterable[int], Iterable[str]],
                                   input_axes: Union[int, Iterable[int]] = None,
                                   output_axes: Union[int, Iterable[int]] = None, concat_output: bool = False)\
        -> Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]]:
    """
    Call a function by splitting its inputs along a given axis, and calling each chunk on a different device.

    :param func: The function to be called.
    :type func: callable
    :param inputs: A list of inputs to pass into the function.
    :type inputs: sequence of arrays or containers
    :param dev_strs: The gpu device strings, in the format "gpu:idx".
    :type dev_strs: int, sequence of ints or sequence of strs
    :param input_axes: The axes along which to split each of the inputs, before passing to the function. Default is 0.
    :type input_axes: int or sequence of ints, optional
    :param output_axes: The axes along which to concat each of the returned outputs. Default is same as fist input axis.
    :type output_axes: int or sequence of ints, optional
    :param concat_output: Whether to concatenate each return values into a single array. Default is False.
    :type concat_output: bool, optional
    :return: The return from the function, following input splitting and re-concattenation across devices.
    """
    if isinstance(input_axes, int):
        input_axes = [input_axes]*len(inputs)
    if isinstance(dev_strs, int):
        dev_strs = ["gpu:{}".format(dev_strs)]
    elif isinstance(dev_strs[0], int):
        dev_strs = ["gpu:{}".format(i) for i in dev_strs]
    input_0 = inputs[0]
    start_dev = ivy.dev_str(input_0) if ivy.is_array(input_0) else input_0.dev_str
    dim_size = input_0.shape[input_axes[0]]
    num_chunks = len(dev_strs)
    chunk_size = dim_size / num_chunks
    chunk_size_rounded = int(np.round(chunk_size))
    chunk_size_diff = chunk_size - chunk_size_rounded
    total_diff = int(np.round(chunk_size_diff*num_chunks))
    chunk_sizes = [chunk_size_rounded]*num_chunks
    for i in range(np.abs(total_diff)):
        chunk_sizes[i] += np.sign(total_diff)
    inputs_split = [ivy.split(inp, chunk_sizes, input_axes[i], True) if ivy.is_array(inp)
                    else inp.split(chunk_sizes, input_axes[i], True) for i, inp in enumerate(inputs)]
    inputs_split_to_devs = [[ivy.to_dev(inp, d_str) if ivy.is_array(inp) else inp.to_dev(d_str)
                             for inp, d_str in zip(inps, dev_strs)] for inps in inputs_split]
    rets = [func(*inps, dev_str=dev_strs[i]) for i, inps in enumerate(zip(*inputs_split_to_devs))]
    # ToDo: make the line below more readable, there is a lot going on
    rets = [[ivy.to_dev(ret, start_dev) if ivy.is_array(ret) else
             (ret.to_dev(start_dev) if isinstance(ret, ivy.Container) else
              ([ivy.to_dev(r, start_dev) if ivy.is_array(r) else r.to_dev(start_dev)
                for r in ret] if isinstance(ret, (list, tuple)) else ret)) for ret in rts] for rts in rets]
    num_outputs = len(rets[0])
    if not concat_output:
        return [[r[i] for r in rets] for i in range(num_outputs)]
    if output_axes is None:
        output_axes = [input_axes[0]] * num_outputs
    elif isinstance(output_axes, int):
        output_axes = [output_axes] * num_outputs
    returns = list()
    ret0 = rets[0]
    # ToDo: possibly make this cleaner using list comprehension or recursion
    for i in range(num_outputs):
        if ivy.is_array(ret0[i]):
            returns.append(ivy.concatenate([r[i] for r in rets], output_axes[i]))
        elif isinstance(ret0[i], ivy.Container):
            returns.append(ivy.Container.concat([r[i] for r in rets], output_axes[i]))
        elif isinstance(ret0[i], (tuple, list)):
            ret0i_len = len(ret0[i])
            if ivy.is_array(ret0[i][0]):
                returns.append([ivy.concatenate([r[i][j] for r in rets], output_axes[i]) for j in range(ret0i_len)])
            elif isinstance(ret0[i][0], ivy.Container):
                returns.append([ivy.Container.concat([r[i][j] for r in rets], output_axes[i])
                                for j in range(ret0i_len)])
            else:
                returns.append([r[i] for r in rets])
        else:
            returns.append([r[i] for r in rets])
    return returns


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
    """
    if check_for_array and not ivy.is_array(x):
        return x
    return [ivy.to_dev(x_sub, d) for x_sub, d in zip(ivy.split(x, len(dev_strs), axis, with_remainder=True), dev_strs)]


# noinspection PyShadowingNames
def unify_array(x, dev_str, axis=0, check_for_array=True):
    """
    Unify a list of sub-arrays, on arbitrary devices, to a single concattenated array on the specified device.

    :param x: The list of sub-arrays to unify onto the specified device.
    :type x: sequence of arrays
    :param dev_str: The device to unify the sub-arrays to.
    :type dev_str: sty
    :param axis: The axis along which to concattenate the array. Default is 0.
    :type axis: int, optional
    :param check_for_array: Whether to check if the input is a list of arrays, and only unify if so. Default is True.
    :type check_for_array: bool, optional
    """
    if check_for_array and not (isinstance(x, list) and ivy.is_array(x[0])):
        return x
    return ivy.concatenate([ivy.to_dev(x_sub, dev_str) for x_sub in x], axis)


def distribute(dev_strs, *args, axis=0, **kwargs):
    """
    Distribute the input arguments across the specified devices.
    """
    if isinstance(dev_strs, str) or len(dev_strs) == 1:
        return args, kwargs
    args_dist = ivy.nested_map(args, lambda x: distribute_array(x, dev_strs, axis))
    kwargs_dist = ivy.nested_map(kwargs, lambda x: distribute_array(x, dev_strs, axis))
    return args_dist, kwargs_dist


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
