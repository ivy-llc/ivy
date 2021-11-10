"""
Collection of multi-node Ivy functions.
"""

# global
import abc
import math
import time
import queue
import inspect
from typing import Union, Type, Callable, Iterable, Dict, Any

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

DEFAULT_NODE = None
NODE_HANDLES = dict()
SPLIT_FACTORS = dict()

'''
# Node Queries #
# -------------#

# Retreival #

def node(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> ivy.Node:
    """
    Get the native node handle for input array x.

    :param x: Tensor for which to get the node handle.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: node handle for the array, in native framework format.
    """
    return _cur_framework(x, f=f).node(x)


def node_str(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> str:
    """
    Get the node string for input array x.

    :param x: Tensor for which to get the node string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: node string for the array, e.g. 'cuda:0', 'cuda:1', 'cpu' etc..
    """
    return _cur_framework(x, f=f).node_str(x)


# Conversions #

def node_to_str(node_in: ivy.Node, f: ivy.Framework = None)\
        -> str:
    """
    Convert native data type to string representation.

    :param node_in: The node handle to convert to string.
    :type node_in: node handle
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: node string e.g. 'cuda:0'.
    """
    return _cur_framework(None, f=f).node_to_str(node_in)


# noinspection PyShadowingNames
def str_to_node(node_str: str, f: ivy.Framework = None)\
        -> ivy.Node:
    """
    Convert node string representation to native node type.

    :param node_str: The node string to conver to native node handle.
    :type node_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Native node handle.
    """
    return _cur_framework(None, f=f).str_to_node(node_str)


# Memory #

# noinspection PyShadowingNames
def total_mem_on_node(node_str: str)\
        -> float:
    """
    Get the total amount of memory (in GB) for a given node string. In case of CPU, the total RAM is returned.

    :param node_str: The node string to conver to native node handle.
    :type node_str: str
    :return: The total memory on the node in GB.
    """
    raise NotImplementedError


# noinspection PyShadowingNames
def used_mem_on_node(node_str: str, process_specific=False)\
        -> float:
    """
    Get the used memory (in GB) for a given node string. In case of CPU, the used RAM is returned.

    :param node_str: The node string to conver to native node handle.
    :type node_str: str
    :param process_specific: Whether the check the memory used by this python process alone. Default is False.
    :type process_specific: bool, optional
    :return: The used memory on the node in GB.
    """
    raise NotImplementedError


# noinspection PyShadowingNames
def percent_used_mem_on_node(node_str: str, process_specific=False)\
        -> float:
    """
    Get the percentage used memory for a given node string. In case of CPU, the used RAM is returned.

    :param node_str: The node string to conver to native node handle.
    :type node_str: str
    :param process_specific: Whether the check the memory used by this python process alone. Default is False.
    :type process_specific: bool, optional
    :return: The percentage used memory on the node.
    """
    raise NotImplementedError


# Utilization #

# noinspection PyShadowingNames
def node_util(node_str: str)\
        -> float:
    """
    Get the current utilization (%) for a given node.

    :param node_str: The node string of the node to query utilization for.
    :type node_str: str
    :return: The node utilization (%)
    """
    raise NotImplementedError


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


# Default node #
# ---------------#

def default_node():
    """
    Return the default node.
    """
    global DEFAULT_NODE
    if not ivy.exists(DEFAULT_NODE):
        DEFAULT_NODE = 'gpu:0' if ivy.gpu_is_available() else 'cpu'
    return DEFAULT_NODE


def set_default_node(node):
    assert node[0:3] in ['gpu', 'tpu', 'cpu']
    if node != 'cpu':
        assert node[3] == ':'
        assert node[4:].isnumeric()
    global DEFAULT_NODE
    DEFAULT_NODE = node


# node Allocation #
# ------------------#

# noinspection PyShadowingNames
def to_node(x: Union[ivy.Array, ivy.NativeArray], node_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Move the input array x to the desired node, specified by node string.

    :param x: Array to move onto the node.
    :type x: array
    :param node_str: node to move the array to 'cuda:0', 'cuda:1', 'cpu' etc. Keep same node if None.
    :type node_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The array x, but now placed on the target node.
    """
    return _cur_framework(x, f=f).to_node(x, node_str)


# Function Splitting #
# -------------------#

# noinspection PyShadowingNames
def split_factor(node_str=None):
    """
    Get the global split factor for a given node, which can be used to scale batch splitting chunk sizes for the
    node across the codebase. Default global value for each node is 1.

    :param node_str: The node to query the split factor for. Sets the default node by default.
    :type node_str: str, optional
    :return: The split factor for the specified node.
    """
    global SPLIT_FACTORS
    node_str = ivy.default(node_str, default_node())
    if node_str in SPLIT_FACTORS:
        return SPLIT_FACTORS[node_str]
    SPLIT_FACTORS[node_str] = 0.
    return SPLIT_FACTORS[node_str]


# noinspection PyShadowingNames
def set_split_factor(factor, node_str=None):
    """
    Set the global split factor for a given node, which can be used to scale batch splitting chunk sizes for the
    node across the codebase.

    :param factor: The factor to set the node-specific split factor to.
    :type factor: float
    :param node_str: The node to set the split factor for. Sets the default node by default.
    :type node_str: str, optional
    """
    assert 0 <= factor
    global SPLIT_FACTORS
    node_str = ivy.default(node_str, default_node())
    SPLIT_FACTORS[node_str] = factor


def split_func_call(func: Callable, inputs: Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]],
                    mode: str, max_chunk_size: int = None, chunk_size: int = None,
                    input_axes: Union[int, Iterable[int]] = 0, output_axes: Union[int, Iterable[int]] = None)\
        -> Iterable[Union[Union[ivy.Array, ivy.NativeArray], ivy.Container]]:
    """
    Call a function by splitting its inputs along a given axis, and calling the function in chunks, rather than feeding
    the entire input array at once. This can be useful to reduce memory usage of the node the arrays are on.
    :param func: The function to be called.
    :type func: callable
    :param inputs: A list of inputs to pass into the function.
    :type inputs: sequence of arrays
    :param mode: The mode by which to unify the return values, must be one of [ concat | mean | sum ]
    :type mode: str
    :param max_chunk_size: The maximum size of each of the chunks to be fed into the function.
    :type max_chunk_size: int
    :param chunk_size: The size of each of the chunks to be fed into the function. Specifying this arg overwrites the
                       global split factor. Default is None.
    :type chunk_size: int, optional
    :param input_axes: The axes along which to split each of the inputs, before passing to the function. Default is 0.
    :type input_axes: int or sequence of ints, optional
    :param output_axes: The axes along which to concat each of the returned outputs. Default is same as fist input axis.
    :type output_axes: int or sequence of ints, optional
    :return: The return from the function, following input splitting and re-concattenation.
    """
    if not ivy.exists(max_chunk_size) and not ivy.exists(chunk_size):
        raise Exception('Either max_chunk_size or chunk_size must be specified, but neither were provided.')
    if isinstance(input_axes, int):
        input_axes = [input_axes]*len(inputs)
    chunk_size = ivy.default(
        chunk_size, lambda: max(int(round(max_chunk_size * ivy.split_factor(ivy.default_node()))), 1), True)
    dim_size = inputs[0].shape[input_axes[0]]
    if chunk_size >= dim_size:
        return func(*inputs)
    num_chunks = dim_size / chunk_size
    num_chunks_floored = math.floor(dim_size / chunk_size)
    chunk_sizes = [chunk_size] * num_chunks_floored
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


# Multi-Node #
# -----------#

class MultiNode:

    def __init__(self, data: Iterable, axis=0):
        if isinstance(data, MultiNode):
            # noinspection PyUnresolvedReferences,PyProtectedMember
            data = data._dict
        self._axis = axis
        self._data = data
        self._length = len(data)
        self._counter = 0

    def __len__(self):
        return self._length

    def __repr__(self):
        return 'MultiNode(' + self._data.__repr__() + ')'


class MultiNodeItem(MultiNode):

    def __init__(self, data: Dict[ivy.Node, Any], axis=0):
        super().__init__(data, axis)

    @property
    def shape(self):
        shapes = [list(x.shape) if hasattr(x, 'shape') else None for x in self._data.values()]
        if not shapes or None in shapes:
            return None
        shape0 = shapes[0]
        for shp in shapes[1:]:
            assert shp == shape0
        shape0[self._axis] = shape0[self._axis]*len(self)
        return tuple(shape0)

    def _slice(self, slice_obj: slice):
        stacked_dim_size = 0
        ret_dict = dict()
        for ns, sub_item in self._data.items():
            if not hasattr(sub_item, 'shape'):
                continue
            shp = sub_item.shape
            rel_slice_obj = slice(slice_obj.start-stacked_dim_size, slice_obj.stop-stacked_dim_size, 1)
            stacked_dim_size += shp[self._axis]
            if slice_obj.start < stacked_dim_size:
                if slice_obj.stop < stacked_dim_size:
                    ret_dict[ns] = sub_item[rel_slice_obj]
                    return MultiNodeItem(ret_dict)
                else:
                    ret_dict[ns] = sub_item[rel_slice_obj.start:]
        return MultiNodeItem(ret_dict)

    def __getitem__(self, query):
        if isinstance(query, str):
            return self._data[query]
        elif isinstance(query, int):
            return self._slice(slice(query, query+1, 1))
        elif isinstance(query, slice):
            return self._slice(query)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return 'MultiNodeItem(' + self._data.__repr__() + ')'


class MultiNodeIter(MultiNode):

    def __init__(self, data: Iterable, node_strs):
        self._node_strs = node_strs
        super().__init__(data)

    # noinspection PyShadowingNames
    def at_node(self, node_str):
        return [x[node_str] if isinstance(x, MultiNodeItem) else x for x in self._data]

    def at_nodes(self):
        return {ns: self.at_node(ns) for ns in self._node_strs}

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
        return 'MultiNodeIter(' + self._data.__repr__() + ')'


class MultiNodeNest(MultiNodeIter):

    def __init__(self, data: Iterable, node_strs, max_depth=1):
        self._max_depth = max_depth
        super().__init__(data, node_strs)

    # noinspection PyShadowingNames
    def at_node(self, node_str):
        return ivy.nested_map(self._data, lambda x: x[node_str] if isinstance(x, MultiNodeItem) else x,
                              max_depth=self._max_depth)

    def __repr__(self):
        return 'MultiNodeNest(' + self._data.__repr__() + ')'


# Node Distribution #
# ------------------#

class NodeDistItem(MultiNodeItem):

    def __repr__(self):
        return 'NodeDistItem(' + self._data.__repr__() + ')'


class NodeDistIter(MultiNodeIter):

    def __repr__(self):
        return 'NodeDistIter(' + self._data.__repr__() + ')'


class NodeDistNest(MultiNodeNest):

    def __repr__(self):
        return 'NodeDistNest(' + self._data.__repr__() + ')'


def node_dist_array(x, node_strs: Union[Iterable[str], Dict[str, int]], axis=0):
    """
    Distribute an array across the specified nodes, returning a list of sub-arrays, each on a different node.

    :param x: The array to distribute across nodes.
    :type x: array
    :param node_strs: The nodes to distribute the array across.
    :type node_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the array. Default is 0.
    :type axis: int, optional
    :return: array distributed across the target nodes
    """
    split_arg = list(node_strs.values()) if isinstance(node_strs, dict) else len(node_strs)
    return NodeDistItem(
        {ns: ivy.to_node(x_sub, ns) for x_sub, ns in zip(ivy.split(x, split_arg, axis, with_remainder=True),
                                                         node_strs)})


def node_dist(x, node_strs: Union[Iterable[str], Dict[str, int]], axis=0):
    """
    Distribute the input item across the specified nodes, returning a list of sub-items, each on a different node.

    :param x: The input array or container to distribute across nodes.
    :type x: array or container
    :param node_strs: The nodes to distribute the input across.
    :type node_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the input. Default is 0.
    :type axis: int, optional
    :return: array or container distributed across the target nodes
    """
    if ivy.is_array(x):
        return node_dist_array(x, node_strs, axis)
    elif isinstance(x, ivy.Container):
        return x.node_dist(node_strs, axis)
    return x


def node_dist_iter(xs, node_strs: Union[Iterable[str], Dict[str, int]], axis=0):
    """
    Distribute elements of the iterbale xs across the specified nodes.

    :param xs: The iterable of items to distribute.
    :type xs: iterable of any
    :param node_strs: The nodes to distribute the iterable elements across.
    :type node_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the arrays in the iterable xs. Default is 0.
    :type axis: int, optional
    :return: iterable with each element distributed to the target nodes
    """
    if isinstance(node_strs, str):
        node_strs = [node_strs]
    return NodeDistIter([node_dist(x, node_strs, axis) for x in xs], node_strs)


def distribute_nest(args, kwargs, node_strs: Union[Iterable[str], Dict[str, int]], axis=0, max_depth=1):
    """
    Distribute the nested input arguments across the specified nodes.

    :param args: The positional nested arguments to distribute.
    :type args: list of any
    :param kwargs: The keyword nested arguments to distribute.
    :type kwargs: dict of any
    :param node_strs: The nodes to distribute the nested arguments across.
    :type node_strs: sequence of strs or dict of split sizes
    :param axis: The axis along which to split the arrays in the arguments. Default is 0.
    :type axis: int, optional
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :return: nested arguments distributed to the target nodes
    """
    if isinstance(node_strs, str):
        node_strs = [node_strs]
    args_dist = ivy.nested_map(args, lambda x: node_dist(x, node_strs, axis), max_depth=max_depth)
    kwargs_dist = ivy.nested_map(kwargs, lambda x: node_dist(x, node_strs, axis), max_depth=max_depth)
    return NodeDistNest(args_dist, node_strs), NodeDistNest(kwargs_dist, node_strs)


# Node Cloning #
# -------------#

class ClonedItem(MultiNodeItem):

    def __repr__(self):
        return 'ClonedItem(' + self._data.__repr__() + ')'


class ClonedIter(MultiNodeIter):

    def __repr__(self):
        return 'ClonedIter(' + self._data.__repr__() + ')'


class ClonedNest(MultiNodeNest):

    def __repr__(self):
        return 'ClonedNest(' + self._data.__repr__() + ')'


def clone_array(x, node_strs):
    """
    Clone an array across the specified nodes, returning a list of cloned arrays, each on a different node.

    :param x: The array to clone across nodes.
    :type x: array
    :param node_strs: The nodes to clone the array to.
    :type node_strs: sequence of strs
    :return: array cloned to each of the target nodes
    """
    return ClonedItem({ns: ivy.stop_gradient(ivy.to_node(x, ns)) for ns in node_strs})


def clone(x, node_strs):
    """
    Clone the input item to each of the specified nodes, returning a list of cloned items, each on a different node.

    :param x: The input array or container to clone to each node.
    :type x: array or container
    :param node_strs: The nodes to clone the input to.
    :type node_strs: sequence of strs
    :return: array or container distributed across the target nodes
    """
    if ivy.is_array(x):
        return clone_array(x, node_strs)
    elif isinstance(x, ivy.Container):
        return x.node_clone(node_strs)
    return x


def clone_iter(xs, node_strs):
    """
    Clone elements of the iterbale xs to each of the specified nodes.

    :param xs: The iterable of items to clone.
    :type xs: iterable of any
    :param node_strs: The nodes to clone each of the iterable elements to.
    :type node_strs: sequence of strs
    :return: iterable with each element cloned to each of the target nodes
    """
    if isinstance(node_strs, str):
        node_strs = [node_strs]
    return ClonedIter([clone(x, node_strs) for x in xs], node_strs)


def clone_nest(args, kwargs, node_strs, max_depth=1):
    """
    Clone the input arguments across the specified nodes.

    :param args: The positional arguments to clone.
    :type args: list of any
    :param kwargs: The keyword arguments to clone.
    :type kwargs: dict of any
    :param node_strs: The nodes to clone the arguments to.
    :type node_strs: sequence of strs
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :return: arguments cloned to each of the target nodes
    """
    if isinstance(node_strs, str):
        node_strs = [node_strs]
    args_cloned = ivy.nested_map(args, lambda x: clone(x, node_strs), max_depth=max_depth)
    kwargs_cloned = ivy.nested_map(kwargs, lambda x: clone(x, node_strs), max_depth=max_depth)
    return ClonedNest(args_cloned, node_strs), ClonedNest(kwargs_cloned, node_strs)


# Node Unification #
# -----------------#

# noinspection PyShadowingNames
def _concat_unify_array(xs, node_str, axis):
    return ivy.concatenate([ivy.to_node(x_sub, node_str) for x_sub in xs.values()], axis)


# noinspection PyShadowingNames
def _sum_unify_array(xs, node_str, _=None):
    return sum([ivy.to_node(x_sub, node_str) for x_sub in xs.values()])


# noinspection PyShadowingNames
def _mean_unify_array(xs, node_str, _=None):
    return _sum_unify_array(xs, node_str) / len(xs)


# noinspection PyShadowingNames
def unify_array(xs, node_str, mode, axis=0):
    """
    Unify a list of sub-arrays, on arbitrary nodes, to a single array on the specified node.

    :param xs: The list of arrays to unify onto the specified node.
    :type xs: sequence of arrays
    :param node_str: The node to unify the arrays to.
    :type node_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the array, if concat mode is set. Default is 0.
    :type axis: int, optional
    :return: array unified to the target node
    """
    return {'concat': _concat_unify_array,
            'sum': _sum_unify_array,
            'mean': _mean_unify_array}[mode](xs, node_str, axis)


# noinspection PyShadowingNames
def unify(xs, node_str, mode, axis=0):
    """
    Unify a list of sub-arrays, on arbitrary nodes, to a single concattenated array on the specified node.

    :param xs: The list of sub-arrays to unify onto the specified node.
    :type xs: sequence of arrays
    :param node_str: The node to unify the sub-arrays to.
    :type node_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the array, if concat mode is set. Default is 0.
    :type axis: int, optional
    :return: array unified to the target node
    """
    if isinstance(xs, ivy.MultiNodeContainer):
        xs = MultiNodeItem(xs.at_nodes())
    elif not isinstance(xs, MultiNodeItem):
        return xs
    # noinspection PyProtectedMember
    xs0 = next(iter(xs.items()))[1]
    if ivy.is_array(xs0):
        return unify_array(xs, node_str, mode, axis)
    elif isinstance(xs0, ivy.Container):
        return ivy.Container.unify(xs, node_str, mode, axis)
    return xs


# noinspection PyShadowingNames
def unify_iter(xs, node_str, mode, axis=0, transpose=False):
    """
    Unify elements of the iterbale xs to a single target node.

    :param xs: The iterable of items to unify.
    :type xs: iterable of any
    :param node_str: The node to unify the elements of the iterable to.
    :type node_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the sub-arrays. Default is 0.
    :type axis: int, optional
    :param transpose: Whether to transpose the first and second dimensions of the iterator. Default is False.
    :type transpose: bool, optional
    :return: iterable with each element unified to a single target nodes
    """
    # noinspection PyProtectedMember
    xs = xs._data if isinstance(xs, MultiNodeIter) else xs
    if transpose:
        # ToDo: make this more elegant, this method should not be responsible for transposing iterators
        xs_t = [MultiNodeItem({ivy.node_str(i) if ivy.is_array(i) else i.node_str: i
                               for i in mdi}) for mdi in list(map(list, zip(*xs)))]
        return [unify(x, node_str, mode, axis) for x in xs_t]
    return unify(xs, node_str, mode, axis)


# noinspection PyShadowingNames,PyProtectedMember
def unify_nest(args: Type[MultiNode], kwargs: Type[MultiNode], node_str, mode, axis=0, max_depth=1):
    """
    Unify the input nested arguments, which consist of sub-arrays spread across arbitrary nodes, to unified arrays
    on the single target node.

    :param args: The nested positional arguments to unify.
    :type args: MultiNode
    :param kwargs: The nested keyword arguments to unify.
    :type kwargs: MultiNode
    :param node_str: The node to unify the nested arguments to.
    :type node_str: str
    :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
    :type mode: str
    :param axis: The axis along which to concattenate the sub-arrays. Default is 0.
    :type axis: int, optional
    :param max_depth: The maximum nested depth to reach. Default is 1. Increase this if the nest is deeper.
    :type max_depth: int, optional
    :return: nested arguments unified to the target node
    """
    args = args._data if isinstance(args, MultiNodeIter) else args
    kwargs = kwargs._data if isinstance(kwargs, MultiNodeIter) else kwargs
    args_uni = ivy.nested_map(args, lambda x: unify(x, node_str, mode, axis), max_depth=max_depth)
    kwargs_uni = ivy.nested_map(kwargs, lambda x: unify(x, node_str, mode, axis), max_depth=max_depth)
    return args_uni, kwargs_uni


# Node Mappers #
# -------------#

class NodeMapper(abc.ABC):

    def __init__(self, fn, ret_fn, queue_class, worker_class, node_strs, timeout=None, constant=None, unique=None):
        """
        Node Mapper base class.

        :param fn: The function which the node mapper parallelises across nodes.
        :type fn: callable
        :param ret_fn: The function which receives the ivy.MultiNodeIter as input, and produces a single node output.
        :type ret_fn: callable
        :param queue_class: The class to use for creating queues.
        :type queue_class: class
        :param worker_class: The class to use for creating parallel workers.
        :type worker_class: class
        :param node_strs: A list of nodes on which to parallelise the function.
        :type node_strs: sequence of str
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
        self._node_strs = node_strs
        self._num_workers = len(node_strs)
        self._timeout = ivy.default(timeout, ivy.queue_timeout())
        self._workers = dict()
        self._input_queues = dict()
        self._output_queues = dict()
        self._worker_class = worker_class
        for i, ns in enumerate(self._node_strs):
            input_queue = queue_class()
            output_queue = queue_class()
            worker_kwargs = dict(**constant_kwargs, **{k: v[i] for k, v in unique_kwargs.items()})
            worker = self._worker_class(target=self._worker_fn, args=(input_queue, output_queue, node_strs[i],
                                                                      worker_kwargs, ivy.current_framework_str()))
            worker.start()
            self._input_queues[ns] = input_queue
            self._output_queues[ns] = output_queue
            self._workers[ns] = worker

    def __getstate__(self):
        # prevent already running processes from being pickled as sent to new processes
        state = self.__dict__.copy()
        state['_workers'] = None
        state['_ret_fn'] = None
        return state

    # noinspection PyShadowingNames
    def _worker_fn(self, input_queue, output_queue, node_str, kwargs, framework_str):
        ivy.set_framework(framework_str)
        ivy.set_default_node(node_str)
        for k, v in kwargs.items():
            if isinstance(v, ivy.Module) and not v.built:
                v.build(node_str=node_str)
        if 'node_str' in inspect.getfullargspec(self._fn).args:
            kwargs['node_str'] = node_str
        while True:
            try:
                loaded_kwargs = input_queue.get(timeout=self._timeout)
            except queue.Empty:
                continue
            if loaded_kwargs is None:
                return
            if 'split_factor' in loaded_kwargs:
                ivy.set_split_factor(loaded_kwargs['split_factor'], node_str)
                del loaded_kwargs['split_factor']
            ret = self._fn(**loaded_kwargs, **kwargs)
            output_queue.put(ret)

    def map(self, used_node_strs=None, split_factors=None, **kwargs):
        """
        Map the function fn to each of the MultiNode args and kwargs, running each function in parallel with CUDA-safe
        multiprocessing.

        :param used_node_strs: The nodes used in the current mapping pass. Default is all node_strs.
        :type used_node_strs: sequence of str, optional
        :param split_factors: The updated split factors 0 < sf < 1 for each node. Default is None.
        :type split_factors: dict of floats, optional
        :param kwargs: The MultiNode keyword arguments to map the function to.
        :type kwargs: dict of any
        :return: The results of the function, returned as a MultiNode instance.
        """
        if ivy.exists(split_factors):
            kwargs['split_factor'] = split_factors
        used_node_strs = ivy.default(used_node_strs, self._node_strs)
        [self._input_queues[ns].put({k: v[ns] for k, v in kwargs.items()}) for ns in used_node_strs]
        return self._ret_fn(
            ivy.MultiNodeIter([self._output_queues[ns].get(timeout=self._timeout) for ns in used_node_strs],
                             self._num_workers))

    @abc.abstractmethod
    def __del__(self):
        raise NotImplementedError


class NodeMapperMultiProc(NodeMapper):

    def __init__(self, fn, ret_fn, node_strs, timeout=None, constant=None, unique=None):
        multiprocessing = ivy.multiprocessing('forkserver')
        super().__init__(fn, ret_fn, multiprocessing.Queue, multiprocessing.Process, node_strs, timeout,
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


# Node Manager #
# -------------#

class NodeManager:

    def __init__(self, node_mapper=None, node_strs: Union[Iterable[str], Dict[str, int]] = None, da_dim_size=None,
                 safety_factor=1.1, min_node_dim_size=0, max_node_dim_step_ratio=0.1, min_unit_node_tune_steps=10,
                 min_sf_tune_steps=10, starting_split_factor=0., max_split_factor_step_size=0.05, tune_node_alloc=True,
                 tune_node_splits=True):
        """
        Create node manager, which unlike the node mapper, handles all argument cloning and distributing internally.
        The node manager only receivess a specification regarding the ratio of the batch each node should consume.

        :param node_mapper: The pre-built node mapper used by the manager internally.
        :type node_mapper: NodeMapper
        :param node_strs: The nodes to distribute and clone the arguments across.
        :type node_strs: sequence of strs or dict of split sizes
        :param da_dim_size: The size of the dimension along which the node allocation splitting is performed.
        :type da_dim_size: int
        :param safety_factor: The factor by which to be safe in the avoidance of OOM GPU errors. Default is 1.1.
        :type safety_factor: float, optional
        :param min_node_dim_size: The minimum dimension size to pass to a node. Default is 0.
        :type min_node_dim_size: int, optional
        :param max_node_dim_step_ratio: The maximum step ratio for changing the dimension for a node. Default is 0.1.
        :type max_node_dim_step_ratio: int, optional
        :param min_unit_node_tune_steps: The minimum number of tune steps to make when optimizing with unit step size.
                                   Default is 10.
        :type min_unit_node_tune_steps: int, optional
        :param min_sf_tune_steps: Minimum number of split factor tune steps. Default is 10.
        :type min_sf_tune_steps: int, optional
        :param starting_split_factor: The initial node-specific split factor. Default is 0.
        :type starting_split_factor: float, optional
        :param max_split_factor_step_size: The maximum step size for changing the split factor for a node.
                                           Default is 0.05.
        :type max_split_factor_step_size: float, optional
        :param tune_node_alloc: Whether to tune the node split sizes internally based on node utilization tracking,
                               and use the provided values for initialization. Default is True.
        :type tune_node_alloc: bool, optional
        :param tune_node_splits: Whether to tune the per-node split sizes internally. Default is True.
        :type tune_node_splits: bool, optional
        """
        with_node_mapping = True if ivy.exists(node_mapper) else False
        tune_node_alloc = False if not with_node_mapping else tune_node_alloc
        self._node_mapper = node_mapper
        node_strs = ivy.default(node_strs, [ivy.default_node()])
        self._num_nodes = len(node_strs)
        self._dim_size = da_dim_size
        assert 1 <= safety_factor
        self._safety_factor = safety_factor
        self._min_node_dim_size = min_node_dim_size
        self._max_node_dim_step_ratio = max_node_dim_step_ratio
        if self._dim_size:
            self._max_node_dim_step_size = max(int(round(self._max_node_dim_step_ratio * self._dim_size)), 1)
        self._min_unit_node_tune_steps = min_unit_node_tune_steps
        self._min_sf_tune_steps = min_sf_tune_steps
        self._max_split_factor_step_size = max_split_factor_step_size
        self._with_node_mappig = with_node_mapping
        self._tune_da = tune_node_alloc
        self._tune_ns = tune_node_splits
        self._tuned = ((not tune_node_alloc or self._num_nodes == 1) and not tune_node_splits)
        self._first_da_tune_step = True
        self._first_ns_tune_step = True
        self._da_tune_count = 0
        self._unit_da_tune_count = 0
        self._ns_tune_count = 0
        if tune_node_alloc:
            self._tune_step = self.da_tune_step
        elif tune_node_splits:
            self._tune_step = self.ns_tune_step
        else:
            self._tune_step = None
        self._observed_configs = set()
        self._da_directions = dict()
        self._da_directions_flipped = dict()
        if isinstance(node_strs, dict):
            self._node_str_da_ratios = node_strs
        else:
            self._node_str_da_ratios = dict(zip(node_strs, [1 / self._num_nodes] * self._num_nodes))
        self._node_strs_keys = self._node_str_da_ratios.keys()
        self._percent_mem_inc_per_unit_da_dim = dict(zip(self._node_strs_keys, [0] * self._num_nodes))
        self._percent_mem_inc_per_sf = dict(zip(self._node_strs_keys, [0] * self._num_nodes))
        self._percent_util_inc_per_unit_da_dim = dict(zip(self._node_strs_keys, [1] * self._num_nodes))
        self._delta_da_dim_sizes = dict(zip(self._node_strs_keys, [0] * self._num_nodes))
        self._delta_sfs = dict(zip(self._node_strs_keys, [0] * self._num_nodes))
        self._node_percent_mems = None
        self._node_utils = None
        if with_node_mapping and ivy.exists(self._dim_size):
            self._compute_node_strs_da()
        self._node_strs_ns = {ns: starting_split_factor for ns in self._node_strs_keys}
        if self._tune_ns and not with_node_mapping:
            [ivy.set_split_factor(starting_split_factor, ns) for ns in self._node_strs_keys]
        self._da_time = time.perf_counter()
        self._da_step_time = 0
        self._ns_time = time.perf_counter()
        self._ns_step_time = 0

    # Node Allocation #

    def _shift_da_splits(self, ordered_node_util_keys, deltas):
        for i in range(math.floor(self._num_nodes / 2)):

            # less and more utilized keys
            less_util_node_str = ordered_node_util_keys[i]
            more_util_node_str = ordered_node_util_keys[-i - 1]

            # less utilized
            delta = max(min(deltas[less_util_node_str],
                            self._node_strs_da[more_util_node_str] - self._min_node_dim_size), 1)
            if ivy.exists(self._max_node_dim_step_size):
                delta = min(delta, self._max_node_dim_step_size)
            self._node_strs_da[less_util_node_str] += delta
            self._delta_da_dim_sizes[less_util_node_str] = delta

            # more utilized
            self._node_strs_da[more_util_node_str] -= delta
            self._delta_da_dim_sizes[more_util_node_str] = -delta

    def _compute_node_strs_da(self):
        split_sizes = [int(round(r * self._dim_size)) for r in self._node_str_da_ratios.values()]
        combined_batch_size = sum(split_sizes)
        excess_size = combined_batch_size - self._dim_size
        if excess_size > 0:
            for i in range(abs(excess_size)):
                split_sizes[i] -= 1
        elif excess_size < 0:
            for i in range(abs(excess_size)):
                split_sizes[i] += 1
        self._node_strs_da = {k: v for k, v in zip(self._node_strs_keys, split_sizes)}

    def _compute_node_da_ratios(self):
        self._node_str_da_ratios = {k: v / self._dim_size for k, v in self._node_strs_da.items()}

    def da_tune_step(self):
        if self._tuned:
            return
        new_node_utils = dict(sorted({k: node_util(k) for k in self._node_strs_keys}.items(), key=lambda item: item[1]))
        new_node_utils_keys = list(new_node_utils.keys())
        highest_util_node = new_node_utils_keys[-1]
        highest_util = new_node_utils[highest_util_node]
        new_node_percent_mems = dict(sorted({k: percent_used_mem_on_node(k) for k in self._node_strs_keys}.items(),
                                           key=lambda item: item[1]))

        # first step
        if self._first_da_tune_step:

            # shift the node splits by 1
            self._shift_da_splits(new_node_utils_keys, {k: 1 for k in self._node_strs_keys})

            # update node percentage memory usages and utilizations
            self._node_percent_mems = new_node_percent_mems
            self._node_utils = new_node_utils

            # increment count, update ratios and tune step, and return
            self._da_tune_count += 1
            self._first_da_tune_step = False
            self._compute_node_da_ratios()
            if self._tune_ns:
                self._tune_step = self.ns_tune_step
            self._da_time = time.perf_counter()
            return

        # otherwise

        # check if all directions have changed, and if so, half the max node dim step size
        if self._max_node_dim_step_size > 1:
            da_directions = {k: 1 if i < math.floor(self._num_nodes / 2) else -1
                             for i, (k, v) in enumerate(new_node_utils.items())}
            if len(self._da_directions) == 0:
                self._da_directions = da_directions
                self._da_directions_flipped = {k: False for k in self._node_strs_keys}
            else:
                self._da_directions_flipped = {k: da_directions[k] * v < 0 for k, v in self._da_directions.items()}
            if sum(self._da_directions_flipped.values()) == self._num_nodes:
                self._da_directions.clear()
                self._max_node_dim_step_size = max(int(round(self._max_node_dim_step_size / 2)), 1)

        # percentage memory increase per unit dim
        delta_percent_mems = {k: new_node_percent_mems[k] - self._node_percent_mems[k] for k in self._node_strs_keys}
        self._percent_mem_inc_per_unit_da_dim = \
            {k: (((self._da_tune_count - 1) * self._percent_mem_inc_per_unit_da_dim[k] +
                  (delta_percent_mems[k]/delta_dim_size)) / self._da_tune_count)
            if delta_dim_size != 0 else self._percent_mem_inc_per_unit_da_dim[k]
             for k, delta_dim_size in self._delta_da_dim_sizes.items()}

        # percentage utility increase per unit dim
        delta_utils = {k: new_node_utils[k] - self._node_utils[k] for k in self._node_strs_keys}
        self._percent_util_inc_per_unit_da_dim = \
            {k: max((((self._da_tune_count - 1) * self._percent_util_inc_per_unit_da_dim[k] +
                      (delta_utils[k]/delta_dim_size)) / self._da_tune_count), 0.1)
            if delta_dim_size != 0 else self._percent_util_inc_per_unit_da_dim[k]
             for k, delta_dim_size in self._delta_da_dim_sizes.items()}

        # shift the node splits
        desired_percent_increases = {k: highest_util - new_node_utils[k] for k in self._node_strs_keys}
        raw_deltas = {k: int(round(desired_percent_increases[k] / self._percent_util_inc_per_unit_da_dim[k]))
                      for k in self._node_strs_keys}
        permissable_steps = \
            {k: min(math.floor(((100-new_node_percent_mems[k]) / max(self._percent_mem_inc_per_unit_da_dim[k], 0.1))
                               / self._safety_factor), self._dim_size) for k in self._node_strs_keys}
        deltas = {k: min(v, pm) for (k, v), pm in zip(raw_deltas.items(), permissable_steps.values())}
        self._shift_da_splits(new_node_utils_keys, deltas)

        # update node utilizations and percentage memory usages
        self._node_utils = new_node_utils
        self._node_percent_mems = new_node_percent_mems

        # increment count, update ratios and tune step
        self._compute_node_da_ratios()
        self._da_tune_count += 1
        if self._tune_ns:
            self._tune_step = self.ns_tune_step

        # if step size is 1, check if tuning is complete, and return if so
        if self._max_node_dim_step_size == 1:

            # check if da tuning is complete
            if self.repeated_config_check() and self._unit_da_tune_count >= self._min_unit_node_tune_steps and \
                    not self._tune_ns or (self._ns_tune_count >= self._min_sf_tune_steps):
                self._observed_configs.clear()
                self._percent_mem_inc_per_unit_da_dim.clear()
                self._delta_da_dim_sizes.clear()
                self._node_percent_mems.clear()
                self._tuned = True

            self._unit_da_tune_count += 1

        # log time
        now = time.perf_counter()
        self._da_step_time = now - self._da_time
        self._da_time = now

    # Node Splitting #

    def _shift_ns(self, deltas):
        for ns, delta in deltas.items():
            clipped_delta = min(delta, self._max_split_factor_step_size)
            self._node_strs_ns[ns] = min(self._node_strs_ns[ns] + clipped_delta, 1)
            self._delta_sfs[ns] = clipped_delta
            if not self._with_node_mappig:
                ivy.set_split_factor(min(self._node_strs_ns[ns] + clipped_delta, 1), ns)

    def ns_tune_step(self):
        if self._tuned:
            return
        new_node_percent_mems = dict(sorted({k: percent_used_mem_on_node(k) for k in self._node_strs_keys}.items(),
                                           key=lambda item: item[1]))

        # first step
        if self._first_ns_tune_step:

            # shift the node splits by 1%
            self._shift_ns({k: 0.01 for k in self._node_strs_keys})

            # update node percentage memory usages and utilizations
            self._node_percent_mems = new_node_percent_mems

            # increment count, update ratios and tune step, and return
            self._ns_tune_count += 1
            self._first_ns_tune_step = False
            if self._tune_da:
                self._tune_step = self.da_tune_step
            self._ns_time = time.perf_counter()
            return

        # otherwise

        # percentage memory increase per unit dim
        delta_percent_mems = {k: new_node_percent_mems[k] - self._node_percent_mems[k] for k in self._node_strs_keys}
        self._percent_mem_inc_per_sf = \
            {k: (((self._ns_tune_count - 1) * self._percent_mem_inc_per_sf[k] +
                  (delta_percent_mems[k]/delta_sf)) / self._ns_tune_count)
            if delta_sf != 0 else self._percent_mem_inc_per_sf[k]
             for k, delta_sf in self._delta_sfs.items()}

        # shift the node splits
        deltas =\
            {k: min((max(100/self._safety_factor-new_node_percent_mems[k], 0)) / max(self._percent_mem_inc_per_sf[k], 1),
                    self._max_split_factor_step_size) for k in self._node_strs_keys}
        self._shift_ns(deltas)

        # update node percentage memory usages
        self._node_percent_mems = new_node_percent_mems

        # increment count, update ratios and tune step
        self._ns_tune_count += 1
        if self._tune_da:
            self._tune_step = self.da_tune_step

        # check whether node allocation tuning is ready to terminate
        da_can_terminate = not self._tune_da or self._max_node_dim_step_size == 1

        # check if ns tuning is complete
        if da_can_terminate and self.repeated_config_check() and self._ns_tune_count >= self._min_sf_tune_steps and \
                not self._tune_da or (self._unit_da_tune_count >= self._min_unit_node_tune_steps):
            self._observed_configs.clear()
            self._percent_mem_inc_per_sf.clear()
            self._node_percent_mems.clear()
            self._tuned = True

        # log time
        now = time.perf_counter()
        self._ns_step_time = now - self._ns_time
        self._ns_time = now

    # Repeated Config Checking #

    def repeated_config_check(self):

        # check if ns tuning is complete, and return if so
        config_list = list()
        if self._tune_da:
            config_list += list(self._node_strs_da.values())
        if self._tune_ns:
            config_list += [self._node_strs_ns[ns] for ns in self._node_strs_keys]
        config = tuple(config_list)
        if config in self._observed_configs:
            return True

        # otherwise add the current config to those observed
        self._observed_configs.add(config)

        return False

    # Mapping #

    def map(self, cloned=None, to_clone=None, distributed=None, to_distribute=None):
        """
        Map the function fn to each of the MultiNode args and kwargs, running each function in parallel with CUDA-safe
        multiprocessing.

        :param cloned: The MutliNode keyword arguments which are already cloned. Default is None.
        :type cloned: dict of any, optional
        :param to_clone: The MutliNode keyword arguments to clone and map to the function. Default is None.
        :type to_clone: dict of any, optional
        :param distributed: The MutliNode keyword arguments which already distributed. Default is None.
        :type distributed: dict of any, optional
        :param to_distribute: The MutliNode keyword arguments to distribute and map to the function. Default is None.
        :type to_distribute: dict of any, optional
        :return: The results of the function, returned as a MultiNode instance.
        """
        used_node_strs_dict = {k: v for k, v in self._node_strs_da.items() if v > 0}
        used_node_strs = list(used_node_strs_dict.keys())
        cloned = ivy.default(cloned, {})
        if ivy.exists(to_clone):
            to_clone = {k: ivy.node_clone(v, used_node_strs) for k, v in to_clone.items()}
        else:
            to_clone = {}
        distributed = ivy.default(distributed, {})
        if ivy.exists(to_distribute):
            to_distribute = {k: ivy.node_dist(v, used_node_strs_dict) for k, v in to_distribute.items()}
        else:
            to_distribute = {}
        if self._tune_ns:
            ret = self._node_mapper.map(**cloned, **to_clone, **distributed, **to_distribute,
                                        used_node_strs=used_node_strs, split_factors=self._node_strs_ns)
        else:
            ret = self._node_mapper.map(**cloned, **to_clone, **distributed, **to_distribute,
                                        used_node_strs=used_node_strs)
        if self._tuned:
            return ret
        self._tune_step()
        return ret

    def __del__(self):
        if ivy.exists(self._node_mapper):
            self._node_mapper.__del__()
            del self._node_mapper

    @property
    def dim_size(self):
        return self._dim_size

    @dim_size.setter
    def dim_size(self, batch_size):
        self._dim_size = batch_size
        if self._tune_da:
            self._max_node_dim_step_size = max(int(round(self._max_node_dim_step_ratio * self._dim_size)), 1)
            self._compute_node_strs_da()

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
'''
