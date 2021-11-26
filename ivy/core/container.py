"""
Base Container Object
"""

# global
import re
import copy
import time
import termcolor
import numpy as _np
import json as _json
import h5py as _h5py
import pickle as _pickle
import random as _random
from operator import lt as _lt
from operator import le as _le
from operator import eq as _eq
from operator import ne as _ne
from operator import gt as _gt
from operator import ge as _ge
from operator import mul as _mul
from operator import pow as _pow
from operator import not_ as _not
from functools import reduce as _reduce
from typing import Union, Iterable, Dict
from operator import truediv as _truediv
from operator import floordiv as _floordiv

# local
import ivy
import ivy.numpy

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
retrieval_key_chain = list()
base_cont = None

def _is_jsonable(x):
    try:
        _json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def _repr(x):
    try:
        return x.__repr__()
    except TypeError:
        return str(x)


# noinspection PyMissingConstructor
class Container(dict):

    def __init__(self, dict_in=None, queues=None, queue_load_sizes=None, container_combine_method='list_join',
                 queue_timeout=None, print_limit=10, key_length_limit=None, print_indent=4, print_line_spacing=0,
                 ivyh=None, default_key_color='green', keyword_color_dict=None, rebuild_child_containers=False,
                 types_to_iteratively_nest=None, alphabetical_keys=True, logging_retrieval_times=False, **kwargs):
        """
        Initialize container object from input dict representation.

        :param dict_in: the dictionary the container should wrap around. Default is None.
        :type dict_in: dict, optional
        :param queues: Sequence of multiprocessing queues, each of which returns containers.
                       This enables the current container to be passed around asynchronously while waiting for data.
                       Default is None.
        :type queues: sequence of multiprocessing queues, optional
        :param queue_load_sizes: Size of leading dimension of the containers returned by each queue. Default is None.
        :type queue_load_sizes: sequence of ints, optional
        :param container_combine_method: The method to use for combining containers arriving from different queues.
                                         Default is ivy.Container.list_join
        :type container_combine_method: str, optional
        :param queue_timeout: The timeout when waiting for containers to arrive from the queues. Default is global.
        :type queue_timeout: float, optional
        :param print_limit: The total array size limit when printing the container. Default is 10.
        :type print_limit: int, optional
        :param key_length_limit: The maximum key length when printing the container. Default is None.
        :type key_length_limit: int, optional
        :param print_indent: The number of whitespaces to use for indenting when printing the container. Default is 4.
        :type print_indent: int, optional
        :param print_line_spacing: The number of extra newlines to use between keys when printing the container.
                                   Default is 0.
        :type print_line_spacing: int, optional
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :param default_key_color: The default key color for printing the container to the terminal. Default is 'green'.
        :type default_key_color: str, optional
        :param keyword_color_dict: A dict mapping keywords to their termcolor color codes for printing the container.
        :type keyword_color_dict: dict, optional
        :param rebuild_child_containers: Whether to rebuild container found in dict_in with these constructor params.
                                         Default is False, in which case the original container are kept as are.
        :type rebuild_child_containers: bool, optional
        :param types_to_iteratively_nest: The data types to nest iteratively in the dict structure, each type must be
                                          iterable. Default is None.
        :type types_to_iteratively_nest: seq of iterable types
        :param alphabetical_keys: Whether to sort the container keys alphabetically, or preserve the dict order.
                                  Default is True.
        :type alphabetical_keys: bool, optional
        :param logging_retrieval_times: Whether retrieval times should be logged. Default is False.
        :type logging_retrieval_times: bool, optional
        :param kwargs: keyword arguments for dict creation. Default is None.
        :type kwargs: keyword arguments.
        """
        self._queues = queues
        self._container_combine_method = container_combine_method
        if ivy.exists(self._queues):
            if isinstance(self._container_combine_method, str):
                self._container_combine_method =\
                    {'list_join': self.list_join,
                     'concat': lambda conts: self.concat(conts, 0)}[self._container_combine_method]
            self._loaded_containers_from_queues = dict()
            self._queue_load_sizes_cum = _np.cumsum(queue_load_sizes)
            self._queue_timeout = ivy.default(queue_timeout, ivy.queue_timeout())
        self._retrieval_times = dict()
        if dict_in is None:
            if kwargs:
                dict_in = dict(**kwargs)
            else:
                dict_in = dict()
        elif kwargs:
            raise Exception('dict_in and **kwargs cannot both be specified for ivy.Container constructor,'
                            'please specify one or the other, not both.')
        self._config_in = dict(
            print_limit=print_limit, print_indent=print_indent, key_length_limit=key_length_limit,
            print_line_spacing=print_line_spacing, ivyh=ivyh, default_key_color=default_key_color,
            keyword_color_dict=keyword_color_dict, rebuild_child_containers=rebuild_child_containers,
            types_to_iteratively_nest=types_to_iteratively_nest, alphabetical_keys=alphabetical_keys,
            logging_retrieval_times=logging_retrieval_times)
        self._config = dict()
        self.inplace_update(dict_in, **self._config_in)

    # Class Methods #
    # --------------#

    @staticmethod
    def list_join(containers, config=None):
        """
        Join containers of lists together along the specified dimension.

        :param containers: containers to list join
        :type containers: sequence of Container objects
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: List joined containers, with each entry being a list of arrays
        """

        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                new_list = list()
                for container in containers:
                    new_list.append(container[key])
                return_dict[key] = Container.list_join(new_list, config)
            return Container(return_dict, **config)
        else:
            return [item for sublist in containers for item in sublist]

    @staticmethod
    def list_stack(containers, dim, config=None):
        """
        List stack containers together along the specified dimension.

        :param containers: containers to list stack
        :type containers: sequence of Container objects
        :param dim: dimension along which to list stack
        :type dim: int
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: Stacked containers, with each entry being a list of arrays
        """

        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.list_stack([container[key] for container in containers], dim, config)
            return Container(return_dict, **config)
        else:
            return containers

    @staticmethod
    def _concat_unify(containers, dev_str, axis=0):
        return Container.concat([cont.to_dev(dev_str) for cont in containers.values()], axis)

    @staticmethod
    def _sum_unify(containers, dev_str, _=None, _1=None):
        return sum([cont.to_dev(dev_str) for cont in containers.values()])

    @staticmethod
    def _mean_unify(containers, dev_str, _=None, _1=None):
        return Container._sum_unify(containers, dev_str) / len(containers)

    @staticmethod
    def unify(containers, dev_str, mode, axis=0):
        """
        Unify a list of containers, on arbitrary devices, to a single container on the specified device.

        :param containers: containers to unify
        :type containers: sequence of Container objects
        :param dev_str: The device to unify the containers to.
        :type dev_str: str
        :param mode: The mode by which to unify, must be one of [ concat | mean | sum ]
        :type mode: str
        :param axis: The axis along which to concattenate the container, if concat mode is set. Default is 0.
        :type axis: int, optional
        :return: Unified container
        """
        return {'concat': Container._concat_unify,
                'sum': Container._sum_unify,
                'mean': Container._mean_unify}[mode](containers, dev_str, axis)

    @staticmethod
    def concat(containers, dim, config=None):
        """
        Concatenate containers together along the specified dimension.

        :param containers: containers to concatenate
        :type containers: sequence of Container objects
        :param dim: dimension along which to concatenate
        :type dim: int
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: Concatenated containers
        """

        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.concat([container[key] for container in containers], dim, config)
            return Container(return_dict, **config)
        else:
            # noinspection PyProtectedMember
            ivyh = ivy.default(lambda: config['ivyh'], ivy, True)
            # noinspection PyBroadException
            try:
                if len(containers[0].shape) == 0:
                    return ivyh.concatenate([ivyh.reshape(item, [1] * (dim + 1)) for item in containers], dim)
                else:
                    return ivyh.concatenate(containers, dim)
            except Exception as e:
                raise Exception(str(e) + '\nContainer concat operation only valid for containers of arrays')

    @staticmethod
    def stack(containers, dim, config=None):
        """
        Stack containers together along the specified dimension.

        :param containers: containers to stack
        :type containers: sequence of Container objects
        :param dim: dimension along which to stack
        :type dim: int
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: Stacked containers
        """

        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.stack([container[key] for container in containers], dim, config)
            return Container(return_dict, **config)
        else:
            # noinspection PyProtectedMember
            ivyh = ivy.default(lambda: config['ivyh'], ivy, True)
            # noinspection PyBroadException
            try:
                if len(containers[0].shape) == 0:
                    return ivyh.stack([ivyh.reshape(item, [1] * (dim + 1)) for item in containers], dim, config)
                else:
                    return ivyh.stack(containers, dim)
            except Exception as e:
                raise Exception(str(e) + '\nContainer stack operation only valid for containers of arrays')

    @staticmethod
    def combine(*containers, config=None):
        """
        Combine keys and values in a sequence of containers, with priority given to the right-most container in the case
        of duplicates.

        :param containers: containers to compare
        :type containers: sequence of Container objects
        :param config: The configuration for the containers. Default is the same as container_rightmost.
        :type config: dict, optional
        :return: Combined containers
        """

        # if inputs are not dicts, then simply return the right-most value
        container_rightmost = containers[-1]
        if not isinstance(container_rightmost, dict):
            return container_rightmost

        if not ivy.exists(config):
            # noinspection PyUnresolvedReferences
            config = container_rightmost.config if isinstance(container_rightmost, Container) else {}

        # return if len==1
        if len(containers) == 1:
            return container_rightmost

        # otherwise, check that the keys are aligned between each container, and apply this method recursively
        return_dict = dict()
        all_keys = set([item for sublist in [list(cont.keys()) for cont in containers] for item in sublist])
        for key in all_keys:
            keys_present = [key in cont for cont in containers]
            return_dict[key] =\
                ivy.Container.combine(*[cont[key] for cont, kp in zip(containers, keys_present) if kp], config=config)
        return ivy.Container(return_dict, **config)

    @staticmethod
    def diff(*containers, mode='all', diff_keys='diff', detect_key_diffs=True, detect_value_diffs=True,
             detect_shape_diffs=True, config=None):
        """
        Compare keys and values in a sequence of containers, returning the single shared values where they are the same,
        and new nested sub-dicts with all values where they are different.

        :param containers: containers to compare
        :type containers: sequence of Container objects
        :param mode: The mode of the diff operation, returning either all keys and values,
                     only those that are consist across the containers, or only the differences. Default is all.
        :type mode: str, optional
        :param diff_keys: The key/keys to add to the returned container when differences are found. Default is "diff".
        :type diff_keys: str or list of strs, optional
        :param detect_key_diffs: Whether to treat different keys as detected differences.
                                 If not, the keys among the input containers are simply combined without flagging
                                 differences. Default is True.
        :type detect_key_diffs: bool, optional
        :param detect_value_diffs: Whether to treat different values as detected differences. Default is True.
        :type detect_value_diffs: bool, optional
        :param detect_shape_diffs: Whether to treat different array shapes as detected differences. Default is True.
        :type detect_shape_diffs: bool, optional
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: Compared containers
        """

        if mode not in ['all', 'same_only', 'diff_only']:
            raise Exception('mode must be one of [ "all" | "same_only" | "diff_only" ], but found {}'.format(mode))

        # if inputs are not dicts, then compare their values to determine the diff dict
        num_containers = len(containers)
        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}
        if not isinstance(container0, dict):
            equal_mat = ivy.equal(*containers, equality_matrix=True)
            if not detect_value_diffs:
                equal_mat = ivy.ones_like(equal_mat)
            if detect_shape_diffs:
                shape_equal_mat = ivy.equal(*[c.shape if ivy.is_array(c) else None for c in containers],
                                            equality_matrix=True)
                equal_mat = ivy.logical_and(equal_mat, shape_equal_mat)
            # noinspection PyTypeChecker
            if ivy.reduce_min(ivy.cast(equal_mat, 'int32')) == 1:
                if mode == 'diff_only':
                    return ivy.Container(**config)
                return container0
            elif mode == 'same_only':
                return ivy.Container(**config)
            else:
                cont_range = range(num_containers)
                diff_dict = dict()
                cont_dict = dict(zip(cont_range, containers))
                idxs_added = list()
                for idx in cont_range:
                    if idx not in idxs_added:
                        idxs_to_add = ivy.indices_where(equal_mat[idx])
                        idxs_to_add_list = sorted(ivy.to_numpy(idxs_to_add).reshape(-1).tolist())
                        if isinstance(diff_keys, str):
                            key = diff_keys + '_' + str(idxs_to_add_list)[1:-1]
                        elif isinstance(diff_keys, (list, tuple)):
                            key = diff_keys[idx]
                        else:
                            raise Exception('diff_keys must be either a string or list of strings,'
                                            'but found {} of type {}'.format(diff_keys, type(diff_keys)))
                        diff_dict[key] = cont_dict[idx]
                        idxs_added += idxs_to_add_list
                return ivy.Container(diff_dict, **config)

        # otherwise, check that the keys are aligned between each container, and apply this method recursively
        return_dict = dict()
        all_keys = set([item for sublist in [list(cont.keys()) for cont in containers] for item in sublist])
        for key in all_keys:
            keys_present = [key in cont for cont in containers]
            all_keys_present = sum(keys_present) == num_containers
            if all_keys_present:
                res = ivy.Container.diff(*[cont[key] for cont in containers],
                                         mode=mode, diff_keys=diff_keys, detect_key_diffs=detect_key_diffs,
                                         detect_value_diffs=detect_value_diffs, detect_shape_diffs=detect_shape_diffs,
                                         config=config)
                if not isinstance(res, dict) or res:
                    return_dict[key] = res
                continue
            elif sum(keys_present) == 1 and not detect_key_diffs:
                if mode == 'all':
                    return_dict[key] = containers[keys_present.index(True)][key]
                continue
            diff_dict = dict()
            for i, (key_present, cont) in enumerate(zip(keys_present, containers)):
                if detect_key_diffs:
                    if key_present and mode != 'same_only':
                        if isinstance(diff_keys, str):
                            diff_dict[diff_keys + '_' + str(i)] = cont[key]
                        elif isinstance(diff_keys, (list, tuple)):
                            diff_dict[diff_keys[i]] = cont[key]
                        else:
                            raise Exception('diff_keys must be either a string or list of strings,'
                                            'but found {} of type {}'.format(diff_keys, type(diff_keys)))
            if diff_dict:
                return_dict[key] = diff_dict
        return ivy.Container(return_dict, **config)

    @staticmethod
    def structural_diff(*containers, mode='all', diff_keys='diff', detect_key_diffs=True, detect_shape_diffs=True,
                        config=None):
        """
        Compare keys and shapes in a sequence of containers, returning the single shared values where they are the same,
        and new nested sub-dicts with all values where they are different.

        :param containers: containers to compare
        :type containers: sequence of Container objects
        :param mode: The mode of the diff operation, returning either all keys and values,
                     only those that are consist across the containers, or only the differences. Default is all.
        :type mode: str, optional
        :param diff_keys: The key/keys to add to the returned container when differences are found. Default is "diff".
        :type diff_keys: str or list of strs, optional
        :param detect_key_diffs: Whether to treat different keys as detected differences.
                                 If not, the keys among the input containers are simply combined without flagging
                                 differences. Default is True.
        :type detect_key_diffs: bool, optional
        :param detect_shape_diffs: Whether to treat different array shapes as detected differences. Default is True.
        :type detect_shape_diffs: bool, optional
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: Compared containers
        """
        return Container.diff(*containers, mode=mode, diff_keys=diff_keys, detect_key_diffs=detect_key_diffs,
                              detect_value_diffs=False, detect_shape_diffs=detect_shape_diffs, config=config)

    @staticmethod
    def multi_map(func, containers, key_chains=None, to_apply=True, prune_unapplied=False, key_chain='', config=None):
        """
        Apply function to all array values from a collection of identically structured containers.

        :param func: Function to apply to each container entry.
        :type func: python function
        :param containers: containers to map.
        :type containers: sequence of Container objects
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied,
                                otherwise the leftmost container value is used. Default is False.
        :type prune_unapplied: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: Contaienr
        """
        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}
        return_dict = dict()
        for key in container0.keys():
            values = [cont[key] for cont in containers]
            value0 = values[0]
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value0, Container):
                ret = ivy.Container.multi_map(
                    func, values, key_chains, to_apply, prune_unapplied, this_key_chain, config)
                if ret:
                    return_dict[key] = ret
            else:
                if key_chains is not None:
                    if (this_key_chain in key_chains and not to_apply) or (
                            this_key_chain not in key_chains and to_apply):
                        if prune_unapplied:
                            continue
                        return_dict[key] = value0
                        continue
                return_dict[key] = func(values, this_key_chain)
        # noinspection PyProtectedMember
        return Container(return_dict, **config)

    @staticmethod
    def common_key_chains(containers):
        """
        Return the key-chains common across all containers.

        :param containers: Containers to check.
        :type containers: list of containers
        :return: list of key-chains.
        """
        if len(containers) == 1:
            return containers[0].all_key_chains()
        sets = [set(cont.all_key_chains()) for cont in containers]
        return list(sets[0].intersection(*sets[1:]))

    @staticmethod
    def identical(containers, check_types=True, check_shapes=True, same_arrays=True, arrays_equal=True, key_chains=None,
                  to_apply=True, partial=False, key_chain=''):
        """
        Returns a single boolean as to whether the input containers have identical key-chains and data types.

        :param containers: containers to check.
        :type containers: sequence of Container objects
        :param check_types: Whether to check if the datatypes of the leaf nodes are the same. Default is True.
        :type check_types: bool, optional
        :param check_shapes: Whether to check if the shapes of the leaf nodes are the same. Default is True.
        :type check_shapes: bool, optional
        :param same_arrays: Whether to check if the arrays are the exact same instances. Default is True.
        :type same_arrays: bool, optional
        :param arrays_equal: Whether to check if the arrays have equal values. Default is True.
        :type arrays_equal: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :return: Boolean
        """
        if partial:
            common_key_chains = Container.common_key_chains(containers)
            if not common_key_chains:
                return False
            containers = [cont.at_key_chains(common_key_chains) for cont in containers]
        keys = set([i for sl in [list(cont.keys()) for cont in containers] for i in sl])
        # noinspection PyProtectedMember
        for key in keys:
            if not min([key in cont for cont in containers]):
                return False
            values = [cont[key] for cont in containers]
            value_0 = values[0]
            type_0 = type(value_0)
            types = [type(val) for val in values]
            if not min([type_n is type_0 for type_n in types]):
                if isinstance(value_0, Container) or check_types:
                    return False
            if ivy.is_array(value_0):
                if check_shapes:
                    shape_0 = value_0.shape
                    shapes = [val.shape for val in values]
                    if not min([shape_n == shape_0 for shape_n in shapes]):
                        return False
                if same_arrays:
                    id_0 = id(value_0)
                    ids = [id(val) for val in values]
                    if not min([id_n == id_0 for id_n in ids]):
                        return False
                elif arrays_equal:
                    if not ivy.arrays_equal(values):
                        return False
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value_0, Container):
                ret = ivy.Container.identical(
                    values, check_types, check_shapes, same_arrays, arrays_equal, key_chains, to_apply, partial,
                    this_key_chain)
                if not ret:
                    return False
        return True

    @staticmethod
    def assert_identical(containers, check_types=True, check_shapes=True, same_arrays=True, arrays_equal=True,
                         key_chains=None, to_apply=True, partial=False):
        """
        Assert whether the input containers are identical. Otherwise, the diff is shown in an exception.

        :param containers: containers to check.
        :type containers: sequence of Container objects
        :param check_types: Whether to check if the datatypes of the leaf nodes are the same. Default is True.
        :type check_types: bool, optional
        :param check_shapes: Whether to check if the shapes of the leaf nodes are the same. Default is True.
        :type check_shapes: bool, optional
        :param same_arrays: Whether to check if the arrays are the exact same instances. Default is True.
        :type same_arrays: bool, optional
        :param arrays_equal: Whether to check if the arrays have equal values. Default is True.
        :type arrays_equal: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        """
        assert Container.identical(
            containers, check_types, check_shapes, same_arrays, arrays_equal, key_chains, to_apply, partial),\
            'Containers were not identical:\n\n{}'.format(Container.diff(*containers))

    @staticmethod
    def identical_structure(containers, check_types=True, check_shapes=True, key_chains=None, to_apply=True,
                            partial=False, key_chain=''):
        """
        Returns a single boolean as to whether the input containers have identical structure.

        :param containers: containers to check.
        :type containers: sequence of Container objects
        :param check_types: Whether to also check whether the datatypes of the leaf nodes are the same. Default is True.
        :type check_types: bool, optional
        :param check_shapes: Whether to also check whether the shapes of the leaf nodes are the same. Default is True.
        :type check_shapes: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :return: Boolean
        """
        return Container.identical(containers, check_types, check_shapes, False, False, key_chains, to_apply, partial,
                                   key_chain)

    @staticmethod
    def assert_identical_structure(containers, check_types=True, check_shapes=True, key_chains=None, to_apply=True,
                                   partial=False):
        """
        Assert whether the input containers have identical structure. Otherwise, the diff is shown in an exception.

        :param containers: containers to check.
        :type containers: sequence of Container objects
        :param check_types: Whether to also check whether the datatypes of the leaf nodes are the same. Default is True.
        :type check_types: bool, optional
        :param check_shapes: Whether to also check whether the shapes of the leaf nodes are the same. Default is True.
        :type check_shapes: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        """
        assert Container.identical_structure(containers, check_types, check_shapes, key_chains, to_apply, partial),\
            'Containers did not have identical structure:\n\n{}'.format(Container.structural_diff(*containers))

    @staticmethod
    def identical_configs(containers):
        """
        Returns a single boolean as to whether the input containers all have identical configs.

        :param containers: containers to check.
        :type containers: sequence of Container objects
        """
        assert len(containers) > 1
        configs = [cont.config for cont in containers]
        config0 = configs[0]
        for k, v in config0.items():
            if not min([config[k] == v for config in configs]):
                return False
        return True

    @staticmethod
    def identical_array_shapes(containers, exclusive=False):
        """
        Determine whether all of the containers have identical number of arrays and identical array shapes,
        regardless of their key-chain structures.

        :param containers: containers to check.
        :type containers: sequence of Container objects
        :param exclusive: Whether to check if the data type is exclusively an array, rather than a variable
                          or traced array.
        :type exclusive: bool, optional
        :return: Boolean
        """
        array_conts = [cont.size_ordered_arrays(exclusive) for cont in containers]
        array_cont0 = array_conts[0]
        array_cont0_len = len(array_cont0)
        for array_cont in array_conts[1:]:
            if len(array_cont) != array_cont0_len:
                return False
            elif not min([a.shape == a0.shape for a, a0 in zip(array_cont.values(), array_cont0.values())]):
                return False
        return True

    @staticmethod
    def from_disk_as_hdf5(h5_obj_or_filepath, slice_obj=slice(None), alphabetical_keys=True, ivyh=None):
        """
        Load container object from disk, as an h5py file, at the specified hdf5 filepath.

        :param h5_obj_or_filepath: Filepath where the container object is saved to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 obj
        :param slice_obj: slice object to slice all h5 elements.
        :type slice_obj: slice or sequence of slices
        :param alphabetical_keys: Whether to sort the container keys alphabetically, or preserve the dict order.
                                  Default is True.
        :type alphabetical_keys: bool, optional
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Container loaded from disk
        """
        container_dict = dict()
        if type(h5_obj_or_filepath) is str:
            h5_obj = _h5py.File(h5_obj_or_filepath, 'r')
        else:
            h5_obj = h5_obj_or_filepath
        items = sorted(h5_obj.items()) if alphabetical_keys else h5_obj.items()
        for key, value in items:
            if isinstance(value, _h5py.Group):
                container_dict[key] = Container.from_disk_as_hdf5(value, slice_obj, ivyh)
            elif isinstance(value, _h5py.Dataset):
                container_dict[key] = ivy.default(ivyh, ivy).array(list(value[slice_obj]))
            else:
                raise Exception('Item found inside h5_obj which was neither a Group nor a Dataset.')
        return Container(container_dict, ivyh=ivyh)

    @staticmethod
    def from_disk_as_pickled(pickle_filepath, ivyh=None):
        """
        Load container object from disk at the specified pickle filepath.

        :param pickle_filepath: Filepath where the container object is saved to disk.
        :type pickle_filepath: str
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Container loaded from disk
        """
        if ivy.wrapped_mode():
            return Container(_pickle.load(open(pickle_filepath, 'rb')), ivyh=ivyh).to_ivy()
        return Container(_pickle.load(open(pickle_filepath, 'rb')), ivyh=ivyh)

    @staticmethod
    def from_disk_as_json(json_filepath, ivyh=None):
        """
        Load container object from disk at the specified json filepath.
        If some objects were not json-able during saving, then they will be loaded as strings.

        :param json_filepath: Filepath where the container object is saved to disk.
        :type json_filepath: str
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Container loaded from disk
        """
        with open(json_filepath) as json_data_file:
            return Container(_json.load(json_data_file), ivyh=ivyh)

    @staticmethod
    def h5_file_size(h5_obj_or_filepath):
        """
        Get file size of h5 file contents.

        :param h5_obj_or_filepath: Filepath where the container object is saved to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 obj
        :return: Size of h5 file contents, and batch size.
        """
        if type(h5_obj_or_filepath) is str:
            h5_obj = _h5py.File(h5_obj_or_filepath, 'r')
        else:
            h5_obj = h5_obj_or_filepath

        size = 0
        batch_size = 0
        for key, value in h5_obj.items():
            if isinstance(value, _h5py.Group):
                size_to_add, batch_size = Container.h5_file_size(value)
                size += size_to_add
            elif isinstance(value, _h5py.Dataset):
                value_shape = value.shape
                size += _reduce(_mul, value_shape, 1) * value.dtype.itemsize
                batch_size = value_shape[0]
            else:
                raise Exception('Item found inside h5_obj which was neither a Group nor a Dataset.')
        return size, batch_size

    @staticmethod
    def shuffle_h5_file(h5_obj_or_filepath, seed_value=0):
        """
        Shuffle entries in all datasets of h5 file, such that they are still aligned along axis 0.

        :param h5_obj_or_filepath: Filepath where the container object is saved to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 obj
        :param seed_value: random seed to use for array shuffling
        :type seed_value: int
        """
        if seed_value is None:
            seed_value = _random.randint(0, 1000)
        if type(h5_obj_or_filepath) is str:
            h5_obj = _h5py.File(h5_obj_or_filepath, 'a')
        else:
            h5_obj = h5_obj_or_filepath

        for key, value in h5_obj.items():
            if isinstance(value, _h5py.Group):
                Container.shuffle_h5_file(value, seed_value)
            elif isinstance(value, _h5py.Dataset):
                _random.seed(seed_value)
                # noinspection PyTypeChecker
                _random.shuffle(value)
            else:
                raise Exception('Item found inside h5_obj which was neither a Group nor a Dataset.')
        if isinstance(h5_obj, _h5py.File):
            h5_obj.close()

    @staticmethod
    def reduce(containers, reduction, config=None):
        """
        Reduce containers.

        :param containers: containers to reduce
        :type containers: sequence of Container objects
        :param reduction: the reduction function
        :type reduction: callable with single list input x
        :param config: The configuration for the containers. Default is the same as container0.
        :type config: dict, optional
        :return: reduced containers
        """
        container0 = containers[0]
        if not ivy.exists(config):
            config = container0.config if isinstance(container0, Container) else {}

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.reduce([container[key] for container in containers], reduction)
            return Container(return_dict, **config)
        else:
            # noinspection PyBroadException
            try:
                return reduction(containers)
            except Exception as e:
                raise Exception(str(e) + '\nContainer reduce operation only valid for containers of arrays')

    @staticmethod
    def flatten_key_chain(key_chain, replacement='__', above_height=None, below_depth=None):
        # noinspection RegExpSingleCharAlternation
        flat_keys = re.split('/|\.', key_chain)
        num_keys = len(flat_keys)
        pre_keys = list()
        post_keys = list()
        if above_height and num_keys > above_height:
            post_keys = flat_keys[-above_height:]
            del flat_keys[-above_height:]
        if below_depth and num_keys > below_depth:
            pre_keys = flat_keys[0:below_depth]
            del flat_keys[0:below_depth]
        return '/'.join([k for k in ['/'.join(pre_keys), replacement.join(flat_keys), '/'.join(post_keys)] if k])

    @staticmethod
    def trim_key(key, max_length):
        key_len = len(key)
        if not ivy.exists(max_length) or key_len <= max_length:
            return key
        idxs =\
            _np.round((key_len-1)/(max_length-1) * _np.linspace(0, max_length-1, max_length)).astype(_np.int32).tolist()
        return ''.join([key[idx] for idx in idxs])

    # Private Methods #
    # ----------------#

    def _get_shape(self):
        if not len(self.keys()):
            if ivy.exists(self._queues):
                return [self._queue_load_sizes_cum[-1]]
            return [0]
        sub_shapes =\
            [v for k, v in self.map(lambda x, kc: list(x.shape) if self._ivy.is_array(x)
                else ([len(x)] if isinstance(x, (list, tuple, ivy.MultiDev)) else None)).to_iterator() if v]
        if not sub_shapes:
            return sub_shapes
        min_num_dims = min([len(sub_shape) for sub_shape in sub_shapes])
        sub_shapes_array = _np.asarray([sub_shape[0:min_num_dims] for sub_shape in sub_shapes])
        sub_shapes_array = _np.where(sub_shapes_array == 0, -1, sub_shapes_array)
        mask = _np.prod(sub_shapes_array / sub_shapes_array[0:1], 0) == 1
        # noinspection PyTypeChecker
        return [None if _np.isnan(i) else int(i)
                for i in _np.where(mask, sub_shapes_array[0], _np.ones(min_num_dims)*float('nan')).tolist()]

    def _get_shapes(self):
        return self.map(lambda x, kc: x.shape if hasattr(x, 'shape') else None)

    def _get_dev_str(self):
        sub_dev_strs =\
            [v for k, v in self.map(lambda x, kc: self._ivy.dev_str(x)
            if self._ivy.is_array(x) else None).to_iterator() if v]
        if len(set(sub_dev_strs)) <= 1:
            return sub_dev_strs[0]
        return None

    def _at_key_chains_input_as_seq(self, key_chains, ignore_key_errors=False):
        return_cont = Container(dict(), **self._config)
        for kc in key_chains:
            val = self.at_key_chain(kc, ignore_key_errors=ignore_key_errors)
            if ignore_key_errors and not ivy.exists(val):
                continue
            return_cont.set_at_key_chain(kc, val, inplace=True)
        return return_cont

    def _at_key_chains_input_as_dict(self, key_chains, current_chain='', ignore_key_errors=False):
        return_dict = dict()
        for k, v in key_chains.items():
            if current_chain == '':
                new_current_chain = k
            else:
                new_current_chain = current_chain + '/' + k
            if isinstance(v, dict):
                return_dict[k] = self._at_key_chains_input_as_dict(v, new_current_chain,
                                                                   ignore_key_errors=ignore_key_errors)
            else:
                val = self.at_key_chain(new_current_chain, ignore_key_errors=ignore_key_errors)
                if ignore_key_errors and not ivy.exists(val):
                    continue
                return_dict[k] = val
        return Container(return_dict, **self._config)

    def _prune_key_chains_input_as_seq(self, key_chains):
        return_cont = self.copy()
        for kc in key_chains:
            return_cont = return_cont.prune_key_chain(kc)
        return return_cont

    def _prune_key_chains_input_as_dict(self, key_chains, return_cont=None):
        if return_cont is None:
            return_cont = self.copy()
        for k, v in key_chains.items():
            if isinstance(v, dict):
                ret_cont = self._prune_key_chains_input_as_dict(v, return_cont[k])
                if ret_cont.shape[0] == 0:
                    del return_cont[k]
            else:
                del return_cont[k]
        return return_cont

    # Public Methods #
    # ---------------#

    def update_config(self, **config):

        new_config = dict()
        for k, v in config.items():
            att_name = '_' + k
            if k in self._config_in:
                if k == 'types_to_iteratively_nest':
                    v = ivy.default(lambda: tuple(v), (), True)
                elif k == 'keyword_color_dict':
                    v = ivy.default(v, {})
                elif k == 'ivyh':
                    att_name = '_local_ivy'
                new_config[k] = v
                self.__setattr__(att_name, v)

        self._config = new_config

    def inplace_update(self, dict_in, **config):
        """
        Update the contents of this container inplace, using either a new dict or container.

        :param dict_in: New dict or container to update the current container inplace with.
        :type dict_in: container or dict
        """

        # update config
        self.update_config(**config)

        # update container values inplace
        if dict_in is None:
            return
        dict_types = tuple([dict] + ivy.container_types())
        if isinstance(dict_in, dict_types):
            dict_in = dict_in
        elif isinstance(dict_in, tuple(self._types_to_iteratively_nest)):
            dict_in = dict(zip(['it_{}'.format(str(i).zfill(len(str(len(dict_in)))))
                                for i in range(len(dict_in))], dict_in))
        else:
            raise Exception('invalid input {}'.format(dict_in))
        items = sorted(dict_in.items()) if self._alphabetical_keys else dict_in.items()
        for key, value in items:
            if (isinstance(value, dict_types) and (not isinstance(value, Container) or
                                                   self._rebuild_child_containers)) or \
                    isinstance(value, tuple(self._types_to_iteratively_nest)):
                self[key] = Container(value, **self._config)
            else:
                self[key] = value

    def set_framework(self, ivyh):
        """
        Update the framework to use for the container.
        """
        self._ivy = ivyh
        self._config['ivyh'] = ivyh
        return self

    def all_true(self, assert_is_bool=False, key_chains=None, to_apply=True, prune_unapplied=False,
                 map_sequences=False):
        """
        Determine whether all the entries in the container boolean evaluate to True.

        :param assert_is_bool: Whether or not to assert each entry is of type Boolean.
        :type assert_is_bool: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Boolean, whether all entries are boolean True.
        """
        return bool(_np.prod([v for k, v in self.as_bools(
            assert_is_bool, key_chains, to_apply, prune_unapplied, map_sequences).to_iterator()]))

    def all_false(self, assert_is_bool=False, key_chains=None, to_apply=True, prune_unapplied=False,
                  map_sequences=False):
        """
        Determine whether all the entries in the container boolean evaluate to False.

        :param assert_is_bool: Whether or not to assert each entry is of type Boolean.
        :type assert_is_bool: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Boolean, whether all entries are boolean False.
        """
        return not bool(_np.sum([v for k, v in self.as_bools(
            assert_is_bool, key_chains, to_apply, prune_unapplied, map_sequences).to_iterator()]))

    def reduce_sum(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                   map_sequences=False):
        """
        Computes sum of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements
                     of the input array. If axis is negative it counts from the last to the first axis. If axis is a
                     tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single
                     axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_sum(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reduce_prod(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                    map_sequences=False):
        """
        Computes product of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a product is performed. The default, axis=None, will multiply all of the
                     elements of the input array. If axis is negative it counts from the last to the first axis. If axis
                     is a tuple of ints, a multiplication is performed on all of the axes specified in the tuple instead
                     of a single axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_prod(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reduce_mean(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                    map_sequences=False):
        """
        Computes mean of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a mean is performed. The default, axis=None, will mean all of the elements
                     of the input array. If axis is negative it counts from the last to the first axis. If axis is a
                     tuple of ints, a mean is performed on all of the axes specified in the tuple instead of a single
                     axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_mean(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reduce_var(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                   map_sequences=False):
        """
        Computes variance of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a var is performed. The default, axis=None, will var all of the elements
                     of the input array. If axis is negative it counts from the last to the first axis. If axis is a
                     tuple of ints, a var is performed on all of the axes specified in the tuple instead of a single
                     axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with the variance computed for all sub-arrays.
        """
        return self.map(lambda x, kc: self._ivy.reduce_var(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reduce_std(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                   map_sequences=False):
        """
        Computes standard deviation of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a var is performed. The default, axis=None, will var all of the elements
                     of the input array. If axis is negative it counts from the last to the first axis. If axis is a
                     tuple of ints, a var is performed on all of the axes specified in the tuple instead of a single
                     axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with the standard deviation computed for all sub-arrays.
        """
        return self.map(lambda x, kc: self._ivy.reduce_std(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reduce_min(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                   map_sequences=False):
        """
        Computes min of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a min is performed. The default, axis=None, will min all of the elements
                     of the input array. If axis is negative it counts from the last to the first axis. If axis is a
                     tuple of ints, a min is performed on all of the axes specified in the tuple instead of a single
                     axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_min(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reduce_max(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                   map_sequences=False):
        """
        Computes max of array elements along a given axis for all sub-arrays of container object.

        :param axis: Axis or axes along which a max is performed. The default, axis=None, will max all of the elements
                     of the input array. If axis is negative it counts from the last to the first axis. If axis is a
                     tuple of ints, a max is performed on all of the axes specified in the tuple instead of a single
                     axis or all the axes as before.
        :type axis: int or sequence of ints
        :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with
                         size one. With this option, the result will broadcast correctly against the input array.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_max(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def minimum(self, other, key_chains=None, to_apply=True, prune_unapplied=False,
                map_sequences=False):
        """
        Computes the elementwise minimum between this container and another container or number.

        :param other: The other container or number to compute the minimum against.
        :type other: Ivy container or number
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays having the minimum values computed.
        """
        is_container = isinstance(other, Container)
        return self.map(lambda x, kc:
                        self._ivy.minimum(x, other[kc] if is_container else other) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def maximum(self, other, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Computes the elementwise maximum between this container and another container or number.

        :param other: The other container or number to compute the maximum against.
        :type other: Ivy container or number
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays having the maximum values computed.
        """
        is_container = isinstance(other, Container)
        return self.map(lambda x, kc:
                        self._ivy.maximum(x, other[kc] if is_container else other) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def clip(self, clip_min, clip_max, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Computes the elementwise clipped values between this container and clip_min and clip_max containers or numbers.

        :param clip_min: The minimum container or number to clip against.
        :type clip_min: Ivy container or number
        :param clip_max: The maximum container or number to clip against.
        :type clip_max: Ivy container or number
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays having the clipped values returned.
        """
        min_is_container = isinstance(clip_min, Container)
        max_is_container = isinstance(clip_max, Container)
        return self.map(lambda x, kc:
                        self._ivy.clip(x, clip_min[kc] if min_is_container else clip_min,
                                       clip_max[kc] if max_is_container else clip_max) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def clip_vector_norm(self, max_norm, p, global_norm=False, key_chains=None, to_apply=True,
                         prune_unapplied=False, map_sequences=False):
        """
        Computes the elementwise clipped values between this container and clip_min and clip_max containers or numbers.

        :param max_norm: The max norm container or number to clip against.
        :type max_norm: Ivy container or number
        :param p: The p-value for computing the p-norm container or number.
        :type p: Ivy container or number
        :param global_norm: Whether to compute the norm across all the concattenated sub-arrays. Default is False.
        :type global_norm: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays having the clipped norms returned.
        """
        max_norm_is_container = isinstance(max_norm, Container)
        p_is_container = isinstance(p, Container)
        if global_norm:
            if max_norm_is_container or p_is_container:
                raise Exception(
                    'global_norm can only be computed for scalar max_norm and p_val arguments,'
                    'but found {} and {} of type {} and {} respectively'.format(
                        max_norm, p, type(max_norm), type(p)))
            vector_norm = self.vector_norm(p, global_norm=True)
            ratio = max_norm/vector_norm
            if ratio < 1:
                return self * ratio
            return self.copy()
        return self.map(lambda x, kc:
                        self._ivy.clip_vector_norm(
                            x, max_norm[kc] if max_norm_is_container else max_norm,
                            p[kc] if p_is_container else p) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def einsum(self, equation, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Sums the product of the elements of the input operands along dimensions specified using a notation based on the
        Einstein summation convention, for each array in the container.

        :param equation: A str describing the contraction, in the same format as numpy.einsum.
        :type equation: str
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.einsum(equation, x) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def vector_norm(self, p=2, axis=None, keepdims=False, global_norm=False, key_chains=None, to_apply=True,
                    prune_unapplied=False, map_sequences=False):
        """
        Compute vector p-norm for each array in the container.

        :param p: Order of the norm. Default is 2.
        :type p: int or str or container, optional
        :param axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms.
                     Default is None, in which case the flattened array is considered.
        :type axis: int or sequence of ints, optional
        :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions
                         with size one. With this option the result will broadcast correctly against the original x.
                         Default is False.
        :type keepdims: bool, optional
        :param global_norm: Whether to compute the norm across all the concattenated sub-arrays. Default is False.
        :type global_norm: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with the vector norms for each sub-array returned.
        """
        p_is_container = isinstance(p, Container)
        if global_norm:
            if p_is_container:
                raise Exception(
                    'global_norm can only be computed for scalar p argument,'
                    'but found {} of type {}'.format(p, type(p)))
            return sum([v for k, v in
                        self.map(lambda x, kc: self._ivy.reduce_sum(x ** p)).to_iterator()]) ** (1/p)
        return self.map(lambda x, kc: self._ivy.vector_norm(x, p[kc] if p_is_container else p, axis, keepdims)
                        if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied, map_sequences)

    def matrix_norm(self, p=2, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False,
                    map_sequences=False):
        """
        Compute matrix p-norm for each array in the container.

        :param p: Order of the norm. Default is 2.
        :type p: int or str, optional
        :param axis: If axis is an integer, it specifies the axis of x along which to compute the matrix norms.
                     Default is None, in which case the flattened array is considered.
        :type axis: int or sequence of ints, optional
        :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions
                         with size one. With this option the result will broadcast correctly against the original x.
                         Default is False.
        :type keepdims: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with the matrix norms for each sub-array returned.
        """
        return self.map(lambda x, kc: self._ivy.matrix_norm(x, p, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def flip(self, axis=None, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Reverses the order of elements in for each array in the container, along the given axis.
        The shape of the array is preserved, but the elements are reordered.

        :param axis: Axis or axes along which to flip over. The default, axis=None, will flip over all axes.
        :type axis: None or int or sequence of ints, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.flip(x, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def shuffle(self, seed_value=None, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False,
                key_chain=''):
        """
        Shuffle entries in all sub-arrays, such that they are still aligned along axis 0.

        :param seed_value: random seed to use for array shuffling
        :type seed_value: int
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        """
        return_dict = dict()
        if seed_value is None:
            seed_value = self._ivy.to_numpy(self._ivy.random.randint(0, 1000, ())).item()
        for key, value in self.items():
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.shuffle(seed_value, key_chains, to_apply, prune_unapplied, map_sequences, this_key_chain)
                if ret:
                    return_dict[key] = ret
            elif isinstance(value, (list, tuple)) and map_sequences:
                def _shuffle(v):
                    self._ivy.seed(seed_value)
                    return self._ivy.shuffle(v)
                ret = ivy.nested_map(value, _shuffle)
                if ret:
                    return_dict[key] = ret
            else:
                if key_chains is not None:
                    if (this_key_chain in key_chains and not to_apply) or (
                            this_key_chain not in key_chains and to_apply):
                        if prune_unapplied:
                            continue
                        return_dict[key] = value
                        continue
                self._ivy.seed(seed_value)
                return_dict[key] = self._ivy.shuffle(value)
        return Container(return_dict, **self._config)

    def slice_via_key(self, slice_key):
        """
        Get slice of container, based on key.

        :param slice_key: key to slice container at.
        :type slice_key: str
        :return: Container object sliced at desired key.
        """
        return_dict = dict()
        for key, value in self.items():
            if key == slice_key:
                return value
            elif isinstance(value, Container):
                return_dict[key] = value.slice_via_key(slice_key)
            else:
                return_dict[key] = value
        return Container(return_dict, **self._config)

    def as_ones(self, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Return arrays of ones for all nested arrays in the container.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays filled with ones.
        """
        return self.map(lambda x, kc: self._ivy.ones_like(x) if self._ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied, map_sequences)

    def as_zeros(self, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Return arrays of zeros for all nested arrays in the container.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays filled with zeros.
        """
        return self.map(lambda x, kc: self._ivy.zeros_like(x) if self._ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied, map_sequences)

    def as_bools(self, assert_is_bool=False, key_chains=None, to_apply=True, prune_unapplied=False,
                 map_sequences=False):
        """
        Return boolean evaluation for all nested items in the container.

        :param assert_is_bool: Whether or not to assert the entry is of type Boolean.
        :type assert_is_bool: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all entries boolean evaluated.
        """

        def _ret_bool(x):
            if assert_is_bool:
                assert isinstance(x, bool)
                return x
            return bool(x)

        return self.map(lambda x, kc: _ret_bool(x), key_chains, to_apply, prune_unapplied, map_sequences)

    def as_random_uniform(self, low=0.0, high=1.0, key_chains=None, to_apply=True, prune_unapplied=False,
                          map_sequences=False):
        """
        Return arrays of random uniform values for all nested arrays in the container.

        :param low: Lower boundary of the output interval. All values generated will be greater than or equal to low.
                    The default value is 0.
        :type low: float
        :param high: Upper boundary of the output interval. All values generated will be less than high.
                    The default value is 1.0.
        :type high: float
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays filled with random uniform values.
        """
        return self.map(lambda x, kc: self._ivy.random_uniform(
            low, high, x.shape, self._ivy.dev_str(x)) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def to_native(self, nested=False, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Return native framework arrays for all nested arrays in the container.

        :param nested: Whether to apply the conversion on arguments in a nested manner. If so, all dicts, lists and
                       tuples will be traversed to their lowest leaves in search of ivy.Array and ivy.Variable
                       instances. Default is False.
        :type nested: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-arrays converted to their native format.
        """
        return self.map(lambda x, kc: self._ivy.to_native(x, nested=nested), key_chains, to_apply, prune_unapplied,
                        map_sequences)

    def to_ivy(self, nested=False, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Return ivy arrays for all nested native framework arrays in the container.

        :param nested: Whether to apply the conversion on arguments in a nested manner. If so, all dicts, lists and
                       tuples will be traversed to their lowest leaves in search of ivy.Array and ivy.Variable
                       instances. Default is False.
        :type nested: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all native sub-arrays converted to their ivy.Array instances.
        """
        return self.map(lambda x, kc: self._ivy.to_ivy(x, nested=nested), key_chains, to_apply, prune_unapplied,
                        map_sequences)

    def expand_dims(self, axis, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Expand dims of all sub-arrays of container object.

        :param axis: Axis along which to expand dimensions of the sub-arrays.
        :type axis: int
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.expand_dims(x, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def dev_clone(self, dev_strs):
        """
        Clone the current container across multiple devices.

        :param dev_strs: The devices on which to clone the container.
        :type dev_strs: sequence of str
        :return: a set of cloned containers across the specified devices.
        """
        return self._ivy.DevClonedItem({dev_str: self.to_dev(dev_str) for dev_str in dev_strs})

    def dev_dist(self, dev_strs: Union[Iterable[str], Dict[str, int]], axis=0):
        """
        Distribute the current container across multiple devices.

        :param dev_strs: The devices along which to distribute the container.
        :type dev_strs: sequence of strs or dict of split sizes
        :param axis: The axis along which to split the arrays at the container leaves. Default is 0.
        :type axis: int, optional
        :return: a set of distributed sub-containers across the specified devices.
        """
        split_arg = list(dev_strs.values()) if isinstance(dev_strs, dict) else len(dev_strs)
        return self._ivy.DevDistItem(
            {dev_str: cont.to_dev(dev_str) for cont, dev_str in
             zip(self.split(split_arg, axis, with_remainder=True), dev_strs)})

    def to_multi_dev(self, dev_strs, axis=0):
        """
        Return a single MultiDevContainer, which shares the same structure as the current container, but replaces arrays
        at the leaves with DistributedArray instances.

        :param dev_strs: The devices along which to distribute each array in the container.
        :type dev_strs: sequence of str
        :param axis: The axis along which to split the arrays at the container leaves. Default is 0.
        :type axis: int, optional
        :return: a MultiDevContainer instance, with all leafs arrays replaced by DistributedArray instances.
        """
        return MultiDevContainer(
            self.map(lambda x, kc: self._ivy.dev_dist_array(x, dev_strs, axis)), dev_strs, **self._config)

    def unstack(self, axis, keepdims=False, dim_size=None):
        """
        Unstack containers along specified dimension.

        :param axis: Dimensions along which to unstack.
        :type axis: int
        :param keepdims: Whether to keep dimension 1 in the unstack dimensions. Default is False.
        :type keepdims: bool, optional
        :param dim_size: Size of the dimension to unstack. Determined from inputs by default.
        :type dim_size: int, optional
        :return: List of containers, unstacked along the specified dimension.
        """
        if dim_size is None:
            dim_size = self.shape[axis]
        if keepdims:
            # noinspection PyTypeChecker
            return [self[slice(i, i+1, 1) if axis == 0
                         else tuple([slice(None, None, None)] * axis + [slice(i, i+1, 1)])] for i in range(dim_size)]
        # noinspection PyTypeChecker
        return [self[i if axis == 0 else tuple([slice(None, None, None)] * axis + [i])] for i in range(dim_size)]

    def split(self, num_or_size_splits=None, axis=0, with_remainder=False, key_chains=None, to_apply=True,
              prune_unapplied=False, map_sequences=False):
        """
        Splits a container into multiple sub-containers, by splitting their constituent arrays.

        :param num_or_size_splits: Number of equal arrays to divide the array into along the given axis if an integer.
                                   The size of each split element if a sequence of integers.
                                   Default is to divide into as many 1-dimensional arrays as the axis dimension.
        :type num_or_size_splits: int, optional
        :param axis: The axis along which to split, default is 0.
        :type axis: int, optional
        :param with_remainder: If the tensor does not split evenly, then store the last remainder entry.
                               Default is False.
        :type with_remainder: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: A list of sub-arrays.
        """
        dim_size = num_or_size_splits if isinstance(num_or_size_splits, int) else len(num_or_size_splits)
        # noinspection PyTypeChecker
        return self.map(
            lambda x, kc: self._ivy.split(x, num_or_size_splits, axis, with_remainder) if self._ivy.is_array(x)
            else x, key_chains, to_apply, prune_unapplied, map_sequences).unstack(0, dim_size=dim_size)

    def gather(self, indices, axis=-1, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Gather slices from all container params at axis according to indices.

        :param indices: Index array.
        :type indices: array
        :param axis: The axis from which to gather from. Default is -1.
        :type axis: int, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions gathered along the axis.
        """
        return self.map(lambda x, kc: self._ivy.gather(x, indices, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def gather_nd(self, indices, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Gather slices from all container params into a arrays with shape specified by indices.

        :param indices: Index array.
        :type indices: array
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: Container object with all sub-array dimensions gathered.
        """
        return self.map(lambda x, kc: self._ivy.gather_nd(x, indices) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def repeat(self, repeats, axis=None, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Repeat values along a given dimension for each array in the container.

        :param repeats: Number of repetitions for each element. repeats is broadcast to fit the shape of the given axis.
        :type repeats: int or sequence of ints.
        :param axis: The axis along which to repeat values.
                      By default, use the flattened input array, and return a flat output array.
        :type axis: int, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: container with each array being repeated along the specified dimension.
        """
        return self.map(lambda x, kc: self._ivy.repeat(x, repeats, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def swapaxes(self, axis0, axis1, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Interchange two axes for each array in the container.

        :param axis0: First axis to be swapped.
        :type axis0: int
        :param axis1: Second axis to be swapped.
        :type axis1: int
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: ivy.Container with each chosen array having the axes swapped.
        """
        return self.map(lambda x, kc: self._ivy.swapaxes(x, axis0, axis1) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def reshape(self, pre_shape=None, shape_slice=None, post_shape=None, key_chains=None, to_apply=True,
                prune_unapplied=False, map_sequences=False):
        """
        Reshapes each array x in the container, to a new shape given by pre_shape + x.shape[shape_slice] + post_shape.
        If shape_slice or post_shape are not specified, then the term is ignored.

        :param pre_shape: The first elements in the new array shape.
        :type pre_shape: int or sequence of ints, optional
        :param shape_slice: The slice of the original shape to use in the new shape. Default is None.
        :type shape_slice: int or sequence of ints, optional
        :param post_shape: The final elements in the new array shape. Default is None.
        :type post_shape: sequence of ints, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: ivy.Container with each array reshaped as specified.
        """
        pre_shape = [] if pre_shape is None else\
            ([pre_shape] if isinstance(pre_shape, int) else list(pre_shape))
        post_shape = [] if post_shape is None else\
            ([post_shape] if isinstance(post_shape, int) else list(post_shape))
        if shape_slice is None:
            return self.map(lambda x, kc: self._ivy.reshape(x, pre_shape + post_shape) if self._ivy.is_array(x) else x,
                            key_chains, to_apply, prune_unapplied, map_sequences)
        shape_slice = slice(shape_slice, shape_slice+1) if isinstance(shape_slice, int) else shape_slice
        return self.map(lambda x, kc:
                        self._ivy.reshape(x, pre_shape + list(x.shape[shape_slice]) + post_shape)
                        if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied, map_sequences)

    def einops_rearrange(self, pattern,  key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False,
                         **axes_lengths):
        """
        Perform einops rearrange operation on each sub array in the container.

        :param pattern: Rearrangement pattern.
        :type pattern: str
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :param axes_lengths: Any additional specifications for dimensions.
        :type axes_lengths: keyword parameter args
        :return: ivy.Container with each array having einops.rearrange applied.
        """
        return self.map(lambda x, kc: ivy.einops_rearrange(x, pattern, **axes_lengths) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def einops_reduce(self, pattern,  reduction, key_chains=None, to_apply=True, prune_unapplied=False,
                      map_sequences=False, **axes_lengths):
        """
        Perform einops reduce operation on each sub array in the container.

        :param pattern: Reduction pattern.
        :type pattern: str
        :param reduction: One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or callable.
        :type reduction: str or callable
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :param axes_lengths: Any additional specifications for dimensions.
        :type axes_lengths: keyword parameter args
        :return: ivy.Container with each array having einops.reduce applied.
        """
        return self.map(lambda x, kc: ivy.einops_reduce(x, pattern, reduction, **axes_lengths) if self._ivy.is_array(x)
                        else x, key_chains, to_apply, prune_unapplied, map_sequences)

    def einops_repeat(self, pattern, key_chains=None, to_apply=True, prune_unapplied=False,
                      map_sequences=False, **axes_lengths):
        """
        Perform einops repeat operation on each sub array in the container.

        :param pattern: Rearrangement pattern.
        :type pattern: str
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :param axes_lengths: Any additional specifications for dimensions.
        :type axes_lengths: keyword parameter args
        :return: ivy.Container with each array having einops.repeat applied.
        """
        return self.map(lambda x, kc: ivy.einops_repeat(x, pattern, **axes_lengths) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def to_dev(self, dev_str, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Move the container arrays to the desired device, specified by device string.

        :param dev_str: device to move the array to 'cuda:0', 'cuda:1', 'cpu' etc. Keep same device if None.
        :type dev_str: str, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: The container, but with each sub-array now placed on the target device.
        """
        return self.map(lambda x, kc: self._ivy.stop_gradient(self._ivy.to_dev(x, dev_str))
            if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied, map_sequences)

    def stop_gradients(self, preserve_type=True, key_chains=None, to_apply=True, prune_unapplied=False,
                       map_sequences=False):
        """
        Stop gradients of all array entries in the container.

        :param preserve_type: Whether to preserve the input type (ivy.Variable or ivy.Array),
                              otherwise an array is always returned. Default is True.
        :param preserve_type: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: container with each array having their gradients stopped.
        """
        return self.map(
            lambda x, kc: self._ivy.stop_gradient(x, preserve_type) if self._ivy.is_variable(x)
            else x, key_chains, to_apply, prune_unapplied, map_sequences)

    def as_variables(self, key_chains=None, to_apply=True, prune_unapplied=False,
                     map_sequences=False):
        """
        Converts all nested arrays to variables, which support gradient computation.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: container with each array converted to a variable.
        """
        return self.map(lambda x, kc: self._ivy.variable(x) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied, map_sequences)

    def as_arrays(self, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Converts all nested variables to arrays, which do not support gradient computation.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: container with each variable converted to an array.
        """
        return self.map(
            lambda x, kc: self._ivy.stop_gradient(x, False) if self._ivy.is_variable(x)
            else (x if self._ivy.is_array(x) else self._ivy.array(x)), key_chains, to_apply, prune_unapplied,
            map_sequences)

    def num_arrays(self, exclusive=False):
        """
        Compute the number of arrays present at the leaf nodes, including variables by default.

        :param exclusive: Whether to check if the data type is exclusively an array,
                          rather than a variable or traced array.
        :type exclusive: bool, optional
        """
        return sum(self.map(lambda x, kc: ivy.is_array(x, exclusive)).to_iterator_values())

    def size_ordered_arrays(self, exclusive=False):
        """
        Return a container with keychains mapped to flat keys, and arrays given in order of smallest to largest.

        :param exclusive: Whether to check if the data type is exclusively an array,
                          rather than a variable or traced array.
        :type exclusive: bool, optional
        """
        array_dict = {Container.flatten_key_chain(kc): v
                      for kc, v in self.to_iterator() if ivy.is_array(v, exclusive)}
        return ivy.Container(dict(sorted(array_dict.items(), key=lambda item: _reduce(_mul, item[1].shape, 1))),
                             alphabetical_keys=False)

    def retrieval_time_ordered(self):
        """
        Return a container with keychains mapped to flat keys, and retrieved values given in order they were retrieved.
        """
        ret_dict = {Container.flatten_key_chain(kc): v for kc, v in self.to_iterator()}
        retrieval_dict = dict()
        for k, v in ret_dict.items():
            if k not in self._retrieval_times:
                continue
            rt = self._retrieval_times[k]
            for i in range(0, len(rt)):
                key = k + '__' + str(i)
                retrieval_dict[key] = (v, rt.pop(0))
        retrieval_dict = {k: v[0] for k, v in sorted(retrieval_dict.items(), key=lambda knv: knv[1][1])}
        return ivy.Container(retrieval_dict, alphabetical_keys=False)

    def to_numpy(self, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False, update_backend=True):
        """
        Converts all nested ivy arrays to numpy arrays.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :param update_backend: Whether to update the ivy backend of the returned container to numpy. Default is True.
        :type update_backend: bool, optional
        :return: container with each ivy array converted to a numpy array.
        """
        ret = self.map(
            lambda x, kc: self._ivy.to_numpy(x) if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied,
            map_sequences)
        if update_backend:
            ret.set_ivy_backend(ivy.numpy)
        return ret

    def from_numpy(self, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Converts all nested numpy arrays to native backend arrays.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: container with each ivy array converted to a numpy array.
        """
        ret = self.map(
            lambda x, kc: self._ivy.array(x) if isinstance(x, _np.ndarray) else x, key_chains, to_apply,
            prune_unapplied, map_sequences)
        return ret

    def arrays_as_lists(self, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False):
        """
        Converts all nested arrays to lists, a useful intermediate step for conversion to other framework array types.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :return: container with each array converted to a list.
        """
        return self.map(
            lambda x, kc: self._ivy.to_list(x) if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied,
            map_sequences)

    def to_disk_as_hdf5(self, h5_obj_or_filepath, starting_index=0, mode='a', max_batch_size=None):
        """
        Save container object to disk, as an h5py file, at the specified filepath.

        :param h5_obj_or_filepath: Filepath for where to save the container to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 object
        :param starting_index: Batch index for which to start writing to file, if it already exists
        :type starting_index: int
        :param mode: H5 read/write mode for writing to disk, ['r', 'r+', 'w', 'w-', 'a'], default is 'a'.
        :type mode: str
        :param max_batch_size: Maximum batch size for the container on disk, this is useful if later appending to file.
        :type max_batch_size: int
        :type h5_obj_or_filepath: str or h5 object
        """
        if type(h5_obj_or_filepath) is str:
            h5_obj = _h5py.File(h5_obj_or_filepath, mode)
        else:
            h5_obj = h5_obj_or_filepath
        for key, value in self.items():
            if isinstance(value, Container):
                if key not in h5_obj.keys():
                    h5_group = h5_obj.create_group(key)
                else:
                    h5_group = h5_obj[key]
                value.to_disk_as_hdf5(h5_group, starting_index, mode, max_batch_size)
            else:
                value_as_np = self._ivy.to_numpy(value)
                value_shape = value_as_np.shape
                this_batch_size = value_shape[0]
                if not max_batch_size:
                    max_batch_size = starting_index + this_batch_size
                if key not in h5_obj.keys():
                    dataset_shape = [max_batch_size] + list(value_shape[1:])
                    maxshape = ([None for _ in dataset_shape])
                    h5_obj.create_dataset(key, dataset_shape, dtype=value_as_np.dtype, maxshape=maxshape)
                space_left = max_batch_size - starting_index
                amount_to_write = min(this_batch_size, space_left)
                h5_obj[key][starting_index:starting_index + amount_to_write] = value_as_np[0:amount_to_write]

    def to_disk_as_pickled(self, pickle_filepath):
        """
        Save container object to disk, as an pickled file, at the specified filepath.

        :param pickle_filepath: Filepath for where to save the container to disk.
        :type pickle_filepath: str
        """
        if ivy.wrapped_mode():
            _pickle.dump(self.to_native().to_dict(), open(pickle_filepath, 'wb'))
        else:
            _pickle.dump(self.to_dict(), open(pickle_filepath, 'wb'))

    def to_jsonable(self, return_dict=None):
        """
        Return container with non-jsonable elements converted to string representations, which are jsonable.
        """
        if return_dict is None:
            return_dict = self.copy()
        for k, v in return_dict.items():
            if not _is_jsonable(v):
                if isinstance(v, dict):
                    return_dict[k] = self.to_jsonable(v)
                else:
                    return_dict[k] = str(v)
        return return_dict

    def to_disk_as_json(self, json_filepath):
        """
        Save container object to disk, as an json file, at the specified filepath.

        :param json_filepath: Filepath for where to save the container to disk.
        :type json_filepath: str
        """

        with open(json_filepath, 'w+') as json_data_file:
            _json.dump(self.to_jsonable().to_dict(), json_data_file, indent=4)

    def to_list(self):
        """
        Return nested list representation of container object.

        :return: Container as nested list.
        """
        return_list = list()
        for key, value in self.items():
            if isinstance(value, Container):
                return_list.append(value.to_list())
            elif value is not None and key != '_f':
                return_list.append(value)
        return return_list

    def to_raw(self):
        """
        Return nested raw representation of container object. This includes restoring lists and tuples passed in the
        constructor to their original form.

        :return: Container data in it's raw form.
        """
        return_item = dict()
        for i, (key, value) in enumerate(self.items()):
            if isinstance(value, Container):
                return_item[key] = value.to_raw()
            elif key[0:3] == 'it_' and tuple(self._types_to_iteratively_nest):
                return_item = list([v.to_raw() if isinstance(v, Container) else v for v in self.values()])
                break
            else:
                return_item[key] = value
        return return_item

    def to_dict(self):
        """
        Return nested pure dict representation of container object.

        :return: Container as nested dict.
        """
        return_dict = dict()
        for key, value in self.items():
            if isinstance(value, Container):
                return_dict[key] = value.to_dict()
            else:
                return_dict[key] = value
        return return_dict

    def to_iterator(self, key_chain='', leaf_keys_only=False, include_empty=False):
        """
        Return iterator for traversing through the nested elements of container object.

        :return: Iterator for the container elements.
        """
        for key, value in self.items():
            if leaf_keys_only:
                kc = key
            else:
                kc = key_chain + '/' + key if key_chain != '' else key
            if isinstance(value, Container) and (not include_empty or value):
                yield from value.to_iterator(kc, leaf_keys_only, include_empty)
            else:
                yield kc, value

    def to_iterator_values(self, include_empty=False):
        """
        Return iterator for traversing through the nested values of container object.

        :return: Iterator for the container values.
        """
        for key, value in self.items():
            if isinstance(value, Container) and (not include_empty or value):
                # noinspection PyCompatibility
                yield from value.to_iterator_values(include_empty)
            else:
                yield value

    def to_iterator_keys(self, key_chain='', leaf_keys_only=False, include_empty=False):
        """
        Return iterator for traversing through the nested keys of container object.

        :return: Iterator for the container elements.
        """
        for key, value in self.items():
            if leaf_keys_only:
                kc = key
            else:
                kc = key_chain + '/' + key if key_chain != '' else key
            if isinstance(value, Container) and (not include_empty or value):
                # noinspection PyCompatibility
                yield from value.to_iterator_keys(kc, leaf_keys_only, include_empty)
            else:
                yield kc

    def to_flat_list(self):
        """
        Return flat list representation of container object.

        :return: Container as flat list.
        """
        return list([item for key, item in self.to_iterator()])

    def from_flat_list(self, flat_list):
        """
        Return new container object with the same hierarchy, but with values replaced from flat list.

        :param flat_list: flat list of values to populate container with.
        :type flat_list: sequence of arrays
        :return: Container.
        """
        new_dict = dict()
        for key, value in self.items():
            if isinstance(value, Container):
                new_value = value.from_flat_list(flat_list)
            else:
                new_value = flat_list.pop(0)
            new_dict[key] = new_value
        return Container(new_dict, **self._config)

    def has_key(self, query_key):
        """
        Determine whether container object has specified key somewhere in the nested structure

        :return: Boolean
        """
        has_key = False

        def map_fn(x, kc):
            nonlocal has_key
            if query_key in kc:
                has_key = True
            return x

        self.map(map_fn)
        return has_key

    def has_key_chain(self, key_chain):
        """
        Determine whether container object has specified key-chain

        :return: Boolean
        """
        keys = re.split('[/.]', key_chain)
        ret = self
        for key in keys:
            try:
                ret = ret[key]
            except KeyError:
                return False
        return True

    def find_sub_container(self, sub_cont_to_find, partial=False):
        """
        Find the sub-container in the current container if it exsits.

        :param sub_cont_to_find: The sub-container to find.
        :type sub_cont_to_find: ivy.Container
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        :return: str
        """

        key_chain_found = False

        def _check_sub_cont(sub_cont, kc):
            sub_cont_key_chains = sub_cont_to_find.all_key_chains()
            kcs_in_sub_cont = [kc in sub_cont for kc in sub_cont_key_chains]
            if kcs_in_sub_cont and min(kcs_in_sub_cont) and \
                    ivy.Container.identical([sub_cont, sub_cont_to_find], partial=partial):
                nonlocal key_chain_found
                key_chain_found = kc
            return sub_cont

        self.map_conts(_check_sub_cont)

        return key_chain_found

    def contains_sub_container(self, sub_cont, partial=False):
        """
        Determine whether the current container contains the sub-container, with matching structure and array values.

        :param sub_cont: The sub-container to check.
        :type sub_cont: ivy.Container
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        :return: Bool
        """
        return True if isinstance(self.find_sub_container(sub_cont, partial), str) else False

    def assert_contains_sub_container(self, sub_cont, partial=False):
        """
        Asserts that the current container contains the sub-container, otherwise exception raised with the
        diff printed to screen.

        :param sub_cont: The sub-container to check.
        :type sub_cont: ivy.Container
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        """
        try:
            assert self.contains_sub_container(sub_cont, partial)
        except AssertionError:
            key_chain = self.find_sub_structure(sub_cont, check_shapes=False, partial=True)
            if not key_chain:
                key_chain = ''
            # noinspection PyTypeChecker
            raise AssertionError('Containers did not have identical structure and values:\n\n{}'.format(
                Container.diff(self[key_chain], sub_cont)))

    def find_sub_structure(self, sub_struc_to_find, check_shapes=True, partial=False):
        """
        Find the sub-container structure in the current container if it exsits.

        :param sub_struc_to_find: The sub-container to find.
        :type sub_struc_to_find: ivy.Container
        :param check_shapes: Whether to check array shapes in the sub-structure. Default is True.
        :type check_shapes: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        :return: str
        """

        key_chain_found = False

        def _check_sub_cont(sub_cont, kc):
            sub_struc_key_chains = sub_struc_to_find.all_key_chains()
            kcs_in_sub_cont = [kc in sub_cont for kc in sub_struc_key_chains]
            if kcs_in_sub_cont and min(kcs_in_sub_cont) and \
                    ivy.Container.identical_structure(
                        [sub_cont, sub_struc_to_find], check_shapes=check_shapes, partial=partial):
                nonlocal key_chain_found
                key_chain_found = kc
            return sub_cont

        self.map_conts(_check_sub_cont)

        return key_chain_found

    def contains_sub_structure(self, sub_cont, check_shapes=True, partial=False):
        """
        Determine whether the current container contains the sub-container structure.

        :param sub_cont: The sub-container to check.
        :type sub_cont: ivy.Container
        :param check_shapes: Whether to check array shapes in the sub-structure. Default is True.
        :type check_shapes: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        :return: Bool
        """
        return True if isinstance(self.find_sub_structure(sub_cont, check_shapes, partial), str) else False

    def assert_contains_sub_structure(self, sub_cont, check_shapes=True, partial=False):
        """
        Asserts that the current container contains the sub-container structure, otherwise exception raised with the
        diff printed to screen.

        :param sub_cont: The sub-container to check.
        :type sub_cont: ivy.Container
        :param check_shapes: Whether to check array shapes in the sub-structure. Default is True.
        :type check_shapes: bool, optional
        :param partial: Whether to also check for partially complete sub-containers. Default is False.
        :type partial: bool, optional
        """
        try:
            assert self.contains_sub_structure(sub_cont, check_shapes, partial)
        except AssertionError:
            key_chain = self.find_sub_structure(sub_cont, check_shapes=False, partial=True)
            if not key_chain:
                key_chain = ''
            # noinspection PyTypeChecker
            raise AssertionError('Containers did not have identical structure:\n\n{}'.format(
                Container.structural_diff(self[key_chain], sub_cont, detect_key_diffs=not partial,
                                          detect_shape_diffs=check_shapes, mode='diff_only' if partial else 'all')))

    def has_nans(self, include_infs=True, leafwise=False):
        """
        Determine whether arrays in the container contain any nans, as well as infs or -infs if specified.

        :param include_infs: Whether to include infs and -infs in the check. Default is True.
        :type include_infs: bool, optional
        :param leafwise: Whether to apply the check leaf-wise, and return a container of booleans. Default is False,
                         in which case the check is applied across the entire container, returning a single boolean.
        :type leafwise: bool, optional
        :return: Whether the container has any nans, applied either leafwise or across the entire container.
        """
        leafwise_res = self.map(lambda x, kc: ivy.has_nans(x, include_infs))
        if leafwise:
            return leafwise_res
        return max([v for k, v in leafwise_res.to_iterator()])

    def at_keys(self, queries, ignore_none=True, containing=False, ignore_key_errors=False):
        """
        Query container object at specified keys, either as list or nested dict.

        :param queries: The keys to query.
        :type queries: sequence of strs or single str
        :param ignore_none: Whether to ignore None input. Default is True.
        :type ignore_none: bool, optional
        :param containing: Whether to include keys which only contain the query substrings. Default is False.
        :type containing: bool, optional
        :param ignore_key_errors: Whether to ignore Key-errors when trying to access the dict. Default is False.
        :type ignore_key_errors: bool, optional
        :return: sub-container containing only key-chains containing the specified keys.
        """
        if queries is None and ignore_none:
            return self
        key_chains_to_keep = list()
        if isinstance(queries, str):
            queries = [queries]

        def map_fn(x, kc):
            nonlocal key_chains_to_keep
            kc_split = re.split('[/.]', kc)
            for query_key in queries:
                if query_key in kc_split or (containing and min([query_key in k for k in kc_split])):
                    key_chains_to_keep.append(kc)
            return x

        self.map(map_fn)
        return self.at_key_chains(key_chains_to_keep, ignore_key_errors=ignore_key_errors)

    def at_key_chain(self, key_chain, ignore_key_errors=False):
        """
        Query container object at a specified key-chain

        :return: sub-container or value at specified key chain
        """
        keys = re.split('[/.]', key_chain)
        ret = self
        for key in keys:
            try:
                ret = ret[key]
            except KeyError as e:
                if ignore_key_errors:
                    return
                raise e
        return ret

    def at_key_chains(self, key_chains, ignore_none=True, ignore_key_errors=False):
        """
        Query container object at specified key-chains, either as list or nested dict.

        :return: sub-container containing only the specified key chains
        """
        if key_chains is None and ignore_none:
            return self
        if isinstance(key_chains, (list, tuple)):
            return self._at_key_chains_input_as_seq(key_chains, ignore_key_errors=ignore_key_errors)
        elif isinstance(key_chains, dict):
            return self._at_key_chains_input_as_dict(key_chains, ignore_key_errors=ignore_key_errors)
        elif isinstance(key_chains, str):
            return self._at_key_chains_input_as_seq([key_chains], ignore_key_errors=ignore_key_errors)
        else:
            raise Exception('Invalid type for input key_chains, must either be a list, tuple, dict, or ivy.Container,'
                            'but found type {}'.format(type(key_chains)))

    def all_key_chains(self, include_empty=False):
        """
        Return all key-chains for the current container.
        """
        return [kc for kc, v in self.to_iterator(include_empty=include_empty)]

    def key_chains_containing(self, sub_str, include_empty=False):
        """
        Return a list of all key-chains containing a given sub-string.
        """
        return [kc for kc, v in self.to_iterator(include_empty=include_empty) if sub_str in kc]

    def set_at_keys(self, target_dict):
        """
        Set values of container object at specified keys

        :return: new container with updated value at each key
        """
        return_dict = dict()
        for key, val in self.items():
            if key in target_dict:
                return_dict[key] = target_dict[key]
            elif isinstance(val, Container):
                return_dict[key] = val.set_at_keys(target_dict)
            else:
                return_dict[key] = val
        return Container(return_dict, **self._config)

    def set_at_key_chain(self, key_chain, val, inplace=False):
        """
        Set value of container object at a specified key-chain

        :return: new container with updated value at key chain
        """
        keys = re.split('[/.]', key_chain)
        if inplace:
            cont = self
        else:
            cont = self.copy()
        sub_cont = cont
        for key in keys[:-1]:
            if key not in sub_cont:
                sub_cont[key] = Container(**self._config)
            sub_cont = sub_cont[key]
        sub_cont[keys[-1]] = val
        return cont

    def overwrite_at_key_chain(self, key_chain, val, inplace=False):
        """
        Overwrite value of container object at a specified key-chain

        :return: new container with updated value at key chain, provided it existed before.
        """
        keys = re.split('[/.]', key_chain)
        if inplace:
            cont = self
        else:
            cont = self.copy()
        sub_cont = cont
        for key in keys[:-1]:
            if key not in sub_cont:
                raise Exception('key chain must already exist in container in order to call overwrite_at_key_chain')
            sub_cont = sub_cont[key]
        if keys[-1] not in sub_cont:
            raise Exception('key chain must already exist in container in order to call overwrite_at_key_chain')
        sub_cont[keys[-1]] = val
        return cont

    def set_at_key_chains(self, target_dict, return_dict=None, inplace=False):
        """
        Set values of container object at specified key-chains

        :return: new container with updated values at the key chains
        """
        if return_dict is None:
            if inplace:
                return_dict = self
            else:
                return_dict = self.copy()
        for k, v in target_dict.items():
            if isinstance(v, dict):
                return_dict[k] = self.set_at_key_chains(v, return_dict[k], inplace)
            else:
                return_dict[k] = v
        return Container(return_dict, **self._config)

    def overwrite_at_key_chains(self, target_dict, return_dict=None, inplace=False):
        """
        Overwrite values of container object at specified key-chains

        :return: new container with updated values at the key chains, provided they existed before.
        """
        if return_dict is None:
            if inplace:
                return_dict = self
            else:
                return_dict = self.copy()
        for k, v in target_dict.items():
            if k not in return_dict:
                raise Exception('key chain must already exist in container in order to call overwrite_at_key_chains')
            if isinstance(v, dict):
                return_dict[k] = self.overwrite_at_key_chains(v, return_dict[k], inplace)
            else:
                return_dict[k] = v
        return Container(return_dict, **self._config)

    def prune_keys(self, query_keys, ignore_none=True):
        """
        Recursively prune set of keys

        :return: Container with key-chains containing the specified keys pruned.
        """
        if query_keys is None and ignore_none:
            return self
        key_chains_to_prune = list()
        if isinstance(query_keys, str):
            query_keys = [query_keys]

        def map_fn(x, kc):
            nonlocal key_chains_to_prune
            for query_key in query_keys:
                if query_key in kc:
                    key_chains_to_prune.append(kc)
            return x

        self.map(map_fn)
        return self.prune_key_chains(key_chains_to_prune)

    def prune_key_chain(self, key_chain):
        """
        Recursively prune chain of keys, specified as 'key1/key2/key3/...'

        :return: Container with keys in key chain pruned.
        """
        keys_in_chain = re.split('[/.]', key_chain)
        out_dict = dict()
        for key, value in self.items():
            if isinstance(value, Container):
                if key == keys_in_chain[0]:
                    if len(keys_in_chain) == 1:
                        new_val = []
                    else:
                        new_val = value.prune_key_chain('/'.join(keys_in_chain[1:]))
                    if len(new_val) > 0:
                        out_dict[key] = new_val
                else:
                    new_val = value.to_dict()
                    if len(new_val) > 0:
                        out_dict[key] = value.to_dict()
            else:
                if len(keys_in_chain) != 1 or key != keys_in_chain[0]:
                    out_dict[key] = value
        return Container(out_dict, **self._config)

    def prune_key_chains(self, key_chains, ignore_none=True):
        """
        Recursively prune set of key chains

        :return: Container with keys in the set of key chains pruned.
        """
        if key_chains is None and ignore_none:
            return self
        if isinstance(key_chains, (list, tuple)):
            return self._prune_key_chains_input_as_seq(key_chains)
        elif isinstance(key_chains, dict):
            return self._prune_key_chains_input_as_dict(key_chains)
        elif isinstance(key_chains, str):
            return self._prune_key_chains_input_as_seq([key_chains])
        else:
            raise Exception('Invalid type for input key_chains, must either be a list, tuple, dict, or ivy.Container,'
                            'but found type {}'.format(type(key_chains)))

    def format_key_chains(self, format_fn):
        """
        Format all key-chains, using the formatting function

        :return: Container with the same key-chain structure, but the key strings formatted.
        """
        return ivy.Container({format_fn(k): v for k, v in self.to_iterator()})

    def sort_by_key(self):
        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, Container):
                v_back = v.sort_by_key()
            else:
                v_back = v
            new_dict[k] = v_back
        return Container(new_dict, **self._config)

    def prune_empty(self, keep_nones=False, base=True):
        """
        Recursively prunes empty keys from the container dict structure.
        Returns None if the entire container is empty.

        :return: Container with empty keys pruned.
        """
        out_dict = dict()
        for key, value in self.items():
            if isinstance(value, Container):
                new_value = value.prune_empty(keep_nones, False)
                if new_value:
                    out_dict[key] = new_value
            elif self._ivy.exists(value) or keep_nones:
                out_dict[key] = value
        if len(out_dict):
            return Container(out_dict, **self._config)
        if base:
            return Container(**self._config)
        return

    def prune_key_from_key_chains(self, absolute=None, containing=None):
        """
        Recursively prune absolute key or key containing a certain substring from all key chains.

        :param absolute: The absolute key to detect in the key chains.
        :type absolute: str, optional
        :param containing: A substring to check each key for, when deciding which keys to prune.
        :type containing: str, optional
        :return: Container with specified key or substring-containing-key from all key chains removed from the chain.
        """
        if not absolute and not containing:
            raise Exception('At least one of absolute or containing arguments must be specified.')
        out_cont = Container(**self._config)
        for key, value in self.items():
            if (absolute and key == absolute) or (containing and containing in key):
                if isinstance(value, Container):
                    out_cont = Container.combine(out_cont, value)
                else:
                    out_cont = value
            elif isinstance(value, Container):
                out_cont[key] = value.prune_key_from_key_chains(absolute, containing)
            else:
                out_cont[key] = value
        return out_cont

    def prune_keys_from_key_chains(self, absolute=None, containing=None):
        """
        Recursively prune absolute keys or keys containing certain substrings from all key chains.

        :param absolute: The absolute key to detect in the key chains.
        :type absolute: sequence of strs, optional
        :param containing: A substring to check each key for, when deciding which keys to prune.
        :type containing: sequence of strs, optional
        :return: Container with specified keys or substring-containing-keys from all key chains removed from the chain.
        """
        if not absolute and not containing:
            raise Exception('At least one of absolute or containing arguments must be specified.')
        out_cont = Container(**self._config)
        for key, value in self.items():
            if (absolute and key in absolute) or (containing and max([con in key for con in containing])):
                if isinstance(value, Container):
                    out_cont = Container.combine(out_cont, value)
                else:
                    out_cont = value
            elif isinstance(value, Container):
                out_cont[key] = value.prune_key_from_key_chains(absolute, containing)
            else:
                out_cont[key] = value
        return out_cont

    def restructure_key_chains(self, keychain_mapping, keep_orig=True, replace=True):
        """
        Create a new container with the same contents, but a new key-chain structure. Given by the mapping with keys as
        old key-chains and values as new key-chains.

        :param keychain_mapping: A dict with keys as old key-chains and values as new key-chains.
        :type keychain_mapping: dict
        :param keep_orig: Whether to keep the original keys, or start from a new empty container. Default is True.
        :type keep_orig: bool, optional
        :param replace: Whether to replace the old key-chains by the new ones. Default is True.
        :type replace: bool, optional
        """
        new_cont = self.copy() if keep_orig else ivy.Container()
        for old_kc, new_kc in keychain_mapping.items():
            if replace and old_kc in new_cont:
                new_cont = new_cont.prune_key_chain(old_kc)
            new_cont = ivy.Container.combine(new_cont, ivy.Container({new_kc: self[old_kc]}))
        return new_cont

    def restructure(self, mapping, keep_orig=True, replace=True):
        """
        Create a new container with the same contents, but a new key-chain structure, and transposes and/or reshaped
        arrays. Given by the mapping with keys as old key-chains and values as new key-chains.

        :param mapping: A dict with keys as old key-chains and values as new key-chains.
        :type mapping: dict
        :param keep_orig: Whether to keep the original keys, are start from a new container. Default is True.
        :type keep_orig: bool, optional
        :param replace: Whether to replace the old key-chains by the new ones. Default is True.
        :type replace: bool, optional
        """
        new_cont = self.copy() if keep_orig else ivy.Container()
        for old_kc, new in mapping.items():
            if replace and old_kc in new_cont:
                new_cont = new_cont.prune_key_chain(old_kc)
            val = self[old_kc]
            if isinstance(new, dict):
                new_kc = new['key_chain']
                if 'pattern' in new:
                    pattern = new['pattern']
                    axes_lengths = new['axes_lengths'] if 'axes_lengths' in new else {}
                    if isinstance(val, Container):
                        val = val.einops_rearrange(pattern, **axes_lengths)
                    else:
                        val = ivy.einops_rearrange(val, pattern, **axes_lengths)
            else:
                new_kc = new
            new_cont = ivy.Container.combine(new_cont, ivy.Container({new_kc: val}))
        return new_cont

    def flatten_key_chains(self, include_empty=False, above_height=None, below_depth=None):
        """
        Return a flat (depth-1) container, which all nested key-chains flattened.
        """
        return Container({Container.flatten_key_chain(kc, above_height=above_height, below_depth=below_depth): v
                          for kc, v in self.to_iterator(include_empty=include_empty)}, **self._config)

    def copy(self):
        """
        Create a copy of this container.

        :return: A copy of the container
        """
        return Container(self.to_dict(), **self._config)

    def deep_copy(self):
        """
        Create a deep copy (copying all internal tensors) of this container.

        :return: A deep copy of the container
        """
        return self.map(lambda x, kc: ivy.copy_array(x) if ivy.is_array(x) else x)

    def map(self, func, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False, inplace=False,
            key_chain=''):
        """
        Apply function to all array values of container

        :param func: Function to apply to each container entry
        :type func: python function
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param map_sequences: Whether to also map method to sequences (lists, tuples). Default is False.
        :type map_sequences: bool, optional
        :param inplace: Whether to apply the mapping inplace, or return a new container. Default is False.
        :type inplace: bool, optional
        :param map_sequences: Whether to also map to sequences (lists and tuples). Default is False.
        :type map_sequences: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :return: New container following the function mapped to each sub-array.
        """
        return_dict = self if inplace else dict()
        for key, value in self.items():
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.map(func, key_chains, to_apply, prune_unapplied, map_sequences, inplace, this_key_chain)
                if prune_unapplied and not ret:
                    continue
                if not inplace:
                    return_dict[key] = ret
            elif isinstance(value, (list, tuple)) and map_sequences:
                ret = ivy.nested_map(value, lambda x: func(x, None), True)
                if prune_unapplied and not ret:
                    continue
                return_dict[key] = ret
            else:
                if key_chains is not None:
                    if (this_key_chain in key_chains and not to_apply) or (
                            this_key_chain not in key_chains and to_apply):
                        if prune_unapplied:
                            continue
                        return_dict[key] = value
                        continue
                return_dict[key] = func(value, this_key_chain)
        if inplace:
            return
        return Container(return_dict, **self._config)

    def map_conts(self, func, key_chains=None, to_apply=True, prune_unapplied=False, inplace=False, key_chain='',
                  include_self=True):
        """
        Apply function to all sub-contains in the container.

        :param func: Function to apply to each sub-container
        :type func: python function
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :param inplace: Whether to apply the mapping inplace, or return a new container. Default is False.
        :type inplace: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :param include_self: Whether to also apply the (possiby in-place) function to this container. Default is True.
        :type include_self: bool, optional
        :return: New container following the function mapped to each sub-container.
        """
        return_dict = self if inplace else dict()
        for key, value in self.items():
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.map_conts(func, key_chains, to_apply, prune_unapplied, inplace, this_key_chain)
                if prune_unapplied and not ret:
                    continue
                if not inplace:
                    return_dict[key] = ret
            else:
                if key_chains is not None and ((this_key_chain in key_chains and not to_apply) or (
                        this_key_chain not in key_chains and to_apply)) and prune_unapplied:
                    continue
                return_dict[key] = value
        ret = return_dict if inplace else Container(return_dict, **self._config)
        if key_chain != '' or include_self:
            ret = func(ret, key_chain)
        if inplace:
            return
        return ret

    def dtype(self):
        """
        Return container, with all entries replaced with their data types.

        :return: New datatype container
        """
        return self.map(lambda x, _: self._ivy.dtype(x))

    def with_entries_as_lists(self):
        """
        Return container object, with each array entry in the container cast to a list
        """
        def to_list(x, _=''):
            try:
                return self._ivy.to_list(x)
            except (AttributeError, ValueError):
                return x
        return self.map(to_list)

    def reshape_like(self, target_dict, leading_shape=None, return_cont=None):
        """
        Set shapes of container entries to shapes specified by new container with the same key structure

        :return: new container with values of updated shapes
        """
        leading_shape = self._ivy.default(leading_shape, list())
        if return_cont is None:
            return_cont = self.copy()
        for (_, v_shape), (k, v) in zip(target_dict.items(), return_cont.items()):
            if isinstance(v_shape, dict):
                return_cont[k] = self.reshape_like(v_shape, leading_shape, return_cont[k])
            else:
                return_cont[k] = self._ivy.reshape(v, leading_shape + list(v_shape))
        return Container(return_cont, **self._config)

    def create_if_absent(self, key, value, inplace=True):
        """
        Add a key to the container with corresponding value, if it is not already present. otherwise, do nothing.
        """
        if key in self:
            return
        self.set_at_key_chain(key, value, inplace)

    def if_exists(self, key):
        """
        Returns the sub-container at the following key if it exists, otherwise None.
        """
        try:
            return self[key]
        except KeyError:
            return

    def try_kc(self, key):
        """
        Tries the following key or key chain, returning self if not present.
        """
        try:
            return self[key]
        except KeyError:
            return self

    def cutoff_at_depth(self, depth_cutoff, inplace=False):
        total_depth = self.depth
        copy = self.copy()
        def _maybe_cutoff(cont, kc):
            if total_depth - copy[kc].depth < depth_cutoff:
                return cont
            if inplace:
                cont.clear()
            return Container()
        ret = self.map_conts(_maybe_cutoff, inplace=inplace)
        if inplace:
            return
        return ret

    def cutoff_at_height(self, height_cutoff, inplace=False):
        copy = self.copy()
        def _maybe_cutoff(cont, kc):
            if copy[kc].depth > height_cutoff:
                return cont
            if inplace:
                cont.clear()
            return Container()
        ret = self.map_conts(_maybe_cutoff, inplace=inplace)
        if inplace:
            return
        return ret

    def _slice_keys(self, key_slice):
        keys = list(self.keys())
        if isinstance(key_slice, str):
            assert len(key_slice) == 3 and key_slice[1] == ':'
            assert self._alphabetical_keys
            start_char = key_slice[0]
            end_char = key_slice[2]
            start_idx = min([i for i, k in enumerate(keys) if k[0] == start_char])
            end_idx = max([i for i, k in enumerate(keys) if k[0] == end_char]) + 1
            key_slice = slice(start_idx, end_idx, 1)
        ret = self.copy()
        desired_keys = keys[key_slice]
        # noinspection PyUnresolvedReferences
        return ret.at_key_chains(desired_keys)

    def slice_keys(self, key_slice, all_depths=False):
        top_depth = self.depth
        if all_depths:
            if isinstance(key_slice, dict):
                first_slice = list(key_slice.values())[0]
                for d in range(0, top_depth+1):
                    if d not in key_slice:
                        key_slice[d] = first_slice
            else:
                key_slice = {d: key_slice for d in range(0, top_depth+1)}
        if isinstance(key_slice, dict):
            def _fn(cont, kc):
                depth = 0 if kc == '' else len(kc.split('/'))
                if depth in key_slice:
                    # noinspection PyProtectedMember
                    return cont._slice_keys(key_slice[depth])
                return cont
            return self.map_conts(_fn)
        return self._slice_keys(key_slice)

    def with_print_limit(self, print_limit, inplace=False):
        def _update_print_limit(cont, _):
            cont._print_limit = print_limit
            return cont
        ret = self.map_conts(_update_print_limit, inplace=inplace)
        if inplace:
            return
        return ret

    # noinspection PyTypeChecker
    def remove_print_limit(self, inplace=False):
        return self.with_print_limit(None, inplace)

    def with_key_length_limit(self, key_length_limit, inplace=False):
        def _update_key_length_limit(cont, _):
            cont._key_length_limit = key_length_limit
            return cont
        ret = self.map_conts(_update_key_length_limit, inplace=inplace)
        if inplace:
            return
        return ret

    def remove_key_length_limit(self, inplace=False):
        return self.with_key_length_limit(None, inplace)

    def with_print_indent(self, print_indent, inplace=False):
        def _update_print_indent(cont, _):
            cont._print_indent = print_indent
            return cont
        ret = self.map_conts(_update_print_indent, inplace=inplace)
        if inplace:
            return
        return ret

    def with_print_line_spacing(self, print_line_spacing, inplace=False):
        def _update_print_line_spacing(cont, _):
            cont._print_line_spacing = print_line_spacing
            return cont
        ret = self.map_conts(_update_print_line_spacing, inplace=inplace)
        if inplace:
            return
        return ret

    def with_default_key_color(self, default_key_color, inplace=False):
        def _update_default_key_color(cont, _):
            cont._default_key_color = default_key_color
            return cont
        ret = self.map_conts(_update_default_key_color, inplace=inplace)
        if inplace:
            return
        return ret

    def with_ivy_backend(self, ivy_backend):
        return Container(self, ivyh=ivy_backend)

    def set_ivy_backend(self, ivy_backend):
        self._local_ivy = ivy_backend

    def start_logging_retrieval_times(self, inplace=True):
        def _flag_logging_retrieval(cont, _):
            cont._logging_retrieval_times = True
            return cont
        ret = self.map_conts(_flag_logging_retrieval, inplace=inplace)
        if inplace:
            return
        return ret

    def stop_logging_retrieval_times(self, inplace=True):
        def _flag_logging_retrieval(cont, _):
            cont._logging_retrieval_times = False
            return cont
        ret = self.map_conts(_flag_logging_retrieval, inplace=inplace)
        if inplace:
            return
        return ret

    def show(self):
        print(self)

    # noinspection PyUnresolvedReferences
    def show_sub_container(self, sub_cont_or_keychain):

        # copy this container
        this_cont = self.copy()

        # get the sub-container
        if isinstance(sub_cont_or_keychain, str):
            sub_cont = self.at_key_chain(sub_cont_or_keychain)
        else:
            sub_cont = sub_cont_or_keychain

        # find the key chain of the sub-container
        sub_cont_kc = self.find_sub_container(sub_cont)

        # show this container if key-chain not found, and return
        if not sub_cont_kc:
            print(self)
            return

        # otherwise, replace sub-container in this container with known key
        this_cont[sub_cont_kc] = ivy.Container({'SUB_CONT': None})

        # get the formatted reprs
        this_repr = this_cont.with_default_key_color('green').__repr__()
        this_repr_red = this_cont.with_default_key_color('red').__repr__()
        this_repr_stripped = ansi_escape.sub('', this_repr)
        sub_repr = sub_cont.with_default_key_color('red').__repr__()

        # remove the outer brackets from the sub repr
        sub_repr = '\n' + '\n'.join(sub_repr.split('\n')[1:-1]) + '\n'

        # find the sub-container placeholder
        idx = this_repr_stripped.find('SUB_CONT: null')

        # count the lines above and below the sub-container
        num_lines_above = this_repr_stripped[0:idx].count('\n')
        num_lines_below = this_repr_stripped[idx:].count('\n')

        # get the str reprs above and below
        this_repr_split = this_repr.split('\n')
        this_repr_red_split = this_repr_red.split('\n')
        this_repr_above = '\n'.join(this_repr_split[0:num_lines_above-1] + [this_repr_red_split[num_lines_above-1]])
        this_repr_below = '\n'.join(this_repr_split[-num_lines_below:])

        # count the number of lines needed to be prepended to the sub-container repr
        cur_num_spaces = 0
        for i, s in enumerate(sub_repr[1:]):
            if s != ' ':
                break
            cur_num_spaces += 1
        exp_num_spaces = 0
        for i, s in enumerate(this_repr.split('\n')[num_lines_above]):
            if s != ' ':
                break
            exp_num_spaces += 1
        num_spaces_to_add = exp_num_spaces - cur_num_spaces

        # prepend these lines to the sub-container
        sub_repr = '\n' + '\n'.join([' '*num_spaces_to_add + s for s in sub_repr[1:-1].split('\n')]) + '\n'

        # show
        print(this_repr_above + sub_repr + this_repr_below)

    # Built-ins #
    # ----------#

    def __repr__(self, as_repr=True):

        indent_str = ' '*self._print_indent

        def _align_array(array_str_in):
            array_str_in_split = array_str_in.split('([')
            leading_str_to_keep = array_str_in_split[0].replace('\\n', '')
            indented_key_size = len(leading_str_to_keep.replace('"', '').split(': ')[0])
            indented_key_str = ' '*(indented_key_size+2)
            padded = False

            def _pre_pad_alpha_line(str_in):
                nonlocal padded
                padded = True
                return '\\n' + indent_str + indented_key_str + str_in

            leading_str_to_keep = ', '.join([_pre_pad_alpha_line(s) if s[0].isalpha() and i != 0 else s
                                             for i, s in enumerate(leading_str_to_keep.split(', '))])
            local_indent_str = '' if padded else indent_str
            leading_str = leading_str_to_keep.split('\\n')[-1].replace('"', '')
            remaining_str = array_str_in_split[1]
            num_extra_dims = 0
            for i, char in enumerate(remaining_str):
                if char != '[':
                    num_extra_dims = i
                    break
            extra_indent = (len(leading_str) + 1 + num_extra_dims) * ' '
            array_str_in = '(['.join([leading_str_to_keep, remaining_str])
            uniform_indent_wo_overflow = array_str_in.replace('\\n[', '\n' + local_indent_str + extra_indent + '[')
            uniform_indent = '\n'.join([local_indent_str + extra_indent + ' ' + s
                                        if (s[0].isnumeric() or s[0] == '-' or s[0:3] == '...' or
                                            max([ss in s[0:6] for ss in ['nan, ', 'inf, ']])) else
                                        (indent_str + indented_key_str + s
                                         if (not s[0].isspace() and s[0] != '"')
                                         else s)
                                        for s in uniform_indent_wo_overflow.split('\\n')])
            indented = uniform_indent
            # 10 dimensions is a sensible upper bound for the number in a single array
            for i in range(2, 10):
                indented = indented.replace(' '*(i-1) + '['*i, '['*i)
                indented = '\n'.join([s for s in indented.split('\n') if bool(s) and not s.isspace()])
            return indented

        def _align_arrays(str_in):
            chunks = str_in.split('\n' + indent_str)
            aligned_array_chunks = {i: _align_array(c) for i, c in enumerate(chunks) if '\\n' in c}
            chunks = [aligned_array_chunks[i] if i in aligned_array_chunks else c_orig
                      for i, c_orig in enumerate(chunks)]
            return ('\n' + indent_str).join(chunks)

        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, Container):
                # noinspection PyArgumentList
                rep = v.__repr__(as_repr=False)
            else:
                if self._ivy.is_array(v) and len(list(v.shape)) > 0 and ivy.exists(self._print_limit) and \
                        _reduce(_mul, v.shape) > self._print_limit:
                    rep = (type(v), "shape=", list(v.shape))
                elif isinstance(v, (list, tuple)) and v and self._ivy.is_array(v[0]):
                    rep = ("list[{}]".format(len(v)), type(v[0]), "shape=", list(v[0].shape))
                else:
                    rep = v
            new_dict[k] = rep
        if as_repr:
            json_dumped_str = _align_arrays(_json.dumps(
                Container(new_dict, **self._config).map(
                    lambda x, kc: x if _is_jsonable(x)
                    else _repr(x).replace(' ', '').replace(',', ', ')).to_dict(),
                indent=self._print_indent))

            def _add_newline(str_in):
                str_in_split = str_in.split('\n')
                str_split_size = len(str_in_split)
                return '\n'.join([('\n'*self._print_line_spacing + ss) if i == (str_split_size-1) else ss
                                  for i, ss in enumerate(str_in_split)])

            json_dumped_str = '":'.join([_add_newline(s) for s in json_dumped_str.split('":')])
            # improve tf formatting
            if ivy.framework_stack and ivy.current_framework_str() == 'tensorflow':
                json_dumped_str_split = json_dumped_str.split("\'Variable:")
                json_dumped_str = json_dumped_str_split[0] + ', ' + ', '.join(["\'".join(ss.split("\'")[1:])
                                                                               for ss in json_dumped_str_split[1:]])
                json_dumped_str = json_dumped_str.replace(':shape', ', shape').replace(')dtype=', '), dtype=').replace(
                    ', ),', ',),')
            # color keys
            json_dumped_str_split = json_dumped_str.split('":')
            split_size = len(json_dumped_str_split)
            json_dumped_str =\
                '":'.join([' "'.join(sub_str.split(' "')[:-1] +
                                     [termcolor.colored(
                                         Container.trim_key(sub_str.split(' "')[-1], self._key_length_limit),
                                         self._default_key_color)])
                           if i < split_size - 1 else sub_str
                           for i, sub_str in enumerate(json_dumped_str_split)])
            # remove quotation marks, shape tuple, and color other elements of the dict
            ret = json_dumped_str.replace('"', '').replace(", 'shape=', [", " shape=[").replace(
                ':', termcolor.colored(':', 'magenta')).replace('{', termcolor.colored('{', 'blue')).replace(
                '}', termcolor.colored('}', 'blue')).replace('shape=', termcolor.colored('shape=', 'magenta')).replace(
                'device=', termcolor.colored('device=', 'magenta')).replace("<class'", "<class '").replace(
                "'", "").replace('<class', '<' + termcolor.colored('class', 'blue'))
            # ToDo: make the solution below more elegant
            for i in range(10):
                ret = ret.replace('diff_{}'.format(i), termcolor.colored('diff_{}'.format(i), 'red'))
            for keyword, color in self._keyword_color_dict.items():
                ret = ret.replace(keyword, termcolor.colored(keyword, color))
            return ret
        return new_dict

    def __dir__(self):
        return list(super.__dir__(self)) + list(self.keys())

    def _log_retrieval(self, item, ret):
        if self._logging_retrieval_times:
            global base_cont
            if not ivy.exists(base_cont):
                base_cont = self
            global retrieval_key_chain
            retrieval_key_chain.append(item)
            if isinstance(ret, Container):
                return ret
            rkc = '__'.join(retrieval_key_chain)
            retrieval_key_chain.clear()
            if rkc not in base_cont._retrieval_times:
                base_cont._retrieval_times[rkc] = list()
            base_cont._retrieval_times[rkc].append(time.perf_counter())
            base_cont = None

    # noinspection PyProtectedMember
    def __getattr__(self, item):
        try:
            ret = dict.__getitem__(self, item)
        except KeyError:
            # noinspection PyUnresolvedReferences
            ret = super.__getattr__(item)
        self._log_retrieval(item, ret)
        return ret

    def __setattr__(self, name, value):
        if name[0] != '_':
            self[name] = value
        else:
            super.__setattr__(self, name, value)

    def _get_queue_item(self, query):
        if isinstance(query, int):
            queue_queries = [query]
        elif isinstance(query, slice):
            queue_queries = list(range(query.start, query.stop, ivy.default(query.step, 1)))
        elif isinstance(query, (list, tuple)):
            queue_queries = list(range(query[0].start, query[0].stop, ivy.default(query[0].step, 1)))
        else:
            raise Exception('Invalid slice type, must be one of integer, slice, or sequences of slices.')
        queue_idxs = set([_np.sum(q >= self._queue_load_sizes_cum).item() for q in queue_queries])
        conts = list()
        for i in queue_idxs:
            if i not in self._loaded_containers_from_queues:
                cont = Container(self._queues[i].get(timeout=self._queue_timeout), **self._config)
                if ivy.wrapped_mode():
                    cont = cont.to_ivy()
                self._loaded_containers_from_queues[i] = cont
            else:
                cont = self._loaded_containers_from_queues[i]
            conts.append(cont)
        combined_cont = self._container_combine_method(conts)
        idx = list(queue_idxs)[0]
        offset = 0 if idx == 0 else self._queue_load_sizes_cum[idx - 1]
        if isinstance(query, int):
            shifted_query = query - offset
        elif isinstance(query, slice):
            shifted_query = slice(query.start-offset, query.stop-offset, query.step)
        elif isinstance(query, (list, tuple)):
            shifted_query = tuple([slice(slc.start-offset, slc.stop-offset, slc.step) for slc in query])
        # noinspection PyUnboundLocalVariable
        return combined_cont[shifted_query]

    def __getitem__(self, query):
        """
        Get slice, key or key chain of container object.

        :param query: slice object, key or key chain to query all container elements.
        :type query: slice or str
        :return: Container object at desired query.
        """
        if isinstance(query, str):
            if query == '':
                return self
            if '/' in query or '.' in query:
                ret = self.at_key_chain(query)
                return ret
            ret = dict.__getitem__(self, query)
            self._log_retrieval(query, ret)
            return ret
        elif ivy.exists(self._queues):
            ret = self._get_queue_item(query)
            return ret
        return_dict = dict()
        for key, value in self.items():
            if isinstance(value, Container):
                return_dict[key] = value[query]
            else:
                # noinspection PyBroadException
                if isinstance(value, list) or isinstance(value, tuple):
                    if len(value) == 0:
                        return_dict[key] = value
                    else:
                        return_dict[key] = value[query]
                elif value is None or hasattr(value, 'shape') and value.shape == ():
                    return_dict[key] = value
                else:
                    return_dict[key] = value[query]
        ret = Container(return_dict, **self._config)
        return ret

    def __setitem__(self, query, val):
        """
        Set key or key chain of container object.

        :param query: slice object, key or key chain at which to set all container elements.
        :type query: slice or str
        :param val: The value to set at the desired query.
        :type val: ivy.Container, array, or other
        :return: New container after updating.
        """
        if isinstance(query, str) and ('/' in query or '.' in query):
            return self.set_at_key_chain(query, val, inplace=True)
        else:
            return dict.__setitem__(self, query, val)

    def __contains__(self, key):
        if isinstance(key, str) and ('/' in key or '.' in key):
            return self.has_key_chain(key)
        elif isinstance(key, Container):
            return self.contains_sub_container(key)
        else:
            return dict.__contains__(self, key)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.map(lambda x, kc: -x)

    def __pow__(self, power):
        if isinstance(power, Container):
            return self.reduce([self, power], lambda x: _reduce(_pow, x))
        return self.map(lambda x, kc: x ** power)

    def __rpow__(self, power):
        return self.map(lambda x, kc: power ** x)

    def __add__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], sum)
        return self.map(lambda x, kc: x + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, -other], sum)
        return self.map(lambda x, kc: x - other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_mul, x))
        return self.map(lambda x, kc: x * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_truediv, x))
        return self.map(lambda x, kc: x / other)

    def __rtruediv__(self, other):
        return self.map(lambda x, kc: other / x)

    def __floordiv__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_floordiv, x))
        return self.map(lambda x, kc: x // other)

    def __rfloordiv__(self, other):
        return self.map(lambda x, kc: other // x)

    def __abs__(self):
        return self.map(lambda x, kc: self._ivy.abs(x))

    def __lt__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_lt, x))
        return self.map(lambda x, kc: x < other)

    def __le__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_le, x))
        return self.map(lambda x, kc: x <= other)

    def __eq__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_eq, x))
        return self.map(lambda x, kc: x == other)

    def __ne__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_ne, x))
        return self.map(lambda x, kc: x != other)

    def __gt__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_gt, x))
        return self.map(lambda x, kc: x > other)

    def __ge__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: _reduce(_ge, x))
        return self.map(lambda x, kc: x >= other)

    def __and__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: x[0] and x[1])
        return self.map(lambda x, kc: x and other)

    def __rand__(self, other):
        return self.map(lambda x, kc: other and x)

    def __or__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: x[0] or x[1])
        return self.map(lambda x, kc: x or other)

    def __ror__(self, other):
        return self.map(lambda x, kc: other or x)

    def __invert__(self):
        return self.map(lambda x, kc: _not(x))

    def __xor__(self, other):
        if isinstance(other, Container):
            return self.reduce([self, other], lambda x: x[0] != x[1])
        return self.map(lambda x, kc: x != other)

    def __rxor__(self, other):
        return self.map(lambda x, kc: other != x)

    def __getstate__(self):
        state_dict = copy.copy(self.__dict__)
        state_dict['_local_ivy'] = ivy.try_else_none(lambda: state_dict['_local_ivy'].current_framework_str())
        config_in = copy.copy(state_dict['_config_in'])
        config_in['ivyh'] = ivy.try_else_none(lambda: config_in['ivyh'].current_framework_str())
        state_dict['_config_in'] = config_in
        config = copy.copy(state_dict['_config'])
        config['ivyh'] = ivy.try_else_none(lambda: config['ivyh'].current_framework_str())
        state_dict['_config'] = config
        return state_dict

    def __setstate__(self, state_dict):
        if '_local_ivy' in state_dict:
            if ivy.exists(state_dict['_local_ivy']):
                state_dict['_local_ivy'] = ivy.get_framework(state_dict['_local_ivy'])
        if '_config_in' in state_dict:
            config_in = copy.copy(state_dict['_config_in'])
            if 'ivyh' in config_in:
                if ivy.exists(config_in['ivyh']):
                    config_in['ivyh'] = ivy.get_framework(config_in['ivyh'])
            state_dict['_config_in'] = config_in
        if '_config' in state_dict:
            config = copy.copy(state_dict['_config'])
            if 'ivyh' in config:
                if ivy.exists(config['ivyh']):
                    config['ivyh'] = ivy.get_framework(config['ivyh'])
            state_dict['_config'] = config
        self.__dict__.update(state_dict)

    # Getters and Setters #
    # --------------------#

    # private

    @property
    def _ivy(self):
        return ivy.default(self._local_ivy, ivy)

    @_ivy.setter
    def _ivy(self, local_ivy):
        self._local_ivy = local_ivy

    # public

    @property
    def shape(self):
        """
        The shape of the arrays in the container, with None placed in indices which are not consistent across arrays
        """
        return self._get_shape()

    @property
    def shapes(self):
        """
        The shapes of each array in the container, with None placed in leaf entries without a shape attribute.
        """
        return self._get_shapes()

    @property
    def dev_str(self):
        """
        The device to which the arrays in the container belong, with None returned if the devices are not consistent
        """
        return self._get_dev_str()

    @property
    def ivy(self):
        return self._ivy

    @property
    def config(self):
        return self._config

    @property
    def depth(self):
        kcs = [kc for kc in self.to_iterator_keys(include_empty=True)]
        if not kcs:
            return 0
        return max([len(kc.split('/')) for kc in kcs])


class MultiDevContainer(Container):

    def __init__(self, dict_in, dev_strs, queues=None, queue_load_sizes=None, container_combine_method='list_join',
                 queue_timeout=None, print_limit=10, print_indent=4, print_line_spacing=0, ivyh=None,
                 keyword_color_dict=None, rebuild_child_containers=False, **kwargs):
        super().__init__(dict_in, queues, queue_load_sizes, container_combine_method, queue_timeout, print_limit,
                         print_indent, print_line_spacing, ivyh, keyword_color_dict, rebuild_child_containers, **kwargs)
        self._dev_strs = dev_strs
        self._num_devs = len(dev_strs)

    def at_dev(self, dev_str):
        return self.map(lambda x, kc: x[dev_str] if isinstance(x, ivy.MultiDevItem) else x)

    def at_devs(self):
        return {ds: self.at_dev(ds) for ds in self._dev_strs}
