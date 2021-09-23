"""
Base Container Object
"""

# global
import re
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
from operator import truediv as _truediv
from operator import floordiv as _floordiv

# local
import ivy as _ivy

INF = float('inf')


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
                 queue_timeout=5.0, print_limit=10, print_indent=4, print_line_spacing=0, ivyh=None,
                 keyword_color_dict=None, rebuild_child_containers=False, **kwargs):
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
        :param queue_timeout: The timeout when waiting for containers to arrive from the queues. Default is 5 seconds.
        :type queue_timeout: float, optional
        :param print_limit: The total array size limit when printing the container. Default is 10.
        :type print_limit: int, optional
        :param print_indent: The number of whitespaces to use for indenting when printing the container. Default is 4.
        :type print_indent: int, optional
        :param print_line_spacing: The number of extra newlines to use between keys when printing the container.
                                   Default is 0.
        :type print_line_spacing: int, optional
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :param keyword_color_dict: A dict mapping keywords to their termcolor color codes for printing the container.
        :type keyword_color_dict: dict, optional
        :param rebuild_child_containers: Whether to rebuild container found in dict_in with these constructor params.
                                         Default is False, in which case the original container are kept as are.
        :type rebuild_child_containers: bool, optional
        :param kwargs: keyword arguments for dict creation. Default is None.
        :type kwargs: keyword arguments.
        """
        self._queues = queues
        self._print_limit = print_limit
        self._print_indent = print_indent
        self._print_line_spacing = print_line_spacing
        self._container_combine_method = container_combine_method
        if _ivy.exists(self._queues):
            if isinstance(self._container_combine_method, str):
                self._container_combine_method =\
                    {'list_join': self.list_join,
                     'concat': lambda conts: self.concat(conts, 0)}[self._container_combine_method]
            self._loaded_containers_from_queues = dict()
            self._queue_load_sizes_cum = _np.cumsum(queue_load_sizes)
            self._queue_timeout = queue_timeout
        self._local_ivy = ivyh
        self._keyword_color_dict = _ivy.default(keyword_color_dict, {})
        if dict_in is None:
            if kwargs:
                dict_in = dict(**kwargs)
            else:
                dict_in = dict()
        elif kwargs:
            raise Exception('dict_in and **kwargs cannot both be specified for ivy.Container constructor,'
                            'please specify one or the other, not both.')
        dict_in = dict_in if isinstance(dict_in, dict) else dict(dict_in)
        for key, value in sorted(dict_in.items()):
            if isinstance(value, dict) and (not isinstance(value, Container) or rebuild_child_containers):
                self[key] = Container(value,
                                      container_combine_method=container_combine_method,
                                      print_limit=print_limit,
                                      print_indent=print_indent,
                                      print_line_spacing=print_line_spacing,
                                      ivyh=ivyh,
                                      keyword_color_dict=keyword_color_dict,
                                      rebuild_child_containers=rebuild_child_containers)
            else:
                self[key] = value

    # Class Methods #
    # --------------#

    @staticmethod
    def list_join(containers, ivyh=None):
        """
        Join containers of lists together along the specified dimension.

        :param containers: containers to list join
        :type containers: sequence of Container objects
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: List joined containers, with each entry being a list of arrays
        """

        container0 = containers[0]

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                new_list = list()
                for container in containers:
                    new_list.append(container[key])
                return_dict[key] = Container.list_join(new_list, ivyh)
            return Container(return_dict, ivyh=ivyh)
        else:
            return [item for sublist in containers for item in sublist]

    @staticmethod
    def list_stack(containers, dim, ivyh=None):
        """
        List stack containers together along the specified dimension.

        :param containers: containers to list stack
        :type containers: sequence of Container objects
        :param dim: dimension along which to list stack
        :type dim: int
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Stacked containers, with each entry being a list of arrays
        """

        container0 = containers[0]

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.list_stack([container[key] for container in containers], dim, ivyh)
            return Container(return_dict, ivyh=ivyh)
        else:
            return containers

    @staticmethod
    def concat(containers, dim, ivyh=None):
        """
        Concatenate containers together along the specified dimension.

        :param containers: containers to concatenate
        :type containers: sequence of Container objects
        :param dim: dimension along which to concatenate
        :type dim: int
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Concatenated containers
        """

        container0 = containers[0]

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.concat([container[key] for container in containers], dim, ivyh)
            return Container(return_dict, ivyh=ivyh)
        else:
            ivyh = _ivy.default(ivyh, _ivy)
            # noinspection PyBroadException
            try:
                if len(containers[0].shape) == 0:
                    return ivyh.concatenate([ivyh.reshape(item, [1] * (dim + 1)) for item in containers], dim)
                else:
                    return ivyh.concatenate(containers, dim)
            except Exception as e:
                raise Exception(str(e) + '\nContainer concat operation only valid for containers of arrays')

    @staticmethod
    def stack(containers, dim, ivyh=None):
        """
        Stack containers together along the specified dimension.

        :param containers: containers to stack
        :type containers: sequence of Container objects
        :param dim: dimension along which to stack
        :type dim: int
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Stacked containers
        """

        container0 = containers[0]

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.stack([container[key] for container in containers], dim, ivyh)
            return Container(return_dict, ivyh=ivyh)
        else:
            ivyh = _ivy.default(ivyh, _ivy)
            # noinspection PyBroadException
            try:
                if len(containers[0].shape) == 0:
                    return ivyh.stack([ivyh.reshape(item, [1] * (dim + 1)) for item in containers], dim)
                else:
                    return ivyh.stack(containers, dim)
            except Exception as e:
                raise Exception(str(e) + '\nContainer stack operation only valid for containers of arrays')

    @staticmethod
    def combine(*containers, ivyh=None):
        """
        Combine keys and values in a sequence of containers, with priority given to the right-most container in the case
        of duplicates.

        :param containers: containers to compare
        :type containers: sequence of Container objects
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Combined containers
        """

        # if inputs are not dicts, then simply return the right-most value
        container_rightmost = containers[-1]
        if not isinstance(container_rightmost, dict):
            return container_rightmost

        # return if len==1
        if len(containers) == 1:
            return container_rightmost

        # otherwise, check that the keys are aligned between each container, and apply this method recursively
        return_dict = dict()
        all_Keys = set([item for sublist in [list(cont.keys()) for cont in containers] for item in sublist])
        for key in all_Keys:
            keys_present = [key in cont for cont in containers]
            return_dict[key] =\
                _ivy.Container.combine(*[cont[key] for cont, kp in zip(containers, keys_present) if kp], ivyh=ivyh)
        return _ivy.Container(return_dict)

    @staticmethod
    def diff(*containers, mode='all', diff_keys='diff', detect_key_diffs=True, ivyh=None):
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
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Compared containers
        """

        if mode not in ['all', 'same_only', 'diff_only']:
            raise Exception('mode must be one of [ "all" | "same_only" | "diff_only" ], but found {}'.format(mode))

        # if inputs are not dicts, then compare their values to determine the diff dict
        num_containers = len(containers)
        container0 = containers[0]
        if not isinstance(container0, dict):
            equal_mat = _ivy.equal(*containers, equality_matrix=True)
            if _ivy.reduce_min(_ivy.cast(equal_mat, 'int32')) == 1:
                if mode == 'diff_only':
                    return _ivy.Container()
                return container0
            elif mode == 'same_only':
                return _ivy.Container()
            else:
                cont_range = range(num_containers)
                diff_dict = dict()
                cont_dict = dict(zip(cont_range, containers))
                idxs_added = list()
                for idx in cont_range:
                    if idx not in idxs_added:
                        idxs_to_add = _ivy.indices_where(equal_mat[idx])
                        idxs_to_add_list = sorted(_ivy.to_numpy(idxs_to_add).reshape(-1).tolist())
                        if isinstance(diff_keys, str):
                            key = diff_keys + '_' + str(idxs_to_add_list)[1:-1]
                        elif isinstance(diff_keys, (list, tuple)):
                            key = diff_keys[idx]
                        else:
                            raise Exception('diff_keys must be either a string or list of strings,'
                                            'but found {} of type {}'.format(diff_keys, type(diff_keys)))
                        diff_dict[key] = cont_dict[idx]
                        idxs_added += idxs_to_add_list
                return _ivy.Container(diff_dict)

        # otherwise, check that the keys are aligned between each container, and apply this method recursively
        return_dict = dict()
        all_Keys = set([item for sublist in [list(cont.keys()) for cont in containers] for item in sublist])
        for key in all_Keys:
            keys_present = [key in cont for cont in containers]
            all_Keys_present = sum(keys_present) == num_containers
            if all_Keys_present:
                res = _ivy.Container.diff(*[cont[key] for cont in containers],
                                          mode=mode, diff_keys=diff_keys, detect_key_diffs=detect_key_diffs, ivyh=ivyh)
                if not isinstance(res, dict) or res:
                    return_dict[key] = res
                continue
            elif sum(keys_present) == 1 and not detect_key_diffs:
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
        return _ivy.Container(return_dict)

    @staticmethod
    def multi_map(func, containers, key_chains=None, to_apply=True, prune_unapplied=False, key_chain=''):
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
        :return: Contaienr
        """
        container0 = containers[0]
        return_dict = dict()
        for key in sorted(container0.keys()):
            values = [cont[key] for cont in containers]
            value0 = values[0]
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value0, Container):
                ret = _ivy.Container.multi_map(func, values, key_chains, to_apply, prune_unapplied, this_key_chain)
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
        return Container(return_dict, ivyh=container0._local_ivy)

    @staticmethod
    def from_disk_as_hdf5(h5_obj_or_filepath, slice_obj=slice(None), ivyh=None):
        """
        Load container object from disk, as an h5py file, at the specified hdf5 filepath.

        :param h5_obj_or_filepath: Filepath where the container object is saved to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 obj
        :param slice_obj: slice object to slice all h5 elements.
        :type slice_obj: slice or sequence of slices
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Container loaded from disk
        """
        container_dict = dict()
        if type(h5_obj_or_filepath) is str:
            h5_obj = _h5py.File(h5_obj_or_filepath, 'r')
        else:
            h5_obj = h5_obj_or_filepath

        for key, value in sorted(h5_obj.items()):
            if isinstance(value, _h5py.Group):
                container_dict[key] = Container.from_disk_as_hdf5(value, slice_obj, ivyh)
            elif isinstance(value, _h5py.Dataset):
                container_dict[key] = _ivy.default(ivyh, _ivy).array(list(value[slice_obj]))
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
        for key, value in sorted(h5_obj.items()):
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

        for key, value in sorted(h5_obj.items()):
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
    def reduce(containers, reduction, ivyh=None):
        """
        Reduce containers.

        :param containers: containers to reduce
        :type containers: sequence of Container objects
        :param reduction: the reduction function
        :type reduction: callable with single list input x
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: reduced containers
        """
        container0 = containers[0]

        if isinstance(container0, Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.reduce([container[key] for container in containers], reduction, ivyh)
            return Container(return_dict, ivyh=ivyh)
        else:
            # noinspection PyBroadException
            try:
                return reduction(containers)
            except Exception as e:
                raise Exception(str(e) + '\nContainer reduce operation only valid for containers of arrays')

    # Private Methods #
    # ----------------#

    def _get_shape(self):
        if not len(self.keys()):
            if _ivy.exists(self._queues):
                return [self._queue_load_sizes_cum[-1]]
            return [0]
        sub_shapes =\
            [v for k, v in self.map(lambda x, kc: list(x.shape) if self._ivy.is_array(x)
                else ([len(x)] if isinstance(x, (list, tuple)) else None)).to_iterator() if v]
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

    def _at_key_chains_input_as_seq(self, key_chains):
        return_cont = Container(dict(), ivyh=self._local_ivy)
        for kc in key_chains:
            return_cont.set_at_key_chain(kc, self.at_key_chain(kc), inplace=True)
        return return_cont

    def _at_key_chains_input_as_dict(self, key_chains, current_chain=''):
        return_dict = dict()
        for k, v in key_chains.items():
            if current_chain == '':
                new_current_chain = k
            else:
                new_current_chain = current_chain + '/' + k
            if isinstance(v, dict):
                return_dict[k] = self._at_key_chains_input_as_dict(v, new_current_chain)
            else:
                return_dict[k] = self.at_key_chain(new_current_chain)
        return Container(return_dict, ivyh=self._local_ivy)

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

    def set_framework(self, ivyh):
        """
        Update the framework to use for the container.

        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        """
        self._ivy = ivyh
        return self

    def all_true(self, assert_is_bool=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Boolean, whether all entries are boolean True.
        """
        return bool(_np.prod([v for k, v in self.as_bools(
            assert_is_bool, key_chains, to_apply, prune_unapplied).to_iterator()]))

    def all_false(self, assert_is_bool=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Boolean, whether all entries are boolean False.
        """
        return not bool(_np.sum([v for k, v in self.as_bools(
            assert_is_bool, key_chains, to_apply, prune_unapplied).to_iterator()]))

    def reduce_sum(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_sum(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reduce_prod(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_prod(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reduce_mean(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_mean(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reduce_var(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with the variance computed for all sub-arrays.
        """
        return self.map(lambda x, kc: self._ivy.reduce_var(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reduce_std(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with the standard deviation computed for all sub-arrays.
        """
        return self.map(lambda x, kc: self._ivy.reduce_std(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reduce_min(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_min(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reduce_max(self, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.reduce_max(x, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def minimum(self, other, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-arrays having the minimum values computed.
        """
        is_container = isinstance(other, Container)
        return self.map(lambda x, kc:
                        self._ivy.minimum(x, other[kc] if is_container else other) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def maximum(self, other, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-arrays having the maximum values computed.
        """
        is_container = isinstance(other, Container)
        return self.map(lambda x, kc:
                        self._ivy.maximum(x, other[kc] if is_container else other) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def clip(self, clip_min, clip_max, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-arrays having the clipped values returned.
        """
        min_is_container = isinstance(clip_min, Container)
        max_is_container = isinstance(clip_max, Container)
        return self.map(lambda x, kc:
                        self._ivy.clip(x, clip_min[kc] if min_is_container else clip_min,
                                       clip_max[kc] if max_is_container else clip_max) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def clip_vector_norm(self, max_norm, p, global_norm=False, key_chains=None, to_apply=True,
                         prune_unapplied=False):
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
                        key_chains, to_apply, prune_unapplied)

    def einsum(self, equation, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.einsum(equation, x) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def vector_norm(self, p=2, axis=None, keepdims=False, global_norm=False, key_chains=None, to_apply=True,
                    prune_unapplied=False):
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
                        if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied)

    def matrix_norm(self, p=2, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with the matrix norms for each sub-array returned.
        """
        return self.map(lambda x, kc: self._ivy.matrix_norm(x, p, axis, keepdims) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def flip(self, axis=None, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.flip(x, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def shuffle(self, seed_value=None, key_chains=None, to_apply=True, prune_unapplied=False, key_chain=''):
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
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        """
        return_dict = dict()
        if seed_value is None:
            seed_value = self._ivy.to_numpy(self._ivy.random.randint(0, 1000, ())).item()
        for key, value in sorted(self.items()):
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.shuffle(seed_value, key_chains, to_apply, prune_unapplied, this_key_chain)
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
        return Container(return_dict, ivyh=self._local_ivy)

    def slice_via_key(self, slice_key):
        """
        Get slice of container, based on key.

        :param slice_key: key to slice container at.
        :type slice_key: str
        :return: Container object sliced at desired key.
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            if key == slice_key:
                return value
            elif isinstance(value, Container):
                return_dict[key] = value.slice_via_key(slice_key)
            else:
                return_dict[key] = value
        return Container(return_dict, ivyh=self._local_ivy)

    def as_ones(self, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Return arrays of ones for all nested arrays in the container.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: Container object with all sub-arrays filled with ones.
        """
        return self.map(lambda x, kc: self._ivy.ones_like(x) if self._ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

    def as_zeros(self, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Return arrays of zeros for all nested arrays in the container.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: Container object with all sub-arrays filled with zeros.
        """
        return self.map(lambda x, kc: self._ivy.zeros_like(x) if self._ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

    def as_bools(self, assert_is_bool=False, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all entries boolean evaluated.
        """

        def _ret_bool(x):
            if assert_is_bool:
                assert isinstance(x, bool)
                return x
            return bool(x)

        return self.map(lambda x, kc: _ret_bool(x), key_chains, to_apply, prune_unapplied)

    def as_random_uniform(self, low=0.0, high=1.0, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-arrays filled with random uniform values.
        """
        return self.map(lambda x, kc: self._ivy.random_uniform(
            low, high, x.shape, self._ivy.dev_str(x)) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def expand_dims(self, axis, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: self._ivy.expand_dims(x, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

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
              prune_unapplied=False):
        """
        Splits a container into multiple sub-containers, by splitting their constituent arrays.

        :param num_or_size_splits: Number of equal arrays to divide the array into along the given axis if an integer.
                                   The size of each split element if a sequence of integers.
                                   Default is to divide into as many 1-dimensional arrays as the axis dimension.
        :type num_or_size_splits: int, optional
        :param axis: The axis along which to split, default is 0.
        :type axis: int, optional
        :param with_remainder: If the tensor does not split evenly, then store the last remainder entry.
                               Defaul is False.
        :type with_remainder: bool, optional
        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: A list of sub-arrays.
        """
        # ToDo: make this more efficient, without so many recursive container calls. For example the splits indices
        #  can be calculated here, and then slices applied directly only once
        dim_size = num_or_size_splits if isinstance(num_or_size_splits, int) else len(num_or_size_splits)
        # noinspection PyTypeChecker
        return self.map(
            lambda x, kc: self._ivy.split(x, num_or_size_splits, axis, with_remainder) if self._ivy.is_array(x)
            else x, key_chains, to_apply, prune_unapplied).unstack(0, dim_size=dim_size)

    def gather(self, indices, axis=-1, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions gathered along the axis.
        """
        return self.map(lambda x, kc: self._ivy.gather(x, indices, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def gather_nd(self, indices, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: Container object with all sub-array dimensions gathered.
        """
        return self.map(lambda x, kc: self._ivy.gather_nd(x, indices) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def repeat(self, repeats, axis=None, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: container with each array being repeated along the specified dimension.
        """
        return self.map(lambda x, kc: self._ivy.repeat(x, repeats, axis) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def swapaxes(self, axis0, axis1, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: ivy.Container with each chosen array having the axes swapped.
        """
        return self.map(lambda x, kc: self._ivy.swapaxes(x, axis0, axis1) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def reshape(self, pre_shape=None, shape_slice=None, post_shape=None, key_chains=None, to_apply=True,
                prune_unapplied=False):
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
        :return: ivy.Container with each array reshaped as specified.
        """
        pre_shape = [] if pre_shape is None else\
            ([pre_shape] if isinstance(pre_shape, int) else list(pre_shape))
        post_shape = [] if post_shape is None else\
            ([post_shape] if isinstance(post_shape, int) else list(post_shape))
        if shape_slice is None:
            return self.map(lambda x, kc: self._ivy.reshape(x, pre_shape + post_shape) if self._ivy.is_array(x) else x,
                            key_chains, to_apply, prune_unapplied)
        shape_slice = slice(shape_slice, shape_slice+1) if isinstance(shape_slice, int) else shape_slice
        return self.map(lambda x, kc:
                        self._ivy.reshape(x, pre_shape + list(x.shape[shape_slice]) + post_shape)
                        if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied)

    def einops_rearrange(self, pattern,  key_chains=None, to_apply=True, prune_unapplied=False, **axes_lengths):
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
        :param axes_lengths: Any additional specifications for dimensions.
        :type axes_lengths: keyword parameter args
        :return: ivy.Container with each array having einops.rearrange applied.
        """
        return self.map(lambda x, kc: _ivy.einops_rearrange(x, pattern, **axes_lengths) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def einops_reduce(self, pattern,  reduction, key_chains=None, to_apply=True, prune_unapplied=False, **axes_lengths):
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
        :param axes_lengths: Any additional specifications for dimensions.
        :type axes_lengths: keyword parameter args
        :return: ivy.Container with each array having einops.reduce applied.
        """
        return self.map(lambda x, kc: _ivy.einops_reduce(x, pattern, reduction, **axes_lengths) if self._ivy.is_array(x)
                        else x, key_chains, to_apply, prune_unapplied)

    def einops_repeat(self, pattern, key_chains=None, to_apply=True, prune_unapplied=False, **axes_lengths):
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
        :param axes_lengths: Any additional specifications for dimensions.
        :type axes_lengths: keyword parameter args
        :return: ivy.Container with each array having einops.repeat applied.
        """
        return self.map(lambda x, kc: _ivy.einops_repeat(x, pattern, **axes_lengths) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def to_dev(self, dev_str, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: The container, but with each sub-array now placed on the target device.
        """
        return self.map(lambda x, kc: self._ivy.to_dev(x, dev_str) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def stop_gradients(self, preserve_type=True, key_chains=None, to_apply=True, prune_unapplied=False):
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
        :return: container with each array having their gradients stopped.
        """
        return self.map(
            lambda x, kc: self._ivy.stop_gradient(x, preserve_type) if self._ivy.is_variable(x)
            else x, key_chains, to_apply, prune_unapplied)

    def as_variables(self, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Converts all nested arrays to variables, which support gradient computation.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: container with each array converted to a variable.
        """
        return self.map(lambda x, kc: self._ivy.variable(x) if self._ivy.is_array(x) else x,
                        key_chains, to_apply, prune_unapplied)

    def as_arrays(self, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Converts all nested variables to arrays, which do not support gradient computation.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: container with each variable converted to an array.
        """
        return self.map(
            lambda x, kc: self._ivy.stop_gradient(x, False) if self._ivy.is_variable(x)
            else (x if self._ivy.is_array(x) else self._ivy.array(x)), key_chains, to_apply, prune_unapplied)

    def to_numpy(self, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Converts all nested ivy arrays to numpy arrays.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: container with each ivy array converted to a numpy array.
        """
        return self.map(
            lambda x, kc: self._ivy.to_numpy(x) if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied)

    def arrays_as_lists(self, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Converts all nested arrays to lists, a useful intermediate step for conversion to other framework array types.

        :param key_chains: The key-chains to apply or not apply the method to. Default is None.
        :type key_chains: list or dict of strs, optional
        :param to_apply: If True, the method will be applied to key_chains, otherwise key_chains will be skipped.
                         Default is True.
        :type to_apply: bool, optional
        :param prune_unapplied: Whether to prune key_chains for which the function was not applied. Default is False.
        :type prune_unapplied: bool, optional
        :return: container with each array converted to a list.
        """
        return self.map(
            lambda x, kc: self._ivy.to_list(x) if self._ivy.is_array(x) else x, key_chains, to_apply, prune_unapplied)

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
        for key, value in sorted(self.items()):
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
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_list.append(value.to_list())
            elif value is not None and key != '_f':
                return_list.append(value)
        return return_list

    def to_dict(self):
        """
        Return nested pure dict representation of container object.

        :return: Container as nested dict.
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_dict[key] = value.to_dict()
            elif key != '_f':
                return_dict[key] = value
        return return_dict

    def to_iterator(self, key_chain='', leaf_keys_only=False):
        """
        Return iterator for traversing through the nested elements of container object.

        :return: Iterator for the container elements.
        """
        for key, value in sorted(self.items()):
            if leaf_keys_only:
                kc = key
            else:
                kc = key_chain + '/' + key if key_chain != '' else key
            if isinstance(value, Container):
                # noinspection PyCompatibility
                yield from value.to_iterator(kc, leaf_keys_only)
            else:
                yield kc, value

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
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                new_value = value.from_flat_list(flat_list)
            else:
                new_value = flat_list.pop(0)
            new_dict[key] = new_value
        return Container(new_dict, ivyh=self._local_ivy)

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

        def _mean_is_nan(x):
            x_scalar = _ivy.to_scalar(x) if _ivy.is_array(x) else x
            if not x_scalar == x_scalar:
                return True
            if include_infs and x_scalar == INF or x_scalar == -INF:
                return True
            return False

        leafwise_res = self.reduce_mean().map(lambda x, kc: _mean_is_nan(x))
        if leafwise:
            return leafwise_res
        return max([v for k, v in leafwise_res.to_iterator()])

    def at_keys(self, queries, ignore_none=True, containing=False):
        """
        Query container object at specified keys, either as list or nested dict.

        :param queries: The keys to query.
        :type queries: sequence of strs or single str
        :param ignore_none: Whether to ignore None input. Default is True.
        :type ignore_none: bool, optional
        :param containing: Whether to include keys which only contain the query substrings. Default is False.
        :type containing: bool, optional
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
        return self.at_key_chains(key_chains_to_keep)

    def at_key_chain(self, key_chain):
        """
        Query container object at a specified key-chain

        :return: sub-container or value at specified key chain
        """
        keys = re.split('[/.]', key_chain)
        ret = self
        for key in keys:
            ret = ret[key]
        return ret

    def at_key_chains(self, key_chains, ignore_none=True):
        """
        Query container object at specified key-chains, either as list or nested dict.

        :return: sub-container containing only the specified key chains
        """
        if key_chains is None and ignore_none:
            return self
        if isinstance(key_chains, (list, tuple)):
            return self._at_key_chains_input_as_seq(key_chains)
        elif isinstance(key_chains, dict):
            return self._at_key_chains_input_as_dict(key_chains)
        elif isinstance(key_chains, str):
            return self._at_key_chains_input_as_seq([key_chains])
        else:
            raise Exception('Invalid type for input key_chains, must either be a list, tuple, dict, or ivy.Container,'
                            'but found type {}'.format(type(key_chains)))

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
        return Container(return_dict, ivyh=self._local_ivy)

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
                sub_cont[key] = Container(ivyh=self._local_ivy)
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
        return Container(return_dict, ivyh=self._local_ivy)

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
        return Container(return_dict, ivyh=self._local_ivy)

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
        for key, value in sorted(self.items()):
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
        return Container(out_dict, ivyh=self._local_ivy)

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

    def restructure_keys(self, key_chain_mapping):
        """
        Restructure the keys of the container.

        :param key_chain_mapping: Sequence of lists/tuples of key chain mapping to apply, with original and new key
                                  chains being the left and right terms respectively.
        :type key_chain_mapping: sequence of len-2 sequences
        :return: New contaienr with the key chains updated.
        """
        ret_cont = self.copy()
        for orig_kc, new_kc in key_chain_mapping:
            if orig_kc == '':
                orig_kc_val = ret_cont
                ret_cont = Container()
            else:
                orig_kc_val = ret_cont[orig_kc]
                ret_cont = ret_cont.prune_key_chain(orig_kc)
            ret_cont[new_kc] = orig_kc_val
        return ret_cont

    def prune_empty(self):
        """
        Recursively prunes empty keys from the container dict structure.
        Returns None if the entire container is empty.

        :return: Container with empty keys pruned.
        """
        out_dict = dict()
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                new_value = value.prune_empty()
                if new_value:
                    out_dict[key] = new_value
            else:
                out_dict[key] = value
        if len(out_dict):
            return Container(out_dict, ivyh=self._local_ivy)
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
        out_cont = Container(ivyh=self._local_ivy)
        for key, value in sorted(self.items()):
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
        out_cont = Container(ivyh=self._local_ivy)
        for key, value in sorted(self.items()):
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

    def copy(self):
        """
        Create a copy of this container.

        :return: A copy of the container
        """
        return Container(self.to_dict(), ivyh=self._local_ivy)

    def map(self, func, key_chains=None, to_apply=True, prune_unapplied=False, key_chain=''):
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
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :return: New container following the function mapped to each sub-array.
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.map(func, key_chains, to_apply, prune_unapplied, this_key_chain)
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
        return Container(return_dict, ivyh=self._local_ivy)

    def map_conts(self, func, key_chains=None, to_apply=True, prune_unapplied=False, include_self=True, key_chain=''):
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
        :param include_self: Whether to also apply the (possiby in-place) function to this container. Default is True.
        :type include_self: bool, optional
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        :return: New container following the function mapped to each sub-container.
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.map_conts(func, key_chains, to_apply, prune_unapplied, key_chain=this_key_chain)
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
                return_dict[key] = value
        ret = Container(return_dict, ivyh=self._local_ivy)
        if key_chain != '' or include_self:
            return func(ret, key_chain)
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
        return Container(return_cont, ivyh=self._local_ivy)

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

    def with_print_limit(self, print_limit):
        return Container(self,
                         container_combine_method=self._container_combine_method,
                         print_limit=print_limit,
                         print_indent=self._print_indent,
                         print_line_spacing=self._print_line_spacing,
                         ivyh=self._local_ivy,
                         keyword_color_dict=self._keyword_color_dict,
                         rebuild_child_containers=True)

    # noinspection PyTypeChecker
    def remove_print_limit(self):
        return self.with_print_limit(None)

    def with_print_indent(self, print_indent):
        return Container(self,
                         container_combine_method=self._container_combine_method,
                         print_limit=self._print_limit,
                         print_indent=print_indent,
                         print_line_spacing=self._print_line_spacing,
                         ivyh=self._local_ivy,
                         keyword_color_dict=self._keyword_color_dict,
                         rebuild_child_containers=True)

    def with_print_line_spacing(self, print_line_spacing):
        return Container(self,
                         container_combine_method=self._container_combine_method,
                         print_limit=self._print_limit,
                         print_indent=self._print_indent,
                         print_line_spacing=print_line_spacing,
                         ivyh=self._local_ivy,
                         keyword_color_dict=self._keyword_color_dict,
                         rebuild_child_containers=True)

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
            aligned_array_chunks = dict([(i, _align_array(c)) for i, c in enumerate(chunks) if '\\n' in c])
            chunks = [aligned_array_chunks[i] if i in aligned_array_chunks else c_orig
                      for i, c_orig in enumerate(chunks)]
            return ('\n' + indent_str).join(chunks)

        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, Container):
                # noinspection PyArgumentList
                rep = v.__repr__(as_repr=False)
            else:
                if self._ivy.is_array(v) and len(list(v.shape)) > 0 and _ivy.exists(self._print_limit) and \
                        _reduce(_mul, v.shape) > self._print_limit:
                    rep = (type(v), "shape=", list(v.shape))
                elif isinstance(v, (list, tuple)) and v and self._ivy.is_array(v[0]):
                    rep = ("list[{}]".format(len(v)), type(v[0]), "shape=", list(v[0].shape))
                else:
                    rep = v
            new_dict[k] = rep
        if as_repr:
            json_dumped_str = _align_arrays(_json.dumps(
                Container(new_dict, print_limit=self._print_limit).map(
                    lambda x, kc: x if _is_jsonable(x)
                    else _repr(x).replace(' ', '').replace(',', ', ')).to_dict(),
                indent=self._print_indent))

            def _add_newline(str_in):
                str_in_split = str_in.split('\n')
                str_split_size = len(str_in_split)
                return '\n'.join([('\n'*self._print_line_spacing + ss) if i == (str_split_size-1) else ss
                                  for i, ss in enumerate(str_in_split)])

            json_dumped_str = '":'.join([_add_newline(s) for s in json_dumped_str.split('":')])
            # make keys green
            json_dumped_str_split = json_dumped_str.split('":')
            split_size = len(json_dumped_str_split)
            json_dumped_str =\
                '":'.join([' "'.join(sub_str.split(' "')[:-1] + [termcolor.colored(sub_str.split(' "')[-1], 'green')])
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

    def __getattr__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            # noinspection PyUnresolvedReferences
            return super.__getattr__(item)

    def __setattr__(self, name, value):
        if name[0] != '_':
            self[name] = value
        else:
            super.__setattr__(self, name, value)

    def _get_queue_item(self, query):
        if isinstance(query, int):
            queue_queries = [query]
        elif isinstance(query, slice):
            queue_queries = list(range(query.start, query.stop, _ivy.default(query.step, 1)))
        elif isinstance(query, (list, tuple)):
            queue_queries = list(range(query[0].start, query[0].stop, _ivy.default(query[0].step, 1)))
        else:
            raise Exception('Invalid slice type, must be one of integer, slice, or sequences of slices.')
        queue_idxs = set([_np.sum(q >= self._queue_load_sizes_cum).item() for q in queue_queries])
        conts = list()
        for i in queue_idxs:
            if i not in self._loaded_containers_from_queues:
                cont = Container(self._queues[i].get(timeout=self._queue_timeout), ivyh=self._local_ivy)
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
        return combined_cont[shifted_query]

    def __getitem__(self, query):
        """
        Get slice, key or key chain of container object.

        :param query: slice object, key or key chain to query all container elements.
        :type query: slice or str
        :return: Container object at desired query.
        """
        if isinstance(query, str):
            if '/' in query or '.' in query:
                return self.at_key_chain(query)
            return dict.__getitem__(self, query)
        elif _ivy.exists(self._queues):
            return self._get_queue_item(query)
        return_dict = dict()
        for key, value in sorted(self.items()):
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

        return Container(return_dict, ivyh=self._local_ivy)

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

    # Getters and Setters #
    # --------------------#

    # private

    @property
    def _ivy(self):
        return _ivy.default(self._local_ivy, _ivy)

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
