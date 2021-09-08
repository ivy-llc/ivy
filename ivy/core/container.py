"""
Base Container Object
"""

# global
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
from operator import or_ as _or
from operator import mul as _mul
from operator import pow as _pow
from operator import and_ as _and
from operator import xor as _xor
from functools import reduce as _reduce
from operator import truediv as _truediv
from operator import floordiv as _floordiv

# local
import ivy as _ivy


def _is_jsonable(x):
    try:
        _json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


# noinspection PyMissingConstructor
class Container(dict):

    def __init__(self, dict_in=None, queues=None, queue_load_sizes=None, container_combine_method='list_join',
                 queue_timeout=5.0, ivyh=None, keyword_color_dict=None, **kwargs):
        """
        Initialize container object from input dict representation.

        :param dict_in: the dictionary the container should wrap around. Default is None.
        :type dict_in: dict, optional
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :param kwargs: keyword arguments for dict creation. Default is None.
        :type kwargs: keyword arguments.
        """
        self._queues = queues
        if _ivy.exists(self._queues):
            self._queue_load_sizes = queue_load_sizes
            self._container_combine_method = {'list_join': self.list_join,
                                              'concat': lambda conts: self.concat(conts, 0)}[container_combine_method]
            self._loaded_containers_from_queues = dict()
            self._queue_load_sizes_cum = _np.cumsum(self._queue_load_sizes)
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
            if isinstance(value, dict):
                self[key] = Container(value, ivyh=ivyh)
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
        Combine keys and values in a sequence of containers, with priority given to the left-most container in the case
        of duplicates.

        :param containers: containers to compare
        :type containers: sequence of Container objects
        :param ivyh: Handle to ivy module to use for the calculations. Default is None, which results in the global ivy.
        :type ivyh: handle to ivy module, optional
        :return: Combined containers
        """

        # if inputs are not dicts, then simply return the left-most value
        container0 = containers[0]
        if not isinstance(container0, dict):
            return container0

        # otherwise, check that the keys are aligned between each container, and apply this method recursively
        return_dict = dict()
        all_Keys = set([item for sublist in [list(cont.keys()) for cont in containers] for item in sublist])
        for key in all_Keys:
            keys_present = [key in cont for cont in containers]
            res = _ivy.Container.combine(*[cont[key] for cont, kp in zip(containers, keys_present) if kp], ivyh=ivyh)
            if not isinstance(res, dict) or res:
                return_dict[key] = res
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
            return_cont.set_at_key_chain(kc, self.at_key_chain(kc))
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

    def clip_norm(self, max_norm, p_val, global_norm=False, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Computes the elementwise clipped values between this container and clip_min and clip_max containers or numbers.

        :param max_norm: The max norm container or number to clip against.
        :type max_norm: Ivy container or number
        :param p_val: The p-value for computing the p-norm container or number.
        :type p_val: Ivy container or number
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
        # ToDo: modify this method to make use of ivy.Container.norm method
        max_norm_is_container = isinstance(max_norm, Container)
        p_val_is_container = isinstance(p_val, Container)
        if global_norm:
            if max_norm_is_container or p_val_is_container:
                raise Exception(
                    'global_norm can only be computed for scalar max_norm and p_val arguments,'
                    'but found {} and {} of type {} and {} respectively'.format(
                        max_norm, p_val, type(max_norm), type(p_val)))
            norm = sum([v for k, v in
                        self.map(lambda x, kc: self._ivy.reduce_sum(x ** p_val)).to_iterator()]) ** (1/p_val)
            ratio = max_norm/norm
            if ratio < 1:
                return self * ratio
            return self.copy()
        return self.map(lambda x, kc:
                        self._ivy.clip_norm(x, max_norm[kc] if max_norm_is_container else max_norm,
                                            p_val[kc] if p_val_is_container else p_val) if self._ivy.is_array(x) else x,
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

    # noinspection PyShadowingBuiltins
    def norm(self, ord=None, axis=None, keepdims=False, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Compute matrix or vector norm for each array in the container.
        This function is able to return ord-1 and ord-2 vector-norms and matrix-norms.

        :param ord: Order of the norm. Default is Frobenius norm.
        :type ord: int or str, optional
        :param axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a
                     2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are
                     computed. Default is None, in which case the axes are inferred from the input shape.
        :type axis: int or 2-sequence of ints, optional
        :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with
                         size one. With this option the result will broadcast correctly against the original x.
                         Default is False.
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
        return self.map(lambda x, kc: self._ivy.norm(x, ord, axis, keepdims) if self._ivy.is_array(x) else x,
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

    def reshape(self, pre_shape, shape_slice=None, post_shape=None, key_chains=None, to_apply=True,
                prune_unapplied=False):
        """
        Reshapes each array x in the container, to a new shape given by pre_shape + x.shape[shape_slice] + post_shape.
        If shape_slice or post_shape are not specified, then the term is ignored.

        :param pre_shape: The first elements in the new array shape.
        :type pre_shape: sequence of ints
        :param shape_slice: The slice of the original shape to use in the new shape. Default is None.
        :type shape_slice: sequence of ints, optional
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
        pre_shape = list(pre_shape)
        post_shape = [] if post_shape is None else list(post_shape)
        if shape_slice is None:
            return self.map(lambda x, kc: self._ivy.reshape(x, pre_shape + post_shape) if self._ivy.is_array(x) else x,
                            key_chains, to_apply, prune_unapplied)
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

    def to_iterator(self):
        """
        Return iterator for traversing through the nested elements of container object.

        :return: Iterator for the container elements.
        """
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                # noinspection PyCompatibility
                yield from value.to_iterator()
            else:
                yield key, value

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
        keys = key_chain.split('/')
        ret = self
        for key in keys:
            try:
                ret = ret[key]
            except KeyError:
                return False
        return True

    def at_keys(self, query_keys, ignore_none=True):
        """
        Query container object at specified keys, either as list or nested dict.

        :return: sub-container containing only key-chains containing the specified keys.
        """
        if query_keys is None and ignore_none:
            return self
        key_chains_to_keep = list()
        if isinstance(query_keys, str):
            query_keys = [query_keys]

        def map_fn(x, kc):
            nonlocal key_chains_to_keep
            for query_key in query_keys:
                if query_key in kc:
                    key_chains_to_keep.append(kc)
            return x

        self.map(map_fn)
        return self.at_key_chains(key_chains_to_keep)

    def at_key_chain(self, key_chain):
        """
        Query container object at a specified key-chain

        :return: sub-container or value at specified key chain
        """
        keys = key_chain.split('/')
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

    def set_at_key_chain(self, key_chain, val):
        """
        Set value of container object at a specified key-chain

        :return: new container with updated value at key chain
        """
        keys = key_chain.split('/')
        cont = self
        for key in keys[:-1]:
            if key not in cont:
                cont[key] = Container(ivyh=self._local_ivy)
            cont = cont[key]
        cont[keys[-1]] = val
        return self

    def set_at_key_chains(self, target_dict, return_dict=None):
        """
        Set values of container object at specified key-chains

        :return: new container with updated value at key chain
        """
        if return_dict is None:
            return_dict = self.copy()
        for k, v in target_dict.items():
            if isinstance(v, dict):
                return_dict[k] = self.set_at_key_chains(v, return_dict[k])
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
        keys_in_chain = key_chain.split('/')
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
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                ret = value.map(func, key_chains, to_apply, prune_unapplied, this_key_chain)
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
                return_dict[key] = func(value, this_key_chain)
        return Container(return_dict, ivyh=self._local_ivy)

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

    def reshape_like(self, target_dict, return_cont=None):
        """
        Set shapes of container entries to shapes specified by new container with the same key structure

        :return: new container with values of updated shapes
        """
        if return_cont is None:
            return_cont = self.copy()
        for (_, v_shape), (k, v) in zip(target_dict.items(), return_cont.items()):
            if isinstance(v_shape, dict):
                return_cont[k] = self.reshape_like(v_shape, return_cont[k])
            else:
                return_cont[k] = self._ivy.reshape(v, v_shape)
        return Container(return_cont, ivyh=self._local_ivy)

    def if_exists(self, key):
        """
        Returns the sub-container at the following key if it exists, otherwise None.
        """
        try:
            return self[key]
        except KeyError:
            return

    # Built-ins #
    # ----------#

    def __repr__(self, as_repr=True):
        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, Container):
                # noinspection PyArgumentList
                rep = v.__repr__(as_repr=False)
            else:
                if self._ivy.is_array(v) and len(list(v.shape)) > 0 and _reduce(_mul, v.shape) > 10:
                    rep = (type(v), "shape=", list(v.shape))
                elif isinstance(v, (list, tuple)) and v and self._ivy.is_array(v[0]):
                    rep = ("list[{}]".format(len(v)), type(v[0]), "shape=", list(v[0].shape))
                else:
                    rep = v
            new_dict[k] = rep
        if as_repr:
            json_dumped_str = _json.dumps(
                Container(new_dict).map(
                    lambda x, kc: x if _is_jsonable(x)
                    else x.__repr__().replace('\n', '').replace(' ', '').replace(',', ', ')).to_dict(),
                indent=4)
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
            if '/' in query:
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
        if isinstance(query, str) and '/' in query:
            return self.set_at_key_chain(query, val)
        else:
            return dict.__setitem__(self, query, val)

    def __contains__(self, key):
        if isinstance(key, str) and '/' in key:
            return self.has_key_chain(key)
        else:
            return dict.__contains__(self, key)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.map(lambda x, kc: -x)

    def __pow__(self, power):
        if isinstance(power, (float, int)):
            return self.map(lambda x, kc: x ** power)
        return self.reduce([self, power], lambda x: _reduce(_pow, x))

    def __rpow__(self, power):
        if not isinstance(power, (float, int)):
            raise Exception('power must be float, int or ivy.Container, but found type: {}'.format(type(power)))
        return self.map(lambda x, kc: power ** x)

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return self.map(lambda x, kc: x + other)
        return self.reduce([self, other], sum)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return self.map(lambda x, kc: x - other)
        return self.reduce([self, -other], sum)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return self.map(lambda x, kc: x * other)
        return self.reduce([self, other], lambda x: _reduce(_mul, x))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return self.map(lambda x, kc: x / other)
        return self.reduce([self, other], lambda x: _reduce(_truediv, x))

    def __rtruediv__(self, other):
        if not isinstance(other, (float, int)):
            raise Exception('power must be float, int or ivy.Container, but found type: {}'.format(type(other)))
        return self.map(lambda x, kc: other / x)

    def __floordiv__(self, other):
        if isinstance(other, (float, int)):
            return self.map(lambda x, kc: x // other)
        return self.reduce([self, other], lambda x: _reduce(_floordiv, x))

    def __rfloordiv__(self, other):
        if not isinstance(other, (float, int)):
            raise Exception('power must be float, int or ivy.Container, but found type: {}'.format(type(other)))
        return self.map(lambda x, kc: other // x)

    def __abs__(self):
        return self.map(lambda x, kc: self._ivy.abs(x))

    def __lt__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_lt, x))

    def __le__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_le, x))

    def __eq__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_eq, x))

    def __ne__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_ne, x))

    def __gt__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_gt, x))

    def __ge__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_ge, x))

    def __and__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_and, x))

    def __or__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_or, x))

    def __invert__(self):
        return self.map(lambda x, kc: self._ivy.logical_not(x))

    def __xor__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_xor, x))

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
    def dev_str(self):
        """
        The device to which the arrays in the container belong, with None returned if the devices are not consistent
        """
        return self._get_dev_str()

    @property
    def ivy(self):
        return self._ivy
