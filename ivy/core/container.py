"""
Base Container Object
"""

# global
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

    def __init__(self, dict_in=None, **kwargs):
        """
        Initialize container object from input dict representation.
        """
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
                self[key] = Container(value)
            else:
                self[key] = value

        self._size = self._get_size()

    # Class Methods #
    # --------------#

    @staticmethod
    def list_join(containers):
        """
        Join containers of lists together along the specified dimension.

        :param containers: containers to list join
        :type containers: sequence of Container objects
        :return: List joined containers, with each entry being a list of arrays
        """

        container0 = containers[0]

        if isinstance(container0, dict):
            return_dict = dict()
            for key in container0.keys():
                new_list = list()
                for container in containers:
                    new_list.append(container[key])
                return_dict[key] = Container.list_join(new_list)
            return Container(return_dict)
        else:
            return [item for sublist in containers for item in sublist]
    
    @staticmethod
    def list_stack(containers, dim):
        """
        List stack containers together along the specified dimension.

        :param containers: containers to list stack
        :type containers: sequence of Container objects
        :param dim: dimension along which to list stack
        :type dim: int
        :return: Stacked containers, with each entry being a list of arrays
        """

        container0 = containers[0]

        if isinstance(container0, dict):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.list_stack([container[key] for container in containers], dim)
            return Container(return_dict)
        else:
            return containers

    @staticmethod
    def concat(containers, dim):
        """
        Concatenate containers together along the specified dimension.

        :param containers: containers to concatenate
        :type containers: sequence of Container objects
        :param dim: dimension along which to concatenate
        :type dim: int
        :return: Concatenated containers
        """

        container0 = containers[0]

        if isinstance(container0, dict):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.concat([container[key] for container in containers], dim)
            return Container(return_dict)
        else:
            # noinspection PyBroadException
            try:
                if len(containers[0].shape) == 0:
                    return _ivy.concatenate([_ivy.reshape(item, [1] * (dim + 1)) for item in containers], dim)
                else:
                    return _ivy.concatenate(containers, dim)
            except Exception as e:
                raise Exception(str(e) + '\nContainer concat operation only valid for containers of arrays')

    @staticmethod
    def from_disk_as_hdf5(h5_obj_or_filepath, slice_obj=slice(None)):
        """
        Load container object from disk, as an h5py file, at the specified hdf5 filepath.

        :param h5_obj_or_filepath: Filepath where the container object is saved to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 obj
        :param slice_obj: slice object to slice all h5 elements.
        :type slice_obj: slice or sequence of slices
        :return: Container loaded from disk
        """
        container_dict = dict()
        if type(h5_obj_or_filepath) is str:
            h5_obj = _h5py.File(h5_obj_or_filepath, 'r')
        else:
            h5_obj = h5_obj_or_filepath

        for key, value in sorted(h5_obj.items()):
            if isinstance(value, _h5py.Group):
                container_dict[key] = Container.from_disk_as_hdf5(value, slice_obj)
            elif isinstance(value, _h5py.Dataset):
                container_dict[key] = _ivy.array(list(value[slice_obj]))
            else:
                raise Exception('Item found inside h5_obj which was neither a Group nor a Dataset.')
        return Container(container_dict)

    @staticmethod
    def from_disk_as_pickled(pickle_filepath):
        """
        Load container object from disk at the specified pickle filepath.

        :param pickle_filepath: Filepath where the container object is saved to disk.
        :type pickle_filepath: str
        :return: Container loaded from disk
        """
        return _pickle.load(open(pickle_filepath, 'rb'))

    @staticmethod
    def from_disk_as_json(json_filepath):
        """
        Load container object from disk at the specified json filepath.
        If some objects were not json-able during saving, then they will be loaded as strings.

        :param json_filepath: Filepath where the container object is saved to disk.
        :type json_filepath: str
        :return: Container loaded from disk
        """
        with open(json_filepath) as json_data_file:
            return Container(_json.load(json_data_file))

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
                _random.shuffle(value)
            else:
                raise Exception('Item found inside h5_obj which was neither a Group nor a Dataset.')
        if isinstance(h5_obj, _h5py.File):
            h5_obj.close()

    @staticmethod
    def reduce(containers, reduction):
        """
        Reduce containers.

        :param containers: containers to reduce
        :type containers: sequence of Container objects
        :param reduction: the reduction function
        :type reduction: callable with single list input x
        :return: reduced containers
        """
        list_size = len(containers)
        container0 = containers[0]
        if isinstance(container0, dict):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.reduce([container[key] for container in containers], reduction)
            return Container(return_dict)
        else:
            # noinspection PyBroadException
            try:
                return reduction(containers)
            except Exception as e:
                raise Exception(str(e) + '\nContainer reduce operation only valid for containers of arrays')

    # Private Methods #
    # ----------------#

    def _get_size(self):
        vals = list(self.values())
        if not vals:
            return 0
        val = vals[0]
        if isinstance(val, Container):
            return val._get_size()
        elif isinstance(val, list):
            return len(val)
        elif isinstance(val, tuple):
            return len(val)
        else:
            try:
                return val.shape[0]
            except (AttributeError, IndexError, TypeError):
                return 0

    def _at_key_chains_input_as_seq(self, key_chains):
        return_cont = Container(dict())
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
        return Container(return_dict)

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
                if ret_cont.size == 0:
                    del return_cont[k]
            else:
                del return_cont[k]
        return return_cont

    # Public Methods #
    # ---------------#

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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.reduce_sum(x, axis, keepdims) if _ivy.is_array(x) else x,
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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.reduce_prod(x, axis, keepdims) if _ivy.is_array(x) else x,
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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.reduce_mean(x, axis, keepdims) if _ivy.is_array(x) else x,
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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.reduce_var(x, axis, keepdims) if _ivy.is_array(x) else x,
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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.reduce_min(x, axis, keepdims) if _ivy.is_array(x) else x,
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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.reduce_max(x, axis, keepdims) if _ivy.is_array(x) else x,
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
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return self.map(lambda x, kc: _ivy.einsum(equation, x) if _ivy.is_array(x) else x,
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
            seed_value = _ivy.to_numpy(_ivy.random.randint(0, 1000, ())).item()
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
                _ivy.seed(seed_value)
                return_dict[key] = _ivy.shuffle(value)
        return Container(return_dict)

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
        return Container(return_dict)

    def ones_like(self, key_chains=None, to_apply=True, prune_unapplied=False):
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
        return self.map(lambda x, kc: _ivy.ones_like(x) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

    def zeros_like(self, key_chains=None, to_apply=True, prune_unapplied=False):
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
        return self.map(lambda x, kc: _ivy.zeros_like(x) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

    def random_uniform_like(self, low=0.0, high=1.0, key_chains=None, to_apply=True, prune_unapplied=False):
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
        return self.map(lambda x, kc: _ivy.random_uniform(
            low, high, x.shape, _ivy.dev_str(x)) if _ivy.is_array(x) else x,
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
        return self.map(lambda x, kc: _ivy.expand_dims(x, axis) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

    def unstack(self, dim, dim_size):
        """
        Unstack containers along specified dimension.

        :param dim: Dimensions along which to unstack.
        :type dim: int
        :param dim_size: Size of the dimension to unstack.
        :type dim_size: int
        :return: List of containers, unstacked along the specified dimension.
        """
        return [self[tuple([slice(None, None, None)] * dim + [slice(i, i + 1, 1)])] for i in range(dim_size)]

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
        return self.map(lambda x, kc: _ivy.gather(x, indices, axis) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

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
        return self.map(lambda x, kc: _ivy.gather_nd(x, indices) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

    def repeat(self, repeats, axis=None, key_chains=None, to_apply=True, prune_unapplied=False):
        """
        Repeat values along a given dimension for each array in the container.

        :param repeats: The number of repetitions for each element. repeats is broadcast to fit the shape of the given axis.
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
        return self.map(lambda x, kc: _ivy.repeat(x, repeats, axis) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

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
        return self.map(lambda x, kc: _ivy.swapaxes(x, axis0, axis1) if _ivy.is_array(x) else x, key_chains, to_apply,
                        prune_unapplied)

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
            return self.map(lambda x, kc: _ivy.reshape(x, pre_shape + post_shape) if _ivy.is_array(x) else x,
                            key_chains, to_apply, prune_unapplied)
        return self.map(lambda x, kc:
                        _ivy.reshape(x, pre_shape + list(x.shape[shape_slice]) + post_shape) if _ivy.is_array(x) else
                        x, key_chains, to_apply, prune_unapplied)

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
            lambda x, kc: _ivy.stop_gradient(x, preserve_type) if _ivy.is_variable(x)
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
        return self.map(lambda x, kc: _ivy.variable(x) if _ivy.is_array(x) else x,
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
        return self.stop_gradients(False, key_chains, to_apply, prune_unapplied)

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
                value_as_np = _ivy.to_numpy(value)
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
        _pickle.dump(self, open(pickle_filepath, 'wb'))

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
        return Container(new_dict)

    def to_random(self):
        """
        Return new container, with all entries having same shape and type, but random values
        """
        def _as_random(value, _=''):
            if hasattr(value, 'shape'):
                return _ivy.random_uniform(0., 1., value.shape)
            return value
        return self.map(_as_random)

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

    def set_at_key_chain(self, key_chain, val):
        """
        Set value of container object at a specified key-chain

        :return: new container with updated value at key chain
        """
        keys = key_chain.split('/')
        conts = list()
        cont = self
        for key in keys[:-1]:
            if key not in cont:
                cont[key] = Container({})
            cont = cont[key]
        cont[keys[-1]] = val
        return self

    def set_at_key_chains(self, target_container, return_dict=None):
        """
        Set values of container object at specified key-chains

        :return: new container with updated value at key chain
        """
        if return_dict is None:
            return_dict = self.copy()
        for k, v in target_container.items():
            if isinstance(v, dict):
                return_dict[k] = self.set_at_key_chains(v, return_dict[k])
            else:
                return_dict[k] = v
        return Container(return_dict)

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
        return Container(out_dict)

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
            return Container(out_dict)
        return

    def copy(self):
        """
        Create a copy of this container.

        :return: A copy of the container
        """
        return Container(self.to_dict())

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
        return Container(return_dict)

    def dtype(self):
        """
        Return container, with all entries replaced with their data types.

        :return: New datatype container
        """
        return self.map(lambda x, _: _ivy.dtype(x))

    def with_entries_as_lists(self):
        """
        Return container object, with each array entry in the container cast to a list
        """
        def to_list(x, _=''):
            try:
                return _ivy.to_list(x)
            except (AttributeError, ValueError):
                return x
        return self.map(to_list)

    def reshape_like(self, target_container, return_cont=None):
        """
        Set shapes of container entries to shapes specified by new container with the same key structure

        :return: new container with values of updated shapes
        """
        if return_cont is None:
            return_cont = self.copy()
        for (_, v_shape), (k, v) in zip(target_container.items(), return_cont.items()):
            if isinstance(v_shape, dict):
                return_cont[k] = self.reshape_like(v_shape, return_cont[k])
            else:
                return_cont[k] = _ivy.reshape(v, v_shape)
        return Container(return_cont)

    # Built-ins #
    # ----------#

    def __repr__(self, as_repr=True):
        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, _ivy.Container):
                rep = v.__repr__(as_repr=False)
            else:
                if _ivy.is_array(v) and _reduce(_mul, v.shape) > 10:
                    rep = (type(v), list(v.shape))
                else:
                    rep = v
            new_dict[k] = rep
        if as_repr:
            return dict.__repr__(new_dict)
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
                elif value is None or value.shape == ():
                    return_dict[key] = value
                else:
                    return_dict[key] = value[query]

        return Container(return_dict)

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
        return self.map(lambda x, kc: _ivy.abs(x))

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
        return self.map(lambda x, kc: _ivy.logical_not(x))

    def __xor__(self, other):
        return self.reduce([self, other], lambda x: _reduce(_xor, x))

    # Getters #
    # --------#

    @property
    def size(self):
        return self._size
