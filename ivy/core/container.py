"""
Base Container Object
"""

# global
import random as _random
import h5py as _h5py
import numpy as _np
from functools import reduce as _reduce
from operator import mul as _mul
try:
    import jax.numpy as _jpn
except ImportError:
    _jpn = None

# local
from ivy.core import general as _ivy_gen
from ivy.core import random as _ivy_rand
from ivy.framework_handler import get_framework as _get_framework


# noinspection PyMissingConstructor
class Container(dict):

    def __init__(self, dict_in=None):
        """
        Initialize container object from input dict representation.
        """
        if dict_in is None:
            dict_in = dict()
        if not isinstance(dict_in, dict):
            dict_in = dict(dict_in)
        for key, value in sorted(dict_in.items()):
            if isinstance(value, dict):
                self[key] = Container(value)
            else:
                self[key] = value

        self._size = self._get_size()

    # Class Methods #
    # --------------#

    @staticmethod
    def concat(containers, dim, f=None):
        """
        Concatenate containers together along the specified dimension.

        :param containers: containers to _concatenate
        :type containers: sequence of Container objects
        :param dim: dimension along which to _concatenate
        :type dim: int
        :param f: Machine learning framework. Inferred from inputs if None.
        :type f: ml_framework, optional
        :return: Concatenated containers
        """

        container0 = containers[0]

        if isinstance(container0, dict):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = Container.concat([container[key] for container in containers], dim)
            return Container(return_dict)
        else:
            f = _get_framework(container0, f=f)
            # noinspection PyBroadException
            try:
                if len(containers[0].shape) == 0:
                    return _ivy_gen.concatenate([_ivy_gen.reshape(item, [1]*(dim+1)) for item in containers], dim, f=f)
                else:
                    return _ivy_gen.concatenate(containers, dim, f=f)
            except Exception as e:
                raise Exception(str(e) + '\nContainer concat operation only valid for containers of arrays')

    @staticmethod
    def from_disk(h5_obj_or_filepath, f, slice_obj=slice(None)):
        """
        Load container object from disk, as an h5py file, at the specified filepath.

        :param h5_obj_or_filepath: Filepath where the container object is saved to disk, or h5 object.
        :type h5_obj_or_filepath: str or h5 obj
        :param f: Machine learning framework.
        :type f: ml_framework, optional
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
                container_dict[key] = Container.from_disk(value, f, slice_obj)
            elif isinstance(value, _h5py.Dataset):
                if f is _np:
                    container_dict[key] = value[slice_obj]
                else:
                    container_dict[key] = _ivy_gen.array(list(value[slice_obj]), f=f)
            else:
                raise Exception('Item found inside h5_obj which was neither a Group nor a Dataset.')
        return Container(container_dict)

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

    # Private Methods #
    # ----------------#

    def _get_size(self):
        vals = list(self.values())
        if not vals:
            return 0
        val = vals[0]
        if isinstance(val, Container):
            return val._get_size()
        else:
            try:
                return val.shape[0]
            except (AttributeError, IndexError, TypeError):
                return 0

    # Public Methods #
    # ---------------#

    def shuffle(self, seed_value=None, f=None):
        """
        Shuffle entries in all sub-arrays, such that they are still aligned along axis 0.

        :param seed_value: random seed to use for array shuffling
        :type seed_value: int
        :param f: Machine learning framework. Inferred from inputs if None.
        :type f: ml_framework, optional
        """
        return_dict = dict()
        if seed_value is None:
            seed_value = _random.randint(0, 1000)
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_dict[key] = value.shuffle(seed_value)
            else:
                f = _get_framework(value, f=f)
                _ivy_rand.seed(seed_value, f=f)
                return_dict[key] = _ivy_rand.shuffle(value, f)
        return Container(return_dict)

    def slice(self, slice_obj):
        """
        Get slice of container object.

        :param slice_obj: slice object to slice all container elements.
        :type slice_obj: slice or sequence of slices
        :return: Container object at desired slice.
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_dict[key] = value.slice(slice_obj)
            else:
                # noinspection PyBroadException
                try:
                    return_dict[key] = value[slice_obj]
                except:
                    return_dict[key] = value

        return Container(return_dict)

    def expand_dims(self, axis):
        """
        Expand dims of all sub-arrays of container object.

        :param axis: Axis along which to expand dimensions of the sub-arrays.
        :type axis: int
        :return: Container object at with all sub-array dimensions expanded along the axis.
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_dict[key] = value.expand_dims(axis)
            else:
                return_dict[key] = _ivy_gen.expand_dims(value, axis)
        return Container(return_dict)

    def unstack(self, dim, dim_size):
        """
        Unstack containers along specified dimension.

        :param dim: Dimensions along which to unstack.
        :type dim: int
        :param dim_size: Size of the dimension to unstack.
        :type dim_size: int
        :return: List of containers, unstacked along the specified dimension.
        """
        return [self.slice(tuple([slice(None, None, None)] * dim + [slice(i, i + 1, 1)])) for i in range(dim_size)]

    def to_disk(self, h5_obj_or_filepath, starting_index=0, mode='a', max_batch_size=None):
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
                value.to_disk(h5_group, starting_index, mode, max_batch_size)
            else:
                value_as_np = _ivy_gen.to_numpy(value)
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

    def to_list(self):
        """
        Return nested list representation of container object.

        :return: Container as nested list.
        """
        return_list = list()
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_list.append(value.to_list())
            elif value is not None and key is not '_f':
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
            elif value is not None and key is not '_f':
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

    def to_random(self, f):
        """
        Return new container, with all entries having same shape and type, but random values

        :param f: Machine learning framework.
        :type f: ml_framework
        :return: Container with random values as entries.
        """
        def _as_random(value, _=''):
            if hasattr(value, 'shape'):
                return _ivy_rand.random_uniform(0., 1., value.shape, f=f)
            return value
        return self.map(_as_random)

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

    def map(self, func, key_chain=''):
        """
        Apply function to all array values of container

        :param func: Function to apply to each container entry
        :type func: python function
        :param key_chain: Chain of keys for this dict entry
        :type key_chain: str
        """
        return_dict = dict()
        for key, value in sorted(self.items()):
            this_key_chain = key if key_chain == '' else (key_chain + '/' + key)
            if isinstance(value, Container):
                return_dict[key] = value.map(func, this_key_chain)
            else:
                return_dict[key] = func(value, this_key_chain)
        return Container(return_dict)

    def dtype(self):
        """
        Return container, with all entries replaced with their data types.

        :return: New datatype container
        """
        return self.map(lambda x, _: _ivy_gen.dtype(x))

    def with_entries_as_lists(self, f):
        """
        Return container object, with each array entry in the container cast to a list

        :param f: Machine learning framework.
        :type f: ml_framework
        :return: New container, with entries not as arrays but as python lists
        """
        def to_list(x, _=''):
            try:
                return _ivy_gen.to_list(x, f)
            except (AttributeError, ValueError):
                return x
        return self.map(to_list)

    # Built-ins #
    # ----------#

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            # noinspection PyUnresolvedReferences
            return super.__getattr__(item)

    # Getters #
    # --------#

    @property
    def size(self):
        return self._size
