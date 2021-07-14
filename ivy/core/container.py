"""
Base Container Object
"""

# global
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
        Load container object from disk, as an h5py file, at the specified filepath.

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
        Load container object from disk at the specified filepath.

        :param pickle_filepath: Filepath where the container object is saved to disk.
        :type pickle_filepath: str
        :return: Container loaded from disk
        """
        return _pickle.load(open(pickle_filepath, 'rb'))

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

    # Public Methods #
    # ---------------#

    def shuffle(self, seed_value=None):
        """
        Shuffle entries in all sub-arrays, such that they are still aligned along axis 0.

        :param seed_value: random seed to use for array shuffling
        :type seed_value: int
        """
        return_dict = dict()
        if seed_value is None:
            seed_value = _random.randint(0, 1000)
        for key, value in sorted(self.items()):
            if isinstance(value, Container):
                return_dict[key] = value.shuffle(seed_value)
            else:
                _ivy.seed(seed_value)
                return_dict[key] = _ivy.shuffle(value)
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
                if isinstance(value, list) or isinstance(value, tuple):
                    if len(value) == 0:
                        return_dict[key] = value
                    else:
                        return_dict[key] = value[slice_obj]
                elif value.shape == ():
                    return_dict[key] = value
                else:
                    return_dict[key] = value[slice_obj]

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
            elif value is not None:
                return_dict[key] = _ivy.expand_dims(value, axis)
            else:
                return_dict[key] = value
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
        Save container object to disk, as an h5py file, at the specified filepath.

        :param pickle_filepath: Filepath for where to save the container to disk.
        :type pickle_filepath: str
        """
        _pickle.dump(self, open(pickle_filepath, 'wb'))

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

    def to_random(self):
        """
        Return new container, with all entries having same shape and type, but random values
        """
        def _as_random(value, _=''):
            if hasattr(value, 'shape'):
                return _ivy.random_uniform(0., 1., value.shape)
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

    # Built-ins #
    # ----------#

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            # noinspection PyUnresolvedReferences
            return super.__getattr__(item)

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
