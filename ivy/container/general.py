# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithGeneral(ContainerBase):

    def __init__(self):
        import ivy.functional.ivy.general as general
        self.add_instance_methods(general, to_ignore=['inplace_update', 'unstack', 'gather', 'gather_nd'])

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
            if (isinstance(value, dict_types) and (not isinstance(value, ivy.Container) or
                                                   self._rebuild_child_containers)) or \
                    isinstance(value, tuple(self._types_to_iteratively_nest)):
                self[key] = ivy.Container(value, **self._config)
            else:
                self[key] = value

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
