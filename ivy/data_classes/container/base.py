"""Base Container Object."""

# global
import inspect
from itertools import chain
import re
import abc
import copy
import termcolor
import numpy as np
import json

from ivy.utils.exceptions import IvyBackendException, IvyException


try:
    # noinspection PyPackageRequirements
    import h5py
except ModuleNotFoundError:
    h5py = None
import pickle
import random
from operator import mul
from functools import reduce as _reduce
from typing import Union, Tuple
from builtins import set

# local
import ivy


ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def _repr(x):
    try:
        return x.__repr__()
    except TypeError:
        return str(x)


# noinspection PyMissingConstructor


class ContainerBase(dict, abc.ABC):
    def __init__(
        self,
        dict_in=None,
        queues=None,
        queue_load_sizes=None,
        container_combine_method="list_join",
        queue_timeout=None,
        print_limit=10,
        key_length_limit=None,
        print_indent=4,
        print_line_spacing=0,
        ivyh=None,
        default_key_color="green",
        keyword_color_dict=None,
        rebuild_child_containers=False,
        types_to_iteratively_nest=None,
        alphabetical_keys=True,
        dynamic_backend=None,
        **kwargs,
    ):
        """
        Initialize container object from input dict representation.

        Parameters
        ----------
        dict_in
            the dictionary the container should wrap around. Default is ``None``.
        queues
            Sequence of multiprocessing queues, each of which returns containers.
            This enables the current container to be passed around asynchronously while
            waiting for data. Default is ``None``.
        queue_load_sizes
            Size of leading dimension of the containers returned by each queue.
            Default is ``None``.
        container_combine_method
            The method to use for combining containers arriving from different queues.
            Default is ivy.Container.cont_list_join
        queue_timeout
            The timeout when waiting for containers to arrive from the queues.
            Default is global.
        print_limit
            The total array size limit when printing the container. Default is 10.
        key_length_limit
            The maximum key length when printing the container. Default is ``None``.
        print_indent
            The number of whitespaces to use for indenting when printing the container.
            Default is 4.
        print_line_spacing
            The number of extra newlines to use between keys when printing the
            container. Default is ``0``.
        ivyh
            Handle to ivy module to use for the calculations. Default is ``None``, which
            results in the global ivy.
        default_key_color
            The default key color for printing the container to the terminal.
            Default is 'green'.
        keyword_color_dict
            A dict mapping keywords to their termcolor color codes for printing the
            container. (Default value = None)
        rebuild_child_containers
            Whether to rebuild container found in dict_in with these constructor params.
            Default is ``False``, in which case the original container are kept as are.
        types_to_iteratively_nest
            The data types to nest iteratively in the dict structure, each type must be
            iterable. Default is ``None``.
        alphabetical_keys
            Whether to sort the container keys alphabetically, or preserve the dict
            order. Default is ``True``.
        kwargs
            keyword arguments for dict creation. Default is ``None``.
        """
        self._queues = queues
        self._container_combine_method = container_combine_method
        if ivy.exists(self._queues):
            if isinstance(self._container_combine_method, str):
                self._container_combine_method = {
                    "list_join": self.cont_list_join,
                    "concat": lambda conts: self.concat(conts, 0),
                }[self._container_combine_method]
            self._loaded_containers_from_queues = dict()
            self._queue_load_sizes_cum = np.cumsum(queue_load_sizes)
            self._queue_timeout = ivy.default(queue_timeout, ivy.queue_timeout)
        if dynamic_backend is not None:
            self._dynamic_backend = dynamic_backend
        else:
            self._dynamic_backend = ivy.dynamic_backend
        if dict_in is None:
            if kwargs:
                dict_in = dict(**kwargs)
            else:
                dict_in = dict()
        elif kwargs:
            raise ivy.utils.exceptions.IvyException(
                "dict_in and **kwargs cannot both be specified for ivy.Container "
                "constructor, please specify one or the other, not both."
            )
        self._config_in = dict(
            print_limit=print_limit,
            print_indent=print_indent,
            key_length_limit=key_length_limit,
            print_line_spacing=print_line_spacing,
            ivyh=ivyh,
            default_key_color=default_key_color,
            keyword_color_dict=keyword_color_dict,
            rebuild_child_containers=rebuild_child_containers,
            types_to_iteratively_nest=types_to_iteratively_nest,
            alphabetical_keys=alphabetical_keys,
        )
        self._config = dict()
        self.cont_inplace_update(dict_in, **self._config_in)

    # Class Methods #
    # --------------#
    @staticmethod
    def cont_multi_map_in_function(
        fn,
        *args,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=None,
        out=None,
        **kwargs,
    ) -> Union[Tuple[ivy.Container, ivy.Container], ivy.Container]:
        inspect_fn = fn
        if isinstance(fn, str):
            inspect_fn = ivy.__dict__[fn]
        # retrieve indices where leaves of args are also nested
        arg_cont_idxs = ivy.nested_argwhere(
            args, ivy.is_ivy_container, to_ignore=ivy.Container
        )
        # retrieve indices where leaves of kwargs are also nested
        kwarg_cont_idxs = ivy.nested_argwhere(
            kwargs, ivy.is_ivy_container, to_ignore=ivy.Container
        )

        # retrieve all the containers in args
        arg_conts = ivy.multi_index_nest(args, arg_cont_idxs)
        num_arg_conts = len(arg_conts)

        # retrieve all the containers in kwargs
        kwarg_conts = ivy.multi_index_nest(kwargs, kwarg_cont_idxs)
        # Combine the retrieved containers from args and kwargs into a single list
        with_out = (
            inspect.signature(inspect_fn).parameters.get("out") is not None
            and out is not None
        )
        if with_out:
            out_conts = [out]
            num_out_conts = 1
            out_cont_idxs = []
            if not ivy.is_array(out) and not ivy.is_ivy_container(out):
                out_cont_idxs = ivy.nested_argwhere(
                    out, ivy.is_ivy_container, to_ignore=ivy.Container
                )
                out_conts = ivy.multi_index_nest(out, out_cont_idxs)
                num_out_conts = len(out_conts)
            conts = arg_conts + kwarg_conts + out_conts
        else:
            conts = arg_conts + kwarg_conts
        ivy.utils.assertions.check_exists(
            conts, message="no containers found in arguments"
        )
        cont0 = conts[0]
        if isinstance(fn, str):
            fn = cont0.cont_ivy.__dict__[fn]
        # Get the function with the name fn_name, enabling containers to specify
        # their backends irrespective of global ivy's backend

        def map_fn(vals, _):
            if with_out:
                out = vals[-num_out_conts:]
                del vals[-num_out_conts:]
            arg_vals = vals[:num_arg_conts]
            a = ivy.copy_nest(args, to_mutable=True)
            ivy.set_nest_at_indices(a, arg_cont_idxs, arg_vals)
            kwarg_vals = vals[num_arg_conts:]
            kw = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.set_nest_at_indices(kw, kwarg_cont_idxs, kwarg_vals)
            if with_out:
                out = out[0] if len(out) == 1 else out
                return fn(*a, out=out, **kw)
            else:
                return fn(*a, **kw)

        # Replace each container in arg and kwarg with the arrays at the leaf
        # levels of that container using map_fn and call fn using those arrays
        # as inputs
        ret = ivy.Container.cont_multi_map(
            map_fn,
            conts,
            key_chains,
            to_apply,
            prune_unapplied,
            map_nests=map_sequences,
        )

        # Multiple containers for functions returning multiple arrays
        if ivy.is_ivy_container(ret):
            for values in ret.values():
                if isinstance(values, (tuple, list)):
                    for v in values:
                        if ivy.is_ivy_array(v):
                            return ret.cont_unstack_conts(0)

        if with_out:
            for out_cont_idx, out_cont in zip(out_cont_idxs, out_conts):
                out_cont.cont_inplace_update(ivy.index_nest(ret, out_cont_idx))
            if len(out_conts) == 1:
                out.cont_inplace_update(ret)
            ret = out

        return ret

    @staticmethod
    def cont_handle_inplace(ret, out):
        """
        Return an inplace update of out, provided it is not None, by updating with the
        values in ret.

        Parameters
        ----------
        ret
            The container with the return values
        out
            The optional out container, which is primed for being overwritten if it
            exists

        Returns
        -------
            The out container, but filled with the values from the ret container
        """
        if ivy.exists(out):
            out.cont_inplace_update(ret)
            ret = out
        return ret

    @staticmethod
    def cont_list_join(containers, config=None):
        """
        Join containers of lists together along the specified dimension.

        Parameters
        ----------
        containers
            containers to list join
        config
            The configuration for the containers. Default is the same as container0.

        Returns
        -------
            List joined containers, with each entry being a list of arrays
        """
        container0 = containers[0]
        if not ivy.exists(config):
            config = (
                container0.cont_config if isinstance(container0, ivy.Container) else {}
            )

        if isinstance(container0, ivy.Container):
            return_dict = dict()
            for key in container0.keys():
                new_list = list()
                for container in containers:
                    new_list.append(container[key])
                return_dict[key] = ivy.Container.cont_list_join(new_list, config)
            return ivy.Container(return_dict, **config)
        else:
            return [item for sublist in containers for item in sublist]

    @staticmethod
    def cont_list_stack(containers, dim, config=None):
        """
        List stack containers together along the specified dimension.

        Parameters
        ----------
        containers
            containers to list stack
        dim
            dimension along which to list stack
        config
            The configuration for the containers. Default is the same as container0.

        Returns
        -------
            Stacked containers, with each entry being a list of arrays
        """
        container0 = containers[0]
        if not ivy.exists(config):
            config = (
                container0.cont_config if isinstance(container0, ivy.Container) else {}
            )

        if isinstance(container0, ivy.Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = ivy.Container.cont_list_stack(
                    [container[key] for container in containers], dim, config
                )
            return ivy.Container(return_dict, **config)
        else:
            return containers

    @staticmethod
    def _cont_concat_unify(containers, device, axis=0):
        return ivy.concat(
            [cont.to_device(device) for cont in containers.values()], axis=axis
        )

    @staticmethod
    def _cont_sum_unify(containers, device, _=None, _1=None):
        return sum(
            [cont.to_device(device) for cont in containers.values()],
            start=ivy.zeros([]),
        )

    @staticmethod
    def _cont_mean_unify(containers, device, _=None, _1=None):
        return ivy.Container._cont_sum_unify(containers, device) / len(containers)

    @staticmethod
    def cont_unify(containers, device, mode, axis=0):
        """
        Unify a list of containers, on arbitrary devices, to a single container on the
        specified device.

        Parameters
        ----------
        containers
            containers to unify
        dev
            The device to unify the containers to.
        mode
            The mode by which to unify, must be one of [ concat | mean | sum ]
        axis
            The axis along which to concattenate the container, if concat mode is set.
            Default is ``0``.

        Returns
        -------
            Unified container
        """
        return {
            "concat": ivy.Container._cont_concat_unify,
            "sum": ivy.Container._cont_sum_unify,
            "mean": ivy.Container._cont_mean_unify,
        }[mode](containers, device, axis)

    @staticmethod
    def cont_combine(*containers, config=None):
        """
        Combine keys and values in a sequence of containers, with priority given to the
        right-most container in the case of duplicates.

        Parameters
        ----------
        containers
            containers to compare
        config
            The configuration for the containers. Default is the same as
            container_rightmost.

        Returns
        -------
            Combined containers
        """
        # if inputs are not dicts, then simply return the right-most value
        container_rightmost = containers[-1]
        if not isinstance(container_rightmost, dict):
            return container_rightmost

        if not ivy.exists(config):
            # noinspection PyUnresolvedReferences
            config = (
                container_rightmost.cont_config
                if isinstance(container_rightmost, ivy.Container)
                else {}
            )

        # return if len==1
        if len(containers) == 1:
            return container_rightmost

        # otherwise, check that the keys are aligned between each container, and apply
        # this method recursively
        return_dict = dict()
        all_keys = set(
            [
                item
                for sublist in [list(cont.keys()) for cont in containers]
                for item in sublist
            ]
        )
        for key in all_keys:
            keys_present = [key in cont for cont in containers]
            return_dict[key] = ivy.Container.cont_combine(
                *[cont[key] for cont, kp in zip(containers, keys_present) if kp],
                config=config,
            )
        return ivy.Container(return_dict, **config)

    @staticmethod
    def cont_diff(
        *containers,
        mode="all",
        diff_keys="diff",
        detect_key_diffs=True,
        detect_value_diffs=True,
        detect_shape_diffs=True,
        config=None,
    ):
        """
        Compare keys and values in a sequence of containers, returning the single shared
        values where they are the same, and new nested sub-dicts with all values where
        they are different.

        Parameters
        ----------
        containers
            containers to compare
        mode
            The mode of the diff operation, returning either all keys and values,
            only those that are consist across the containers, or only the differences.
            Default is all.
        diff_keys
            The key/keys to add to the returned container when differences are found.
            Default is ``"diff"``.
        detect_key_diffs
            Whether to treat different keys as detected differences. If not, the keys
            among the input containers are simply combined without flagging differences.
            Default is ``True``.
        detect_value_diffs
            Whether to treat different values as detected differences.
            Default is ``True``.
        detect_shape_diffs
            Whether to treat different array shapes as detected differences.
            Default is ``True``.
        config
            The configuration for the containers. Default is the same as container0.
        *containers


        Returns
        -------
            Compared containers
        """
        ivy.utils.assertions.check_elem_in_list(mode, ["all", "same_only", "diff_only"])

        # if inputs are not dicts, then compare their values to determine the diff dict
        num_containers = len(containers)
        container0 = containers[0]
        if not ivy.exists(config):
            config = (
                container0.cont_config if isinstance(container0, ivy.Container) else {}
            )
        if not isinstance(container0, dict):
            equal_mat = ivy.all_equal(*containers, equality_matrix=True)
            if not detect_value_diffs:
                equal_mat = ivy.ones_like(equal_mat)
            if detect_shape_diffs:
                shape_equal_mat = ivy.all_equal(
                    *[c.shape if ivy.is_array(c) else None for c in containers],
                    equality_matrix=True,
                )
                equal_mat = ivy.logical_and(equal_mat, shape_equal_mat)
            # noinspection PyTypeChecker
            if ivy.min(ivy.astype(equal_mat, "int32")) == 1:
                if mode == "diff_only":
                    return ivy.Container(**config)
                return container0
            elif mode == "same_only":
                return ivy.Container(**config)
            else:
                cont_range = range(num_containers)
                diff_dict = dict()
                cont_dict = dict(zip(cont_range, containers))
                idxs_added = list()
                for idx in cont_range:
                    if idx not in idxs_added:
                        idxs_to_add = ivy.argwhere(equal_mat[idx])
                        idxs_to_add_list = sorted(
                            ivy.to_numpy(idxs_to_add).reshape(-1).tolist()
                        )
                        if isinstance(diff_keys, str):
                            key = diff_keys + "_" + str(idxs_to_add_list)[1:-1]
                        elif isinstance(diff_keys, (list, tuple)):
                            key = diff_keys[idx]
                        else:
                            raise ivy.utils.exceptions.IvyException(
                                "diff_keys must be either a string or list of strings,"
                                "but found {} of type {}".format(
                                    diff_keys, type(diff_keys)
                                )
                            )
                        diff_dict[key] = cont_dict[idx]
                        idxs_added += idxs_to_add_list
                return ivy.Container(diff_dict, **config)

        # otherwise, check that the keys are aligned between each container, and apply
        # this method recursively
        return_dict = dict()
        all_keys = set(
            [
                item
                for sublist in [list(cont.keys()) for cont in containers]
                for item in sublist
            ]
        )
        for key in all_keys:
            keys_present = [key in cont for cont in containers]
            all_keys_present = sum(keys_present) == num_containers
            if all_keys_present:
                res = ivy.Container.cont_diff(
                    *[cont[key] for cont in containers],
                    mode=mode,
                    diff_keys=diff_keys,
                    detect_key_diffs=detect_key_diffs,
                    detect_value_diffs=detect_value_diffs,
                    detect_shape_diffs=detect_shape_diffs,
                    config=config,
                )
                if not isinstance(res, dict) or res:
                    return_dict[key] = res
                continue
            elif sum(keys_present) == 1 and not detect_key_diffs:
                if mode == "all":
                    return_dict[key] = containers[keys_present.index(True)][key]
                continue
            diff_dict = dict()
            for i, (key_present, cont) in enumerate(zip(keys_present, containers)):
                if detect_key_diffs:
                    if key_present and mode != "same_only":
                        if isinstance(diff_keys, str):
                            diff_dict[diff_keys + "_" + str(i)] = cont[key]
                        elif isinstance(diff_keys, (list, tuple)):
                            diff_dict[diff_keys[i]] = cont[key]
                        else:
                            raise ivy.utils.exceptions.IvyException(
                                "diff_keys must be either a string or list of strings,"
                                "but found {} of type {}".format(
                                    diff_keys, type(diff_keys)
                                )
                            )
            if diff_dict:
                return_dict[key] = diff_dict
        return ivy.Container(return_dict, **config)

    @staticmethod
    def cont_structural_diff(
        *containers,
        mode="all",
        diff_keys="diff",
        detect_key_diffs=True,
        detect_shape_diffs=True,
        config=None,
    ):
        """
        Compare keys and shapes in a sequence of containers, returning the single shared
        values where they are the same, and new nested sub-dicts with all values where
        they are different.

        Parameters
        ----------
        containers
            containers to compare
        mode
            The mode of the diff operation, returning either all keys and values,
            only those that are consist across the containers, or only the differences.
            Default is all.
        diff_keys
            The key/keys to add to the returned container when differences are found.
            Default is "diff".
        detect_key_diffs
            Whether to treat different keys as detected differences.
            If not, the keys among the input containers are simply combined without
            flagging differences. Default is ``True``.
        detect_shape_diffs
            Whether to treat different array shapes as detected differences.
            Default is ``True``.
        config
            The configuration for the containers. Default is the same as container0.
        *containers

        Returns
        -------
            Compared containers
        """
        return ivy.Container.cont_diff(
            *containers,
            mode=mode,
            diff_keys=diff_keys,
            detect_key_diffs=detect_key_diffs,
            detect_value_diffs=False,
            detect_shape_diffs=detect_shape_diffs,
            config=config,
        )

    @staticmethod
    def cont_multi_map(
        func,
        containers,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        key_chain="",
        config=None,
        map_nests=False,
        assert_identical=False,
    ):
        """
        Apply function to all array values from a collection of containers.

        Parameters
        ----------
        func
            Function to apply to each container entry.
        containers
            containers to map.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied,
            otherwise the leftmost container value is used. Default is ``False``.
        key_chain
            Chain of keys for this dict entry (Default value = '')
        config
            The configuration for the containers. Default is the same as container0.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        assert_identical
            Whether to assert that the input containers are identical or not.

        Returns
        -------
            Container
        """
        # retrieve all keys and the first container if it exists
        keys = set([])
        container0 = None
        for container in containers:
            if isinstance(container, ivy.Container):
                if container0 is None:
                    container0 = container
                keys = keys.union(container.keys())

        ivy.utils.assertions.check_exists(
            container0,
            message="No containers found in the inputs to ivy.Container.cont_multi_map",
        )
        if not ivy.exists(config):
            config = (
                container0.cont_config if isinstance(container0, ivy.Container) else {}
            )
        return_dict = dict()

        for key in keys:
            values = []
            for cont in containers:
                if isinstance(cont, (ivy.Container, list, tuple)) and key in cont:
                    values.append(cont[key])
                elif not isinstance(cont, (ivy.Container, list, tuple)):
                    values.append(cont)
            value0 = values[0]
            if len(values) >= 1:
                this_key_chain = key if key_chain == "" else (key_chain + "/" + key)
                is_container = [ivy.is_ivy_container(x) for x in values]

                def _found_in_key_chains(this_key_chain, key_chains):
                    if key_chains is None:
                        return False
                    for key_chain in key_chains:
                        if this_key_chain.startswith(key_chain):
                            return True
                    return False

                if not assert_identical and not all(is_container) and any(is_container):
                    found = _found_in_key_chains(this_key_chain, key_chains)
                    if key_chains is not None:
                        if (found and not to_apply) or (not found and to_apply):
                            if prune_unapplied:
                                continue
                            return_dict[key] = value0
                            continue
                    return_dict[key] = func(values, this_key_chain)
                else:
                    if isinstance(value0, ivy.Container):
                        ret = ivy.Container.cont_multi_map(
                            func,
                            values,
                            key_chains,
                            to_apply,
                            prune_unapplied,
                            this_key_chain,
                            config,
                            map_nests,
                            assert_identical,
                        )
                        if ret:
                            return_dict[key] = ret
                    elif (
                        any(isinstance(x, (list, tuple)) for x in values) and map_nests
                    ):
                        ret = ivy.nested_multi_map(
                            lambda x, _: func(x, None), values, to_ivy=False
                        )
                        if prune_unapplied and not ret:
                            continue
                        return_dict[key] = ret
                    else:
                        found = _found_in_key_chains(this_key_chain, key_chains)
                        if key_chains is not None:
                            if (found and not to_apply) or (not found and to_apply):
                                if prune_unapplied:
                                    continue
                                return_dict[key] = value0
                                continue
                        return_dict[key] = func(values, this_key_chain)
            else:
                return_dict[key] = value0

            # noinspection PyProtectedMember
        return ivy.Container(return_dict, **config)

    @staticmethod
    def cont_common_key_chains(containers):
        """
        Return the key-chains common across all containers.

        Parameters
        ----------
        containers
            Containers to check.

        Returns
        -------
            list of key-chains.
        """
        if len(containers) == 1:
            return containers[0].cont_all_key_chains()
        sets = [set(cont.cont_all_key_chains()) for cont in containers]
        return list(sets[0].intersection(*sets[1:]))

    @staticmethod
    def cont_identical(
        containers,
        check_types=True,
        check_shapes=True,
        same_arrays=True,
        arrays_equal=True,
        key_chains=None,
        to_apply=True,
        partial=False,
        key_chain="",
        build_callable=False,
        assert_and_assign=False,
    ):
        """
        Return a single boolean as to whether the input containers have identical key-
        chains and data types.

        Parameters
        ----------
        containers
            containers to check.
        check_types
            Whether to check if the datatypes of the leaf nodes are the same.
            Default is ``True``.
        check_shapes
            Whether to check if the shapes of the leaf nodes are the same.
            Default is ``True``.
        same_arrays
            Whether to check if the arrays are the exact same instances.
            Default is ``True``.
        arrays_equal
            Whether to check if the arrays have equal values. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        key_chain
            Chain of keys for this dict entry (Default value = '')

        Returns
        -------
        Boolean
        """
        if partial:
            common_key_chains = ivy.Container.cont_common_key_chains(containers)
            if not common_key_chains:
                return False
            containers = [
                cont.cont_at_key_chains(common_key_chains) for cont in containers
            ]
        keys = set([i for sl in [list(cont.keys()) for cont in containers] for i in sl])

        # noinspection PyProtectedMember
        for key in keys:
            if not min([key in cont for cont in containers]):
                return False
            if build_callable:
                # call the callable if encountered
                for cont in containers:
                    cont[key] = cont[key]() if callable(cont[key]) else cont[key]
            values = [cont[key] for cont in containers]
            value_0 = values[0]
            type_0 = type(value_0)
            types = [type(val) for val in values]
            if not min([type_n is type_0 for type_n in types]):
                if isinstance(value_0, ivy.Container) or check_types:
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
                    if not ivy.all_equal(*values):
                        return False
                if assert_and_assign:
                    containers[0].cont_set_at_key_chains(
                        target_dict=containers[1], inplace=True
                    )
            this_key_chain = key if key_chain == "" else (key_chain + "/" + key)
            if isinstance(value_0, ivy.Container):
                ret = ivy.Container.cont_identical(
                    values,
                    check_types,
                    check_shapes,
                    same_arrays,
                    arrays_equal,
                    key_chains,
                    to_apply,
                    partial,
                    this_key_chain,
                    build_callable=build_callable,
                    assert_and_assign=assert_and_assign,
                )
                if not ret:
                    return False
                if assert_and_assign:
                    try:
                        containers[0][key].cont_set_at_key_chains(
                            target_dict=containers[1][key], inplace=True
                        )
                    except:
                        print("why it didn't work")
                        print(containers)

        return True

    @staticmethod
    def cont_assert_identical(
        containers,
        check_types=True,
        check_shapes=True,
        same_arrays=True,
        arrays_equal=True,
        key_chains=None,
        to_apply=True,
        partial=False,
    ):
        """
        Assert whether the input containers are identical. Otherwise, the diff is shown
        in an exception.

        Parameters
        ----------
        containers
            containers to check.
        check_types
            Whether to check if the datatypes of the leaf nodes are the same.
            Default is ``True``.
        check_shapes
            Whether to check if the shapes of the leaf nodes are the same.
            Default is ``True``.
        same_arrays
            Whether to check if the arrays are the exact same instances.
            Default is ``True``.
        arrays_equal
            Whether to check if the arrays have equal values. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        ivy.utils.assertions.check_true(
            ivy.Container.cont_identical(
                containers,
                check_types,
                check_shapes,
                same_arrays,
                arrays_equal,
                key_chains,
                to_apply,
                partial,
            ),
            "Containers were not identical:\n\n{}".format(
                ivy.Container.cont_diff(*containers)
            ),
        )

    @staticmethod
    def cont_identical_structure(
        containers,
        check_types=True,
        check_shapes=True,
        key_chains=None,
        to_apply=True,
        partial=False,
        key_chain="",
        build_callable=False,
        assert_and_assign=False,
    ):
        """
        Return a single boolean as to whether the input containers have identical
        structure.

        Parameters
        ----------
        containers
            containers to check.
        check_types
            Whether to also check whether the datatypes of the leaf nodes are the same.
            Default is ``True``.
        check_shapes
            Whether to also check whether the shapes of the leaf nodes are the same.
            Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        key_chain
            Chain of keys for this dict entry (Default value = '')

        Returns
        -------
            Boolean
        """
        return ivy.Container.cont_identical(
            containers,
            check_types,
            check_shapes,
            False,
            False,
            key_chains,
            to_apply,
            partial,
            key_chain,
            build_callable=build_callable,
            assert_and_assign=assert_and_assign,
        )

    @staticmethod
    def cont_assert_identical_structure(
        containers,
        check_types=True,
        check_shapes=True,
        key_chains=None,
        to_apply=True,
        partial=False,
        build_callable=False,
        assert_and_assign=False,
    ):
        """
        Assert whether the input containers have identical structure. Otherwise, the
        diff is shown in an exception.

        Parameters
        ----------
        containers
            containers to check.
        check_types
            Whether to also check whether the datatypes of the leaf nodes are the same.
            Default is ``True``.
        check_shapes
            Whether to also check whether the shapes of the leaf nodes are the same.
            Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        ivy.utils.assertions.check_true(
            ivy.Container.cont_identical_structure(
                containers,
                check_types,
                check_shapes,
                key_chains,
                to_apply,
                partial,
                build_callable=build_callable,
                assert_and_assign=assert_and_assign,
            ),
            "Containers did not have identical structure:\n\n{}".format(
                ivy.Container.cont_structural_diff(*containers)
            ),
        )

    @staticmethod
    def cont_identical_configs(containers):
        """
        Return a single boolean as to whether the input containers all have identical
        configs.

        Parameters
        ----------
        containers
            containers to check.
        """
        ivy.utils.assertions.check_greater(len(containers), 1, as_array=False)
        configs = [cont.cont_config for cont in containers]
        config0 = configs[0]
        for k, v in config0.items():
            if not min([config[k] == v for config in configs]):
                return False
        return True

    @staticmethod
    def cont_identical_array_shapes(containers, exclusive=False):
        """
        Determine whether all of the containers have identical number of arrays and
        identical array shapes, regardless of their key-chain structures.

        Parameters
        ----------
        containers
            containers to check.
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array. (Default value = False)

        Returns
        -------
            Boolean
        """
        array_conts = [cont.cont_size_ordered_arrays(exclusive) for cont in containers]
        array_cont0 = array_conts[0]
        array_cont0_len = len(array_cont0)
        for array_cont in array_conts[1:]:
            if len(array_cont) != array_cont0_len:
                return False
            elif not min(
                [
                    a.shape == a0.shape
                    for a, a0 in zip(array_cont.values(), array_cont0.values())
                ]
            ):
                return False
        return True

    @staticmethod
    def cont_load(filepath, format="h5py"):
        if format == "json":
            return ivy.Container.cont_from_disk_as_json(filepath)
        elif format == "pickle":
            return ivy.Container.cont_from_disk_as_pickled(filepath)
        elif format == "h5py":
            return ivy.Container.cont_from_disk_as_hdf5(filepath)
        else:
            raise ivy.utils.exceptions.IvyException("Unsupported format")

    @staticmethod
    def cont_from_disk_as_hdf5(
        h5_obj_or_filepath, slice_obj=slice(None), alphabetical_keys=True, ivyh=None
    ):
        """
        Load container object from disk, as an h5py file, at the specified hdf5
        filepath.

        Parameters
        ----------
        h5_obj_or_filepath
            Filepath where the container object is saved to disk, or h5 object.
        slice_obj
            slice object to slice all h5 elements. (Default value = slice(None))
        alphabetical_keys
            Whether to sort the container keys alphabetically, or preserve the dict
            order. Default is ``True``.
        ivyh
            Handle to ivy module to use for the calculations. Default is ``None``, which
            results in the global ivy.

        Returns
        -------
            Container loaded from disk
        """
        ivy.utils.assertions.check_exists(
            h5py,
            message=(
                "You must install python package h5py in order to load hdf5 "
                "files from disk into a container."
            ),
        )
        container_dict = dict()
        if type(h5_obj_or_filepath) is str:
            h5_obj = h5py.File(h5_obj_or_filepath, "r")
        else:
            h5_obj = h5_obj_or_filepath
        items = sorted(h5_obj.items()) if alphabetical_keys else h5_obj.items()
        for key, value in items:
            if isinstance(value, h5py.Group):
                container_dict[key] = ivy.Container.cont_from_disk_as_hdf5(
                    value, slice_obj, ivyh
                )
            elif isinstance(value, h5py.Dataset):
                container_dict[key] = ivy.default(ivyh, ivy).array(
                    list(value[slice_obj]), dtype=str(value[slice_obj].dtype)
                )
            else:
                raise ivy.utils.exceptions.IvyException(
                    "Item found inside h5_obj which was neither a Group nor a Dataset."
                )
        return ivy.Container(container_dict, ivyh=ivyh)

    @staticmethod
    def cont_from_disk_as_pickled(pickle_filepath, ivyh=None):
        """
        Load container object from disk at the specified pickle filepath.

        Parameters
        ----------
        pickle_filepath
            Filepath where the container object is saved to disk.
        ivyh
            Handle to ivy module to use for the calculations. Default is ``None``, which
            results in the global ivy.

        Returns
        -------
            Container loaded from disk
        """
        return ivy.Container(
            pickle.load(open(pickle_filepath, "rb")),
            rebuild_child_containers=True,
            ivyh=ivyh,
        ).to_ivy()

    @staticmethod
    def cont_from_disk_as_json(json_filepath, ivyh=None):
        """
        Load container object from disk at the specified json filepath. If some objects
        were not json-able during saving, then they will be loaded as strings.

        Parameters
        ----------
        json_filepath
            Filepath where the container object is saved to disk.
        ivyh
            Handle to ivy module to use for the calculations. Default is ``None``, which
            results in the global ivy.

        Returns
        -------
            Container loaded from disk
        """
        with open(json_filepath) as json_data_file:
            return ivy.Container(json.load(json_data_file), ivyh=ivyh)

    @staticmethod
    def h5_file_size(h5_obj_or_filepath):
        """
        Get file size of h5 file contents.

        Parameters
        ----------
        h5_obj_or_filepath
            Filepath where the container object is saved to disk, or h5 object.

        Returns
        -------
            Size of h5 file contents, and batch size.
        """
        ivy.utils.assertions.check_exists(
            h5py,
            message=(
                "You must install python package h5py in order to determine "
                "the size of hdf5 files."
            ),
        )
        if type(h5_obj_or_filepath) is str:
            h5_obj = h5py.File(h5_obj_or_filepath, "r")
        else:
            h5_obj = h5_obj_or_filepath

        size = 0
        batch_size = 0
        for key, value in h5_obj.items():
            if isinstance(value, h5py.Group):
                size_to_add, batch_size = ivy.Container.h5_file_size(value)
                size += size_to_add
            elif isinstance(value, h5py.Dataset):
                value_shape = value.shape
                size += _reduce(mul, value_shape, 1) * value.dtype.itemsize
                batch_size = value_shape[0]
            else:
                raise ivy.utils.exceptions.IvyException(
                    "Item found inside h5_obj which was neither a Group nor a Dataset."
                )
        return size, batch_size

    @staticmethod
    def shuffle_h5_file(h5_obj_or_filepath, seed_value=0):
        """
        Shuffle entries in all datasets of h5 file, such that they are still aligned
        along axis 0.

        Parameters
        ----------
        h5_obj_or_filepath
            Filepath where the container object is saved to disk, or h5 object.
        seed_value
            random seed to use for array shuffling (Default value = 0)
        """
        ivy.utils.assertions.check_exists(
            h5py,
            message=(
                "You must install python package h5py in order to shuffle "
                "hdf5 files on disk."
            ),
        )
        if seed_value is None:
            seed_value = random.randint(0, 1000)
        if type(h5_obj_or_filepath) is str:
            h5_obj = h5py.File(h5_obj_or_filepath, "a")
        else:
            h5_obj = h5_obj_or_filepath

        for key, value in h5_obj.items():
            if isinstance(value, h5py.Group):
                ivy.Container.shuffle_h5_file(value, seed_value)
            elif isinstance(value, h5py.Dataset):
                random.seed(seed_value)
                # noinspection PyTypeChecker
                random.shuffle(value)
            else:
                raise ivy.utils.exceptions.IvyException(
                    "Item found inside h5_obj which was neither a Group nor a Dataset."
                )
        if isinstance(h5_obj, h5py.File):
            h5_obj.close()

    @staticmethod
    def cont_reduce(containers, reduction, config=None):
        """
        Reduce containers.

        Parameters
        ----------
        containers
            containers to reduce
        reduction
            the reduction function
        config
            The configuration for the containers. Default is the same as container0.

        Returns
        -------
            reduced containers
        """
        container0 = containers[0]
        if not ivy.exists(config):
            config = (
                container0.cont_config if isinstance(container0, ivy.Container) else {}
            )

        if isinstance(container0, ivy.Container):
            return_dict = dict()
            for key in container0.keys():
                return_dict[key] = ivy.Container.cont_reduce(
                    [container[key] for container in containers], reduction
                )
            return ivy.Container(return_dict, **config)
        else:
            # noinspection PyBroadException
            try:
                return reduction(containers)
            except Exception as e:
                raise ivy.utils.exceptions.IvyException(
                    str(e)
                    + "\nContainer reduce operation only valid for containers of arrays"
                )

    @staticmethod
    def cont_flatten_key_chain(
        key_chain, replacement="__", above_height=None, below_depth=None
    ):
        """
        Summary.

        Parameters
        ----------
        key_chain
            param replacement: (Default value = '__')
        above_height
            Default value = None)
        below_depth
            Default value = None)
        replacement
             (Default value = '__')
        """
        # noinspection RegExpSingleCharAlternation
        flat_keys = re.split("/|\.", key_chain)  # noqa
        num_keys = len(flat_keys)
        pre_keys = list()
        post_keys = list()
        if above_height and num_keys > above_height:
            post_keys = flat_keys[-above_height:]
            del flat_keys[-above_height:]
        if below_depth and num_keys > below_depth:
            pre_keys = flat_keys[0:below_depth]
            del flat_keys[0:below_depth]
        return "/".join(
            [
                k
                for k in [
                    "/".join(pre_keys),
                    replacement.join(flat_keys),
                    "/".join(post_keys),
                ]
                if k
            ]
        )

    @staticmethod
    def cont_trim_key(key, max_length):
        """
        Summary. Returns a trimmed key with a maximum length of max_length.

        Parameters
        ----------
        key
            key to trim
        max_length
            maximum length of key
        """
        key_len = len(key)
        if not ivy.exists(max_length) or key_len <= max_length:
            return key
        idxs = (
            np.round(
                (key_len - 1)
                / (max_length - 1)
                * np.linspace(0, max_length - 1, max_length)
            )
            .astype(np.int32)
            .tolist()
        )
        return "".join([key[idx] for idx in idxs])

    # Private Methods #
    # ----------------#

    def _cont_call_static_method_with_flexible_args(
        self,
        static_method,
        *args,
        kw,
        required,
        defaults,
        self_idx=0,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=None,
        out=None,
    ) -> ivy.Container:
        if args:
            num_args = len(args)
            kw = {
                k: defaults[k] if k in defaults else v
                for i, (k, v) in enumerate(kw.items())
                if i > num_args
            }
            args = list(args)
            if self_idx > num_args:
                k = list(kw.keys())[self_idx - num_args - 1]
                kw[k] = self
            else:
                args.insert(self_idx, self)
            return static_method(
                *args,
                **kw,
                key_chains=key_chains,
                to_apply=to_apply,
                prune_unapplied=prune_unapplied,
                map_sequences=map_sequences,
                out=out,
            )
        self_set = False
        # set to leftmost non-specified required arg, if present
        for k in required:
            if kw[k] is None:
                kw[k] = self
                self_set = True
                break
        # go through each key and value of the keyword arguments
        for k, v in kw.items():
            if v is None:
                if self_set:
                    if k in defaults:
                        # if self is set and a default value exists, set it
                        kw[k] = defaults[k]
                else:
                    # otherwise set self to this argument
                    kw[k] = self
                    self_set = True
        # call the static method
        return static_method(
            **kw,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def _cont_get_shape(self):
        if not len(self.keys()):
            if ivy.exists(self._queues):
                return [self._queue_load_sizes_cum[-1]]
            return [0]
        sub_shapes = [
            v
            for k, v in self.cont_map(
                lambda x, kc: (
                    list(x.shape)
                    if self._cont_ivy.is_native_array(x) or isinstance(x, ivy.Array)
                    else ([len(x)] if isinstance(x, (list, tuple)) else None)
                )
            ).cont_to_iterator()
            if v
        ]
        if not sub_shapes:
            return sub_shapes
        min_num_dims = min([len(sub_shape) for sub_shape in sub_shapes])
        sub_shapes_array = np.asarray(
            [sub_shape[0:min_num_dims] for sub_shape in sub_shapes]
        )
        sub_shapes_array = np.where(sub_shapes_array == 0, -1, sub_shapes_array)
        mask = np.prod(sub_shapes_array / sub_shapes_array[0:1], 0) == 1
        # noinspection PyTypeChecker
        return [
            None if np.isnan(i) else int(i)
            for i in np.where(
                mask, sub_shapes_array[0], np.ones(min_num_dims) * float("nan")
            ).tolist()
        ]

    def _cont_get_shapes(self):
        return self.cont_map(lambda x, kc: x.shape if hasattr(x, "shape") else None)

    def _cont_get_dev(self, as_native=False):
        sub_devs = [
            v
            for k, v in self.cont_map(
                lambda x, kc: (
                    self._cont_ivy.dev(x, as_native=as_native)
                    if self._cont_ivy.is_native_array(x) or isinstance(x, ivy.Array)
                    else None
                )
            ).cont_to_iterator()
            if v
        ]
        if len(set(sub_devs)) <= 1:
            return sub_devs[0]
        return None

    def _cont_at_key_chains_input_as_seq(self, key_chains, ignore_key_errors=False):
        return_cont = ivy.Container(dict(), **self._config)
        for kc in key_chains:
            val = self.cont_at_key_chain(kc, ignore_key_errors=ignore_key_errors)
            if ignore_key_errors and not ivy.exists(val):
                continue
            return_cont.cont_set_at_key_chain(kc, val, inplace=True)
        return return_cont

    def _cont_at_key_chains_input_as_dict(
        self, key_chains, current_chain="", ignore_key_errors=False
    ):
        return_dict = dict()
        for k, v in key_chains.items():
            if current_chain == "":
                new_current_chain = k
            else:
                new_current_chain = current_chain + "/" + k
            if isinstance(v, dict):
                return_dict[k] = self._cont_at_key_chains_input_as_dict(
                    v, new_current_chain, ignore_key_errors=ignore_key_errors
                )
            else:
                val = self.cont_at_key_chain(
                    new_current_chain, ignore_key_errors=ignore_key_errors
                )
                if ignore_key_errors and not ivy.exists(val):
                    continue
                return_dict[k] = val
        return ivy.Container(return_dict, **self._config)

    def _cont_prune_key_chains_input_as_seq(self, key_chains):
        return_cont = self.cont_copy()
        for kc in key_chains:
            return_cont = return_cont.cont_prune_key_chain(kc)
        return return_cont

    def _cont_prune_key_chains_input_as_dict(self, key_chains, return_cont=None):
        if return_cont is None:
            return_cont = self.cont_copy()
        for k, v in key_chains.items():
            if isinstance(v, dict):
                ret_cont = self._cont_prune_key_chains_input_as_dict(v, return_cont[k])
                if ret_cont.cont_shape[0] == 0:
                    del return_cont[k]
            else:
                del return_cont[k]
        return return_cont

    # Public Methods #
    # ---------------#

    def cont_duplicate_array_keychains(self):
        duplciates = ()
        key_chains = self.cont_all_key_chains()
        skips = set()
        for i in range(len(key_chains)):
            temp_duplicates = ()
            if key_chains[i] in skips:
                continue
            for j in range(i + 1, len(key_chains)):
                if key_chains[j] in skips:
                    continue
                if self[key_chains[i]] is self[key_chains[j]]:
                    if key_chains[i] not in temp_duplicates:
                        temp_duplicates += (key_chains[i],)
                    if key_chains[j] not in temp_duplicates:
                        temp_duplicates += (key_chains[j],)
            if len(temp_duplicates) > 0:
                duplciates += (temp_duplicates,)
            skips = chain.from_iterable(duplciates)
        return duplciates

    def cont_update_config(self, **config):
        new_config = dict()
        for k, v in config.items():
            att_name = "_" + k
            if k in self._config_in:
                if k == "types_to_iteratively_nest":
                    v = ivy.default(lambda: tuple(v), (), catch_exceptions=True)
                elif k == "keyword_color_dict":
                    v = ivy.default(v, {})
                elif k == "ivyh":
                    att_name = "_local_ivy"
                new_config[k] = v
                self.__setattr__(att_name, v)

        self._config = new_config

    def cont_inplace_update(
        self, dict_in: Union[ivy.Container, dict], **config
    ) -> ivy.Container:
        """
        Update the contents of this container inplace, using either a new dict or
        container.

        Parameters
        ----------
        dict_in
            New dict or container to update the current container inplace with.
        **config
        """
        # # update config
        self.cont_update_config(**config)

        # update container values inplace
        if dict_in is None:
            return
        dict_types = tuple([dict] + ivy.container_types())
        if isinstance(dict_in, dict_types):
            dict_in = dict_in
        elif isinstance(dict_in, tuple(self._types_to_iteratively_nest)):
            dict_in = dict(
                zip(
                    [
                        "it_{}".format(str(i).zfill(len(str(len(dict_in)))))
                        for i in range(len(dict_in))
                    ],
                    dict_in,
                )
            )
        else:
            raise ivy.utils.exceptions.IvyException("invalid input {}".format(dict_in))
        items = sorted(dict_in.items()) if self._alphabetical_keys else dict_in.items()
        for key, value in items:
            if (
                isinstance(value, dict_types)
                and (
                    not isinstance(value, ivy.Container)
                    or self._rebuild_child_containers
                )
            ) or isinstance(value, tuple(self._types_to_iteratively_nest)):
                self[key] = ivy.Container(value, **self._config)
            else:
                if key in self and isinstance(self[key], ivy.Container):
                    self[key].cont_inplace_update(value)
                else:
                    self[key] = value

    def cont_all_true(
        self,
        assert_is_bool=False,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
    ):
        """
        Determine whether all the entries in the container boolean evaluate to True.

        Parameters
        ----------
        assert_is_bool
            Whether or not to assert each entry is of type Boolean.
            (Default value = False)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
            Boolean, whether all entries are boolean True.
        """
        return bool(
            np.prod(
                [
                    v
                    for k, v in self.cont_as_bools(
                        assert_is_bool,
                        key_chains,
                        to_apply,
                        prune_unapplied,
                        map_sequences,
                    ).cont_to_iterator()
                ]
            )
        )

    def cont_all_false(
        self,
        assert_is_bool=False,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
    ):
        """
        Determine whether all the entries in the container boolean evaluate to False.

        Parameters
        ----------
        assert_is_bool
            Whether or not to assert each entry is of type Boolean.
            (Default value = False)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
            Boolean, whether all entries are boolean False.
        """
        return not bool(
            np.sum(
                [
                    v
                    for k, v in self.cont_as_bools(
                        assert_is_bool,
                        key_chains,
                        to_apply,
                        prune_unapplied,
                        map_sequences,
                    ).cont_to_iterator()
                ]
            )
        )

    def cont_slice_via_key(self, slice_key):
        """
        Get slice of container, based on key.

        Parameters
        ----------
        slice_key
            key to slice container at.

        Returns
        -------
            Container object sliced at desired key.
        """
        return_dict = dict()
        for key, value in self.items():
            if key == slice_key:
                return value
            elif isinstance(value, ivy.Container):
                return_dict[key] = value.cont_slice_via_key(slice_key)
            else:
                return_dict[key] = value
        return ivy.Container(return_dict, **self._config)

    def cont_as_bools(
        self,
        assert_is_bool=False,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
    ):
        """
        Return boolean evaluation for all nested items in the container.

        Parameters
        ----------
        assert_is_bool
            Whether or not to assert the entry is of type Boolean.
            (Default value = False)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
            Container object with all entries boolean evaluated.
        """

        def _ret_bool(x):
            if assert_is_bool:
                ivy.utils.assertions.check_isinstance(x, bool)
                return x
            return bool(x)

        return self.cont_map(
            lambda x, kc: _ret_bool(x),
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    def cont_unstack_conts(self, axis, keepdims=False, dim_size=None):
        """
        Unstack containers along specified dimension.

        Parameters
        ----------
        axis
            Dimensions along which to unstack.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is ``False``.
        dim_size
            Size of the dimension to unstack. Determined from inputs by default.

        Returns
        -------
            List of containers, unstacked along the specified dimension.
        """
        if dim_size is None:
            dim_size = self.cont_shape[axis]
        if keepdims:
            # noinspection PyTypeChecker
            return [
                self[
                    (
                        slice(i, i + 1, 1)
                        if axis == 0
                        else tuple(
                            [slice(None, None, None)] * axis + [slice(i, i + 1, 1)]
                        )
                    )
                ]
                for i in range(dim_size)
            ]
        # noinspection PyTypeChecker
        return [
            self[i if axis == 0 else tuple([slice(None, None, None)] * axis + [i])]
            for i in range(dim_size)
        ]

    def split_conts(
        self,
        num_or_size_splits=None,
        axis=0,
        with_remainder=False,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
    ):
        """
        Split a container into multiple sub-containers.

        The function does that by splitting their constituent arrays.

        Parameters
        ----------
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers. Default
            is to divide into as many 1-dimensional arrays as the axis dimension.
        axis
            The axis along which to split, default is ``0``.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
            A list of sub-arrays.
        """
        dim_size = (
            num_or_size_splits
            if isinstance(num_or_size_splits, int)
            else len(num_or_size_splits)
        )
        # noinspection PyTypeChecker
        return self.cont_map(
            lambda x, kc: (
                self._cont_ivy.split(
                    x,
                    num_or_size_splits=num_or_size_splits,
                    axis=axis,
                    with_remainder=with_remainder,
                )
                if self._cont_ivy.is_native_array(x) or isinstance(x, ivy.Array)
                else x
            ),
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        ).cont_unstack_conts(0, dim_size=dim_size)

    def cont_num_arrays(self, exclusive=False):
        """
        Compute the number of arrays present at the leaf nodes, including variables by
        default.

        Parameters
        ----------
        exclusive
            Whether to check if the data type is exclusively an array,
            rather than a variable or traced array. (Default value = False)
        """
        return sum(
            self.cont_map(
                lambda x, kc: ivy.is_array(x, exclusive=exclusive)
            ).cont_to_iterator_values()
        )

    def cont_size_ordered_arrays(self, exclusive=False):
        """
        Return a container with keychains mapped to flat keys, and arrays given in order
        of smallest to largest.

        Parameters
        ----------
        exclusive
            Whether to check if the data type is exclusively an array,
            rather than a variable or traced array. (Default value = False)
        """
        array_dict = {
            ivy.Container.cont_flatten_key_chain(kc): v
            for kc, v in self.cont_to_iterator()
            if ivy.is_array(v, exclusive=exclusive)
        }
        return ivy.Container(
            dict(
                sorted(
                    array_dict.items(), key=lambda item: _reduce(mul, item[1].shape, 1)
                )
            ),
            alphabetical_keys=False,
        )

    def cont_save(self, filepath, format="h5py"):
        if format == "json":
            self.cont_to_disk_as_json(filepath)
        elif format == "pickle":
            self.cont_to_disk_as_pickled(filepath)
        elif format == "h5py":
            self.cont_to_disk_as_hdf5(filepath)
        else:
            raise ValueError("Unsupported format")

    def cont_to_disk_as_hdf5(
        self, h5_obj_or_filepath, starting_index=0, mode="a", max_batch_size=None
    ):
        """
        Save container object to disk, as an h5py file, at the specified filepath.

        Parameters
        ----------
        h5_obj_or_filepath
            Filepath for where to save the container to disk, or h5 object.
        starting_index
            Batch index for which to start writing to file, if it already exists
            (Default value = 0)
        mode
            H5 read/write mode for writing to disk, ['r', 'r+', 'w', 'w-', 'a'],
            default is 'a'.
        max_batch_size
            Maximum batch size for the container on disk, this is useful if later
            appending to file. (Default value = None)
        """
        ivy.utils.assertions.check_exists(
            h5py,
            message=(
                "You must install python package h5py in order to save "
                "containers to disk as hdf5 files."
            ),
        )
        if type(h5_obj_or_filepath) is str:
            h5_obj = h5py.File(h5_obj_or_filepath, mode)
        else:
            h5_obj = h5_obj_or_filepath
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                if key not in h5_obj.keys():
                    h5_group = h5_obj.create_group(key)
                else:
                    h5_group = h5_obj[key]
                value.cont_to_disk_as_hdf5(
                    h5_group, starting_index, mode, max_batch_size
                )
            else:
                value_as_np = self._cont_ivy.to_numpy(value)
                value_shape = value_as_np.shape
                this_batch_size = value_shape[0]
                max_bs = (
                    starting_index + this_batch_size
                    if not max_batch_size
                    else max_batch_size
                )
                if key not in h5_obj.keys():
                    dataset_shape = [max_bs] + list(value_shape[1:])
                    maxshape = [None for _ in dataset_shape]
                    h5_obj.create_dataset(
                        key, dataset_shape, dtype=value_as_np.dtype, maxshape=maxshape
                    )
                space_left = max_bs - starting_index
                amount_to_write = min(this_batch_size, space_left)
                for i in range(amount_to_write):
                    h5_obj[key][starting_index + i : starting_index + i + 1] = (
                        value_as_np[i : i + 1]
                    )

    def cont_to_disk_as_pickled(self, pickle_filepath):
        """
        Save container object to disk, as an pickled file, at the specified filepath.

        Parameters
        ----------
        pickle_filepath
            Filepath for where to save the container to disk.
        """
        pickle.dump(self.to_native().cont_to_dict(), open(pickle_filepath, "wb"))

    def cont_to_jsonable(self, return_dict=None):
        """

        Parameters
        ----------
        return_dict
            Default value = None)

        """
        if return_dict is None:
            return_dict = self.cont_copy()
        for k, v in return_dict.items():
            if not _is_jsonable(v):
                if isinstance(v, dict):
                    return_dict[k] = self.cont_to_jsonable(v)
                else:
                    return_dict[k] = str(v)
        return return_dict

    def cont_to_disk_as_json(self, json_filepath):
        """
        Save container object to disk, as an json file, at the specified filepath.

        Parameters
        ----------
        json_filepath
            Filepath for where to save the container to disk.
        """
        with open(json_filepath, "w+") as json_data_file:
            json.dump(self.cont_to_jsonable().cont_to_dict(), json_data_file, indent=4)

    def cont_to_nested_list(self):
        return_list = list()
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                return_list.append(value.cont_to_nested_list())
            elif value is not None and key != "_f":
                return_list.append(value)
        return return_list

    def cont_to_raw(self):
        """
        Convert container to its original form.

        Returns
        -------
        ret
             Container data in its raw form.
        """
        return_item = dict()
        for i, (key, value) in enumerate(self.items()):
            if isinstance(value, ivy.Container):
                return_item[key] = value.cont_to_raw()
            elif key[0:3] == "it_" and tuple(self._types_to_iteratively_nest):
                return_item = list(
                    [
                        v.cont_to_raw() if isinstance(v, ivy.Container) else v
                        for v in self.values()
                    ]
                )
                break
            else:
                return_item[key] = value
        return return_item

    def cont_to_dict(self):
        """
        Summary.

        Returns
        -------
            ret Container as nested dict.
        """
        return_dict = dict()
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                return_dict[key] = value.cont_to_dict()
            else:
                return_dict[key] = value
        return return_dict

    def cont_to_iterator(self, key_chain="", leaf_keys_only=False, include_empty=False):
        """

        Parameters
        ----------
        key_chain
            Default value = '')
        leaf_keys_only
            Default value = False)
        include_empty
            Default value = False)

        Returns
        -------
            Iterator for the container elements.

        """
        for key, value in self.items():
            if leaf_keys_only:
                kc = key
            else:
                kc = key_chain + "/" + key if key_chain != "" else key
            if isinstance(value, ivy.Container) and (not include_empty or value):
                yield from value.cont_to_iterator(kc, leaf_keys_only, include_empty)
            else:
                yield kc, value

    def cont_to_iterator_values(self, include_empty=False):
        """

        Parameters
        ----------
        include_empty
            Default value = False)

        Returns
        -------
            Iterator for the container values.

        """
        for key, value in self.items():
            if isinstance(value, ivy.Container) and (not include_empty or value):
                # noinspection PyCompatibility
                yield from value.cont_to_iterator_values(include_empty)
            else:
                yield value

    def cont_to_iterator_keys(
        self, key_chain="", leaf_keys_only=False, include_empty=False
    ):
        """

        Parameters
        ----------
        key_chain
            Default value = '')
        leaf_keys_only
            Default value = False)
        include_empty
            Default value = False)

        Returns
        -------
            Iterator for the container elements.

        """
        for key, value in self.items():
            if leaf_keys_only:
                kc = key
            else:
                kc = key_chain + "/" + key if key_chain != "" else key
            if isinstance(value, ivy.Container) and (not include_empty or value):
                # noinspection PyCompatibility
                yield from value.cont_to_iterator_keys(
                    kc, leaf_keys_only, include_empty
                )
            else:
                yield kc

    def cont_to_flat_list(self):
        """
        Summary.

        Returns
        -------
        ret
            Container as flat list.
        """
        return list([item for key, item in self.cont_to_iterator()])

    def cont_from_flat_list(self, flat_list):
        """
        Return new container object with the same hierarchy, but with values replaced
        from flat list.

        Parameters
        ----------
        flat_list
            flat list of values to populate container with.

        Returns
        -------
            Container.
        """
        new_dict = dict()
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                new_value = value.cont_from_flat_list(flat_list)
            else:
                new_value = flat_list.pop(0)
            new_dict[key] = new_value
        return ivy.Container(new_dict, **self._config)

    def cont_has_key(self, query_key):
        """
        Determine whether container object has specified key somewhere in the nested
        structure.

        Parameters
        ----------
        query_key


        Returns
        -------
        ret
            Boolean
        """
        has_key = False

        def map_fn(x, kc):
            """

            Parameters
            ----------
            x
                param kc:
            kc

            """
            nonlocal has_key
            if query_key in kc:
                has_key = True
            return x

        self.cont_map(map_fn)
        return has_key

    def cont_has_key_chain(self, key_chain):
        """
        Determine whether container object has specified key-chain.

        Parameters
        ----------
        key_chain


        Returns
        -------
        ret
            Boolean
        """
        keys = re.split("[/.]", key_chain)
        ret = self
        for key in keys:
            try:
                ret = ret[key]
            except KeyError:
                return False
        return True

    def cont_find_sub_container(self, sub_cont_to_find, partial=False):
        """
        Find the sub-container in the current container if it exsits.

        Parameters
        ----------
        sub_cont_to_find
            The sub-container to find.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        key_chain_found = False

        def _check_sub_cont(sub_cont, kc):
            sub_cont_key_chains = sub_cont_to_find.cont_all_key_chains()
            kcs_in_sub_cont = [kc in sub_cont for kc in sub_cont_key_chains]
            if (
                kcs_in_sub_cont
                and min(kcs_in_sub_cont)
                and ivy.Container.cont_identical(
                    [sub_cont, sub_cont_to_find], partial=partial
                )
            ):
                nonlocal key_chain_found
                key_chain_found = kc
            return sub_cont

        self.cont_map_sub_conts(_check_sub_cont)

        return key_chain_found

    def cont_contains_sub_container(self, sub_cont, partial=False):
        """
        Determine whether the current container contains the sub-container, with
        matching structure and array values.

        Parameters
        ----------
        sub_cont
            The sub-container to check.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.

        Returns
        -------
            Bool
        """
        return (
            True
            if isinstance(self.cont_find_sub_container(sub_cont, partial), str)
            else False
        )

    def cont_assert_contains_sub_container(self, sub_cont, partial=False):
        """
        Assert that the current container contains the sub-container, otherwise
        exception raised with the diff printed to screen.

        Parameters
        ----------
        sub_cont
            The sub-container to check.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        try:
            ivy.utils.assertions.check_true(
                self.cont_contains_sub_container(sub_cont, partial)
            )
        except ivy.utils.exceptions.IvyException:
            key_chain = self.cont_find_sub_structure(
                sub_cont, check_shapes=False, partial=True
            )
            if not key_chain:
                key_chain = ""
            # noinspection PyTypeChecker
            raise ivy.utils.exceptions.IvyException(
                "Containers did not have identical structure and values:\n\n{}".format(
                    ivy.Container.cont_diff(self[key_chain], sub_cont)
                )
            )

    def cont_find_sub_structure(
        self, sub_struc_to_find, check_shapes=True, partial=False
    ):
        """
        Find the sub-container structure in the current container if it exsits.

        Parameters
        ----------
        sub_struc_to_find
            The sub-container to find.
        check_shapes
            Whether to check array shapes in the sub-structure. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        key_chain_found = False

        def _check_sub_cont(sub_cont, kc):
            """

            Parameters
            ----------
            sub_cont
                param kc:
            kc

            """
            sub_struc_key_chains = sub_struc_to_find.cont_all_key_chains()
            kcs_in_sub_cont = [kc in sub_cont for kc in sub_struc_key_chains]
            if (
                kcs_in_sub_cont
                and min(kcs_in_sub_cont)
                and ivy.Container.cont_identical_structure(
                    [sub_cont, sub_struc_to_find],
                    check_shapes=check_shapes,
                    partial=partial,
                )
            ):
                nonlocal key_chain_found
                key_chain_found = kc
            return sub_cont

        self.cont_map_sub_conts(_check_sub_cont)

        return key_chain_found

    def cont_contains_sub_structure(self, sub_cont, check_shapes=True, partial=False):
        """
        Determine whether the current container contains the sub-container structure.

        Parameters
        ----------
        sub_cont
            The sub-container to check.
        check_shapes
            Whether to check array shapes in the sub-structure. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        return (
            True
            if isinstance(
                self.cont_find_sub_structure(sub_cont, check_shapes, partial), str
            )
            else False
        )

    def cont_assert_contains_sub_structure(
        self, sub_cont, check_shapes=True, partial=False
    ):
        """
        Assert that the current container contains the sub-container structure,
        otherwise exception raised with the diff printed to screen.

        Parameters
        ----------
        sub_cont
            The sub-container to check.
        check_shapes
            Whether to check array shapes in the sub-structure. Default is ``True``.
        partial
            Whether to also check for partially complete sub-containers.
            Default is ``False``.
        """
        try:
            ivy.utils.assertions.check_true(
                self.cont_contains_sub_structure(sub_cont, check_shapes, partial)
            )
        except ivy.utils.exceptions.IvyException:
            key_chain = self.cont_find_sub_structure(
                sub_cont, check_shapes=False, partial=True
            )
            if not key_chain:
                key_chain = ""
            # noinspection PyTypeChecker
            raise ivy.utils.exceptions.IvyException(
                "Containers did not have identical structure:\n\n{}".format(
                    ivy.Container.cont_structural_diff(
                        self[key_chain],
                        sub_cont,
                        detect_key_diffs=not partial,
                        detect_shape_diffs=check_shapes,
                        mode="diff_only" if partial else "all",
                    )
                )
            )

    def cont_at_keys(
        self, queries, ignore_none=True, containing=False, ignore_key_errors=False
    ):
        """
        Query container object at specified keys, either as list or nested dict.

        Parameters
        ----------
        queries
            The keys to query.
        ignore_none
            Whether to ignore None input. Default is ``True``.
        containing
            Whether to include keys which only contain the query substrings.
            Default is ``False``.
        ignore_key_errors
            Whether to ignore Key-errors when trying to access the dict.
            Default is ``False``.

        Returns
        -------
            sub-container containing only key-chains containing the specified keys.
        """
        if queries is None and ignore_none:
            return self
        key_chains_to_keep = list()
        if isinstance(queries, str):
            queries = [queries]

        def map_fn(x, kc):
            nonlocal key_chains_to_keep
            kc_split = re.split("[/.]", kc)
            for query_key in queries:
                if query_key in kc_split or (
                    containing and min([query_key in k for k in kc_split])
                ):
                    key_chains_to_keep.append(kc)
            return x

        self.cont_map(map_fn)
        return self.cont_at_key_chains(
            key_chains_to_keep, ignore_key_errors=ignore_key_errors
        )

    def cont_at_key_chain(self, key_chain, ignore_key_errors=False):
        """
        Query container object at a specified key-chain.

        Parameters
        ----------
        key_chain
            param ignore_key_errors: (Default value = False)
        ignore_key_errors
             (Default value = False)

        Returns
        -------
        ret
            sub-container or value at specified key chain
        """
        keys = re.split("[/.]", key_chain)
        ret = self
        for key in keys:
            try:
                ret = ret[key]
            except KeyError as e:
                if ignore_key_errors:
                    return
                raise ivy.utils.exceptions.IvyException(repr(e))
        return ret

    def cont_at_key_chains(self, key_chains, ignore_none=True, ignore_key_errors=False):
        """
        Query container object at specified key-chains, either as list or nested dict.

        Parameters
        ----------
        key_chains
            param ignore_none: (Default value = True)
        ignore_key_errors
            Default value = False)
        ignore_none
             (Default value = True)

        Returns
        -------
        type
            sub-container containing only the specified key chains
        """
        if key_chains is None and ignore_none:
            return self
        if isinstance(key_chains, (list, tuple)):
            return self._cont_at_key_chains_input_as_seq(
                key_chains, ignore_key_errors=ignore_key_errors
            )
        elif isinstance(key_chains, dict):
            return self._cont_at_key_chains_input_as_dict(
                key_chains, ignore_key_errors=ignore_key_errors
            )
        elif isinstance(key_chains, str):
            return self._cont_at_key_chains_input_as_seq(
                [key_chains], ignore_key_errors=ignore_key_errors
            )
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid type for input key_chains, must either be a list, tuple, dict"
                " or ivy.Container, but found type {}".format(type(key_chains))
            )

    def cont_all_key_chains(self, include_empty=False):
        """

        Parameters
        ----------
        include_empty
            Default value = False)

        """
        return [kc for kc, v in self.cont_to_iterator(include_empty=include_empty)]

    def cont_key_chains_containing(self, sub_str, include_empty=False):
        """

        Parameters
        ----------
        sub_str
            param include_empty: (Default value = False)
        include_empty
             (Default value = False)

        """
        return [
            kc
            for kc, v in self.cont_to_iterator(include_empty=include_empty)
            if sub_str in kc
        ]

    def cont_set_at_keys(self, target_dict):
        """
        Set values of container object at specified keys.

        Parameters
        ----------
        target_dict


        Returns
        -------
        type
            new container with updated value at each key
        """
        return_dict = dict()
        for key, val in self.items():
            if key in target_dict:
                return_dict[key] = target_dict[key]
            elif isinstance(val, ivy.Container):
                return_dict[key] = val.cont_set_at_keys(target_dict)
            else:
                return_dict[key] = val
        return ivy.Container(return_dict, **self._config)

    def cont_set_at_key_chain(self, key_chain, val, inplace=False):
        """
        Set value of container object at a specified key-chain.

        Parameters
        ----------
        key_chain
            param val:
        inplace
            Default value = False)
        val


        Returns
        -------
        ret
            new container with updated value at key chain
        """
        keys = re.split("[/.]", key_chain)
        if inplace:
            cont = self
        else:
            cont = self.cont_copy()
        sub_cont = cont
        for key in keys[:-1]:
            if key not in sub_cont:
                sub_cont[key] = ivy.Container(**self._config)
            sub_cont = sub_cont[key]
        sub_cont[keys[-1]] = val
        return cont

    def cont_overwrite_at_key_chain(self, key_chain, val, inplace=False):
        """
        Overwrite value of container object at a specified key-chain.

        Parameters
        ----------
        key_chain
            param val:
        inplace
            Default value = False)
        val


        Returns
        -------
        ret
            new container with updated value at key chain, provided it existed before.
        """
        keys = re.split("[/.]", key_chain)
        if inplace:
            cont = self
        else:
            cont = self.cont_copy()
        sub_cont = cont
        for key in keys[:-1]:
            ivy.utils.assertions.check_elem_in_list(
                key,
                sub_cont,
                message=(
                    "key chain must already exist in container in order to "
                    "call cont_overwrite_at_key_chain"
                ),
            )
            sub_cont = sub_cont[key]
        ivy.utils.assertions.check_elem_in_list(
            keys[-1],
            sub_cont,
            message=(
                "key chain must already exist in container in order to call "
                "cont_overwrite_at_key_chain"
            ),
        )
        sub_cont[keys[-1]] = val
        return cont

    def cont_set_at_key_chains(self, target_dict, return_dict=None, inplace=False):
        """
        Set values of container object at specified key-chains.

        Parameters
        ----------
        target_dict
            param return_dict: (Default value = None)
        inplace
            Default value = False)
        return_dict
             (Default value = None)

        Returns
        -------
        ret
            new container with updated values at the key chains
        """
        if return_dict is None:
            if inplace:
                return_dict = self
            else:
                return_dict = self.cont_copy()
        for k, v in target_dict.items():
            if isinstance(v, dict):
                return_dict[k] = self.cont_set_at_key_chains(v, return_dict[k], inplace)
            else:
                return_dict[k] = v
        return return_dict

    def cont_overwrite_at_key_chains(
        self, target_dict, return_dict=None, inplace=False
    ):
        """
        Overwrite values of container object at specified key-chains.

        Parameters
        ----------
        target_dict
            param return_dict: (Default value = None)
        inplace
            Default value = False)
        return_dict
             (Default value = None)

        Returns
        -------
        ret
            new container with updated values at the key chains, provided they
            existed before.
        """
        if return_dict is None:
            if inplace:
                return_dict = self
            else:
                return_dict = self.cont_copy()
        for k, v in target_dict.items():
            ivy.utils.assertions.check_elem_in_list(
                k,
                return_dict,
                message=(
                    "key chain must already exist in container in order to "
                    "call cont_overwrite_at_key_chain"
                ),
            )
            if isinstance(v, dict):
                return_dict[k] = self.cont_overwrite_at_key_chains(
                    v, return_dict[k], inplace
                )
            else:
                return_dict[k] = v
        return ivy.Container(return_dict, **self._config)

    def cont_prune_keys(self, query_keys, ignore_none=True):
        """
        Recursively prune set of keys.

        Parameters
        ----------
        query_keys
            param ignore_none: (Default value = True)
        ignore_none
             (Default value = True)

        Returns
        -------
        ret
            Container with key-chains containing the specified keys pruned.
        """
        if query_keys is None and ignore_none:
            return self
        key_chains_to_prune = list()
        if isinstance(query_keys, str):
            query_keys = [query_keys]

        def map_fn(x, kc):
            """

            Parameters
            ----------
            x
                param kc:
            kc

            """
            nonlocal key_chains_to_prune
            for query_key in query_keys:
                if query_key in kc:
                    key_chains_to_prune.append(kc)
            return x

        self.cont_map(map_fn)
        return self.cont_prune_key_chains(key_chains_to_prune)

    def cont_prune_key_chain(self, key_chain):
        """
        Recursively prune chain of keys, specified as 'key1/key2/key3/...'.

        Parameters
        ----------
        key_chain

        Returns
        -------
        ret
            Container with keys in key chain pruned.
        """
        keys_in_chain = re.split("[/.]", key_chain)
        out_dict = dict()
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                if key == keys_in_chain[0]:
                    if len(keys_in_chain) == 1:
                        new_val = []
                    else:
                        new_val = value.cont_prune_key_chain(
                            "/".join(keys_in_chain[1:])
                        )
                    if len(new_val) > 0:
                        out_dict[key] = new_val
                else:
                    new_val = value.cont_to_dict()
                    if len(new_val) > 0:
                        out_dict[key] = value.cont_to_dict()
            else:
                if len(keys_in_chain) != 1 or key != keys_in_chain[0]:
                    out_dict[key] = value
        return ivy.Container(out_dict, **self._config)

    def cont_prune_key_chains(self, key_chains, ignore_none=True):
        """
        Recursively prune set of key chains.

        Parameters
        ----------
        key_chains
            param ignore_none: (Default value = True)
        ignore_none
             (Default value = True)

        Returns
        -------
        ret
            Container with keys in the set of key chains pruned.
        """
        if key_chains is None and ignore_none:
            return self
        if isinstance(key_chains, (list, tuple)):
            return self._cont_prune_key_chains_input_as_seq(key_chains)
        elif isinstance(key_chains, dict):
            return self._cont_prune_key_chains_input_as_dict(key_chains)
        elif isinstance(key_chains, str):
            return self._cont_prune_key_chains_input_as_seq([key_chains])
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid type for input key_chains, must either be a list, tuple, dict "
                "or ivy.Container, but found type {}".format(type(key_chains))
            )

    def cont_format_key_chains(self, format_fn):
        """
        Format all key-chains, using the formatting function.

        Parameters
        ----------
        format_fn


        Returns
        -------
        ret
            Container with the same key-chain structure, but the key strings formatted.
        """
        return ivy.Container({format_fn(k): v for k, v in self.cont_to_iterator()})

    def cont_sort_by_key(self):
        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, ivy.Container):
                v_back = v.cont_sort_by_key()
            else:
                v_back = v
            new_dict[k] = v_back
        return ivy.Container(new_dict, **self._config)

    def cont_prune_empty(self, keep_nones=False, base=True):
        """
        Recursively prunes empty keys from the container dict structure. Returns None if
        the entire container is empty.

        Parameters
        ----------
        keep_nones
            Default value = False)
        base
            Default value = True)

        Returns
        -------
        ret
            Container with empty keys pruned.
        """
        out_dict = dict()
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                new_value = value.cont_prune_empty(keep_nones, False)
                if new_value:
                    out_dict[key] = new_value
            elif self._cont_ivy.exists(value) or keep_nones:
                out_dict[key] = value
        if len(out_dict):
            return ivy.Container(out_dict, **self._config)
        if base:
            return ivy.Container(**self._config)
        return

    def cont_prune_key_from_key_chains(self, absolute=None, containing=None):
        """
        Recursively prune absolute key or key containing a certain substring from all
        key chains.

        Parameters
        ----------
        absolute
            The absolute key to detect in the key chains. (Default value = None)
        containing
            A substring to check each key for, when deciding which keys to prune.
            (Default value = None)

        Returns
        -------
            Container with specified key or substring-containing-key from all key chains
            removed from the chain.
        """
        ivy.utils.assertions.check_all_or_any_fn(
            absolute,
            containing,
            fn=ivy.exists,
            type="any",
            limit=[1, 2],
            message="at least one of absolute or containing must be specified",
            as_array=False,
        )
        out_cont = ivy.Container(**self._config)
        for key, value in self.items():
            if (absolute and key == absolute) or (containing and containing in key):
                if isinstance(value, ivy.Container):
                    out_cont = ivy.Container.cont_combine(out_cont, value)
                else:
                    out_cont = value
            elif isinstance(value, ivy.Container):
                out_cont[key] = value.cont_prune_key_from_key_chains(
                    absolute, containing
                )
            else:
                out_cont[key] = value
        return out_cont

    def cont_prune_keys_from_key_chains(self, absolute=None, containing=None):
        """
        Recursively prune absolute keys or keys containing certain substrings from all
        key chains.

        Parameters
        ----------
        absolute
            The absolute key to detect in the key chains. (Default value = None)
        containing
            A substring to check each key for, when deciding which keys to prune.
            (Default value = None)

        Returns
        -------
            Container with specified keys or substring-containing-keys from all
            key chains removed from the chain.
        """
        ivy.utils.assertions.check_all_or_any_fn(
            absolute,
            containing,
            fn=ivy.exists,
            type="any",
            limit=[1, 2],
            message="at least one of absolute or containing must be specified",
            as_array=False,
        )
        out_cont = ivy.Container(**self._config)
        for key, value in self.items():
            if (absolute and key in absolute) or (
                containing and max([con in key for con in containing])
            ):
                if isinstance(value, ivy.Container):
                    out_cont = ivy.Container.cont_combine(out_cont, value)
                else:
                    out_cont = value
            elif isinstance(value, ivy.Container):
                out_cont[key] = value.cont_prune_key_from_key_chains(
                    absolute, containing
                )
            else:
                out_cont[key] = value
        return out_cont

    def cont_restructure_key_chains(
        self, keychain_mapping, keep_orig=True, replace=True
    ):
        """
        Create a new container with the same contents, but a new key-chain structure.
        Given by the mapping with keys as old key-chains and values as new key-chains.

        Parameters
        ----------
        keychain_mapping
            A dict with keys as old key-chains and values as new key-chains.
        keep_orig
            Whether to keep the original keys, or start from a new empty container.
            Default is ``True``.
        replace
            Whether to replace the old key-chains by the new ones. Default is ``True``.
        """
        new_cont = self.cont_copy() if keep_orig else ivy.Container()
        for old_kc, new_kc in keychain_mapping.items():
            if replace and old_kc in new_cont:
                new_cont = new_cont.cont_prune_key_chain(old_kc)
            new_cont = ivy.Container.cont_combine(
                new_cont, ivy.Container({new_kc: self[old_kc]})
            )
        return new_cont

    def cont_restructure(self, mapping, keep_orig=True, replace=True):
        """
        Create a new container with the same contents, but a new key-chain structure,
        and transposes and/or reshaped arrays. Given by the mapping with keys as old
        key-chains and values as new key-chains.

        Parameters
        ----------
        mapping
            A dict with keys as old key-chains and values as new key-chains.
        keep_orig
            Whether to keep the original keys, are start from a new container.
            Default is ``True``.
        replace
            Whether to replace the old key-chains by the new ones. Default is ``True``.
        """
        new_cont = self.cont_copy() if keep_orig else ivy.Container()
        for old_kc, new in mapping.items():
            if replace and old_kc in new_cont:
                new_cont = new_cont.cont_prune_key_chain(old_kc)
            val = self[old_kc]
            if isinstance(new, dict):
                new_kc = new["key_chain"]
                if "pattern" in new:
                    pattern = new["pattern"]
                    axes_lengths = new["axes_lengths"] if "axes_lengths" in new else {}
                    if isinstance(val, ivy.Container):
                        val = val.einops_rearrange(pattern, **axes_lengths)
                    else:
                        val = ivy.einops_rearrange(val, pattern, **axes_lengths)
            else:
                new_kc = new
            new_cont = ivy.Container.cont_combine(
                new_cont, ivy.Container({new_kc: val})
            )
        return new_cont

    def cont_flatten_key_chains(
        self, include_empty=False, above_height=None, below_depth=None
    ):
        """
        Summary.

        Parameters
        ----------
        include_empty
            Default value = False)
        above_height
            Default value = None)
        below_depth
            Default value = None)
        """
        return ivy.Container(
            {
                ivy.Container.cont_flatten_key_chain(
                    kc, above_height=above_height, below_depth=below_depth
                ): v
                for kc, v in self.cont_to_iterator(include_empty=include_empty)
            },
            **self._config,
        )

    def cont_copy(self):
        """
        Create a copy of this container.

        Returns
        -------
            A copy of the container
        """
        return ivy.Container(self.cont_to_dict(), **self._config)

    def cont_deep_copy(self):
        """
        Create a deep copy (copying all internal tensors) of this container.

        return: A deep copy of the container
        """
        return self.cont_map(
            lambda x, kc: (
                ivy.copy_array(x) if ivy.is_array(x) and not isinstance(x, str) else x
            )
        )

    def __deepcopy__(self, memo):
        return self.cont_deep_copy()

    def cont_map(
        self,
        func,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
        inplace=False,
        key_chain="",
    ):
        """
        Apply function to all array values of container.

        Parameters
        ----------
        func
            Function to apply to each container entry
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        inplace
            Whether to apply the mapping inplace, or return a new container.
            Default is ``False``.
        key_chain
            Chain of keys for this dict entry (Default value = '')

        Returns
        -------
            New container following the function mapped to each sub-array.
        """
        return_dict = self if inplace else dict()
        for key, value in self.items():
            this_key_chain = (
                key if key_chain == "" else (str(key_chain) + "/" + str(key))
            )
            if isinstance(value, ivy.Container):
                ret = value.cont_map(
                    func,
                    key_chains,
                    to_apply,
                    prune_unapplied,
                    map_sequences,
                    inplace,
                    this_key_chain,
                )
                if prune_unapplied and not ret:
                    continue
                if not inplace:
                    return_dict[key] = ret
            elif isinstance(value, (list, tuple)) and map_sequences:
                ret = ivy.nested_map(
                    value, lambda x: func(x, None), True, shallow=False
                )
                if prune_unapplied and not ret:
                    continue
                return_dict[key] = ret
            else:
                if key_chains is not None:
                    if (this_key_chain in key_chains and not to_apply) or (
                        this_key_chain not in key_chains and to_apply
                    ):
                        if prune_unapplied:
                            continue
                        return_dict[key] = value
                        continue
                return_dict[key] = func(value, this_key_chain)
        if inplace:
            return self
        return ivy.Container(return_dict, **self._config)

    def cont_map_sub_conts(
        self,
        func,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        inplace=False,
        key_chain="",
        include_self=True,
    ):
        """
        Apply function to all sub-contains in the container.

        Parameters
        ----------
        func
            Function to apply to each sub-container
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        inplace
            Whether to apply the mapping inplace, or return a new container.
            Default is ``False``.
        key_chain
            Chain of keys for this dict entry (Default value = '')
        include_self
            Whether to also apply the (possiby in-place) function to this container.
            Default is ``True``.

        Returns
        -------
            New container following the function mapped to each sub-container.
        """
        return_dict = self if inplace else dict()
        for key, value in self.items():
            this_key_chain = key if key_chain == "" else (key_chain + "/" + key)
            if isinstance(value, ivy.Container):
                ret = value.cont_map_sub_conts(
                    func, key_chains, to_apply, prune_unapplied, inplace, this_key_chain
                )
                if prune_unapplied and not ret:
                    continue
                if not inplace:
                    return_dict[key] = ret
            else:
                if (
                    key_chains is not None
                    and (
                        (this_key_chain in key_chains and not to_apply)
                        or (this_key_chain not in key_chains and to_apply)
                    )
                    and prune_unapplied
                ):
                    continue
                return_dict[key] = value
        ret = return_dict if inplace else ivy.Container(return_dict, **self._config)
        if key_chain != "" or include_self:
            ret = func(ret, key_chain)
        if inplace:
            return
        return ret

    def cont_with_entries_as_lists(self):
        def to_list(x, _=""):
            try:
                return self._cont_ivy.to_list(x)
            except (IvyException, IvyBackendException):
                return x

        return self.cont_map(to_list)

    def cont_reshape_like(self, target_dict, leading_shape=None, return_cont=None):
        """
        Set shapes of container entries to shapes specified by new container with the
        same key structure.

        Parameters
        ----------
        target_dict
            param leading_shape: (Default value = None)
        return_cont
            Default value = None)
        leading_shape
             (Default value = None)

        Returns
        -------
        ret
            new container with values of updated shapes
        """
        leading_shape = self._cont_ivy.default(leading_shape, list())
        if return_cont is None:
            return_cont = self.cont_copy()
        for (_, v_shape), (k, v) in zip(target_dict.items(), return_cont.items()):
            if isinstance(v_shape, dict):
                return_cont[k] = self.cont_reshape_like(
                    v_shape, leading_shape, return_cont[k]
                )
            else:
                return_cont[k] = self._cont_ivy.reshape(
                    v, leading_shape + list(v_shape)
                )
        return ivy.Container(return_cont, **self._config)

    def cont_create_if_absent(self, key, value, inplace=True):
        """
        Add a key to the container with corresponding value, if it is not already
        present. otherwise, do nothing.

        Parameters
        ----------
        key
            param value:
        inplace
            Default value = True)
        value
        """
        if key in self:
            return
        self.cont_set_at_key_chain(key, value, inplace)

    def cont_if_exists(self, key):
        """
        Return the sub-container at the following key if it exists, otherwise None.

        Parameters
        ----------
        key
        """
        try:
            return self[key]
        except KeyError:
            return

    def cont_try_kc(self, key):
        """
        Try the following key or key chain, returning self if not present.

        Parameters
        ----------
        key
        """
        try:
            return self[key]
        except IvyException:
            return self

    def cont_cutoff_at_depth(self, depth_cutoff, inplace=False):
        """
        Summary.

        Parameters
        ----------
        depth_cutoff
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """
        total_depth = self.cont_max_depth
        copy = self.cont_copy()

        def _maybe_cutoff(cont, kc):
            if total_depth - copy[kc].cont_max_depth < depth_cutoff:
                return cont
            if inplace:
                cont.clear()
            return ivy.Container()

        ret = self.cont_map_sub_conts(_maybe_cutoff, inplace=inplace)
        if inplace:
            return
        return ret

    def cont_cutoff_at_height(self, height_cutoff, inplace=False):
        """
        Summary.

        Parameters
        ----------
        height_cutoff
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """
        copy = self.cont_copy()

        def _maybe_cutoff(cont, kc):
            if copy[kc].cont_max_depth > height_cutoff:
                return cont
            if inplace:
                cont.clear()
            return ivy.Container()

        ret = self.cont_map_sub_conts(_maybe_cutoff, inplace=inplace)
        if inplace:
            return
        return ret

    def _cont_slice_keys(self, key_slice):
        keys = list(self.keys())
        if isinstance(key_slice, str):
            ivy.utils.assertions.check_true(len(key_slice) == 3 and key_slice[1] == ":")
            ivy.utils.assertions.check_true(self._alphabetical_keys)
            start_char = key_slice[0]
            end_char = key_slice[2]
            start_idx = min([i for i, k in enumerate(keys) if k[0] == start_char])
            end_idx = max([i for i, k in enumerate(keys) if k[0] == end_char]) + 1
            key_slice = slice(start_idx, end_idx, 1)
        ret = self.cont_copy()
        desired_keys = keys[key_slice]
        # noinspection PyUnresolvedReferences
        return ret.cont_at_key_chains(desired_keys)

    def cont_slice_keys(self, key_slice, all_depths=False):
        """
        Summary.

        Parameters
        ----------
        key_slice
            param all_depths: (Default value = False)
        all_depths
             (Default value = False)
        """
        top_depth = self.cont_max_depth
        if all_depths:
            if isinstance(key_slice, dict):
                first_slice = list(key_slice.values())[0]
                for d in range(0, top_depth + 1):
                    if d not in key_slice:
                        key_slice[d] = first_slice
            else:
                key_slice = {d: key_slice for d in range(0, top_depth + 1)}
        if isinstance(key_slice, dict):

            def _fn(cont, kc):
                depth = 0 if kc == "" else len(kc.split("/"))
                if depth in key_slice:
                    # noinspection PyProtectedMember
                    return cont._cont_slice_keys(key_slice[depth])
                return cont

            return self.cont_map_sub_conts(_fn)
        return self._cont_slice_keys(key_slice)

    def cont_with_print_limit(self, print_limit, inplace=False):
        """
        Summary.

        Parameters
        ----------
        print_limit
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """

        def _update_print_limit(cont, _):
            cont._print_limit = print_limit
            return cont

        ret = self.cont_map_sub_conts(_update_print_limit, inplace=inplace)
        if inplace:
            return
        return ret

    # noinspection PyTypeChecker
    def cont_remove_print_limit(self, inplace=False):
        """
        Summary.

        Parameters
        ----------
        inplace
            Default value = False)
        """
        return self.cont_with_print_limit(None, inplace)

    def cont_with_key_length_limit(self, key_length_limit, inplace=False):
        """
        Summary.

        Parameters
        ----------
        key_length_limit
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """

        def _update_key_length_limit(cont, _):
            cont._key_length_limit = key_length_limit
            return cont

        ret = self.cont_map_sub_conts(_update_key_length_limit, inplace=inplace)
        if inplace:
            return
        return ret

    def cont_remove_key_length_limit(self, inplace=False):
        """
        Summary.

        Parameters
        ----------
        inplace
            Default value = False)
        """
        return self.cont_with_key_length_limit(None, inplace)

    def cont_with_print_indent(self, print_indent, inplace=False):
        """
        Summary.

        Parameters
        ----------
        print_indent
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """

        def _update_print_indent(cont, _):
            cont._print_indent = print_indent
            return cont

        ret = self.cont_map_sub_conts(_update_print_indent, inplace=inplace)
        if inplace:
            return
        return ret

    def cont_with_print_line_spacing(self, print_line_spacing, inplace=False):
        """
        Summary.

        Parameters
        ----------
        print_line_spacing
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """

        def _update_print_line_spacing(cont, _):
            cont._print_line_spacing = print_line_spacing
            return cont

        ret = self.cont_map_sub_conts(_update_print_line_spacing, inplace=inplace)
        if inplace:
            return
        return ret

    def cont_with_default_key_color(self, default_key_color, inplace=False):
        """
        Summary.

        Parameters
        ----------
        default_key_color
            param inplace: (Default value = False)
        inplace
             (Default value = False)
        """

        def _update_default_key_color(cont, _):
            cont._default_key_color = default_key_color
            return cont

        ret = self.cont_map_sub_conts(_update_default_key_color, inplace=inplace)
        if inplace:
            return
        return ret

    def cont_with_ivy_backend(self, ivy_backend: str, inplace=False):
        """
        Summary.

        Parameters
        ----------
        self
            input Container
        ivy_backend
            backend to use
        inplace
            whether to modify the container or return a copy
        """
        if inplace:
            self._cont_ivy = ivy_backend
            self._config["ivyh"] = ivy_backend
            return self
        else:
            return ivy.Container(self, ivyh=ivy_backend)

    def cont_show(self):
        print(self)

    # noinspection PyUnresolvedReferences
    def cont_show_sub_container(self, sub_cont_or_keychain):
        """
        Summary.

        Parameters
        ----------
        sub_cont_or_keychain
        """
        # copy this container
        this_cont = self.cont_copy()

        # get the sub-container
        if isinstance(sub_cont_or_keychain, str):
            sub_cont = self.cont_at_key_chain(sub_cont_or_keychain)
        else:
            sub_cont = sub_cont_or_keychain

        # find the key chain of the sub-container
        sub_cont_kc = self.cont_find_sub_container(sub_cont)

        # show this container if key-chain not found, and return
        if not sub_cont_kc:
            print(self)
            return

        # otherwise, replace sub-container in this container with known key
        this_cont[sub_cont_kc] = ivy.Container({"SUB_CONT": None})

        # get the formatted reprs
        this_repr = this_cont.cont_with_default_key_color("green").__repr__()
        this_repr_red = this_cont.cont_with_default_key_color("red").__repr__()
        this_repr_stripped = ansi_escape.sub("", this_repr)
        sub_repr = sub_cont.cont_with_default_key_color("red").__repr__()

        # remove the outer brackets from the sub repr
        sub_repr = "\n" + "\n".join(sub_repr.split("\n")[1:-1]) + "\n"

        # find the sub-container placeholder
        idx = this_repr_stripped.find("SUB_CONT: null")

        # count the lines above and below the sub-container
        num_lines_above = this_repr_stripped[0:idx].count("\n")
        num_lines_below = this_repr_stripped[idx:].count("\n")

        # get the str reprs above and below
        this_repr_split = this_repr.split("\n")
        this_repr_red_split = this_repr_red.split("\n")
        this_repr_above = "\n".join(
            this_repr_split[0 : num_lines_above - 1]
            + [this_repr_red_split[num_lines_above - 1]]
        )
        this_repr_below = "\n".join(this_repr_split[-num_lines_below:])

        # count the number of lines needed to be prepended to the sub-container repr
        cur_num_spaces = 0
        for i, s in enumerate(sub_repr[1:]):
            if s != " ":
                break
            cur_num_spaces += 1
        exp_num_spaces = 0
        for i, s in enumerate(this_repr.split("\n")[num_lines_above]):
            if s != " ":
                break
            exp_num_spaces += 1
        num_spaces_to_add = exp_num_spaces - cur_num_spaces

        # prepend these lines to the sub-container
        sub_repr = (
            "\n"
            + "\n".join(
                [" " * num_spaces_to_add + s for s in sub_repr[1:-1].split("\n")]
            )
            + "\n"
        )

        # show
        print(this_repr_above + sub_repr + this_repr_below)

    # Built-ins #
    # ----------#

    def __repr__(self, as_repr=True):
        indent_str = " " * self._print_indent

        def _align_array(array_str_in):
            split_phrase_dict = {
                "": "([",
                "jax": "([",
                "numpy": "([",
                "tensorflow": "([",
                "torch": "([",
                "paddle": "])",
                "mxnet": "])",
            }
            split_phrase = split_phrase_dict[self._cont_ivy.current_backend_str()]
            array_str_in_split = array_str_in.split(split_phrase)
            leading_str_to_keep = array_str_in_split[0].replace("\\n", "")
            indented_key_size = len(leading_str_to_keep.replace('"', "").split(": ")[0])
            indented_key_str = " " * (indented_key_size + 2)
            padded = False

            def _pre_pad_alpha_line(str_in):
                nonlocal padded
                padded = True
                return "\\n" + indent_str + indented_key_str + str_in

            leading_str_to_keep = ", ".join(
                [
                    _pre_pad_alpha_line(s) if s[0].isalpha() and i != 0 else s
                    for i, s in enumerate(leading_str_to_keep.split(", "))
                ]
            )
            local_indent_str = "" if padded else indent_str
            leading_str = leading_str_to_keep.split("\\n")[-1].replace('"', "")
            remaining_str = array_str_in_split[1]
            num_extra_dims = 0
            for i, char in enumerate(remaining_str):
                if char != "[":
                    num_extra_dims = i
                    break
            extra_indent = (len(leading_str) + 1 + num_extra_dims) * " "
            array_str_in = split_phrase.join([leading_str_to_keep, remaining_str])
            uniform_indent_wo_overflow = array_str_in.replace(
                "\\n[", "\n" + local_indent_str + extra_indent + "["
            )
            uniform_indent_wo_overflow_list = list(
                filter(None, uniform_indent_wo_overflow.split("\\n"))
            )
            uniform_indent = "\n".join(
                [
                    (
                        local_indent_str + extra_indent + " " + s
                        if (
                            s[0].isnumeric()
                            or s[0] == "-"
                            or s[0:3] == "..."
                            or max([ss in s[0:6] for ss in ["nan, ", "inf, "]])
                        )
                        else (
                            indent_str + indented_key_str + s
                            if (not s[0].isspace() and s[0] != '"')
                            else s
                        )
                    )
                    for s in uniform_indent_wo_overflow_list
                ]
            )
            indented = uniform_indent
            # 10 dimensions is a sensible upper bound for the number in a single array
            for i in range(2, 10):
                indented = indented.replace(" " * (i - 1) + "[" * i, "[" * i)
                indented = "\n".join(
                    [s for s in indented.split("\n") if bool(s) and not s.isspace()]
                )
            return indented

        def _align_arrays(str_in):
            chunks = str_in.split("\n" + indent_str)
            aligned_array_chunks = {
                i: _align_array(c) for i, c in enumerate(chunks) if "\\n" in c
            }
            chunks = [
                aligned_array_chunks[i] if i in aligned_array_chunks else c_orig
                for i, c_orig in enumerate(chunks)
            ]
            return ("\n" + indent_str).join(chunks)

        new_dict = dict()
        for k, v in self.items():
            if isinstance(v, ivy.Container):
                # noinspection PyArgumentList
                rep = v.__repr__(as_repr=False)
            else:
                if (
                    (self._cont_ivy.is_native_array(v) or isinstance(v, ivy.Array))
                    and len(list(v.shape)) > 0
                    and ivy.exists(self._print_limit)
                    and _reduce(mul, v.shape) > self._print_limit
                ):
                    rep = (type(v), "shape=", list(v.shape))
                elif (
                    isinstance(v, (list, tuple))
                    and v
                    and (
                        self._cont_ivy.is_native_array(v[0])
                        or isinstance(v[0], ivy.Array)
                    )
                ):
                    if (
                        isinstance(v, tuple)
                        and hasattr(v, "_asdict")
                        and hasattr(v, "_fields")
                    ):
                        if len(v) <= self._print_limit:
                            rep = tuple(
                                [
                                    (
                                        "{} = {}".format(name, v[i])
                                        if v[i].size < self._print_limit
                                        else "{} = {}, shape={}".format(
                                            name, type(v[i]), list(v[i].shape)
                                        )
                                    )
                                    for i, name in enumerate(v._fields)
                                ],
                            )
                        else:
                            rep = (
                                "NamedTuple({})".format(len(v)),
                                type(v[0]),
                                "shape={}".format(list(v[0].shape)),
                            )

                    elif isinstance(v, tuple):
                        rep = (
                            "tuple({})".format(len(v)),
                            type(v[0]),
                            "shape={}".format(list(v[0].shape)),
                        )
                    else:
                        rep = (
                            "list[{}]".format(len(v)),
                            type(v[0]),
                            "shape={}".format(list(v[0].shape)),
                        )

                else:
                    rep = v
            new_dict[k] = rep
        if as_repr:
            json_dumped_str = _align_arrays(
                json.dumps(
                    ivy.Container(new_dict, **self._config)
                    .cont_map(
                        lambda x, kc: (
                            x
                            if _is_jsonable(x)
                            else _repr(x).replace(" ", "").replace(",", ", ")
                        )
                    )
                    .cont_to_dict(),
                    indent=self._print_indent,
                )
            )
            json_dumped_str = json_dumped_str.replace("true", "True").replace(
                "false", "False"
            )

            def _add_newline(str_in):
                str_in_split = str_in.split("\n")
                str_split_size = len(str_in_split)
                return "\n".join(
                    [
                        (
                            ("\n" * self._print_line_spacing + ss)
                            if i == (str_split_size - 1)
                            else ss
                        )
                        for i, ss in enumerate(str_in_split)
                    ]
                )

            json_dumped_str = '":'.join(
                [_add_newline(s) for s in json_dumped_str.split('":')]
            )
            # improve tf formatting
            if ivy.backend_stack and ivy.current_backend_str() == "tensorflow":
                json_dumped_str_split = json_dumped_str.split("'Variable:")
                json_dumped_str = (
                    json_dumped_str_split[0]
                    + ", "
                    + ", ".join(
                        [
                            "'".join(ss.split("'")[1:])
                            for ss in json_dumped_str_split[1:]
                        ]
                    )
                )
                json_dumped_str = (
                    json_dumped_str.replace(":shape", ", shape")
                    .replace(")dtype=", "), dtype=")
                    .replace(", ),", ",),")
                )
                json_dumped_str = re.sub("}, $", "}", json_dumped_str)
            # color keys
            json_dumped_str_split = json_dumped_str.split('":')
            split_size = len(json_dumped_str_split)
            json_dumped_str = '":'.join(
                [
                    (
                        ' "'.join(
                            sub_str.split(' "')[:-1]
                            + [
                                termcolor.colored(
                                    ivy.Container.cont_trim_key(
                                        sub_str.split(' "')[-1], self._key_length_limit
                                    ),
                                    self._default_key_color,
                                )
                            ]
                        )
                        if i < split_size - 1
                        else sub_str
                    )
                    for i, sub_str in enumerate(json_dumped_str_split)
                ]
            )
            # remove quotation marks, shape tuple, and color other elements of the dict
            ret = (
                json_dumped_str.replace('"', "")
                .replace(", 'shape=', [", " shape=[")
                .replace(":", termcolor.colored(":", "magenta"))
                .replace("{", termcolor.colored("{", "blue"))
                .replace("}", termcolor.colored("}", "blue"))
                .replace("shape=", termcolor.colored("shape=", "magenta"))
                .replace("device=", termcolor.colored("device=", "magenta"))
                .replace("<class'", "<class '")
                .replace("'", "")
                .replace("<class", "<" + termcolor.colored("class", "blue"))
            )
            # ToDo: make the solution below more elegant
            for i in range(10):
                ret = ret.replace(
                    "diff_{}".format(i), termcolor.colored("diff_{}".format(i), "red")
                )
            for keyword, color in self._keyword_color_dict.items():
                ret = ret.replace(keyword, termcolor.colored(keyword, color))
            return ret
        return new_dict

    def __dir__(self):
        return list(super.__dir__(self)) + list(self.keys())

    # noinspection PyProtectedMember
    def __getattr__(self, item, *args, **kwargs):
        try:
            ret = dict.__getitem__(self, item)
        except KeyError:
            # noinspection PyUnresolvedReferences
            ret = ivy.Container()
            for k, v in self.items():
                if isinstance(v, ivy.Container):
                    result = v.__getattr__(item, *args, **kwargs)
                else:
                    # raise error
                    if not hasattr(v, item):
                        raise AttributeError(
                            "'{}' object has no attribute '{}'".format(
                                type(v).__module__, item
                            )
                        )
                    attr = getattr(v, item)
                    result = attr(*args, **kwargs) if callable(attr) else attr
                ret.__setitem__(k, result)
        return ret

    def __setattr__(self, name, value):
        if name[0] != "_":
            self[name] = value
        else:
            super.__setattr__(self, name, value)

    def _get_queue_item(self, query):
        if isinstance(query, int):
            queue_queries = [query]
        elif isinstance(query, slice):
            queue_queries = list(
                range(query.start, query.stop, ivy.default(query.step, 1))
            )
        elif isinstance(query, (list, tuple)):
            queue_queries = list(
                range(query[0].start, query[0].stop, ivy.default(query[0].step, 1))
            )
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid slice type, must be one of integer, slice "
                "or sequences of slices."
            )
        queue_idxs = set(
            [np.sum(q >= self._queue_load_sizes_cum).item() for q in queue_queries]
        )
        conts = list()
        for i in queue_idxs:
            if i not in self._loaded_containers_from_queues:
                cont = ivy.Container(
                    self._queues[i].get(timeout=self._queue_timeout), **self._config
                ).to_ivy()
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
            shifted_query = slice(query.start - offset, query.stop - offset, query.step)
        elif isinstance(query, (list, tuple)):
            shifted_query = tuple(
                [
                    slice(slc.start - offset, slc.stop - offset, slc.step)
                    for slc in query
                ]
            )
        # noinspection PyUnboundLocalVariable
        return combined_cont[shifted_query]

    def __getitem__(self, query):
        """
        Get slice, key or key chain of container object.

        Parameters
        ----------
        query slice or str
            slice object, key or key chain to query all container elements.

        Returns
        -------
            Container object at desired query.
        """
        if isinstance(query, str):
            if query == "":
                return self
            if "/" in query or "." in query:
                ret = self.cont_at_key_chain(query)
                return ret
            ret = dict.__getitem__(self, query)
            return ret
        elif ivy.exists(self._queues):
            ret = self._get_queue_item(query)
            return ret
        return_dict = dict()
        for key, value in self.items():
            if isinstance(value, ivy.Container):
                return_dict[key] = value[query]
            else:
                # noinspection PyBroadException
                if isinstance(value, list) or isinstance(value, tuple):
                    if len(value) == 0:
                        return_dict[key] = value
                    else:
                        return_dict[key] = value[query]
                elif value is None or hasattr(value, "shape") and value.shape == ():
                    return_dict[key] = value
                else:
                    return_dict[key] = value[query]
        ret = ivy.Container(return_dict, **self._config)
        return ret

    def __setitem__(self, query, val):
        """
        Set key or key chain of container object.

        Parameters
        ----------
        query slice or str
            slice object, key or key chain at which to set all container elements.
        val ivy.Container, array, or other
            The value to set at the desired query.

        Returns
        -------
            New container after updating.
        """

        def _map_fn(fn, x):
            x = x.data if isinstance(x, ivy.Array) else x
            return fn(x)

        if query == "_backend":
            self._backend = val
            return

        if query == "dynamic_backend":
            from ivy.functional.ivy.gradients import _variable
            from ivy.utils.backend.handler import _determine_backend_from_args

            if not val:
                self._backend = _determine_backend_from_args(self)
            else:
                is_variable = self._backend.is_variable
                to_numpy = self._backend.to_numpy
                variable_data = self._backend.variable_data

                def _is_var(x):
                    x = x.data if isinstance(x, ivy.Array) else x
                    return is_variable(x)

                is_var = self.cont_map(lambda x, kc: _is_var(x)).cont_all_true()
                if is_var and not (
                    str(self._backend).__contains__("jax")
                    or str(self._backend).__contains__("numpy")
                ):
                    self.cont_map(lambda x, kc: _map_fn(variable_data, x), inplace=True)
                    self.cont_map(lambda x, kc: _map_fn(to_numpy, x), inplace=True)
                    self.cont_map(lambda x, kc: _map_fn(ivy.array, x), inplace=True)
                    self.cont_map(lambda x, kc: _map_fn(_variable, x), inplace=True)

                else:
                    self.cont_map(lambda x, kc: _map_fn(to_numpy, x), inplace=True)
                    self.cont_map(lambda x, kc: _map_fn(ivy.array, x), inplace=True)

            def _set_dyn_backend(obj, val):
                if isinstance(obj, ivy.Array):
                    obj._dynamic_backend = val
                    return

                if isinstance(obj, ivy.Container):
                    for item in obj.values():
                        _set_dyn_backend(item, val)

                    obj._dynamic_backend = val

            _set_dyn_backend(self, val)
            return

        if isinstance(query, str) and ("/" in query or "." in query):
            return self.cont_set_at_key_chain(query, val, inplace=True)
        else:
            return dict.__setitem__(self, query, val)

    def __contains__(self, key):
        if isinstance(key, str) and ("/" in key or "." in key):
            return self.cont_has_key_chain(key)
        elif isinstance(key, ivy.Container):
            return self.cont_contains_sub_container(key)
        else:
            return dict.__contains__(self, key)

    def __getstate__(self):
        state_dict = copy.copy(self.__dict__)
        state_dict["_local_ivy"] = (
            state_dict["_local_ivy"].current_backend_str()
            if state_dict["_local_ivy"] is not None
            else None
        )
        config_in = copy.copy(state_dict["_config_in"])
        config_in["ivyh"] = (
            config_in["ivyh"].current_backend_str()
            if config_in["ivyh"] is not None
            else None
        )
        state_dict["_config_in"] = config_in
        config = copy.copy(state_dict["_config"])
        config["ivyh"] = (
            config["ivyh"].current_backend_str() if config["ivyh"] is not None else None
        )
        state_dict["_config"] = config
        return state_dict

    def __setstate__(self, state_dict):
        if "_local_ivy" in state_dict:
            if ivy.exists(state_dict["_local_ivy"]):
                if len(state_dict["_local_ivy"]) > 0:
                    state_dict["_local_ivy"] = ivy.with_backend(
                        state_dict["_local_ivy"], cached=True
                    )
                else:
                    state_dict["_local_ivy"] = ivy
        if "_config_in" in state_dict:
            config_in = copy.copy(state_dict["_config_in"])
            if "ivyh" in config_in:
                if ivy.exists(config_in["ivyh"]):
                    if len(config_in["ivyh"]) > 0:
                        config_in["ivyh"] = ivy.with_backend(
                            config_in["ivyh"], cached=True
                        )
                    else:
                        config_in["ivyh"] = ivy
            state_dict["_config_in"] = config_in
        if "_config" in state_dict:
            config = copy.copy(state_dict["_config"])
            if "ivyh" in config:
                if ivy.exists(config["ivyh"]):
                    if len(config["ivyh"]) > 0:
                        config["ivyh"] = ivy.with_backend(config["ivyh"], cached=True)
                    else:
                        config["ivyh"] = ivy
            state_dict["_config"] = config
        self.__dict__.update(state_dict)

    # Getters and Setters #
    # --------------------#

    # private

    @property
    def _cont_ivy(self):
        return ivy.default(self._local_ivy, ivy)

    @_cont_ivy.setter
    def _cont_ivy(self, local_ivy):
        self._local_ivy = local_ivy

    # public

    @property
    def cont_shape(self):
        """
        The shape of the arrays in the container.

        None is placed in indices which are not consistent across
        arrays.
        """
        return self._cont_get_shape()

    @property
    def cont_shapes(self):
        """
        The shapes of each array in the container.

        None is placed in leaf entries without a shape attribute.
        """
        return self._cont_get_shapes()

    @property
    def cont_dev(self):
        """
        The device to which the arrays in the container belong.

        None returned if the devices are not consistent.
        """
        return self._cont_get_dev()

    @property
    def cont_dev_str(self):
        """
        The device to which the arrays in the container belong.

        None returned if the devices are not consistent.
        """
        return self._cont_get_dev()

    @property
    def cont_ivy(self):
        return self._cont_ivy

    @property
    def cont_config(self):
        return self._config

    @property
    def cont_max_depth(self):
        kcs = [kc for kc in self.cont_to_iterator_keys(include_empty=True)]
        if not kcs:
            return 0
        return max([len(kc.split("/")) for kc in kcs])

    @property
    def dynamic_backend(self):
        return self._dynamic_backend

    @dynamic_backend.setter
    def dynamic_backend(self, value):
        self._dynamic_backend = value
