"""Base class for deriving trainable modules"""

# global
import os
import abc
import ivy.functional.backends.numpy
import termcolor
import numpy as np
from types import SimpleNamespace

try:
    import haiku as hk
    from haiku._src.data_structures import FlatMapping
    import jax
except ImportError:
    hk = SimpleNamespace()
    hk.Module = SimpleNamespace
    hk.transform = SimpleNamespace
    hk.get_parameter = SimpleNamespace
    FlatMapping = SimpleNamespace
    jax = SimpleNamespace()
    jax.random = SimpleNamespace()
    jax.random.PRNGKey = SimpleNamespace

try:
    import torch
except ImportError:
    torch = SimpleNamespace()
    torch.nn = SimpleNamespace()
    torch.nn.Parameter = SimpleNamespace
    torch.nn.Module = SimpleNamespace

try:
    import tensorflow as tf
except ImportError:
    tf = SimpleNamespace()
    tf.keras = SimpleNamespace()
    tf.keras.Model = SimpleNamespace

import re
import inspect
from collections import OrderedDict
from typing import Optional, Dict, List

# local
import ivy
from ivy.container import Container
from ivy.func_wrapper import _get_first_array
from ivy.functional.ivy.gradients import _is_variable


# Base #
# -----#
class Module(abc.ABC):
    """Module is a base class for deriving trainable modules."""

    def __init__(
        self,
        /,
        *args,
        device=None,
        v=None,
        build_mode="on_init",
        compile_on_next_step=False,
        store_vars=True,
        stateful=None,
        arg_stateful_idxs=None,
        kwarg_stateful_idxs=None,
        fallback_to_non_compiled=False,
        with_partial_v=False,
        devices=None,
        dtype=None,
        **kwargs,
    ):
        """
        Initialize Ivy layer, which is a stateful object consisting of trainable
        variables.

        Parameters
        ----------
        device
            device on which to create the module's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        v
            Ivy container of trainable variables. Created internally by default.
        build_mode
            How the Module is built, either on initialization (now),
            explicitly by the user by calling
            build(), or the first time the __call__ method is run.
            Default is on initialization.
        compile_on_next_step
            Whether to compile the network on the next forward pass.
            Default is ``False``.
        store_vars
            Whether or not to store the variables created. Default is ``True``.
        stateful
            The constant id stateful items to track as part of the forward pass.
            Used when graph compiling, default is ``None``.
        arg_stateful_idxs
            The nested argument indices of stateful items to track as part of
            the forward pass.
            Used when graph compiling, default is ``None``.
        kwarg_stateful_idxs
            The nested keyword argument indices of stateful items to track as part of
            the forward pass. Used when graph compiling, default is ``None``.
        fallback_to_non_compiled
            Whether to fall back to non-compiled forward call in the case that an error
            is raised during the compiled forward pass. Default is ``True``.
        with_partial_v
            Whether to allow partial specification of variables. Default is ``False``.
        devices
            devices on which to distribute the module's variables
            'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)
        """
        valid_build_modes = ["on_init", "explicit", "on_call"]
        ivy.assertions.check_elem_in_list(build_mode, valid_build_modes)
        self._dev = ivy.default(
            device,
            ivy.default(
                lambda: devices[0],
                default_val=ivy.default_device(),
                catch_exceptions=True,
            ),
        )
        self._devs = ivy.default(devices, [self._dev])
        self._build_mode = build_mode
        self._stateful = stateful
        self._arg_stateful_idxs = arg_stateful_idxs
        self._kwarg_stateful_idxs = kwarg_stateful_idxs
        self._fallback_to_non_compiled = fallback_to_non_compiled
        self._with_partial_v = with_partial_v
        self._store_vars = store_vars
        self._built = False
        self._compiled = False
        self._compiled_fn = None
        self._compile_on_next_step = compile_on_next_step
        self._v_in = v if isinstance(v, Container) or v is None else Container(v)
        self.v = v
        self.top_v = None
        self.top_mod = None
        self._track_submod_rets = False
        self._submod_depth = None
        self._submods_to_track = None
        self._track_submod_call_order = False
        self.submod_rets = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend(backend="numpy")
        )
        self.expected_submod_rets = None
        self.submod_dict = dict()
        self.submod_call_order = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend(backend="numpy")
        )
        self._sub_mods = set()
        self._dtype = dtype
        self._args = args
        self._kwargs = kwargs
        if build_mode != "on_init":
            return
        self.build(*args, **kwargs)

    # Private #
    # --------#

    def _fn_with_var_arg(self, fn, v_fn, /):
        def new_fn(*a, with_grads=None, **kw):
            with_grads = ivy.with_grads(with_grads=with_grads)
            if "v" in kw.keys():
                del kw["v"]
            v = v_fn(self.v)
            if not with_grads:
                v = v.stop_gradient()
            return fn(*a, **kw, v=v)

        new_fn.wrapped = True
        return new_fn

    def _top_v_fn(self, /, *, depth=None, flatten_key_chains=False):
        """
        Helps in visualising the top view of a nested network upto
        a certain depth

        Parameters
        ----------
        depth
            depth upto which we want to visualise
        flatten_key_chains
            If set True, will return a flat (depth-1) container,
            which all nested key-chains flattened. Default is ``False``.

        Returns
        -------
        ret


        """
        if ivy.exists(self.top_v):
            if ivy.exists(depth):
                ret = self.top_v(depth - 1) if depth > 1 else self.v
            else:
                ret = self.top_v()
        else:
            ret = self.v
        if flatten_key_chains:
            return ret.cont_flatten_key_chains()
        return ret

    def _top_mod_fn(self, /, *, depth=None):
        """
        Find the top module at specific depth.

        Parameters
        ----------
        depth
            The number of modules we want to trace back.

        Returns
        -------
        ret
            The module we want to track down. Return current layer if no top
            module exists.
        """
        if ivy.exists(self.top_mod):
            if ivy.exists(depth):
                return self.top_mod(depth - 1) if depth > 1 else self
            return self.top_mod()
        return self

    # noinspection PyProtectedMember
    def track_submod_rets(self):
        """
        Tracks the returns of the submodules if track_submod_returns
        argument is set to True during call

        Returns
        -------
        ret
            True if the current module gets tracked in the computation
            graph.
        """
        if not ivy.exists(self.top_mod):
            return False
        top_mod = self.top_mod()
        submods = top_mod._submods_to_track
        if ivy.exists(submods):
            if self not in submods:
                return False
        depth = top_mod._submod_depth
        if ivy.exists(depth):
            return (
                self.top_mod(depth - 1)._track_submod_rets
                if depth > 0
                else self._track_submod_rets
            )
        return top_mod._track_submod_rets

    def check_submod_rets(self):
        """
        Compares the submodule returns with the expected submodule
        returns passed during call

        Returns
        -------
        ret
            True if the top module has expected_submod_rets.
        """
        if not ivy.exists(self.top_mod):
            return False
        if ivy.exists(self.top_mod().expected_submod_rets):
            return True
        return False

    # noinspection PyProtectedMember
    def track_submod_call_order(self):
        """
        Tracks the order in which the submodules are called.


        Returns
        -------
        ret
            True if the current module allows call order tracking.
        """
        if not ivy.exists(self.top_mod):
            return False
        top_mod = self.top_mod()
        submods = top_mod._submods_to_track
        if ivy.exists(submods):
            if self not in submods:
                return False
        depth = top_mod._submod_depth
        if ivy.exists(depth):
            return (
                self.top_mod(depth - 1)._track_submod_call_order
                if depth > 0
                else self._track_submod_call_order
            )
        return top_mod._track_submod_call_order

    def mod_depth(self):
        """
        Return the depth of the current module.

        Returns
        -------
        ret
            The depth of the module in the network. Return 0 for root module.
        """
        depth = 0
        mod_above = self
        while True:
            if ivy.exists(mod_above.top_mod):
                mod_above = mod_above.top_mod(1)
            else:
                break
            depth += 1
        return depth

    def mod_height(self):
        """
        Return the height of the current module.

        Returns
        -------
        ret
            The height of the network. Return 0 for leaf module.
        """
        return self.sub_mods().cont_max_depth - 1

    def _find_variables(self, /, *, obj=None):
        """
        Find all interval variables in obj. Return empty Container if obj is None.

        Parameters
        ----------
        obj
            The submodule whose internal variables are to be returned. Default
            is None.

        Returns
        -------
        ret
            The internal variables of the submodule passed in the argument.
        """
        vs = Container()
        # ToDo: add support for finding local variables, if/when JAX supports
        #  uniquely flagging variables
        if isinstance(obj, Module) and obj is not self:
            obj.top_v = lambda depth=None, flatten_key_chains=False: self._top_v_fn(
                depth=depth, flatten_key_chains=flatten_key_chains
            )
            obj.top_mod = lambda depth=None: self._top_mod_fn(depth=depth)
            self._sub_mods.add(obj)
            return obj.v
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(obj=v)
                if ret:
                    vs["v" + str(i)] = ret
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ret = self._find_variables(obj=v)
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
            return vs
        elif not hasattr(obj, "__dict__"):
            return vs
        for k, v in obj.__dict__.items():
            if v is not None and k[0:2] != "__":
                ret = self._find_variables(obj=v)
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
        return vs

    @staticmethod
    def _extract_v(v, keychain_mappings: dict, orig_key_chain, /):
        """


        Parameters
        ----------
        v
        keychain_mappings
        orig_key_chain


        Returns
        -------
        ret_cont
        """
        if v.cont_has_key_chain(orig_key_chain):
            ret_cont = v.cont_at_key_chain(orig_key_chain)
        else:
            ret_cont = ivy.Container()
        for old_kc, new_kc in keychain_mappings.items():
            if orig_key_chain in old_kc:
                ret_cont = ret_cont.cont_set_at_key_chain(
                    "/".join(new_kc.split("/")[1:]), v.cont_at_key_chain(new_kc)
                )
        return ret_cont

    def _wrap_call_methods(self, keychain_mappings, /, *, key="", obj=None):
        """
        Wraps the call methods of the Module object

        Parameters
        ----------
        keychain_mappings
            The keychain mappings of the object
        key

        obj
            the object whose __call__ method is to be wrapped


        Returns
        -------
        None
        """
        if isinstance(obj, Module) and obj is not self:
            orig_key_chain = key[1:] if key[0] == "_" else key

            obj.__call__ = self._fn_with_var_arg(
                obj.__call__,
                lambda v_: self._extract_v(v_, keychain_mappings, orig_key_chain),
            )
            return
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                self._wrap_call_methods(
                    keychain_mappings, key=key + "/v" + str(i), obj=val
                )
            return
        elif isinstance(obj, dict):
            for k, val in obj.items():
                k = (key + "/" + k) if key != "" and isinstance(k, str) else k
                self._wrap_call_methods(keychain_mappings, key=k, obj=val)
            return
        if not hasattr(obj, "__dict__"):
            return
        for k, val in obj.__dict__.items():
            if k[0:2] == "__":
                continue
            k = (key + "/" + k) if key != "" else k
            if val is not None:
                self._wrap_call_methods(keychain_mappings, key=k, obj=val)
        return

    @staticmethod
    def _remove_duplicate_variables(vs, created, /):
        """
        Remove duplicate variables in `vs` referring to `created`.

        Parameters
        ----------
        vs
            The container that needs to be pruned.
        created
            The container as the duplication reference.

        Returns
        -------
        vs
            The container after removing duplicate variables.
        keychain_mappings
            Dict storing those keys and ids being removed.
        """
        created_ids = created.cont_map(lambda x, kc: id(x))
        vs_ids = vs.cont_map(lambda x, kc: id(x))
        ids = dict()
        duplicate_keychains = list()
        keychain_mappings = dict()

        def unique_callback(x, kc):
            ids[x] = kc

        def found_dup_callback(x, kc):
            if ids[x] == kc:
                return
            duplicate_keychains.append(kc)
            keychain_mappings[kc] = ids[x]

        created_ids.cont_map(lambda x, kc: unique_callback(x, kc))
        vs_ids.cont_map(
            lambda x, kc: unique_callback(x, kc)
            if x not in ids
            else found_dup_callback(x, kc)
        )
        for dup_kc in duplicate_keychains:
            vs = vs.cont_prune_key_chain(dup_kc)
        return vs, keychain_mappings

    # Overridable #

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _create_variables(self, *, device=None, dtype=None):
        """
        Create internal trainable variables, and return as arbitrary nested dict.
        Overridable.

        Parameters
        ----------
        device
            The device string, specifying the device on which to create the variables.

        Returns
        -------
        ret
            An empty set.
        """
        return {}

    def _build(self, *args, **kwargs) -> bool:
        """
        Build the internal layers and variables for this module. Overridable.

        Returns
        -------
        ret
            False or empty Container if the build only partially completed (i.e. some
            child Modules have "on_call" build mode). Alternatively, return True or a
            container of the built variables if the module is built.
        """
        return True

    # Abstract #

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        """
        Forward pass of the layer,
        called after handling the optional input variables.

        Raises
        ------
        NotImplementedError
        """
        raise ivy.exceptions.IvyNotImplementedException

    def _forward_with_tracking(self, *args, **kwargs):
        """
        Forward pass while optionally tracking submodule returns
        and call order.

        Returns
        -------
        ret
            Result of the forward pass of the layer.
        """
        if self.track_submod_call_order():
            self._add_submod_enter()
        ret = self._forward(*args, **kwargs)
        track_submod_rets = self.track_submod_rets()
        check_submod_rets = self.check_submod_rets()
        if track_submod_rets or check_submod_rets:
            self._add_submod_ret(ret)
        if check_submod_rets:
            self._check_submod_ret()
        return ret

    def _call(self, *args, v=None, with_grads=None, **kwargs):
        """
        The forward pass of the layer,
        treating layer instance as callable function.

        Parameters
        ----------
        v
            Replace `v` of current layer when forwarding. Restore
            after the forward finished.
        with_grads
            Whether to forward with gradients.

        Returns
        -------
        ret
            Result of the forward pass of the layer.
        """
        with_grads = ivy.with_grads(with_grads=with_grads)
        if not self._built:
            self.build(
                *args,
                **kwargs,
                from_call=True,
                dtype=_get_first_array(*args, **kwargs).dtype,
            )
        if v is not None:
            v_orig = self.v
            if not with_grads:
                v = v.stop_gradient()
            self.v = (
                Container(v, **v.config) if isinstance(v, Container) else Container(v)
            )
            ret = self._forward_with_tracking(*args, **kwargs)
            self.v = v_orig
            return ret
        elif hasattr(self.__call__, "wrapped"):
            return self.__call__(*args, with_grads=with_grads, **kwargs)
        elif not with_grads:
            v_orig = self.v
            self.v = v_orig.stop_gradient()
            ret = self._forward_with_tracking(*args, **kwargs)
            self.v = v_orig
            return ret
        return self._forward_with_tracking(*args, **kwargs)

    # Public #
    # -------#

    def sub_mods(self, /, *, show_v=True, depth=None, flatten_key_chains=False):
        """
        Return a container composing of all submodules.

        Parameters
        ----------
        show_v
            If set True, will return values of all submodule variables.
            Default is ``True``.
        depth
            How many layers we step in before beginning enumerating submodules.
            None for current layer. Default is ``None``.
        flatten_key_chains
            If set True, will return a flat (depth-1) container,
            which all nested key-chains flattened. Default is ``False``.

        Returns
        -------
        ret
            A container composing of all submodules.
        """
        if self._sub_mods:
            if ivy.exists(depth):
                if depth == 0:
                    if show_v:
                        return self.v
                    return ""
                next_depth = depth - 1
            else:
                next_depth = None
            ret = ivy.Container(
                {
                    ivy.Container.cont_flatten_key_chain(
                        sm.__repr__(), replacement="_"
                    ): sm.sub_mods(show_v=show_v, depth=next_depth)
                    for sm in self._sub_mods
                }
            )
            if flatten_key_chains:
                return ret.cont_flatten_key_chains()
            return ret
        if show_v:
            return self.v
        return ""

    def show_v_in_top_v(self, /, *, depth=None):
        """
        Show sub containers from the perspective of value of top layer.
        Will give prompt if either of `v` and `top_v` is not initialized.

        Parameters
        ----------
        depth
            The number of modules we want to step in. None for the value of
            current module. Default is ``None``.
        """
        if ivy.exists(self.top_v) and ivy.exists(self.v):
            self.top_v(depth).cont_show_sub_container(self.v)
        else:
            print(
                "both self.top_v and self.v must be initialized in order to show v in "
                "top_v, "
                "but found\n\ntop_v: {}\n\nv: {}.".format(self.top_v, self.v)
            )

    def v_with_top_v_key_chains(self, /, *, depth=None, flatten_key_chains=False):
        """
        Show current layer from the perspective of value of top layer.
        Will give prompt if either of `v` and `top_v` is not initialized.

        Parameters
        ----------
        depth
            The number of modules we want to step in. None for the value of
            current module. Default is ``None``.
        flatten_key_chains
            If set True, will return a flat (depth-1) container,
            which all nested key-chains flattened. Default is ``False``.
        """
        if ivy.exists(self.top_v) and ivy.exists(self.v):
            kc = self.top_v(depth).cont_find_sub_container(self.v)
            if kc:
                ret = self.v.cont_restructure_key_chains({"": kc}, keep_orig=False)
            else:
                ret = self.v
            if flatten_key_chains:
                return ret.cont_flatten_key_chains()
            return ret
        else:
            print(
                "both self.top_v and self.v must be initialized in order to show v in "
                "top_v, "
                "but found\n\ntop_v: {}\n\nv: {}.".format(self.top_v, self.v)
            )

    def mod_with_top_mod_key_chain(self, /, *, depth=None, flatten_key_chain=False):
        """
        (TODO)

        Parameters
        ----------
        depth

        flatten_key_chain
            If set True, will return return a flat (depth-1) container,
            with all nested key-chains flattened. Default is ``False``.
        """
        if not ivy.exists(self.top_mod) or depth == 0:
            return self.__repr__()
        max_depth = depth
        depth = 1
        top_mod = self
        mods = [
            ivy.Container.cont_flatten_key_chain(top_mod.__repr__(), replacement="_")
        ]
        while True:
            if not ivy.exists(top_mod.top_mod):
                break
            top_mod = top_mod.top_mod(1)
            mods.append(
                ivy.Container.cont_flatten_key_chain(
                    top_mod.__repr__(), replacement="_"
                )
            )
            if depth == max_depth:
                break
            depth += 1
        if flatten_key_chain:
            return "__".join(reversed(mods))
        return [mod for mod in reversed(mods)]

    def show_mod_in_top_mod(
        self, /, *, upper_depth=None, lower_depth=None, flatten_key_chains=False
    ):
        """
        Show lower submodules in the top module. `upper_depth` and `lower_depth`
        are for controlling the coverage of upper and lower modules.
        Will give prompt if no top module found.

        Parameters
        ----------
        upper_depth
            How many modules it tracks up as upper module. None for current module.
            Default is ``None``. Will be truncated to mod_depth.
        lower_depth
            How many modules it tracks down. None for current module.
            Default is ``None``. Will be truncated to mod_height.
        flatten_key_chains
            If set True, will return a flat (depth-1) container,
            which all nested key-chains flattened. Default is ``False``.
        """
        if ivy.exists(self.top_mod):
            upper_depth = ivy.default(upper_depth, self.mod_depth())
            lower_depth = ivy.default(lower_depth, self.mod_height())
            mid_depth = upper_depth + lower_depth
            upper_sub_mods = self.top_mod(upper_depth).sub_mods(depth=mid_depth)
            lower_sub_mods = self.sub_mods(depth=lower_depth)
            if flatten_key_chains:
                upper_sub_mods = upper_sub_mods.cont_flatten_key_chains()
                lower_sub_mods = lower_sub_mods.cont_flatten_key_chains()
            upper_sub_mods.cont_show_sub_container(lower_sub_mods)
        else:
            print(
                "self.top_mod must be initialized in order to show mod in top_mod,"
                "but found\n\ntop_mod: {}".format(self.top_mod)
            )

    def _set_submod_flags(
        self,
        track_submod_rets,
        submod_depth,
        submods_to_track,
        track_submod_call_order,
        expected_submod_rets,
        /,
    ):
        """
        Set flags of the submodule.

        Parameters
        ----------
        track_submod_rets
            If True, will track the returns of submodules.
        submod_depth
            The depth of tracked submodules.
        submods_to_track
            If given, will only track submodules in `submods_to_track`.
        track_submod_call_order
            If True, will track the call order of submodules.
        expected_submod_rets
            If given, will raise exception if submodule returns are
            different from expected returns.
        """
        self._track_submod_rets = track_submod_rets
        self._submod_depth = submod_depth
        self._submods_to_track = submods_to_track
        self._track_submod_call_order = track_submod_call_order
        self.expected_submod_rets = (
            ivy.Container(expected_submod_rets).to_numpy(map_sequences=True)
            if ivy.exists(expected_submod_rets)
            else expected_submod_rets
        )

    def _unset_submod_flags(self):
        """Unset flags of the submodule."""
        self._track_submod_rets = False
        self._submod_depth = None
        self._submods_to_track = None
        self._track_submod_call_order = False
        self.expected_submod_rets = None

    def get_mod_key(self, /, *, top_mod=None):
        """
        Get the key of current module.

        Parameters
        ----------
        top_mod
            Explicit indicate the top module. None for the top
            module of current module. Default is ``None``.

        Returns
        -------
            A string of current module key.
        """
        if top_mod is None:
            top_mod = self.top_mod()
        submod_dict = top_mod.submod_dict
        full_key = self.__repr__().split(".")[-1]
        name_key = full_key.split(" ")[0]
        if name_key not in submod_dict:
            submod_dict[name_key] = dict()
        id_str = full_key.split(" ")[-1][:-1]
        if id_str not in submod_dict[name_key]:
            submod_dict[name_key][id_str] = str(len(submod_dict[name_key]))
        idx_key = submod_dict[name_key][id_str]
        return " " * self.mod_depth() + "_".join([name_key, idx_key])

    def _add_submod_ret(self, ret, /):
        """
        Add returns in the submodule return of the top module.

        Parameters
        ----------
        ret
            The return you want to add.

        """
        top_mod = self.top_mod()
        sr = top_mod.submod_rets
        ret = ivy.to_numpy(ret)
        key = self.get_mod_key(top_mod=top_mod)
        if key in sr:
            sr[key].append(ret)
        else:
            sr[key] = [ret]

    def _check_submod_ret(self):
        """
        Check submodule returns with expected submodule returns.
        Raise AssertError if returns are not close enough.
        """
        top_mod = self.top_mod()
        esr = top_mod.expected_submod_rets
        key = self.get_mod_key(top_mod=top_mod)
        esr_key = key
        if key not in esr:
            esr_key = key.replace(" ", "")
            if esr_key not in esr:
                return
        sr = self.top_mod().submod_rets
        rets = sr[key]
        esr_ret = esr[esr_key]
        if isinstance(esr_ret, dict):
            expected_rets = esr_ret["val"]
            atols = esr_ret["atol"] if "atol" in esr_ret else None
            if not isinstance(atols, list):
                atols = [atols] * len(expected_rets)
            rtols = esr_ret["rtol"] if "rtol" in esr_ret else None
            if not isinstance(rtols, list):
                rtols = [rtols] * len(expected_rets)
        else:
            expected_rets = esr_ret
            atols = [None] * len(expected_rets)
            rtols = [None] * len(expected_rets)
        for ret, expected_ret, atol, rtol in zip(rets, expected_rets, atols, rtols):
            if expected_ret is None:
                continue
            kwargs = {}
            if atol:
                kwargs["atol"] = atol
            if rtol:
                kwargs["rtol"] = rtol
            ivy.assertions.check_true(
                np.allclose(ret, expected_ret, **kwargs),
                message="ret: {} and expected_ret: {} were not close enough".format(
                    ret, expected_ret
                ),
            )

    # noinspection PyProtectedMember
    def _is_submod_leaf(self):
        """
        Checks if the submodule is the leaf node of the network.

        Returns
        -------
        ret
            True if the submodule is the leaf node of the network.
        """
        submod_depth = self.top_mod()._submod_depth
        submods_to_track = self.top_mod()._submods_to_track
        return (
            (ivy.exists(submod_depth) and self.mod_depth() == submod_depth)
            or self.mod_height() == 0
            or (ivy.exists(submods_to_track) and self in submods_to_track)
        )

    def _add_submod_enter(self):
        """
        (TODO)

        Returns
        -------
        None
        """
        sco = self.top_mod().submod_call_order
        key_chain = self.mod_with_top_mod_key_chain()
        for key in key_chain[:-1]:
            kcs = sco.cont_key_chains_containing(key, include_empty=True)
            if kcs:
                max_key = sorted(
                    kcs,
                    key=lambda kc: int(
                        kc.split("/")[
                            -2 if isinstance(sco[kc], np.ndarray) else -1
                        ].split("_")[-1]
                    ),
                )[-1].split("/")[0]
            else:
                max_key = key + "_0"
                sco[max_key] = ivy.Container(
                    alphabetical_keys=False, ivyh=ivy.get_backend(backend="numpy")
                )
            sco = sco[max_key]
        final_key = key_chain[-1]
        kcs = sco.cont_key_chains_containing(final_key, include_empty=True)
        if kcs:
            sorted_kcs = sorted(
                kcs,
                key=lambda kc: int(
                    kc.split("/")[-2 if isinstance(sco[kc], np.ndarray) else -1].split(
                        "_"
                    )[-1]
                ),
            )
            chosen_kc = sorted_kcs[-1]
            max_key_idx = int(
                chosen_kc.split("/")[
                    -2 if isinstance(sco[chosen_kc], np.ndarray) else -1
                ].split("_")[-1]
            )
            new_key = final_key + "_{}".format(max_key_idx + 1)
        else:
            new_key = final_key + "_0"
        if self._is_submod_leaf():
            sco[new_key] = self.v_with_top_v_key_chains(
                flatten_key_chains=True
            ).to_numpy()
        else:
            sco[new_key] = ivy.Container(
                alphabetical_keys=False, ivyh=ivy.get_backend(backend="numpy")
            )

    def __call__(
        self,
        *args,
        v=None,
        with_grads=None,
        stateful=None,
        arg_stateful_idxs=None,
        kwarg_stateful_idxs=None,
        track_submod_rets=False,
        submod_depth=None,
        submods_to_track=None,
        track_submod_call_order=False,
        expected_submod_rets=None,
        **kwargs,
    ):
        """
        Forward an input through current module.

        Parameters
        ----------
        v
            If given, use this container as internal varibles temporarily.
            Default is ``None``.
        with_grads
            If True, forward this pass with gradients.
        track_submod_rets
            If True, will track the returns of submodules.
        submod_depth
            The depth of tracked submodules.
        submods_to_track
            If given, will only track submodules in `submods_to_track`.
        track_submod_call_order
            If True, will track the call order of submodules.
        expected_submod_rets
            If given, will raise exception if submodule returns are
            different from expected returns.

        Returns
        -------
        ret
        """
        with_grads = ivy.with_grads(with_grads=with_grads)
        self.submod_rets = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend(backend="numpy")
        )
        self.submod_call_order = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend(backend="numpy")
        )
        self._set_submod_flags(
            track_submod_rets,
            submod_depth,
            submods_to_track,
            track_submod_call_order,
            expected_submod_rets,
        )

        # convert variables to native arrays so that they can be tracked
        v = ivy.to_native(v)
        ret = self._call(*args, v=v, with_grads=with_grads, **kwargs)
        self._unset_submod_flags()
        return ret

    def save_weights(self, weights_path, /):
        """
        Save the weights on the Module.

        Parameters
        ----------
        weights_path
            The hdf5 file for saving the weights.

        Returns
        -------
        None
        """
        os.makedirs("/".join(weights_path.split("/")[:-1]), exist_ok=True)
        self.v.cont_to_disk_as_hdf5(weights_path)

    def build(self, *args, from_call=False, device=None, dtype=None, **kwargs):
        """
        Build the internal layers and variables for this module.

        Parameters
        ----------
        from_call
            If True, denote that this build is triggered by calling. Otherwise,
            triggered by initializing the module. Default is ``False``.
        device
            The device we want to build module on. None for default device.
            Default is ``None``.
        dtype
            The data type for building the module. Default is ``None``.

        Returns
        -------
        ret
            True for successfully built a module.
        """
        self._dev = ivy.default(device, self._dev)
        # return False if not from_call but build_mode is on_call

        if not from_call and self._build_mode == "on_call":
            return self.v
        if dtype:
            dtype = ivy.default_dtype(dtype=dtype, as_native=True)
        else:
            dtype = ivy.default_dtype(dtype=self._dtype, as_native=True)

        # TODO: this line causes error when calling consturctor
        # kwargs["dtype"] = dtype
        # build local Module, and any child modules flagged with "explicit" build mode
        built = ivy.default(self._build(*args, **kwargs), True)

        # build variables based on locally built layers, if v not passed in constructor
        v_from_constructor = self._v_in
        created = Container(self._create_variables(device=self._dev, dtype=dtype))
        created_n_found = Container(dict(**self._find_variables(obj=self), **created))
        if ivy.exists(v_from_constructor):
            if self._with_partial_v:
                if v_from_constructor:
                    created_n_found.cont_assert_contains_sub_structure(
                        v_from_constructor, partial=True
                    )
                self.v = created_n_found.cont_set_at_key_chains(v_from_constructor)
            else:
                created_n_found, _ = self._remove_duplicate_variables(
                    created_n_found, created
                )
                ivy.Container.cont_assert_identical_structure(
                    [created_n_found, v_from_constructor]
                )
                self.v = v_from_constructor
        else:
            self.v = created_n_found
        # remove duplicates
        self.v, keychain_mappings = self._remove_duplicate_variables(self.v, created)
        # build any child 'on_call' layers
        if not built and from_call:
            # update child modules to share the same device
            for k, v in self.__dict__.items():
                if isinstance(v, ivy.Module):
                    v._dev = self._dev

            # build during forward pass
            self._forward(*args, **kwargs)

            # re-build variables based on additional child on-call layers, if v not
            # passed in constructor
            if not ivy.exists(v_from_constructor):
                created_n_found = Container(
                    dict(
                        **self._find_variables(obj=self),
                        **self._create_variables(device=self._dev, dtype=dtype),
                    )
                )
                self.v = created_n_found

            # remove further duplicates with self.v
            self.v, keychain_mappings = self._remove_duplicate_variables(
                self.v, created
            )

            # set built flag
            built = True

        # wrap call methods if the module is fully built
        if built:
            self._wrap_call_methods(keychain_mappings, obj=self)

        # flag built and remove local variables if specified
        self._built = bool(built)
        v_ret = self.v
        if not self._store_vars:
            # ToDo: verify variables in self.v are released once this method exits
            self.v = ivy.Container()
        return v_ret if bool(v_ret) or isinstance(built, bool) else built

    def show_structure(self):
        """
        Prints the structure of the layer network.

        Returns
        -------
        this_repr
            String of the structure of the module.
        """
        this_repr = termcolor.colored(object.__repr__(self), "green")
        sub_mod_repr = self.sub_mods(show_v=False).__repr__()
        if sub_mod_repr == "''":
            return this_repr
        print("\n".join([this_repr, sub_mod_repr]))

    def __repr__(self):
        return object.__repr__(self)

    # Properties #
    # -----------#

    @property
    def build_mode(self):
        return self._build_mode

    @property
    def built_(self):
        return self._built

    # Module Converters #
    def to_haiku_module(self):
        """
        Converts an ivy Module instance to a Haiku Module instance.

        Parameters
        ----------
        ivy_module
            The ivy module instance to convert

        Returns
        -------
        ret
            The new trainable hk.Module instance.
        """
        ivy_module = self

        class MyHaikuModel(hk.Module):
            def __init__(self):
                super(MyHaikuModel, self).__init__()
                self._ivy_module = ivy_module

            def __call__(self, *args, **kwargs):
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: hk.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: ivy.to_native(self._ivy_module.v[kc]),
                    )
                )
                a, kw = ivy.args_to_native(*args, **kwargs)
                ret = self._ivy_module._forward(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        return MyHaikuModel

    def to_keras_module(self):
        """
        Converts an ivy Module instance to a Keras Module instance.

        Parameters
        ----------
        self
            The ivy module instance to convert

        Returns
        -------
        ret
            The new trainable tf.keras.Module instance.
        """
        return MyTFModule(self)

    def to_torch_module(self):
        """
        Converts an ivy Module instance to a Torch Module instance.

        Parameters
        ----------
        self
            The ivy module instance to convert

        Returns
        -------
        ret
            The new trainable torch.nn.Module instance.
        """
        return MyTorchModule(self)

    @staticmethod
    def from_haiku_module(
        native_module,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Converts a Haiku module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance).
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.

        Returns
        -------
        ret
            The new trainable torch module instance.

        """
        RNG = jax.random.PRNGKey(42)

        def _hk_flat_map_to_dict(hk_flat_map):
            ret_dict = dict()
            for k, v in hk_flat_map.items():
                new_k = k.replace("/", "|")
                if isinstance(v, FlatMapping):
                    ret_dict[new_k] = _hk_flat_map_to_dict(v)
                else:
                    ret_dict[new_k] = v
            return ret_dict

        def _dict_to_hk_flat_map(dict_in):
            ret_flat_map = dict()
            for k, v in dict_in.items():
                new_k = k.replace("|", "/")
                if isinstance(v, dict):
                    ret_flat_map[new_k] = _dict_to_hk_flat_map(v)
                else:
                    ret_flat_map[new_k] = v
            return FlatMapping(ret_flat_map)

        class HaikuIvyModule(ivy.Module):
            def __init__(self, *args, native_module, device, devices, **kwargs):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs
                ivy.Module.__init__(
                    self,
                    *args,
                    build_mode="on_init",
                    device=device,
                    devices=devices,
                    **kwargs,
                )

            def _create_variables(self, device, dtype):
                return self._hk_params

            def _build(self, *args, **kwargs):
                args, kwargs = ivy.args_to_native(*args, **kwargs)
                # noinspection PyUnresolvedReferences
                params_hk = self._native_module.init(RNG, *args, **kwargs)
                params_dict = _hk_flat_map_to_dict(params_hk)
                self._hk_params = ivy.Container(params_dict)
                param_iterator = self._hk_params.cont_to_iterator()
                _, param0 = next(param_iterator)
                self._dev = ivy.as_ivy_dev(param0.device())

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                params_hk = _dict_to_hk_flat_map(self.v.cont_to_dict())
                ret = self._native_module.apply(params_hk, None, *a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})
        i_args, i_kwargs = ivy.args_to_native(*i_args, **i_kwargs)
        transformed_module = native_module

        if inspect.isclass(native_module):

            if len(i_args) == 0 and len(i_kwargs) == 0:
                raise ivy.exceptions.IvyException(
                    "both instance_args and instance_kwargs cannot be none"
                    " when passing a native class"
                )

            def forward_fn(*a, **kw):
                model = native_module(*c_args, **c_kwargs)
                return model(*i_args, **i_kwargs)

            transformed_module = hk.transform(forward_fn)

        return HaikuIvyModule(
            *i_args,
            native_module=transformed_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_keras_module(
        native_module=None,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Converts a Keras module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance).
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.

        Returns
        -------
        ret
            The new trainable ivy.Module instance.
        """

        class KerasIvyModule(ivy.Module):
            def __init__(self, *args, native_module, device, devices, **kwargs):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs

                ivy.Module.__init__(
                    self, *args, device=device, devices=devices, **kwargs
                )

            def _create_variables(self, device=None, dtype=None):
                return self._native_params

            def _build(self, *args, **kwargs):
                self._native_params = ivy.Container(
                    OrderedDict(
                        sorted(
                            [
                                (param.name, param)
                                for param in self._native_module.variables
                            ]
                        )
                    )
                )

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                ret = self._native_module(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):

            if len(i_args) == 0 and len(i_kwargs) == 0:
                raise ivy.exceptions.IvyException(
                    "both instance_args and instance_kwargs cannot be none"
                    " when passing a native class"
                )
            native_module = native_module(*c_args, **c_kwargs)
            input_shape = i_args[0].shape
            native_module.build((input_shape[-1],))

        return KerasIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_torch_module(
        native_module=None,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
        inplace_update=False,
    ):
        """
        Converts a Torch module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance)
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.
        inplace_update
            For backends with dedicated variable classes, whether to update these
            inplace. Default is ``False``.

        Returns
        -------
        ret
            The new trainable ivy.Module instance.
        """

        class TorchIvyModule(ivy.Module):
            def __init__(
                self, *args, native_module, device, devices, inplace_update, **kwargs
            ):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs
                self._update_v = (
                    self._inplace_update_v if inplace_update else self._replace_update_v
                )
                ivy.Module.__init__(
                    self, *args, device=device, devices=devices, **kwargs
                )

            def _create_variables(self, device=None, dtype=None):
                return self._native_params

            def _build(self, *args, **kwargs):
                self._native_params = ivy.Container(
                    OrderedDict(
                        sorted(
                            [
                                (k.replace(".", "/"), v)
                                for k, v in dict(
                                    self._native_module.named_parameters()
                                ).items()
                            ]
                        )
                    )
                )

            @staticmethod
            def _inplace_update(p, v):
                p.data = v.data

            def _inplace_update_v(self, new_v):
                ivy.Container.cont_multi_map(
                    lambda xs, kc: self._inplace_update(xs[0], xs[1]),
                    [self._native_params, new_v],
                )

            def _replace_update_v(self, new_v, native=None):
                native = ivy.default(native, self._native_module)
                for k, v in new_v.items():
                    if isinstance(v, ivy.Container):
                        # noinspection PyProtectedMember
                        native._modules[k] = self._replace_update_v(
                            v, native._modules[k]
                        )
                    elif _is_variable(v):
                        if isinstance(v, torch.nn.Parameter):
                            # noinspection PyProtectedMember
                            native.__setattr__(k, v)
                        else:
                            # noinspection PyProtectedMember
                            native.__setattr__(k, torch.nn.Parameter(v.data))
                    else:
                        raise ivy.exceptions.IvyException(
                            "found item in variable container {} which was neither a "
                            "sub ivy.Container nor a variable.".format(v)
                        )
                return native

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                self._update_v(self.v)
                ret = self._native_module(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)

        return TorchIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            inplace_update=inplace_update,
            **i_kwargs,
        )


class MyTorchModule(torch.nn.Module):
    def __init__(self, ivy_module):
        torch.nn.Module.__init__(self)
        self._ivy_module = ivy_module
        self._assign_variables()

    def _assign_variables(self):
        self._ivy_module.v.cont_map(
            lambda x, kc: self.register_parameter(
                name=kc, param=torch.nn.Parameter(ivy.to_native(x))
            )
        )
        self._ivy_module.v = self._ivy_module.v.cont_map(
            lambda x, kc: self._parameters[kc]
        )

    def forward(self, *args, **kwargs):
        a, kw = ivy.args_to_native(*args, **kwargs)
        ret = self._ivy_module._forward(*a, **kw)
        if isinstance(ret, tuple):
            return ivy.args_to_native(*ret)
        return ivy.to_native(ret)


class MyTFModule(tf.keras.Model):
    def __init__(self, ivy_module):
        super(MyTFModule, self).__init__()
        self._ivy_module = ivy_module
        self._assign_variables()

    def _assign_variables(self):
        self._ivy_module.v.cont_map(
            lambda x, kc: self.add_weight(
                name=kc, shape=x.shape, dtype=x.dtype, trainable=True
            )
        )
        model_weights = list()
        self._ivy_module.v.cont_map(lambda x, kc: model_weights.append(ivy.to_numpy(x)))
        self.set_weights(model_weights)
        params = {re.sub(":\\d+", "", param.name): param for param in self.variables}
        self._ivy_module.v = self._ivy_module.v.cont_map(lambda x, kc: params[kc])

    def call(self, *args, **kwargs):
        a, kw = ivy.args_to_native(*args, **kwargs)
        ret = self._ivy_module._forward(*a, **kw)
        if isinstance(ret, tuple):
            return ivy.args_to_native(*ret)

        return ivy.to_native(ret)
