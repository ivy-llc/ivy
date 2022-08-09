"""Base class for deriving trainable modules"""

# global
import os
import abc
import ivy.functional.backends.numpy
import termcolor
import numpy as np

# local
import ivy
from ivy.container import Container
from ivy.func_wrapper import _get_first_array


# Base #
# -----#


class Module(abc.ABC):
    def __init__(
        self,
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
    ):
        """
        Initialze Ivy layer, which is a stateful object consisting of trainable
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
            Whether to compile the network on the next forward pass. Default is False.
        store_vars
            Whether or not to store the variables created. Default is True.
        stateful
            The constant id stateful items to track as part of the forward pass.
            Used when graph compiling, default is None.
        arg_stateful_idxs
            The nested argument indices of stateful items to track as part of
            the forward pass.
            Used when graph compiling, default is None.
        kwarg_stateful_idxs
            The nested keyword argument indices of stateful items to track as part of
            the forward pass. Used when graph compiling, default is None.
        fallback_to_non_compiled
            Whether to fall back to non-compiled forward call in the case that an error
            is raised during the compiled forward pass. Default is True.
        with_partial_v
            Whether to allow partial specification of variables. Default is False.
        devices
            devices on which to distribute the module's variables
            'cuda:0', 'cuda:1', 'cpu' etc. (Default value = None)
        """
        valid_build_modes = ["on_init", "explicit", "on_call"]
        if build_mode not in valid_build_modes:
            raise Exception(
                "build_mode must be one of {} of type str, but found "
                "{} of type {}".format(valid_build_modes, build_mode, type(build_mode))
            )
        self._dev = ivy.default(
            device, ivy.default(lambda: devices[0], ivy.default_device(), True)
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
        self._v_in = v if (isinstance(v, Container) or v is None) else Container(v)
        self.v = v
        self.top_v = None
        self.top_mod = None
        self._track_submod_rets = False
        self._submod_depth = None
        self._submods_to_track = None
        self._track_submod_call_order = False
        self.submod_rets = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend("numpy")
        )
        self.expected_submod_rets = None
        self.submod_dict = dict()
        self.submod_call_order = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend("numpy")
        )
        self._sub_mods = set()
        self._dtype = dtype
        if build_mode != "on_init":
            return
        self.build()

    # Private #
    # --------#

    def _fn_with_var_arg(self, fn, v_fn):
        def new_fn(*a, with_grads=None, **kw):
            with_grads = ivy.with_grads(with_grads)
            if "v" in kw.keys():
                del kw["v"]
            v = v_fn(self.v)
            if not with_grads:
                v = v.stop_gradient()
            return fn(*a, **kw, v=v)

        new_fn.wrapped = True
        return new_fn

    def _top_v_fn(self, depth=None, flatten_key_chains=False):
        if ivy.exists(self.top_v):
            if ivy.exists(depth):
                ret = self.top_v(depth - 1) if depth > 1 else self.v
            else:
                ret = self.top_v()
        else:
            ret = self.v
        if flatten_key_chains:
            return ret.flatten_key_chains()
        return ret

    def _top_mod_fn(self, depth=None):
        if ivy.exists(self.top_mod):
            if ivy.exists(depth):
                return self.top_mod(depth - 1) if depth > 1 else self
            return self.top_mod()
        return self

    # noinspection PyProtectedMember
    def track_submod_rets(self):
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
        if not ivy.exists(self.top_mod):
            return False
        if ivy.exists(self.top_mod().expected_submod_rets):
            return True
        return False

    # noinspection PyProtectedMember
    def track_submod_call_order(self):
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
        return self.sub_mods().max_depth - 1

    def _find_variables(self, obj=None):
        vs = Container()
        # ToDo: add support for finding local variables, if/when JAX supports
        #  uniquely flagging variables
        if isinstance(obj, Module) and obj is not self:
            obj.top_v = lambda depth=None, flatten_key_chains=False: self._top_v_fn(
                depth, flatten_key_chains
            )
            obj.top_mod = lambda depth=None: self._top_mod_fn(depth)
            self._sub_mods.add(obj)
            return obj.v
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(v)
                if ret:
                    vs["v" + str(i)] = ret
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ret = self._find_variables(v)
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
            return vs
        elif not hasattr(obj, "__dict__"):
            return vs
        for k, v in obj.__dict__.items():
            if v is not None and k[0:2] != "__":
                ret = self._find_variables(v)
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
        return vs

    @staticmethod
    def _extract_v(v, keychain_mappings, orig_key_chain):
        if v.has_key_chain(orig_key_chain):
            ret_cont = v.at_key_chain(orig_key_chain)
        else:
            ret_cont = ivy.Container()
        for old_kc, new_kc in keychain_mappings.items():
            if orig_key_chain in old_kc:
                ret_cont = ret_cont.set_at_key_chain(
                    "/".join(new_kc.split("/")[1:]), v.at_key_chain(new_kc)
                )
        return ret_cont

    def _wrap_call_methods(self, keychain_mappings, key="", obj=None):
        if isinstance(obj, Module) and obj is not self:
            orig_key_chain = key[1:] if key[0] == "_" else key

            obj.__call__ = self._fn_with_var_arg(
                obj.__call__,
                lambda v_: self._extract_v(v_, keychain_mappings, orig_key_chain),
            )
            return
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                self._wrap_call_methods(keychain_mappings, key + "/v" + str(i), val)
            return
        elif isinstance(obj, dict):
            for k, val in obj.items():
                k = (key + "/" + k) if key != "" else k
                self._wrap_call_methods(keychain_mappings, k, val)
            return
        if not hasattr(obj, "__dict__"):
            return
        for k, val in obj.__dict__.items():
            if k[0:2] == "__":
                continue
            k = (key + "/" + k) if key != "" else k
            if val is not None:
                self._wrap_call_methods(keychain_mappings, k, val)
        return

    @staticmethod
    def _remove_duplicate_variables(vs, created):
        created_ids = created.map(lambda x, kc: id(x))
        vs_ids = vs.map(lambda x, kc: id(x))
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

        created_ids.map(lambda x, kc: unique_callback(x, kc))
        vs_ids.map(
            lambda x, kc: unique_callback(x, kc)
            if x not in ids
            else found_dup_callback(x, kc)
        )
        for dup_kc in duplicate_keychains:
            vs = vs.prune_key_chain(dup_kc)
        return vs, keychain_mappings

    # Overridable #

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _create_variables(self, device, dtype):
        """
        Create internal trainable variables, and return as arbitrary nested dict.
        Overridable.

        Parameters
        ----------
        device
            The device string, specifying the device on which to create the variables.
        """
        return {}

    def _build(self, *args, **kwargs) -> bool:
        """
        Build the internal layers and variables for this module. Overridable. Return
        False or empty Container if the build only partially completed (i.e. some
        child Modules have "on_call" build mode). Alternatviely, return True or a
        container of the built variables if the module is built.
        """
        return True

    # Abstract #

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        """
        Forward pass of the layer,
        called after handling the optional input variables.
        """
        raise NotImplementedError

    def _forward_with_tracking(self, *args, **kwargs):
        """Forward pass while optionally tracking submodule returns and call order"""
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
        """
        with_grads = ivy.with_grads(with_grads)
        if not self._built:
            self.build(
                *args,
                **kwargs,
                from_call=True,
                dtype=_get_first_array(*args, **kwargs).dtype
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

    def sub_mods(self, show_v=True, depth=None, flatten_key_chains=False):
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
                    ivy.Container.flatten_key_chain(sm.__repr__(), "_"): sm.sub_mods(
                        show_v, next_depth
                    )
                    for sm in self._sub_mods
                }
            )
            if flatten_key_chains:
                return ret.flatten_key_chains()
            return ret
        if show_v:
            return self.v
        return ""

    def show_v_in_top_v(self, depth=None):
        if ivy.exists(self.top_v) and ivy.exists(self.v):
            self.top_v(depth).show_sub_container(self.v)
        else:
            print(
                "both self.top_v and self.v must be initialized in order to show v in "
                "top_v, "
                "but found\n\ntop_v: {}\n\nv: {}.".format(self.top_v, self.v)
            )

    def v_with_top_v_key_chains(self, depth=None, flatten_key_chains=False):
        if ivy.exists(self.top_v) and ivy.exists(self.v):
            kc = self.top_v(depth).find_sub_container(self.v)
            if kc:
                ret = self.v.restructure_key_chains({"": kc}, keep_orig=False)
            else:
                ret = self.v
            if flatten_key_chains:
                return ret.flatten_key_chains()
            return ret
        else:
            print(
                "both self.top_v and self.v must be initialized in order to show v in "
                "top_v, "
                "but found\n\ntop_v: {}\n\nv: {}.".format(self.top_v, self.v)
            )

    def mod_with_top_mod_key_chain(self, depth=None, flatten_key_chain=False):
        if not ivy.exists(self.top_mod) or depth == 0:
            return self.__repr__()
        max_depth = depth
        depth = 1
        top_mod = self
        mods = [ivy.Container.flatten_key_chain(top_mod.__repr__(), "_")]
        while True:
            if not ivy.exists(top_mod.top_mod):
                break
            top_mod = top_mod.top_mod(1)
            mods.append(ivy.Container.flatten_key_chain(top_mod.__repr__(), "_"))
            if depth == max_depth:
                break
            depth += 1
        if flatten_key_chain:
            return "__".join(reversed(mods))
        return [mod for mod in reversed(mods)]

    def show_mod_in_top_mod(
        self, upper_depth=None, lower_depth=None, flatten_key_chains=False
    ):
        if ivy.exists(self.top_mod):
            upper_depth = ivy.default(upper_depth, self.mod_depth())
            lower_depth = ivy.default(lower_depth, self.mod_height())
            mid_depth = upper_depth + lower_depth
            upper_sub_mods = self.top_mod(upper_depth).sub_mods(depth=mid_depth)
            lower_sub_mods = self.sub_mods(depth=lower_depth)
            if flatten_key_chains:
                upper_sub_mods = upper_sub_mods.flatten_key_chains()
                lower_sub_mods = lower_sub_mods.flatten_key_chains()
            upper_sub_mods.show_sub_container(lower_sub_mods)
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
    ):
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
        self._track_submod_rets = False
        self._submod_depth = None
        self._submods_to_track = None
        self._track_submod_call_order = False
        self.expected_submod_rets = None

    def get_mod_key(self, top_mod=None):
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

    def _add_submod_ret(self, ret):
        top_mod = self.top_mod()
        sr = top_mod.submod_rets
        ret = ivy.to_numpy(ret)
        key = self.get_mod_key(top_mod)
        if key in sr:
            sr[key].append(ret)
        else:
            sr[key] = [ret]

    def _check_submod_ret(self):
        top_mod = self.top_mod()
        esr = top_mod.expected_submod_rets
        key = self.get_mod_key(top_mod)
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
            assert np.allclose(
                ret, expected_ret, **kwargs
            ), "ret\n\n{}\n\nand expected_ret\n\n{}\n\nwere not close enough".format(
                ret, expected_ret
            )

    # noinspection PyProtectedMember
    def _is_submod_leaf(self):
        submod_depth = self.top_mod()._submod_depth
        submods_to_track = self.top_mod()._submods_to_track
        return (
            (ivy.exists(submod_depth) and self.mod_depth() == submod_depth)
            or self.mod_height() == 0
            or (ivy.exists(submods_to_track) and self in submods_to_track)
        )

    def _add_submod_enter(self):
        sco = self.top_mod().submod_call_order
        key_chain = self.mod_with_top_mod_key_chain()
        for key in key_chain[:-1]:
            kcs = sco.key_chains_containing(key, include_empty=True)
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
                    alphabetical_keys=False, ivyh=ivy.get_backend("numpy")
                )
            sco = sco[max_key]
        final_key = key_chain[-1]
        kcs = sco.key_chains_containing(final_key, include_empty=True)
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
                alphabetical_keys=False, ivyh=ivy.get_backend("numpy")
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
        **kwargs
    ):

        with_grads = ivy.with_grads(with_grads)
        self.submod_rets = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend("numpy")
        )
        self.submod_call_order = ivy.Container(
            alphabetical_keys=False, ivyh=ivy.get_backend("numpy")
        )
        self._set_submod_flags(
            track_submod_rets,
            submod_depth,
            submods_to_track,
            track_submod_call_order,
            expected_submod_rets,
        )
        ret = self._call(*args, v=v, with_grads=with_grads, **kwargs)
        self._unset_submod_flags()
        return ret

    def save_weights(self, weights_path):
        """Save the weights on the Module.

        Parameters
        ----------
        weights_path
            The hdf5 file for saving the weights.
        """
        os.makedirs("/".join(weights_path.split("/")[:-1]), exist_ok=True)
        self.v.to_disk_as_hdf5(weights_path)

    def build(self, *args, from_call=False, device=None, dtype=None, **kwargs):
        """Build the internal layers and variables for this module."""
        self._dev = ivy.default(device, self._dev)
        # return False if not from_call but build_mode is on_call

        if not from_call and self._build_mode == "on_call":
            return self.v
        if dtype:
            dtype = ivy.default_dtype(dtype=dtype, as_native=True)
        else:
            dtype = ivy.default_dtype(self._dtype, as_native=True)

        kwargs["dtype"] = dtype
        # build local Module, and any child modules flagged with "explicit" build mode
        built = ivy.default(self._build(*args, **kwargs), True)

        # build variables based on locally built layers, if v not passed in constructor
        v_from_constructor = self._v_in
        created = Container(self._create_variables(self._dev, dtype=dtype))
        created_n_found = Container(dict(**self._find_variables(self), **created))
        if ivy.exists(v_from_constructor):
            if self._with_partial_v:
                if v_from_constructor:
                    created_n_found.assert_contains_sub_structure(
                        v_from_constructor, partial=True
                    )
                self.v = created_n_found.set_at_key_chains(v_from_constructor)
            else:
                created_n_found, _ = self._remove_duplicate_variables(
                    created_n_found, created
                )
                ivy.Container.assert_identical_structure(
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
                        **self._find_variables(self),
                        **self._create_variables(self._dev, dtype=dtype)
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
        this_repr = termcolor.colored(object.__repr__(self), "green")
        sub_mod_repr = self.sub_mods(False).__repr__()
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
    def built(self):
        return self._built
