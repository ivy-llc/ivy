"""Base class for helper module methods."""

# global
import abc
import numpy as np
import termcolor

# local
import ivy


class ModuleHelpers(abc.ABC):
    # Private #
    # --------#
    def _top_v_fn(self, /, depth=None, flatten_key_chains=False):
        """
        Return the variables at a specific depth, with depth 1 returning the variables
        of the current layer.

        Parameters
        ----------
        depth
            depth of the variables to return. 1 for current layer, None for the
            topmost layer. Default is ``None``.
        flatten_key_chains
            If set True, will return a flat container which all nested key-chains
            flattened. Default is ``False``.

        Returns
        -------
        ret
            The variables of the submodule at the specified depth.
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

    def _top_mod_fn(self, /, depth=None):
        """
        Find the top (parent) module at specific depth, starting with depth 1 to return
        the current submodule.

        Parameters
        ----------
        depth
            The number of modules we want to trace back. 1 for the current module, None
            for the topmost module. Default is ``None``.

        Returns
        -------
        ret
            The module we want to track down. Return current module if no top
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
        Return True if the current module should have its returns tracked as set by the
        user during the call.

        Returns
        -------
        ret
            True if the returned values of the current module should be
            tracked.
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
        Return True if there is an expected submodule return value set by the user
        during the call.

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
        Return the depth of the current module. Return 0 for root module.

        Returns
        -------
        ret
            The depth of the module in the network.
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
        Return the height of the network, with the current level being 0.

        Returns
        -------
        ret
            The height of the network. 0 if the are no submodules.
        """
        return self.sub_mods().cont_max_depth - 1

    # Public #
    # ------#

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
        Get the key of current module to be used when checking or tracking the return
        values of a submodule.

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

    def sub_mods(self, /, *, show_v=True, depth=None, flatten_key_chains=False):
        """
        Return a container comoposed of all submodules.

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
            in which all nested key-chains flattened. Default is ``False``.

        Returns
        -------
        ret
            A container composed of all submodules.
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
        Show sub containers from the perspective of the top layer. Will give prompt if
        either of `v` or `top_v` is not initialized.

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
        Show the network's variables from the perspective of value of top layer. Will
        give prompt if either of `v` and `top_v` is not initialized.

        Parameters
        ----------
        depth
            The number of modules we want to step in. None for the value of
            current module. Default is ``None``.
        flatten_key_chains
            If set True, will return a flat container,
            with all nested key-chains flattened. Default is ``False``.
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
        Return a list containing the modules of the network starting from the top
        module, and ending with the current module.

        Parameters
        ----------
        depth
            If specified, will return a list of modules of length starting at
            the current module and ending at the module at the specified depth.
            0 for the current module. 1 for the iimediate parent module. None for
            the top module. Default is ``None``.

        flatten_key_chain
            If set True, will return return a flat container,
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
        Show lower submodules in the top module. `upper_depth` and `lower_depth` are for
        controlling the coverage of upper and lower modules. Will give prompt if no top
        module found.

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

    def _add_submod_ret(self, ret, /):
        """
        Add returns to submod_rets variable of the top module.

        Parameters
        ----------
        ret
            The return to be added.
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
        Check the actual submodule returns with the expected submodule return values.

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
            ivy.utils.assertions.check_true(
                np.allclose(ret, expected_ret, **kwargs),
                message="ret: {} and expected_ret: {} were not close enough".format(
                    ret, expected_ret
                ),
            )

    # noinspection PyProtectedMember
    def _is_submod_leaf(self):
        """
        Check if the submodule is the leaf node of the network.

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
        """Add key chains to submod_call_order variable of the top module."""
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
                    alphabetical_keys=False, ivyh=ivy.with_backend("numpy", cached=True)
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
                alphabetical_keys=False, ivyh=ivy.with_backend("numpy", cached=True)
            )

    def show_structure(self):
        """
        Print the structure of the layer network.

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

    def _convert_tensors_to_numpy(self):
        """
        Recursively traverses the _sub_mods attribute of a Module object and converts
        every container containing tensors to numpy using the to_numpy() method.

        Returns
        -------
        Module
            The converted Module object.
        """
        if len(self._sub_mods) > 0:
            for sub_mod in self._sub_mods:
                sub_mod._convert_tensors_to_numpy()
            self.v = self.v.to_numpy()
        else:
            self.v = self.v.to_numpy()

    def _convert_numpy_to_tensors(self):
        """
        Recursively traverses the _sub_mods attribute of a Module object and converts
        every container containing tensors to numpy using the to_numpy() method.

        Returns
        -------
        Module
            The converted Module object.
        """
        if len(self._sub_mods) > 0:
            for sub_mod in self._sub_mods:
                sub_mod._convert_numpy_to_tensors()
                self.v = self.v.to_ivy()
        else:
            self.v = self.v.to_ivy()
