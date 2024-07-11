"""Base class for helper module methods."""

# global
import functools
import logging

# local
import ivy
from ivy.data_classes.container import Container
from ivy.func_wrapper import _get_first_array


class ModuleHelpers:
    def _find_variables(
        self,
        /,
        *,
        obj=None,
        without_initialisation=False,
        _visited=None,
    ):
        """Find all internal variables in obj. Return empty Container if obj is
        None.

        Parameters
        ----------
        obj
            The submodule whose internal variables are to be returned. Default
            is None.
        without_initialization
            Whether or not to initialize the variables, to avoid initialization when
            the model variables are passed in the input directly.
        _visited
            Placeholder for tracking the visited nodes, do not set this parameter.

        Returns
        -------
        ret
            The internal variables of the submodule passed in the argument.
        """
        _visited = ivy.default(_visited, {})
        vs = Container()
        if id(obj) in _visited:
            return vs
        _visited[id(obj)] = True
        # ToDo: add support for finding local variables, if/when JAX supports
        #  uniquely flagging variables
        if isinstance(obj, ModuleHelpers) and obj is not self:

            if not obj.built and without_initialisation:
                return lambda: obj._build_and_return_v(
                    *obj._args, dynamic_backend=self._dynamic_backend, **obj._kwargs
                )

            return obj._build_and_return_v(
                *obj._args, dynamic_backend=obj._dynamic_backend, **obj._kwargs
            )
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                )
                if ret:
                    vs[f"v{str(i)}"] = ret
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                )
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
            return vs
        elif not hasattr(obj, "__dict__"):
            return vs
        for k, v in obj.__dict__.items():
            if v is not None and k[0:2] != "__" and k != "_module_dict":
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                )
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
        return vs

    def _find_buffers(self):
        if hasattr(self, "_module_dict"):
            for key, sub_module in self._module_dict.items():
                if len(sub_module._buffers) > 0:
                    self._buffers[key] = sub_module._buffers

    def _build_and_return_v(self, *args, **kwargs):
        self.build(*args, **kwargs)
        return self.v

    @staticmethod
    def _extract_v(v, keychain_mappings: dict, orig_key_chain, /):
        """Extract the variables from the variables container v using the key
        orig_key_chain and reinstantiate the duplicate variables that were
        removed by _remove_duplicate_variables in their correct locations using
        keychain_mappings.

        Parameters
        ----------
        v
            The variables container
        keychain_mappings
            The keychain mappings of duplicate vatriables
        orig_key_chain
            keychain of the variables to be extracted


        Returns
        -------
        ret_cont
            container with the extracted variables.
        """
        if v.cont_has_key_chain(orig_key_chain):
            ret_cont = v.cont_at_key_chain(orig_key_chain)
        else:
            ret_cont = Container()
        for old_kc, new_kc in keychain_mappings.items():
            if orig_key_chain in old_kc:
                # Check if `v` contains `new_kc` before replacing in `ret_cont`
                if v.cont_has_key_chain(new_kc):
                    ret_cont = ret_cont.cont_set_at_key_chain(
                        "/".join(old_kc.split("/")[1:]), v.cont_at_key_chain(new_kc)
                    )
                else:
                    continue
        return ret_cont

    @staticmethod
    def _remove_duplicate_variables(vs, created, /):
        """Remove duplicate variables in `vs` referring to `created`.

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
        ids = {}
        duplicate_keychains = []
        keychain_mappings = {}

        def unique_callback(x, kc):
            ids[x] = kc

        def found_dup_callback(x, kc):
            if ids[x] == kc:
                return
            duplicate_keychains.append(kc)
            keychain_mappings[kc] = ids[x]

        created_ids.cont_map(lambda x, kc: unique_callback(x, kc))
        vs_ids.cont_map(
            lambda x, kc: (
                unique_callback(x, kc) if x not in ids else found_dup_callback(x, kc)
            )
        )
        for dup_kc in duplicate_keychains:
            vs = vs.cont_prune_key_chain(dup_kc)
        return vs, keychain_mappings

    def _wrap_call_methods(
        self, keychain_mappings, /, *, key="", obj=None, _visited=None
    ):
        """Wrap the call methods of the Module object by looping over all the
        items within the module, wrapping the __call__ methods of all
        submodules using _fn_with_var_arg.

        Parameters
        ----------
        keychain_mappings
            The keychain mappings of the object
        key
            The keychain of the object obj, used for recursion.
        obj
            the object whose __call__ method is to be wrapped
        _visited
            Placeholder for tracking the visited nodes, do not set this parameter.

        Returns
        -------
        None
        """
        _visited = ivy.default(_visited, {})
        if id(obj) in _visited or not isinstance(key, str):
            return
        _visited[id(obj)] = True
        if isinstance(obj, ModuleHelpers) and obj is not self:
            orig_key_chain = key[1:] if key[0] == "_" else key

            obj.__call__ = self._fn_with_var_arg(
                obj.__call__, self._extract_v, keychain_mappings, orig_key_chain
            )
            return
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                self._wrap_call_methods(
                    keychain_mappings,
                    key=f"{key}/v{str(i)}",
                    obj=val,
                    _visited=_visited,
                )
            return
        elif isinstance(obj, dict):
            for k, val in obj.items():
                k = f"{key}/{k}" if key != "" and isinstance(k, str) else k
                self._wrap_call_methods(
                    keychain_mappings, key=k, obj=val, _visited=_visited
                )
            return
        if not hasattr(obj, "__dict__"):
            return
        for k, val in obj.__dict__.items():
            if k[0:2] == "__":
                continue
            k = f"{key}/{k}" if key != "" else k
            if val is not None:
                self._wrap_call_methods(
                    keychain_mappings, key=k, obj=val, _visited=_visited
                )
        return

    def _call(self, *args, v=None, buffers=None, **kwargs):
        """Compute forward pass of the layer, treating layer instance as
        callable function.

        Parameters
        ----------
        args
            Positional arguments to the _build method.
        v
            Replace `v` of current layer when forwarding. Restore
            after the forward finished.
        buffers
            Replace `v` of current layer when forwarding. Restore
            after the forward finished.
        kwargs
            Keyword arguments to the _build method.

        Returns
        -------
        ret
            Result of the forward pass of the layer.
        """
        if not self._built:
            first_arr = _get_first_array(*args, **kwargs)
            self.build(
                *args,
                **kwargs,
                from_call=True,
                dtype=first_arr.dtype if ivy.exists(first_arr) else ivy.default_dtype(),
            )

        # If `v` was provided, replace with the module's v
        replace_v = False
        if v is not None:
            v_orig = self.v
            self._v = v
            replace_v = True

        # If `buffers` were provided, replace with the module's buffers
        replace_buffers = False
        if buffers is not None:
            buffers_orig = self.buffers
            self._buffers = buffers
            replace_buffers = True

        if replace_v or replace_buffers:
            # Call the forward pass
            ret = self._forward(*args, **kwargs)
            # Replace v, buffers if needed
            self._v = v_orig if replace_v else self._v
            self._buffers = buffers_orig if replace_buffers else self._buffers
            return ret
        elif hasattr(self.__call__, "wrapped"):
            return self.__call__(*args, **kwargs)
        return self._forward(*args, **kwargs)

    def _rebuild(self):
        logging.warning(
            "Building the module again as a trainable module was modified, "
            'please use the "explicit" or "on_call" build_modes instead '
            'of "on_init" to avoid repetitive building after each addition'
        )
        self._v = Container()
        self._built = False
        self.build(*self._args, **self._kwargs)

    def _compute_module_dict(self):
        self._module_dict = Container()
        for key, value in self.__dict__.items():
            if isinstance(value, ivy.Module):
                if "stateful" in value.__module__ or hasattr(value, "_frontend_module"):
                    self._module_dict[key] = value
                else:
                    self._module_dict[key] = value._module_dict

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def _fn_with_var_arg_wrapper(
        self, *a, fn, v_fn, keychain_mappings, orig_key_chain, **kw
    ):
        if "v" in kw:
            del kw["v"]
        v = v_fn(self.v, keychain_mappings, orig_key_chain)
        return fn(*a, **kw, v=v)

    def _fn_with_var_arg(self, fn, v_fn, /, keychain_mappings, orig_key_chain):
        """Extract variables from `v_fn` and use it as inputs for `fn`.

        Use `v_fn` to extract the variables and use the extracted
        variables as inputs to the call function fn of the module.
        """
        _fn_with_var_arg_wrapper = functools.partial(
            self._fn_with_var_arg_wrapper,
            fn=fn,
            v_fn=v_fn,
            keychain_mappings=keychain_mappings,
            orig_key_chain=orig_key_chain,
        )
        _fn_with_var_arg_wrapper.wrapped = True
        return _fn_with_var_arg_wrapper

    def _convert_tensors_to_numpy(self):
        """Recursively traverses the module_dict attribute of a Module object
        and converts every container containing tensors to numpy using the
        to_numpy() method.

        Returns
        -------
        Module
            The converted Module object.
        """
        if self.module_dict:
            for module in self.module_dict.values():
                module._convert_tensors_to_numpy()
        self.v = self.v.to_numpy()

    def _convert_numpy_to_tensors(self):
        """Recursively traverses the module_dict attribute of a Module object
        and converts every container containing tensors to numpy using the
        to_numpy() method.

        Returns
        -------
        Module
            The converted Module object.
        """
        if self.module_dict:
            for module in self.module_dict.values():
                module._convert_numpy_to_tensors()
                self.v = self.v.to_ivy()
        else:
            self.v = self.v.to_ivy()
