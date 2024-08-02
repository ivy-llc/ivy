"""Base class for deriving trainable modules."""

# global
from collections import OrderedDict
import os
import copy
from packaging import version
import dill
from typing import Optional, Tuple, Dict

# local
import ivy
from ivy.data_classes.container import Container
from ivy.functional.ivy.gradients import _is_variable
from ivy.stateful.helpers import ModuleHelpers
from ivy.stateful.converters import ModuleConverters


class ModuleMeta:
    def __new__(cls, *args, **kwargs):
        # check the module of the class
        # if it's stateful, it's internal
        # we leave this untouched
        if "stateful" in cls.__module__:
            # we are not assigning it a variable
            pass
        else:
            # first check if a var is already assigned
            # this would mean it is a nested custom class
            if not hasattr(Module, "_init_var"):
                # if not , create it and add
                Module._init_var = [cls]
            else:
                Module._init_var.append(cls)
        instance = super().__new__(cls)
        return instance


class Module(ModuleHelpers, ModuleConverters, ModuleMeta):
    """Module is a base class for deriving trainable modules."""

    def __init__(
        self,
        /,
        *args,
        v=None,
        buffers=None,
        build_mode="on_init",
        store_vars=True,
        with_partial_v=False,
        dynamic_backend=None,
        training=True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        """Initialize Ivy layer, which is a stateful object consisting of
        trainable variables.

        Parameters
        ----------
        args
            Positional arguments to the _build method.
        v
            Ivy container of trainable variables. Created internally by default.
        buffers
            Ivy container of buffers/non-trainable arrays in the state_dict.
        build_mode
            How the Module is built, either on initialization (now),
            explicitly by the user by calling build(), or the first
            time the __call__ method is run. Default is on initialization.
        store_vars
            Whether or not to store the variables created. Default is ``True``.
        with_partial_v
            Whether to allow partial specification of variables. Default is ``False``.
        dynamic_backend
            When the value is true, allow conversion of arrays from a different backend
            to the current backend if v passed in the input contains arrays created with
            different backend.
        training
            specifies whether the module is in training or evaluation mode. Default is
            ``True``.
        dtype
            Data type to be used for creating model variables. (Default value = None).
        device
            Device on which to create the module's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None).
        kwargs
            Keyword arguments to the _build method.
        """
        valid_build_modes = ["on_init", "explicit", "on_call"]
        ivy.utils.assertions.check_elem_in_list(build_mode, valid_build_modes)
        self._build_mode = build_mode
        self._with_partial_v = with_partial_v
        self._store_vars = store_vars
        self._built = False
        self._v_from_constructor = (
            v if isinstance(v, Container) or v is None else Container(v)
        )
        self._v = v if v is not None else Container()
        self._buffers = Container(ivy.default(buffers, {}))
        self._module_dict = Container()
        self._args = args
        self._kwargs = kwargs
        self._module_graph = None
        self._target = None
        self._lazy_traced = False
        self._training = training
        self._dynamic_backend = dynamic_backend
        self._device = ivy.default(device, ivy.default_device())
        self._dtype = ivy.default(dtype, ivy.default_dtype())
        if build_mode != "on_init":
            return
        if hasattr(Module, "_init_var"):
            if "stateful" in self.__module__:
                # we know we are operating within the
                # context of another class, and it's a
                # stateful class internally defined
                # so we freeze weight generation
                # unless `v` or `with_partial_v` is passed

                if v or with_partial_v:
                    # build only if `v` or `with_partial_v`
                    self.build(*args, dynamic_backend=dynamic_backend, **kwargs)
                # we don't want to delete the class variable now
                # since there could be other child modules
                return
            # we know this is the custom class that has triggered the
            # class var, so we do the building, and after that delete
            # the class variable, but before that we check if it's a
            # nested scenario, because if it's another custom class initialised
            # within another one, then we have to hold variable initialisation
            # here too, unless `v` or `with_partial_v`
            if len(Module._init_var) > 1 and not v and not with_partial_v:
                # hold off initialisation, delete key for this class and
                # move on
                Module._init_var.pop()
                return
            self.build(*args, dynamic_backend=dynamic_backend, **kwargs)
            if Module._init_var[-1] == self.__class__.__name__:
                # you delete it, only if this is the class that caused it's creation
                Module._init_var.pop()

            # do a final check if _init_var  becomes empty, then delete it all together
            if not Module._init_var:
                del Module._init_var

            return
        self.build(*args, dynamic_backend=dynamic_backend, **kwargs)

    # Public Methods #
    # ---------------#

    def build(
        self,
        *args,
        from_call=False,
        device=None,
        dtype=None,
        dynamic_backend=None,
        **kwargs,
    ):
        """Build the internal layers and variables for this module.

        Parameters
        ----------
        args
            Positional arguments to the _build method.
        from_call
            If True, denote that this build is triggered by calling. Otherwise,
            triggered by initializing the module. Default is ``False``.
        device
            The device we want to build module on. None for default device.
            Default is ``None``.
        dtype
            The data type for building the module. Default is ``None``.
        dynamic_backend
            Whether to use dynamic backend setting to deal if variables are passed as
            input and created with a different backend to the current backend.
        kwargs
            Keyword arguments to the _build method.

        Returns
        -------
        ret
            True for successfully built a module.
        """
        self._device = ivy.default(device, self._device)
        self._dtype = ivy.default(dtype, self._dtype)
        self._dynamic_backend = ivy.default(dynamic_backend, self._dynamic_backend)
        # return False if not from_call but build_mode is on_call
        if not from_call and self._build_mode == "on_call":
            return self.v

        # why are we adding this kwarg in user-defined build ?
        # it results in the error while doing `from_haiku_module` if haiku's forward
        # therefore leaving it commented out
        # kwargs["dtype"] = dtype

        # build local Module, and any child modules flagged with "explicit" build mode
        # this gets the child modules initialised at best, their weights
        # remain un-generated
        built = ivy.default(self._build(*args, **kwargs), True)

        # this creates weights for this Module only
        created = Container(
            self._create_variables(device=self._device, dtype=dtype),
            dynamic_backend=self._dynamic_backend,
        )

        # build variables based on locally built layers, if v not passed in constructor
        created_n_found = Container(
            dict(
                **self._find_variables(
                    obj=self,
                    without_initialisation=(
                        True
                        if self._v_from_constructor and not self._with_partial_v
                        else False
                    ),
                ),
                **created,
            ),
            dynamic_backend=self._dynamic_backend,
        )
        created_n_found.cont_config["build_callable"] = True
        if ivy.exists(self._v_from_constructor):
            if self._with_partial_v:
                if self._v_from_constructor:
                    created_n_found.cont_assert_contains_sub_structure(
                        self._v_from_constructor, partial=True
                    )
                self._v = created_n_found.cont_set_at_key_chains(
                    self._v_from_constructor
                )
            else:
                created_n_found, _ = self._remove_duplicate_variables(
                    created_n_found, created
                )

                ivy.Container.cont_assert_identical_structure(
                    [created_n_found, self._v_from_constructor],
                    assert_and_assign=True,
                )

                self._v = created_n_found
        else:
            self._v = created_n_found
        # remove duplicates
        self._v, keychain_mappings = self._remove_duplicate_variables(self._v, created)
        # build any child 'on_call' layers
        if not built and from_call:
            # update child modules to share the same device
            for v in self.__dict__.values():
                if isinstance(v, ivy.Module):
                    v._device = self._device

            # build during forward pass
            self._forward(*args, **kwargs)

            # re-build variables based on additional child on-call layers, if v not
            # passed in constructor
            if not ivy.exists(self._v_from_constructor):
                created_n_found = Container(
                    dict(
                        **self._find_variables(obj=self),
                        **self._create_variables(device=self._device, dtype=dtype),
                    ),
                    dynamic_backend=self._dynamic_backend,
                )
                self._v = created_n_found

            # remove further duplicates with self.v
            self._v, keychain_mappings = self._remove_duplicate_variables(
                self._v, created
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
            self._v = ivy.Container()

        # compute the module dict
        self._compute_module_dict()

        # once all variables built, find and assign buffers
        self._find_buffers()

        return v_ret if bool(v_ret) or isinstance(built, bool) else built

    def trace_graph(
        self,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
        **trace_kwargs,
    ):
        """Trace the `ivy.Module`'s `_unified_ivy_graph` or `_call` method to
        the target backend.

        Parameters
        ----------
        args:
            arguments used to trace. Defaults to None.
        kwargs:
            keyword arguments used to trace. Defaults to None.
        trace_kwargs:
            keyword arguments passed to the trace function.
        """
        # no arguments given to trace, so delay the compilation
        if not (args or kwargs):
            self._lazy_traced = True
            return

        # we do not need convert the args to source
        args = ivy.default(args, ())
        kwargs = ivy.default(kwargs, {})

        # shallow copy the kwargs dict
        kwargs = copy.copy(kwargs)
        kwargs["v"] = self.v

        fn_to_trace = ivy.default(self._module_graph, self._call)

        self._module_graph = ivy.trace_graph(
            fn_to_trace, **trace_kwargs, args=args, kwargs=kwargs
        )

        self._lazy_traced = False

    def register_buffer(self, name, value):
        """Register a buffer.

        Parameters
        ----------
        name
            Name of the buffer
        value
            Value of the buffer
        """
        if value is not None:
            self._buffers.update({name: value})
        else:
            super().__setattr__(name, value)

    def register_parameter(self, name, value):
        """Register a parameter.

        Parameters
        ----------
        name
            Name of the parameter
        value
            Value of the parameter
        """
        self._v.update({name: value})

    def train(self, mode: bool = True):
        """Enable or disable training mode."""
        self._training = mode
        for module in self.v:
            module = getattr(self, module, None)
            if isinstance(module, ivy.Module):
                module.train(mode=mode)
        return self

    def eval(self):
        """Disable training mode."""
        return self.train(mode=False)

    def to_device(self, device):
        """Move the weights and buffers  to the specified device."""
        self._device = ivy.default(device, self._device)
        for obj in self.state_dict.values():
            if isinstance(obj, ivy.Module):
                obj.to_device(device)
            elif ivy.is_array(obj) or ivy.is_ivy_container(obj):
                ivy.to_device(obj, device, out=obj)
        return self

    def show_graph(
        self,
        randomness_factor: float = 0.1,
        save_to_disk: bool = False,
        notebook: bool = False,
        with_edge_labels: bool = True,
        with_arg_labels: bool = True,
        with_output_labels: bool = True,
        output_connected_only: bool = True,
        highlight_subgraph: Optional[int] = None,
        fname: Optional[str] = None,
    ):
        if not ivy.exists(self._module_graph):
            raise ValueError("You must trace the module to display the graph.")

        return self._module_graph.show(
            save_to_disk=save_to_disk,
            notebook=notebook,
            with_edge_labels=with_edge_labels,
            with_arg_labels=with_arg_labels,
            with_output_labels=with_output_labels,
            output_connected_only=output_connected_only,
            randomness_factor=randomness_factor,
            highlight_subgraph=highlight_subgraph,
            fname=fname,
        )

    def save_weights(self, weights_path, /):
        """Save the weights on the Module.

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

    def save(self, filename):
        """Save the module object to disk using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to save the module object to.
        """
        if ivy.current_backend_str() == "paddle":
            self._convert_tensors_to_numpy()
        with open(filename, "wb") as f:
            dill.dump(self, f)
        if ivy.current_backend_str() == "paddle":
            self._convert_numpy_to_tensors()

    @staticmethod
    def load(filename):
        """Load a module object from disk using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to load the module object from.

        Returns
        -------
        Module
            The loaded module object.
        """
        with open(filename, "rb") as f:
            loaded = dill.load(f)
        if ivy.current_backend_str() == "paddle":
            loaded._convert_numpy_to_tensors()
        return loaded

    # Dunder Methods #
    # ---------------#

    def __call__(
        self,
        *args,
        v=None,
        buffers=None,
        **kwargs,
    ):
        """Forward an input through current module.

        Parameters
        ----------
        args
            Positional args to the build method.
        v
            If given, use this container as internal variables temporarily.
            Default is ``None``.
        buffers
            If given, use this container as internal buffers temporarily.
            Default is ``None``.
        kwargs
            Keyword arguments to the build method.

        Returns
        -------
        ret
        """
        if self._lazy_traced:
            # we are creating graph since we want to transpile module,
            # so set the appropriate backend
            if self._target:
                ivy.set_backend(self._target)
            self.trace_graph(args=args, kwargs=kwargs)
            if self._target:
                ivy.previous_backend()

        if self._module_graph:
            # we need `v` in kwargs, since this is a traced call
            v = v if v else self.v
            return self._module_graph(*args, v=v, **kwargs)

        # convert variables to native arrays so that they can be tracked
        v = ivy.to_native(v)
        ret = self._call(*args, v=v, buffers=buffers, **kwargs)
        return ret

    def __getattribute__(self, name):
        if name == "v":
            if not super().__getattribute__("_v") and not self.built:
                self._build_and_return_v(
                    *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
                )
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in ["v", "buffers"]:
            name = "_" + name
        if isinstance(value, Module):
            ret = super().__setattr__(name, value)
            if (
                hasattr(self, "_build_mode")
                and self.build_mode == "on_init"
                and self.built
            ):
                self._rebuild()
            return ret
        return super().__setattr__(name, value)

    def __delattr__(self, name):
        if hasattr(self, name):
            if isinstance(getattr(self, name), Module):
                super().__delattr__(name)
                if self.build_mode == "on_init":
                    self._rebuild()
                return
        super().__delattr__(name)

    def __repr__(self):
        extra_lines = []
        extra_repr = self._extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key in self.v.keys():
            if isinstance(getattr(self, key, None), Module):
                mod_str = repr(getattr(self, key))
                mod_str = self._addindent(mod_str, 2)
                child_lines.append(f"({key}): {mod_str}")
        lines = extra_lines + child_lines

        main_str = f"{self.__class__.__name__}("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    # Methods to be Optionally Overridden #
    # -----------------------------------#

    def _create_variables(self, *, device=None, dtype=None):
        """Create internal trainable variables, and return as arbitrary nested
        dict. Overridable.

        Parameters
        ----------
        device
            The device string, specifying the device on which to create the variables.
        dtype
            The dtype string, specifying the dtype on which to create the variables.

        Returns
        -------
        ret
            An empty set.
        """
        return {}

    def _build(self, *args, **kwargs) -> bool:
        """Build the internal layers and variables for this module.
        Overridable.

        Returns
        -------
        ret
            False or empty Container if the build only partially completed (i.e. some
            child Modules have "on_call" build mode). Alternatively, return True or a
            container of the built variables if the module is built.
        """
        return True

    def _forward(self, *args, **kwargs):
        """Forward pass of the layer, called after handling the optional input
        variables.

        Raises
        ------
        NotImplementedError
        """
        raise ivy.utils.exceptions.IvyNotImplementedException

    def _extra_repr(self) -> str:
        """Set the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    # Properties #
    # -----------#

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def build_mode(self):
        return self._build_mode

    @property
    def built(self):
        return self._built

    @property
    def training(self):
        return self._training

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    @property
    def state_dict(self):
        """Return the state_dict which is a collection of the variables and
        buffers."""
        return {**self.v, **self.buffers}

    @property
    def module_dict(self):
        return self._module_dict


class _HaikuIvyModule(Module):
    def __init__(self, *args, params_hk, native_module, device, devices, **kwargs):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        ivy.Module.__init__(
            self,
            params_hk,
            *args,
            build_mode="on_init",
            device=device,
            devices=devices,
            **kwargs,
        )

    def _create_variables(self, device, dtype):
        return self._hk_params

    def _build(self, params_hk, *args, **kwargs):
        ivy.set_jax_backend()
        args, kwargs = ivy.args_to_native(*args, **kwargs)
        # noinspection PyUnresolvedReferences
        params_dict = self._hk_flat_map_to_dict(params_hk)
        self._hk_params = ivy.Container(params_dict, dynamic_backend=False)
        param_iterator = self._hk_params.cont_to_iterator()
        _, param0 = next(param_iterator, ["_", 0])
        if hasattr(param0, "device"):
            import jax

            if version.parse(jax.__version__) >= version.parse("0.4.31"):
                self._device = ivy.as_ivy_dev(param0.device)
            else:
                self._device = ivy.as_ivy_dev(param0.device())
        else:
            self._device = ivy.as_ivy_dev("cpu")
        ivy.previous_backend()

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        params_hk = self._dict_to_hk_flat_map(self.v.cont_to_dict())
        ret = self._native_module.apply(params_hk, 0, *a, **kw)
        nested = isinstance(ret, tuple)
        return ivy.to_native(ret, nested=nested)

    def _hk_flat_map_to_dict(self, hk_flat_map):
        from haiku._src.data_structures import FlatMapping

        ret_dict = {}
        for k, v in hk_flat_map.items():
            new_k = k.replace("/", "|")
            if isinstance(v, FlatMapping):
                ret_dict[new_k] = self._hk_flat_map_to_dict(v)
            else:
                ret_dict[new_k] = v
        return ret_dict

    def _dict_to_hk_flat_map(self, dict_in):
        from haiku._src.data_structures import FlatMapping

        ret_flat_map = {}
        for k, v in dict_in.items():
            new_k = k.replace("|", "/")
            if isinstance(v, dict):
                ret_flat_map[new_k] = self._dict_to_hk_flat_map(v)
            else:
                ret_flat_map[new_k] = v
        return FlatMapping(ret_flat_map)


class _FlaxIvyModule(Module):
    def __init__(self, *args, params_fx, native_module, device, devices, **kwargs):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        ivy.Module.__init__(
            self,
            params_fx,
            *args,
            build_mode="on_init",
            device=device,
            devices=devices,
            **kwargs,
        )

    def _create_variables(self, device, dtype):
        return self._fx_params

    def _build(self, params_fx, *args, **kwargs):
        import flax

        args, kwargs = ivy.args_to_native(*args, **kwargs)
        # noinspection PyUnresolvedReferences
        params_dict = flax.core.unfreeze(params_fx)
        self._fx_params = ivy.Container(params_dict, dynamic_backend=False)
        param_iterator = self._fx_params.cont_to_iterator()
        _, param0 = next(param_iterator, ["_", 0])
        self._device = ivy.as_ivy_dev(ivy.dev(param0))

    def _forward(self, *a, **kw):
        import flax

        a, kw = ivy.args_to_native(*a, **kw)
        params_fx = flax.core.freeze(self.v.cont_to_dict())
        ret = self._native_module.apply(params_fx, *a, **kw)
        nested = isinstance(ret, tuple)
        return ivy.to_native(ret, nested=nested)


class _KerasIvyModule(Module):
    def __init__(self, *args, native_module, device, devices, **kwargs):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs

        ivy.Module.__init__(self, *args, device=device, devices=devices, **kwargs)

    def _create_variables(self, device=None, dtype=None):
        return self._native_params

    def _build(self, *args, **kwargs):
        import tensorflow as tf

        def _get_variable_name(variable):
            return variable.path.split("/")[-2] + "/" + variable.name + ":0"

        self._native_params = ivy.Container(
            OrderedDict(
                sorted(
                    [
                        (
                            (
                                param.name
                                if tf.__version__ < "2.16.0"
                                else _get_variable_name(param)
                            ),
                            param,
                        )
                        for param in self._native_module.variables
                    ],
                    key=lambda kv: kv[0],
                )
            ),
            dynamic_backend=False,
        )

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        ret = self._native_module(*a, **kw)
        nested = isinstance(ret, tuple)
        return ivy.to_native(ret, nested=nested)


class _PaddleIvyModule(Module):
    def __init__(self, *args, native_module, device, devices, **kwargs):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs

        ivy.Module.__init__(self, *args, device=device, devices=devices, **kwargs)

    def _create_variables(self, device=None, dtype=None):
        return self._native_params

    def _build(self, *args, **kwargs):
        self._native_params = ivy.Container(
            OrderedDict(
                sorted(
                    [
                        (k.replace(".", "/"), v)
                        for k, v in dict(self._native_module.named_parameters()).items()
                    ]
                )
            ),
            dynamic_backend=False,
        )

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        ret = self._native_module(*a, **kw)
        nested = isinstance(ret, tuple)
        return ivy.to_native(ret, nested=nested)


class _TorchIvyModule(Module):
    def __init__(self, *args, native_module, device, devices, inplace_update, **kwargs):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        self._update_v = (
            self._inplace_update_v if inplace_update else self._replace_update_v
        )
        ivy.Module.__init__(self, *args, device=device, devices=devices, **kwargs)

    def _create_variables(self, device=None, dtype=None):
        return self._native_params

    def _build(self, *args, **kwargs):
        self._native_params = ivy.Container(
            OrderedDict(
                sorted(
                    [
                        (k.replace(".", "/"), v)
                        for k, v in dict(self._native_module.named_parameters()).items()
                    ]
                )
            ),
            dynamic_backend=False,
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
        import torch

        native = ivy.default(native, self._native_module)
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                # noinspection PyProtectedMember
                native._modules[k] = self._replace_update_v(v, native._modules[k])
            elif _is_variable(v):
                # noinspection PyProtectedMember
                native.__setattr__(k, v)
            elif isinstance(v, torch.Tensor):
                # noinspection PyProtectedMember
                native.__setattr__(
                    k, torch.nn.Parameter(v, requires_grad=v.requires_grad)
                )
            else:
                raise ivy.utils.exceptions.IvyException(
                    f"found item in variable container {v} which was neither a sub"
                    " ivy.Container nor a variable."
                )
        return native

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        self._update_v(self.v)
        ret = self._native_module(*a, **kw)
        nested = isinstance(ret, tuple)
        return ivy.to_native(ret, nested=nested)
