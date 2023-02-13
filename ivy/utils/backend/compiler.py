import sys
from . import _importlib
from . import ast_helpers

# We shouldn't be able to set the backend on a local Ivy
# modules_to_remove = ["utils.backend.handler"]


def with_backend(backend: str):
    finder = ast_helpers.IvyPathFinder()
    sys.meta_path.insert(0, finder)
    _importlib.path_hooks.insert(0, finder)
    ivy_pack = _importlib._ivy_import_module("ivy")
    ivy_pack._is_local = True
    backend_module = _importlib._ivy_import_module(
        ivy_pack.utils.backend.handler._backend_dict[backend], ivy_pack.__package__
    )
    # TODO temporary
    if backend == "numpy":
        ivy_pack.set_default_device("cpu")
    elif backend == "jax":
        ivy_pack.set_global_attr("RNG", ivy_pack.functional.backends.jax.random.RNG)
    # We know for sure that the backend stack is empty, no need to do backend unsetting
    ivy_pack.utils.backend.handler._set_backend_as_ivy(
        ivy_pack.__dict__.copy(), ivy_pack, backend_module
    )
    # Remove access to specific modules on local Ivy
    # TODO this doesn't work properly atm due to not handling nested module name
    # for module in modules_to_remove:
    #    for fn in inspect.getmembers(ivy_pack.__dict__[module], inspect.isfunction):
    #        if fn[1].__module__ != module:
    #            continue
    #        if hasattr(ivy_pack, fn[0]):
    #            del ivy_pack.__dict__[fn[0]]
    #    del ivy_pack.__dict__[module]
    ivy_pack.backend_stack.append(backend_module)
    # TODO use an init function
    ivy_pack.__setattr__("import_module", ivy_pack.utils.backend._ivy_import_module)
    _importlib.path_hooks.remove(finder)
    sys.meta_path.remove(finder)
    _importlib._clear_cache()
    return ivy_pack
