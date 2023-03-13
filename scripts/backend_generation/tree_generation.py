_backend_generation_path = "/ivy/functional/backends/"
_backend_reference_path = ""
_target_backend = ""


def _parse_module(module_path: str):
    pass


def _copy_tree(backend_root_path: str):
    pass


def generate(backend_reference: str, target_backend: str):
    global _target_backend, _backend_reference_path
    _backend_reference_path = f"{_backend_generation_path}/{backend_reference}"
    _target_backend = target_backend
