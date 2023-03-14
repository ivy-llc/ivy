import sys
import os

_reference_backend = ""
_target_backend = ""


def _parse_module(module_path: str):
    pass


def _copy_tree(backend_reference_path: str, backend_generation_path: str):
    for root, _, files in os.walk(backend_reference_path):
        # Skip pycache dirs
        if root.endswith("__pycache__"):
            continue

        relative_path = os.path.relpath(root, backend_reference_path)
        os.makedirs(os.path.join(backend_generation_path, relative_path))

        for name in files:
            # Skip pycache modules
            if name.endswith("pyc"):
                continue

            with open(os.path.join(root, name)) as ref_file:
                ref_str = ref_file.read()

            with open(
                os.path.join(backend_generation_path, relative_path, name), "w"
            ) as generated_file:
                generated_file.write(ref_str)


def generate(backend_reference: str, target_backend: str):
    global _target_backend, _reference_backend
    _target_backend = target_backend
    _reference_backend = backend_reference

    backends_root = "../../ivy/functional/backends/"
    backend_reference_path = backends_root + backend_reference
    backend_generation_path = backends_root + target_backend
    _copy_tree(backend_reference_path, backend_generation_path)


if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2])
