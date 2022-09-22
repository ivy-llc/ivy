import inspect
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Dict, List

__all__ = [
    "name_to_func",
    "array_methods",
    "category_to_funcs",
    "EXTENSIONS",
    "extension_to_funcs",
]


spec_dir = Path(__file__).parent.parent / "array-api" / "spec" / "API_specification"
assert spec_dir.exists(), f"{spec_dir} not found - try `git submodule update --init`"
sigs_dir = spec_dir / "signatures"
assert sigs_dir.exists()

spec_abs_path: str = str(spec_dir.resolve())
sys.path.append(spec_abs_path)
assert find_spec("signatures") is not None

name_to_mod: Dict[str, ModuleType] = {}
for path in sigs_dir.glob("*.py"):
    name = path.name.replace(".py", "")
    name_to_mod[name] = import_module(f"signatures.{name}")

array = name_to_mod["array_object"].array
array_methods = [
    f for n, f in inspect.getmembers(array, predicate=inspect.isfunction)
    if n != "__init__"  # probably exists for Sphinx
]

category_to_funcs: Dict[str, List[FunctionType]] = {}
for name, mod in name_to_mod.items():
    if name.endswith("_functions"):
        category = name.replace("_functions", "")
        objects = [getattr(mod, name) for name in mod.__all__]
        assert all(isinstance(o, FunctionType) for o in objects)  # sanity check
        category_to_funcs[category] = objects

all_funcs = []
for funcs in [array_methods, *category_to_funcs.values()]:
    all_funcs.extend(funcs)
name_to_func: Dict[str, FunctionType] = {f.__name__: f for f in all_funcs}

EXTENSIONS: str = ["linalg"]
extension_to_funcs: Dict[str, List[FunctionType]] = {}
for ext in EXTENSIONS:
    mod = name_to_mod[ext]
    objects = [getattr(mod, name) for name in mod.__all__]
    assert all(isinstance(o, FunctionType) for o in objects)  # sanity check
    funcs = []
    for func in objects:
        if "Alias" in func.__doc__:
            funcs.append(name_to_func[func.__name__])
        else:
            funcs.append(func)
    extension_to_funcs[ext] = funcs

for funcs in extension_to_funcs.values():
    for func in funcs:
        if func.__name__ not in name_to_func.keys():
            name_to_func[func.__name__] = func
