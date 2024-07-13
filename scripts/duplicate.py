import ast
import importlib
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_all_functions_from_directory(root_dir, startswith="test"):
    if not os.path.exists(root_dir):
        print("Invalid directory")
        sys.exit(1)
    functions_names = []
    for filename in glob.iglob(f"{root_dir}/**/*.py", recursive=True):
        if len(filename) >= 2 and filename[:2] == "./":
            filename = filename[2:]
        filename = filename.replace(".py", "")
        filename = filename.replace("/", ".")
        module = importlib.import_module(filename)
        module_functions_names = [
            obj for obj in dir(module) if obj.startswith(startswith)
        ]
        functions_names.extend(module_functions_names)
    return functions_names


def check_duplicate_functional_experimental_tests():
    fn_test_core = get_all_functions_from_directory(
        "ivy_tests/test_ivy/test_functional/test_core"
    )
    fn_test_nn = get_all_functions_from_directory(
        "ivy_tests/test_ivy/test_functional/test_nn"
    )
    fn_test_experimental = get_all_functions_from_directory(
        "ivy_tests/test_ivy/test_functional/test_experimental"
    )
    fn_ivy_test = set(fn_test_core).union(set(fn_test_nn))
    common_list = fn_ivy_test.intersection(set(fn_test_experimental))

    # returns True if duplicate found, False if no duplicates found
    return len(common_list) > 0


def find_duplicate_functions(root_dir):
    """Searches for any duplicate frontend functions within ivy's frontend
    api."""
    fns = []
    duplicates = []
    current_class_fns = []
    exclude = [
        # functions to exclude
        "dtype",
        "device",
        "is_leaf",
        "ivy_array",
        "numel",  # torch.Size
        "requires_grad",
        "symmetrize",
        # files to exclude
        "base.py",
        "func_wrapper.py",
        "loss_functions.py",
    ]

    # NOTE: Size and Tensor are currently defined in the same file, which
    # causes duplication overlapping as the class defs override each
    # other when doing the breadth-first ast walk. Not causing any
    # problems right now (excluded some methods), but worth noting.

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_name = file_path.split("/")[-1]
                if file_name in exclude or file_name.startswith("_"):
                    continue
                with open(file_path, "r") as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Module):
                            current_class = ""
                            current_class_fns = []
                        if not hasattr(node, "col_offset") or node.col_offset == 0:
                            current_class = ""
                            current_class_fns = []
                        if isinstance(node, ast.ClassDef):
                            current_class = node.name
                            current_class_fns = []
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            if (
                                func_name in exclude  # ignore any fns in `exclude`
                                or func_name.startswith(
                                    "_"
                                )  # ignore any private or dunder methods
                            ):
                                continue
                            func_path = file_path
                            full_name = func_path + "::" + func_name
                            if (
                                node.decorator_list
                                and hasattr(node.decorator_list[0], "id")
                                and node.decorator_list[0].id == "property"
                            ):
                                # ignore properties
                                continue
                            if len(current_class) == 0:
                                if full_name not in fns:
                                    fns.append(full_name)
                                else:
                                    duplicates.append(full_name)
                            else:
                                if full_name not in current_class_fns:
                                    current_class_fns.append(full_name)
                                else:
                                    duplicates.append(full_name)
    return duplicates


if __name__ == "__main__":
    duplicated_frontends = find_duplicate_functions("ivy/functional/frontends/")
    if duplicated_frontends:
        print("Duplicate functions found:")
        print(duplicated_frontends)
        sys.exit(1)

    if check_duplicate_functional_experimental_tests():
        sys.exit(1)

    print("No duplicate functions found")
