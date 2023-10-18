import importlib
import os
import sys
import glob


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


def check_duplicate():
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
    return common_list


if __name__ == "__main__":
    common_set = check_duplicate()
    if len(common_set) != 0:
        print("This function already exists in the functional API.")
        sys.exit(1)
