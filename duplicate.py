# from . import hypothesis_helpers
# from .hypothesis_helpers import number_helpers as nh
# import inspect
# import ivy_tests.test_ivy.test_functional.test_core.test_linalg as hello
# print(inspect.getmembers(hello, inspect.isfunction))
# print(dir(hello))
import importlib
import os
import glob


def get_all_functions_from_directory(root_dir, startswith='test'):
    if not os.path.exists(root_dir):
        print('Invalid directory')
        exit(1)
    functions_names = []
    for filename in glob.iglob(root_dir + '/**/*.py', recursive=True):
        if len(filename) >= 2 and filename[:2] == './':
            filename = filename[2:]
        filename = filename.replace('.py', '')
        filename = filename.replace('/', '.')
        module = importlib.import_module(filename)
        module_functions_names = [obj for obj in dir(module) if obj.startswith(startswith)]
        functions_names.extend(module_functions_names)
    return functions_names

def check_duplicate():
    fn_test_core = get_all_functions_from_directory('ivy_tests/test_ivy/test_functional/test_core')
    fn_test_nn = get_all_functions_from_directory('ivy_tests/test_ivy/test_functional/test_nn')
    fn_test_experimental = get_all_functions_from_directory('ivy_tests/test_ivy/test_functional/test_experimental')
    fn_ivy_test = set(fn_test_core).union(set(fn_test_nn))
    common_list = fn_ivy_test.intersection(set(fn_test_experimental))
    return common_list

if __name__ == "__main__":
    common_set = check_duplicate()
    if len(common_set) != 0:
        exit(1)

# print()
#
# # print(os)
# def get_all_tests():
#     os.system(
#         "docker run -v `pwd`:/ivy -v `pwd`/.hypothesis:/.hypothesis unifyai/ivy:latest"
#         " python3 -m pytest --disable-pytest-warnings ivy_tests/test_ivy "
#         "--my_test_dump true > test_names "
#         # noqa
#     )
#     test_names_without_backend = []
#     test_names = []
#     with open("test_names") as f:
#         for line in f:
#             if "ERROR" in line:
#                 break
#             if not line.startswith("ivy_tests"):
#                 continue
#             test_name = line[:-1]
#             pos = test_name.find("[")
#             if pos != -1:
#                 test_name = test_name[:pos]
#             test_names_without_backend.append(test_name)
#
#     for test_name in test_names_without_backend:
#         for backend in BACKENDS:
#             test_backend = test_name + "," + backend
#             test_names.append(test_backend)
#
#     test_names = list(set(test_names))
#     return test_names
#
# print(get_all_tests())

# def num_positional_args(draw, *, fn_name: str = None):
#     """Draws an integers randomly from the minimum and maximum number of positional
#     arguments a given function can take.
#
#     Parameters
#     ----------
#     draw
#         special function that draws data randomly (but is reproducible) from a given
#         data-set (ex. list).
#     fn_name
#         name of the function.
#
#     Returns
#     -------
#     A strategy that can be used in the @given hypothesis decorator.
#
#     Examples
#     --------
#     @given(
#         num_positional_args=num_positional_args(fn_name="floor_divide")
#     )
#     @given(
#         num_positional_args=num_positional_args(fn_name="add")
#     )
#     """
#     num_positional_only = 0
#     num_keyword_only = 0
#     total = 0
#     fn = None
#     for i, fn_name_key in enumerate(fn_name.split(".")):
#         if i == 0:
#             fn = ivy.__dict__[fn_name_key]
#         else:
#             fn = fn.__dict__[fn_name_key]
#     for param in inspect.signature(fn).parameters.values():
#         if param.name == "self":
#             continue
#         total += 1
#         if param.kind == param.POSITIONAL_ONLY:
#             num_positional_only += 1
#         elif param.kind == param.KEYWORD_ONLY:
#             num_keyword_only += 1
#         elif param.kind == param.VAR_KEYWORD:
#             num_keyword_only += 1
#     return draw(
#         nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
#     )
#
#     num_positional_args([1 ,2, 3], str = "eigh")