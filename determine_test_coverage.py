import os
from pydriller import Repository
import pickle
from tqdm import tqdm

# Shared Map
tests = {}

# os.system("pytest --disable-pytest-warnings ivy_tests/test_ivy/ --my_test_dump true > test_names") # noqa
test_names = []
# with open("test_names") as f:
#     i = 0
#     for line in f:
#         i += 1
#         if i <= 5:
#             continue
#         test_names.append(line[:-1])
#
# test_names = test_names[:-3]

test_names = [
    "ivy_tests/test_ivy/test_functional/test_core/test_elementwise.py::test_abs",
]
directories = [
    "ivy",
    "ivy/array",
    "ivy/container",
    "ivy/functional",
    "ivy/functional/backends",
    "ivy/functional/backends/jax",
    "ivy/functional/backends/numpy",
    "ivy/functional/backends/torch",
    "ivy/functional/backends/tensorflow",
    "ivy/functional/frontends",
    "ivy/functional/frontends/jax",
    "ivy/functional/frontends/numpy",
    "ivy/functional/frontends/torch",
    "ivy/functional/frontends/tensorflow",
    "ivy/functional/ivy",
    "ivy/stateful",
    "ivy_tests",
    "ivy_tests/test_ivy",
    "ivy_tests/test_ivy/test_frontends",
    "ivy_tests/test_ivy/test_frontends/test_jax",
    "ivy_tests/test_ivy/test_frontends/test_numpy",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_creation_routines",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_fft",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_indexing_routines",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_linear_algebra",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_logic",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_ma",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_manipulation_routines",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_matrix",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_ndarray",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_random",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_sorting_searching_counting",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_statistics",
    "ivy_tests/test_ivy/test_frontends/test_numpy/test_ufunc",
    "ivy_tests/test_ivy/test_frontends/test_tensorflow",
    "ivy_tests/test_ivy/test_frontends/test_torch",
    "ivy_tests/test_ivy/test_functional",
    "ivy_tests/test_ivy/test_functional/test_core",
    "ivy_tests/test_ivy/test_functional/test_nn",
    "ivy_tests/test_ivy/test_stateful",
]

if __name__ == "__main__":
    for test_name in tqdm(test_names):
        os.system(
            f"coverage run -m pytest {test_name} --disable-warnings > coverage_output"
        )
        os.system("coverage annotate > coverage_output")
        for directory in directories:
            for file_name in os.listdir(directory):
                if file_name.endswith("cover"):
                    file_name = directory + "/" + file_name
                    if file_name not in tests:
                        tests[file_name] = []
                        with open(file_name) as f:
                            for line in f:
                                tests[file_name].append(set())
                    with open(file_name) as f:
                        i = 0
                        for line in f:
                            if line[0] == ">":
                                tests[file_name][i].add(test_name)
                            i += 1
        os.system("find . -name \\*cover -type f -delete")


commit_hash = ""
for commit in Repository(".", order="reverse").traverse_commits():
    commit_hash = commit.hash
    break
tests["commit"] = commit_hash
with open("tests.pkl", "wb") as f:
    pickle.dump(tests, f)
