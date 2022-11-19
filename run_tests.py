# Run Tests
import os
import sys
from pymongo import MongoClient
from automation_tools.dashboard_automation.update_tests import update_test_result

submodules = (
    "test_core",
    "test_nn",
    "test_stateful",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy/test_creation_routines",
    "test_numpy/test_fft",
    "test_numpy/test_indexing_routines",
    "test_numpy/test_linear_algebra",
    "test_numpy/test_logic",
    "test_numpy/test_ma",
    "test_numpy/test_manipulation_routines",
    "test_numpy/test_mathematical_functions",
    "test_numpy/test_matrix",
    "test_numpy/test_ndarray",
    "test_numpy/test_random",
    "test_numpy/test_sorting_searching_counting",
    "test_numpy/test_statistics",
    "test_numpy/test_ufunc",
)
db_dict = {
    "test_core": ["intelligent_test_core", 10],
    "test_nn": ["intelligent_test_nn", 11],
    "test_stateful": ["intelligent_test_stateful", 12],
    "test_torch": ["intelligent_test_torch", 13],
    "test_jax": ["intelligent_test_jax", 14],
    "test_tensorflow": ["intelligent_test_tensorflow", 15],
}


def get_submodule(test_path):
    test_path = test_path.split("/")
    for name in submodules:
        if name in test_path:
            coll = db_dict[name]
            break
    submod_test = test_path[-1]
    submod, test = submod_test.split("::")
    submod = submod.split("_")[1].strip(".py")
    return coll, submod, test


if __name__ == "__main__":
    if len(sys.argv) > 2:
        redis_url = sys.argv[1]
        redis_pass = sys.argv[2]
        mongo_key = sys.argv[3]
    failed = False
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Ivy_tests"]
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            print(test, backend)
            coll, submod, test = get_submodule(test)
            if len(sys.argv) > 2:
                ret = os.system(
                    f'docker run --rm --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest {test} --backend {backend}'  # noqa
                )
            else:
                ret = os.system(
                    f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest {test} --backend {backend}'  # noqa
                )
            if ret != 0:
                update_test_result(
                    db[coll[0]], coll[1], submod, backend, test, "failure"
                )
                failed = True

    if failed:
        exit(1)
