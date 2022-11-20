# Run Tests
import os
import sys
from pymongo import MongoClient

submodules = (
    "test_core",
    "test_nn",
    "test_stateful",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy",
)
db_dict = {
    "test_functional/test_core": ["intelligent_core", 10],
    "test_experimental/test_core": ["intelligent_exp_core", 11],
    "test_functional/test_nn": ["intelligent_nn", 12],
    "test_experimental/test_nn": ["intelligent_exp_nn", 13],
    "test_stateful": ["intelligent_stateful", 14],
    "test_torch": ["intelligent_torch", 15],
    "test_jax": ["intelligent_jax", 16],
    "test_tensorflow": ["intelligent_tensorflow", 17],
    "test_numpy": ["intelligent_numpy", 18],
}
result_config = {
    "success": "https://img.shields.io/badge/-success-success",
    "failure": "https://img.shields.io/badge/-failure-red",
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


def update_individual_test_results(collection, id, submod, backend, test, result):
    collection.update_one(
        {"_id": id},
        {"$set": {submod + "." + backend + "." + test: result["result"]}},
        upsert=True,
    )
    return


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
                update_individual_test_results(
                    db[coll[0]], coll[1], submod, backend, test, "failure"
                )
                failed = True
            else:
                update_individual_test_results(
                    db[coll[0]], coll[1], submod, backend, test, "success"
                )

    if failed:
        exit(1)
