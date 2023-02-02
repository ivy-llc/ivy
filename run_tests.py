# Run Tests
import os
import sys
from pymongo import MongoClient


submodules = (
    "test_functional",
    "test_experimental",
    "test_stateful",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy",
    "test_misc",
)
db_dict = {
    "test_functional/test_core": ["core", 10],
    "test_experimental/test_core": ["exp_core", 11],
    "test_functional/test_nn": ["nn", 12],
    "test_experimental/test_nn": ["exp_nn", 13],
    "test_stateful": ["stateful", 14],
    "test_torch": ["torch", 15],
    "test_jax": ["jax", 16],
    "test_tensorflow": ["tensorflow", 17],
    "test_numpy": ["numpy", 18],
    "test_misc": ["misc", 19],
}
result_config = {
    "success": "https://img.shields.io/badge/-success-success",
    "failure": "https://img.shields.io/badge/-failure-red",
}


def make_clickable(url, name):
    return '<a href="{}" rel="noopener noreferrer" '.format(
        url
    ) + 'target="_blank"><img src={}></a>'.format(name)


def get_submodule(test_path):
    test_path = test_path.split("/")
    for name in submodules:
        if name in test_path:
            if name == "test_functional":
                coll = db_dict["test_functional/" + test_path[-2]]
            elif name == "test_experimental":
                coll = db_dict["test_experimental/" + test_path[-2]]
            else:
                coll = db_dict[name]
    submod_test = test_path[-1]
    submod, test_fn = submod_test.split("::")
    submod = submod.replace("test_", "").replace(".py", "")
    return coll, submod, test_fn


def update_individual_test_results(collection, id, submod, backend, test, result):
    collection.update_one(
        {"_id": id},
        {"$set": {submod + "." + backend + "." + test: result}},
        upsert=True,
    )
    return


def remove_from_db(collection, id, submod, backend, test):
    collection.update_one(
        {"_id": id},
        {"$unset": {submod + "." + backend + ".": test}},
    )
    return


def run_multiversion_testing(failed):
    with open("tests_to_run", "r") as f:
        for line in f:
            test, frontend, backend = line.split(",")
        ret = os.system(
            f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/multiversion /opt/miniconda/envs/multienv/bin/python -m pytest --tb=short {test} --frontend={frontend} --backend={backend}'  # noqa
        )
        if ret != 0:
            failed = True

        if failed:
            exit(1)


if __name__ == "__main__":
    redis_url = sys.argv[1]
    redis_pass = sys.argv[2]
    mongo_key = sys.argv[3]
    version_flag = sys.argv[4]
    if len(sys.argv) > 4:
        run_id = sys.argv[5]
    else:
        run_id = "https://github.com/unifyai/ivy/actions/"
    failed = False
    # multiversion testing
    if version_flag == "true":
        run_multiversion_testing(failed)
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Ivy_tests"]
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            coll, submod, test_fn = get_submodule(test)
            print(coll, submod, test_fn)
            if len(sys.argv) > 2:
                ret = os.system(
                    f'docker run --rm --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --tb=short {test} --backend {backend}'  # noqa
                )
            else:
                ret = os.system(
                    f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --tb=short {test} --backend {backend}'  # noqa
                )
            if ret != 0:
                res = make_clickable(run_id, result_config["failure"])
                update_individual_test_results(
                    db[coll[0]], coll[1], submod, backend, test_fn, res
                )
                failed = True
            else:
                res = make_clickable(run_id, result_config["success"])
                update_individual_test_results(
                    db[coll[0]], coll[1], submod, backend, test_fn, res
                )

    try:
        with open("tests_to_remove", "r") as f:
            for line in f:
                test, backend = line.split(",")
                coll, submod, test_fn = get_submodule(test)
                print(coll, submod, test_fn)
                remove_from_db(db[coll[0]], coll[1], submod, backend, test_fn)
    except Exception:
        pass

    if failed:
        exit(1)
