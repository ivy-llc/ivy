# Run Tests
import os
import sys
from pymongo import MongoClient

submodules = (
    "test_paddle",
    "test_functional",
    "test_experimental",
    "test_stateful",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy",
    "test_misc",
    "test_scipy",
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
    "test_paddle": ["paddle", 20],
    "test_scipy": ["scipy", 21],
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


def update_individual_test_results(
    collection,
    id,
    submod,
    backend,
    test,
    result,
    backend_version=None,
    frontend_version=None,
    device=None,
):
    key = submod + "." + backend
    if backend_version is not None:
        backend_version = backend_version.replace(".", "_")
        key += "." + backend_version
    if frontend_version is not None:
        frontend_version = frontend_version.replace(".", "_")
        key += "." + frontend_version
    key += "." + test
    if device:
        key += "." + device
    collection.update_one(
        {"_id": id},
        {"$set": {key: result}},
        upsert=True,
    )
    return


def remove_from_db(collection, id, submod, backend, test):
    collection.update_one(
        {"_id": id},
        {"$unset": {submod + "." + backend + ".": test}},
    )
    return


def run_multiversion_testing():
    failed = False
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Ivy_tests_multi"]
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            backend = backend.strip("\n")
            coll, submod, test_fn = get_submodule(test)
            frontend_version = None
            if ";" in backend:
                # This is a frontend test
                backend, frontend = backend.split(";")
                frontend_version = "/".join(frontend.split("/")[1:])
                command = f'docker run --rm --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/multiversion:base /bin/bash -c "/opt/miniconda/envs/multienv/bin/python docker/multiversion_framework_directory.py {backend} {frontend} numpy/1.23.1; /opt/miniconda/envs/multienv/bin/python -m pytest --tb=short {test} --backend={backend} --frontend={frontend}" '  # noqa
                ret = os.system(command)
            else:
                ret = os.system(
                    f"docker run --rm --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                    ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/multiversion:base'
                    " /opt/miniconda/envs/multienv/bin/python"
                    " docker/multiversion_framework_directory.py backend"
                    f" {backend};/opt/miniconda/envs/multienv/bin/python pytest"
                    f" --tb=short {test} --backend={backend.split('/')[0]}"
                    # noqa
                )
            if ret != 0:
                res = make_clickable(run_id, result_config["failure"])
                failed = True
            else:
                res = make_clickable(run_id, result_config["success"])
            backend_list = backend.split("/")
            backend_name = backend_list[0] + "\n"
            backend_version = "/".join(backend_list[1:])
            update_individual_test_results(
                db[coll[0]],
                coll[1],
                submod,
                backend_name,
                test_fn,
                res,
                backend_version,
                frontend_version,
            )
    if failed:
        exit(1)
    exit(0)


if __name__ == "__main__":
    redis_url = sys.argv[1]
    redis_pass = sys.argv[2]
    mongo_key = sys.argv[3]
    version_flag = sys.argv[4]
    gpu_flag = sys.argv[5]
    workflow_id = sys.argv[6]
    priority_flag = sys.argv[7]
    if len(sys.argv) > 8 and sys.argv[8] != "null":
        run_id = sys.argv[8]
    else:
        run_id = "https://github.com/unifyai/ivy/actions/runs/" + workflow_id
    failed = False
    # GPU Testing
    with_gpu = False
    if gpu_flag == "true":
        with_gpu = True
    if priority_flag == "true":
        priority_flag = True
    # Multi Version Testing
    if version_flag == "true":
        run_multiversion_testing()
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db_multi = cluster["Ivy_tests_multi"]
    db_gpu = cluster["Ivy_tests_multi_gpu"]
    db_priority = cluster["Ivy_tests_priority"]
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            coll, submod, test_fn = get_submodule(test)
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")
            sys.stdout.flush()
            if with_gpu:
                ret = os.system(
                    f"docker run --rm --gpus all --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                    ' "$(pwd)"/.hypothesis:/.hypothesis'
                    " unifyai/multicuda:base_and_requirements python3 -m pytest"
                    f" --tb=short {test} --device=gpu:0 -B={backend}"
                    # noqa
                )
            else:
                ret = os.system(
                    f"docker run --rm --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                    ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m'
                    f" pytest --tb=short {test} --backend {backend}"
                    # noqa
                )
            if ret != 0:
                res = make_clickable(run_id, result_config["failure"])
                failed = True
            else:
                res = make_clickable(run_id, result_config["success"])
            frontend_version = None
            if (
                coll[0] == "numpy"
                or coll[0] == "jax"
                or coll[0] == "tensorflow"
                or coll[0] == "torch"
                or coll[0] == "paddle"
            ):
                frontend_version = "latest-stable"
            if priority_flag:
                print("Updating Priority DB")
                update_individual_test_results(
                    db_priority[coll[0]],
                    coll[1],
                    submod,
                    backend,
                    test_fn,
                    res,
                    "latest-stable",
                    frontend_version,
                    "gpu" if with_gpu else "cpu",
                )
            else:
                if not with_gpu:
                    update_individual_test_results(
                        db_multi[coll[0]],
                        coll[1],
                        submod,
                        backend,
                        test_fn,
                        res,
                        "latest-stable",
                        frontend_version,
                    )
                update_individual_test_results(
                    db_gpu[coll[0]],
                    coll[1],
                    submod,
                    backend,
                    test_fn,
                    res,
                    "latest-stable",
                    frontend_version,
                    "gpu" if with_gpu else "cpu",
                )

    if failed:
        exit(1)
