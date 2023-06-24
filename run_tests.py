import os
import sys
from pymongo import MongoClient

# Configuration
CONFIG = {
    "mongo_url": "mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority",
    "result_config": {
        "success": "https://img.shields.io/badge/-success-success",
        "failure": "https://img.shields.io/badge/-failure-red",
    },
    "docker_command": {
        "gpu": 'docker run --rm --gpus all --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/multicuda:base_and_requirements python3 -m pytest --tb=short {test} --device=gpu:0 -B={backend}',
        "non_gpu": 'docker run --rm --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --tb=short {test} --backend {backend}',
        "multi_version": 'docker run --rm --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/multiversion:base /bin/bash -c "/opt/miniconda/envs/multienv/bin/python docker/multiversion_framework_directory.py {backend} {frontend} numpy/1.23.1; /opt/miniconda/envs/multienv/bin/python -m pytest --tb=short {test} --backend={backend} --frontend={frontend}"',
    },
    "submodules": (
        "test_functional",
        "test_experimental",
        "test_stateful",
        "test_tensorflow",
        "test_torch",
        "test_jax",
        "test_numpy",
        "test_misc",
        "test_paddle",
        "test_scipy",
    ),
}


def make_clickable(url, name):
    return '<a href="{}" rel="noopener noreferrer" target="_blank"><img src={}></a>'.format(url, name)


def get_submodule(test_path):
    test_path = test_path.split("/")
    for name in CONFIG["submodules"]:
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


def update_individual_test_results(collection, id, submod, backend, test, result, backend_version=None, frontend_version=None):
    key = submod + "." + backend
    if backend_version is not None:
        backend_version = backend_version.replace(".", "_")
        key += "." + backend_version
    if frontend_version is not None:
        frontend_version = frontend_version.replace(".", "_")
        key += "." + frontend_version
    key += "." + test
    collection.update_one({"_id": id}, {"$set": {
