import sys
from pymongo import MongoClient
from get_all_tests import get_all_tests


module_map = {
    "core": "test_functional/test_core",
    "exp_core": "test_functional/test_experimental/test_core",
    "nn": "test_functional/test_experimental/test_nn",
    "exp_nn": "test_experimental/test_nn",
    "stateful": "test_stateful",
    "torch": "test_frontends/test_torch",
    "jax": "test_frontends/test_jax",
    "tensorflow": "test_frontends/test_tensorflow",
    "numpy": "test_frontends/test_numpy",
    "misc": "test_misc",
    "paddle": "test_frontends/test_paddle",
    "scipy": "test_frontends/test_scipy",
    "torchvision": "test_frontends/test_torchvision",
}


def keys_to_delete_from_db(all_tests, module, data, current_key=""):
    """Recursively navigate and identify keys not in the list."""
    keys_for_deletion = []

    for key, value in data.items():
        new_key = f"{current_key}.{key}" if current_key else key

        # If this is a dictionary, recurse deeper
        if isinstance(value, dict):
            keys_for_deletion.extend(
                keys_to_delete_from_db(all_tests, module, value, new_key)
            )
        elif key != "_id":
            components = new_key.split(".")
            submodule = components[0]
            function = components[-2]
            test = f"{module}/{submodule}::{function}"
            if test not in all_tests:
                keys_for_deletion.append(".".join(components[:-1]))

    return keys_for_deletion


submodules = (
    "test_paddle",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy",
    "test_functional",
    "test_experimental",
    "test_stateful",
    "test_misc",
    "test_scipy",
    "test_pandas",
    "test_mindspore",
    "test_onnx",
    "test_sklearn",
    "test_xgboost",
    "test_torchvision",
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
    "test_pandas": ["pandas", 22],
    "test_mindspore": ["mindspore", 23],
    "test_onnx": ["onnx", 24],
    "test_sklearn": ["sklearn", 25],
    "test_xgboost": ["xgboost", 26],
    "test_torchvision": ["torchvision", 27],
}


def get_submodule(test_path):
    test_path = test_path.split("/")
    for name in submodules:
        if name in test_path:
            if name == "test_functional":
                if test_path[3] == "test_experimental":
                    coll = db_dict[f"test_experimental/{test_path[4]}"]
                else:
                    coll = db_dict[f"test_functional/{test_path[-2]}"]
            else:
                coll = db_dict[name]
            break
    submod_test = test_path[-1]
    submod, test_fn = submod_test.split("::")
    submod = submod.replace("test_", "").replace(".py", "")
    return coll, submod, test_fn


def process_test(test):
    coll, submod, test_fn = get_submodule(test)
    return f"{coll[0]}/{submod}::{test_fn}"


def remove_empty_objects(document, key_prefix=""):
    # Base case: if the document is not a dictionary, return an empty list
    if not isinstance(document, dict):
        return []

    # List to store keys associated with empty objects
    empty_keys = []

    for key, value in document.items():
        # Generate the full key path
        full_key = f"{key_prefix}.{key}" if key_prefix else key

        # If the value is a dictionary, recursively check for empty objects
        if isinstance(value, dict):
            # If the dictionary is empty, store its key
            if not value:
                empty_keys.append(full_key)
            else:
                empty_keys.extend(remove_empty_objects(value, full_key))

    return empty_keys


def main():
    all_tests = get_all_tests()
    all_tests = {process_test(test.split(",")[0].strip()) for test in all_tests}
    mongo_key = sys.argv[1]
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Ivy_tests_multi_gpu"]
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        for document in collection.find({}):
            undesired_keys = keys_to_delete_from_db(
                all_tests, collection_name, document
            )
            for key in undesired_keys:
                collection.update_one({"_id": document["_id"]}, {"$unset": {key: 1}})

    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        break_flag = False
        while True:
            for document in collection.find({}):
                keys_to_remove = remove_empty_objects(document)
                if keys_to_remove:
                    update_operation = {"$unset": {key: 1 for key in keys_to_remove}}
                    collection.update_one({"_id": document["_id"]}, update_operation)
                else:
                    break_flag = True
                    break
            if break_flag:
                break_flag = False
                break


if __name__ == "__main__":
    main()
