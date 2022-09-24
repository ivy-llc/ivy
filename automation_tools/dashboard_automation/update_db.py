import sys
from pymongo import MongoClient


test_configs = {
    "test-array-api": ["array_api", 0],
    "test-ivy-core": ["ivy_core", 1],
    "test-ivy_nn": ["ivy_nn", 2],
    "test_ivy_stateful": ["ivy_stateful", 3],
}


def update_test_results():
    key, workflow, backend, submodule, result = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
    )
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"
    )
    db = cluster["Ivy_tests"]
    collection = db[test_configs[workflow]]
    collection.update_one(
        {"_id": test_configs[workflow][1]},
        {"$set": {backend + "." + submodule: result}},
    )
    return


if __name__ == "__main__":
    update_test_results()
