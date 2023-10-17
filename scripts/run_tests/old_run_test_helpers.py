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
}

result_config = {
    "success": "https://img.shields.io/badge/-success-success",
    "failure": "https://img.shields.io/badge/-failure-red",
}


def make_clickable(url, name):
    return (
        f'<a href="{url}" rel="noopener noreferrer" '
        + f'target="_blank"><img src={name}></a>'
    )


def get_submodule(test_path):
    test_path = test_path.split("/")
    for name in submodules:
        if name in test_path:
            if name == "test_functional":
                if len(test_path) > 3 and test_path[3] == "test_experimental":
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
    key = f"{submod}.{backend}"
    if backend_version is not None:
        backend_version = backend_version.replace(".", "_")
        key += f".{backend_version}"
    if frontend_version is not None:
        frontend_version = frontend_version.replace(".", "_")
        key += f".{frontend_version}"
    key += f".{test}"
    if device:
        key += f".{device}"
    collection.update_one(
        {"_id": id},
        {"$set": {key: result}},
        upsert=True,
    )
