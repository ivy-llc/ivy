# Run Tests
import os
import sys
import requests

url = "https://ivy-dynamical-dashboards.onrender.com/api/test"

modules = (
    "test_functional",
    "test_experimental",
    "test_stateful",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy",
    "test_misc",
)

module_map = {
    "test_functional/test_core": "core",
    "test_experimental/test_core": "exp_core",
    "test_functional/test_nn": "nn",
    "test_experimental/test_nn": "exp_nn",
    "test_stateful": "stateful",
    "test_torch": "torch",
    "test_jax": "jax",
    "test_tensorflow": "tensorflow",
    "test_numpy": "numpy",
    "test_misc": "misc",
}


def get_mod_submod_test(test_path):
    test_path = test_path.split("/")
    module = ""
    for name in modules:
        if name in test_path:
            if name == "test_functional":
                module = module_map["test_functional/" + test_path[-2]]
            elif name == "test_experimental":
                module = module_map["test_experimental/" + test_path[-2]]
            else:
                module = module_map[name]
            break
    submod_test = test_path[-1]
    submod, test_fn = submod_test.split("::")
    submod = submod.replace("test_", "").replace(".py", "")
    return module, submod, test_fn


if __name__ == "__main__":
    failed = False
    f_write = open(sys.argv[1], "w")
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")
            sys.stdout.flush()
            ret = os.system(
                f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --tb=short {test} --backend {backend}'  # noqa
            )
            if ret != 0:
                failed = True
                module, submodule, test = get_mod_submod_test(test)
                params = {
                    "module": module,
                    "submodule": submodule,
                    "backend": backend[:-1],
                    "test": test,
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    if response.json():
                        # The test passes on master but fails in this fork/branch
                        f_write.write(line)

    f_write.close()
    if failed:
        exit(1)
