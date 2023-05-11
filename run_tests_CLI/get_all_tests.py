import os
import random


def get_all_tests():
    BACKENDS = ["numpy", "jax", "tensorflow", "torch", "paddle"]
    os.system(
        "docker run -v `pwd`:/ivy -v `pwd`/.hypothesis:/.hypothesis unifyai/ivy:latest"
        " python3 -m pytest --disable-pytest-warnings ivy_tests/test_ivy --my_test_dump"
        " true > test_names"
    )
    test_names_without_backend = []
    test_names = []
    with open("test_names") as f:
        for line in f:
            if "ERROR" in line:
                break
            if not line.startswith("ivy_tests"):
                continue
            test_name = line[:-1]
            pos = test_name.find("[")
            if pos != -1:
                test_name = test_name[:pos]
            test_names_without_backend.append(test_name)

    test_names_without_backend = list(set(test_names_without_backend))
    test_names_without_backend.sort()
    random.Random(4).shuffle(test_names_without_backend)

    for test_name in test_names_without_backend:
        for backend in BACKENDS:
            test_backend = test_name + "," + backend
            test_names.append(test_backend)

    return test_names
