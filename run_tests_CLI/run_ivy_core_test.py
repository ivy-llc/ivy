import sys
import os

run = int(sys.argv[1])
backends = ["numpy", "torch", "jax", "tensorflow"]
submodules = [
    "creation",
    "device",
    "dtype",
    "elementwise",
    "general",
    "gradients",
    "linalg",
    "manipulation",
    "meta",
    "nest",
    "random",
    "searching",
    "set",
    "sorting",
    "statistical",
    "utility",
]

N = len(backends)
M = len(submodules)

num_tests = N * M
run = run % num_tests

i = run // M
j = run % M

backend = backends[0]
submodule = submodules[0]

# print(backend, submodule)
os.environ["BACKEND"] = str(backend)
os.environ["SUBMODULE"] = str(submodule)
print(os.environ["BACKEND"], os.environ["SUBMODULE"])
# os.system(f"./run_tests_CLI/test_ivy_core.sh {backend} test_{submodule}")
