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

backend = backends[i]
submodule = submodules[j]

print(backend, submodule)
os.system(f"./run_tests_CLI/test_ivy_core.sh {backend} test_{submodule}")
