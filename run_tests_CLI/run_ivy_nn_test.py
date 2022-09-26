import sys
import os

run = int(sys.argv[1])
backends = ["numpy", "torch", "jax", "tensorflow"]
submodules = [
    "activations",
    "layers",
    "losses",
    "norms",
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
os.system(f"./run_tests_CLI/test_ivy_nn.sh {backend} test_{submodule}")
