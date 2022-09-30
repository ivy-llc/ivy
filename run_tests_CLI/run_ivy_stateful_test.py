import sys
import os


backends = ["numpy", "torch", "jax", "tensorflow"]
submodules = [
    "activations",
    "converters",
    "layers",
    "modules",
    "norms",
    "optimizers",
    "sequential",
]


run = int(sys.argv[1])
N = len(backends)
M = len(submodules)

num_tests = N * M
run = run % num_tests

i = run // M
j = run % M

backend = backends[i]
submodule = submodules[j]


with open("./fwsubmod.txt", "w") as outfile:
    outfile.write(f"{backend}-{submodule}")


os.system(f"./run_tests_CLI/test_ivy_stateful.sh {backend} test_{submodule}")
