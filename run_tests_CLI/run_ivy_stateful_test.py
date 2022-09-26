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


def get_my_print(file=sys.stdout):
    _original_print = print

    def my_print(*objects, sep="", end="\n", file=file, flush=False):
        _original_print(*objects, sep="", end="\n", file=file, flush=False)

    return my_print


def cron_job():
    run = int(sys.argv[1])
    N = len(backends)
    M = len(submodules)

    num_tests = N * M
    run = run % num_tests

    i = run // M
    j = run % M

    backend = backends[i]
    submodule = submodules[j]
    os.system(f"./run_tests_CLI/test_ivy_stateful.sh {backend} test_{submodule}")
    print = get_my_print(sys.stderr)
    print(backend, submodule)


if __name__ == "__main__":
    cron_job()
