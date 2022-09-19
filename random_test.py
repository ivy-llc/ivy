import random
import os

fw = ["numpy", "tensorflow", "jax", "torch"]
submodule = [
    "creation",
    "device",
    "dtype, elementwise",
    "manipulation" "meta",
    "nest",
    "random",
    "searching",
    "set",
    "sorting",
    "statistical",
    "utility",
]

submod_temp = submodule


def main():
    global submod_temp
    if not submod_temp:
        submod_temp = submodule
    backend = random.choice(fw)
    submod = random.choice(submod_temp)
    submod_temp.remove(submod)
    return backend, submod


if __name__ == "__main__":
    backend, submodule = main()
    os.environ["BACKEND"] = backend
    os.environ["SUBMODULE"] = submodule
