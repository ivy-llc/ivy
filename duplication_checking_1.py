import os
import sys
from pydriller import Repository
import pickle  # noqa
from tqdm import tqdm
from random import shuffle
import bz2
import _pickle as cPickle

# Shared Map
tests = {}
BACKENDS = ["numpy", "jax", "tensorflow", "torch"]

os.system("git config --global --add safe.directory /ivy")
N = 32
run_iter = int(sys.argv[1])

os.system(
    "docker run -v `pwd`:/ivy -v `pwd`/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --disable-pytest-warnings ivy_tests/test_ivy --my_test_dump true > test_names"  # noqa
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
