import os
import sys

BACKENDS = ["numpy", "jax", "tensorflow", "torch"]
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

for test_name in test_names_without_backend:
    for backend in BACKENDS:
        test_backend = test_name + "," + backend
        test_names.append(test_backend)

test_names = list(set(test_names))

# We run 10 tests in each iteration of the cron job
num_tests = len(test_names)
tests_per_run = 10
start = run_iter * tests_per_run
end = (run_iter + 1) * tests_per_run
with open("tests_to_run", "w") as f:
    for i in range(start, end):
        i = i % num_tests
        test = test_names[i]
        f.write(test + "\n")
