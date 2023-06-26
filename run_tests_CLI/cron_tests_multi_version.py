import os
import sys

# BACKENDS = ["numpy", "jax", "tensorflow", "torch"]
torch_req = [
    # "torch/1.4.0",
    # "torch/1.5.0",
    # "torch/1.10.1",
    # "torch/1.10.2",
    # "torch/2.0.1",
    # "torch/1.12.0",
    "torch/1.12.1",
    "torch/1.13.0",
]
tensorflow_req = [
    # "tensorflow/2.2.0",
    # "tensorflow/2.2.1",
    # "tensorflow/2.2.2",
    # "tensorflow/2.4.4",
    # "tensorflow/2.9.0",
    # "tensorflow/2.12.0",
    "tensorflow/2.12.0",
    "tensorflow/2.9.2",
]
jax_only_req = [
    # "jax/0.1.60",
    # "jax/0.1.61",
    # "jax/0.3.10",
    # "jax/0.3.13",
    # "jax/0.4.10",
    # "jax/0.4.10",
    # "jax/0.3.15",
    "jax/0.3.16",
    "jax/0.3.17",
]
jaxlib_req = [
    # "jaxlib/0.1.50",
    # "jaxlib/0.1.60",
    # "jaxlib/0.1.61",
    # "jaxlib/0.3.10",
    # "jaxlib/0.4.10",
    # "jaxlib/0.3.15",
    "jaxlib/0.3.20",
    "jaxlib/0.3.22",
]
numpy_req = [
    # "numpy/1.17.3",
    # "numpy/1.17.4",
    # "numpy/1.23.1",
    # "numpy/1.24.0",
    "numpy/1.24.1",
    "numpy/1.24.2",
]
jax_req = [
    jax_ver + "/" + jaxlib_ver for jax_ver in jax_only_req for jaxlib_ver in jaxlib_req
]

framework_versions = {
    "numpy": numpy_req,
    "torch": torch_req,
    "jax": jax_req,
    "tensorflow": tensorflow_req,
}

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
    for backend, backend_versions in framework_versions.items():
        for backend_version in backend_versions:
            test_backend = test_name + "," + backend_version
            if "test_frontends" in test_name:
                frontend = test_name[39:]
                frontend = frontend[: frontend.find("/")]
                frontend_versions = framework_versions.get(frontend, [])
                for frontend_version in frontend_versions:
                    test_names.append(test_backend + ";" + frontend_version)
            else:
                test_names.append(test_backend)

test_names = list(set(test_names))
test_names.sort()

# Run 150 tests in each iteration of the cron job
num_tests = len(test_names)
print(num_tests)
tests_per_run = 150
start = run_iter * tests_per_run
end = (run_iter + 1) * tests_per_run
print("Running Tests:")
with open("tests_to_run", "w") as f:
    for i in range(start, end):
        i = i % num_tests
        test = test_names[i]
        print(test)
        f.write(test + "\n")
