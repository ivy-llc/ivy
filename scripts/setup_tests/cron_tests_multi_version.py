import sys
from get_all_tests import get_all_tests

torch_req = ["torch/2.0.0", "torch/2.0.1"]
tensorflow_req = [
    "tensorflow/2.13.0",
    "tensorflow/2.14.0",
]
jax_req = [
    "jax/0.4.10",
    "jax/0.4.14",
]
numpy_req = [
    "numpy/1.25.0",
    "numpy/1.24.0",
]
framework_versions = {
    "numpy": numpy_req,
    "torch": torch_req,
    "jax": jax_req,
    "tensorflow": tensorflow_req,
}

run_iter = int(sys.argv[1])
all_tests = get_all_tests()
test_names_without_backend = [test.split(",")[0].strip() for test in all_tests]
test_names = []
for test_name in test_names_without_backend:
    for backend, backend_versions in framework_versions.items():
        for backend_version in backend_versions:
            test_backend = test_name + "," + backend_version
            test_names.append(test_backend)

# Run 150 tests in each iteration of the cron job
num_tests = len(test_names)
tests_per_run = 5
start = run_iter * tests_per_run
end = (run_iter + 1) * tests_per_run
print("Running Tests:")
with open("tests_to_run", "w") as f:
    for i in range(start, end):
        i = i % num_tests
        test = test_names[i]
        if "test_frontends" in test:
            continue  # skip frontend tests (No support from testing)
        print(test)
        f.write(test + "\n")
