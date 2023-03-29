import os
import sys
import itertools

# Define the list of backends and backend versions
BACKENDS = ["numpy", "jax", "tensorflow", "torch"]
BACKEND_VERSIONS = {
    "numpy": ["1.17.3", "1.17.4", "1.23.1", "1.24.0", "1.24.1", "1.24.2"],
    "jax": ["0.1.60", "0.1.61", "0.3.10", "0.3.13", "0.3.14", "0.3.15", "0.3.16", "0.3.17"],
    "jaxlib": ["0.1.50", "0.1.60", "0.1.61", "0.3.10", "0.3.14", "0.3.15", "0.3.20", "0.3.22"],
    "tensorflow": ["2.2.0", "2.2.1", "2.2.2", "2.4.4", "2.9.0", "2.9.1", "2.9.1", "2.9.2"],
    "torch": ["1.4.0", "1.5.0", "1.10.1", "1.10.2", "1.11.0", "1.12.0", "1.12.1", "1.13.0"]
}

# Generate all combinations of backends and backend versions
backend_combinations = [
    {"backend": backend, "version": version}
    for backend, versions in BACKEND_VERSIONS.items()
    for version in versions
    if backend in BACKENDS
]

run_iter = int(sys.argv[1])
os.system(
    "docker run -v `pwd`:/ivy -v `pwd`/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --disable-pytest-warnings ivy_tests/test_ivy --my_test_dump true > test_names" # noqa
)
test_names_without_backend = []
test_names = []

# Parse the test names from the output file
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

# Generate all test names with backend/version combinations
for test_name in test_names_without_backend:
    if "test_frontends" in test_name:
        frontend = test_name[39:].split("/")[0]
        if frontend in BACKENDS:
            frontend_versions = BACKEND_VERSIONS.get(frontend, [])
            for backend_version in backend_combinations:
                if backend_version["backend"] != frontend:
                    continue
                for frontend_version in frontend_versions:
                    test_names.append(
                        f"{test_name},{backend_version['backend']}/{backend_version['version']};{frontend}/{frontend_version}"
                    )
    else:
        for backend_version in backend_combinations:
            test_names.append(
                f"{test_name},{backend_version['backend']}/{backend_version['version']
