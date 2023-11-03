# Run Array API Tests for PRs
import os
import subprocess
import sys

BACKENDS = ["numpy", "jax", "tensorflow", "torch"]


def main():
    failed = False
    k_flag = {}
    subprocess.run(
        ["python3", "ivy_tests/array_api_testing/write_array_api_tests_k_flag.py"],
        check=True,
    )
    for backend in BACKENDS:
        k_flag_file = f"ivy_tests/array_api_testing/.array_api_tests_k_flag_{backend}"
        with open(k_flag_file, "r") as f:
            array_api_tests_k_flag = f.read().strip()
        if backend == "torch":
            array_api_tests_k_flag += " and not (uint16 or uint32 or uint64)"
        k_flag[backend] = array_api_tests_k_flag

    with open(sys.argv[1], "w") as f_write:
        with open("tests_to_run", "r") as f:
            for line in f:
                test, backend = line.split(",")
                backend = backend.strip("\n")
                command = f'docker run --rm --env IVY_BACKEND={backend} --env ARRAY_API_TESTS_MODULE="ivy" -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest timeout 30m python3 -m pytest {test} -k "{k_flag[backend]}" --tb=short -vv'  # noqa
                print(f"\n{'*' * 100}")
                print(f"{line[:-1]}")
                print(f"{'*' * 100}\n")
                sys.stdout.flush()
                ret = os.system(command)
                if ret != 0:
                    failed = True
                    f_write.write(line)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
