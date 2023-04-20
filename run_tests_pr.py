# Run Tests
import os
import sys

if __name__ == "__main__":
    failed = False
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")
            sys.stdout.flush()
            ret = os.system(
                f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --tb=short {test} --backend {backend}'  # noqa
            )
            if ret != 0:
                failed = True

    if failed:
        exit(1)
