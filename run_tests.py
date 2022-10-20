# Run Tests
import os

if __name__ == "__main__":
    failed = False
    with open("tests_to_run", "r") as f:
        for line in f:
            test = line[:-1]
            print(test)
            ret = os.system(
                f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest {test}'  # noqa
            )
            if ret != 0:
                failed = True

    if failed:
        exit(1)
