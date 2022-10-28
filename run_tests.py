# Run Tests
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) > 2:
        redis_url = sys.argv[1]
        redis_pass = sys.argv[2]
    failed = False
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            print(test, backend)
            if len(sys.argv) > 2:
                ret = os.system(
                    f'docker run --rm --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest {test} --backend {backend}'  # noqa
                )
            else:
                ret = os.system(
                    f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest {test} --backend {backend}'  # noqa
                )
            if ret != 0:
                failed = True

    if failed:
        exit(1)
