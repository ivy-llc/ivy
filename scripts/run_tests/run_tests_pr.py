# Run Tests
import os
import sys


if __name__ == "__main__":
    failed = False
    with open(sys.argv[1], "w") as f_write:
        with open("tests_to_run", "r") as f:
            for line in f:
                test_path, backend = line.strip().split(",")
                print(f"\n{'*' * 100}")
                print(f"{line[:-1]}")
                print(f"{'*' * 100}\n")
                sys.stdout.flush()
                ret = os.system(
                    f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis ivyllc/ivy:latest python3 -m pytest --tb=short {test_path} --skip-trace-testing --skip-trace-testing-each --backend {backend}'  # noqa
                )
                if ret != 0:
                    failed = True
                    f_write.write(line)

    if failed:
        sys.exit(1)
