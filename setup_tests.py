import sys
from run_tests_CLI.get_all_tests import BACKENDS


def main():
    if len(sys.argv) < 2:
        return
    test = sys.argv[1]
    if "," in test:
        with open("tests_to_run", "w") as f:
            f.write(test + "\n")
    else:
        with open("tests_to_run", "w") as f:
            for backend in BACKENDS:
                f.write(f"{test},{backend}\n")


if __name__ == "__main__":
    main()
