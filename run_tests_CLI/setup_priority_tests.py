import sys

from get_all_tests import BACKENDS


def main():
    with open("tests_to_run", "w") as write_file:
        with open(sys.argv[1], "r") as f:
            for test in f:
                test = test.strip()
                if test.startswith("ivy/"):
                    test = test[4:]
                for backend in BACKENDS:
                    write_file.write(f"{test},{backend}\n")


if __name__ == "__main__":
    main()
