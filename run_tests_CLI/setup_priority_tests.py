import sys
from get_all_tests import BACKENDS


def main():
    write_file = open("tests_to_run", "w")
    with open(sys.argv[1], "r") as f:
        for test in f:
            if test.startswith("ivy/"):
                test = test[4:]
            for backend in BACKENDS:
                write_file.write(f"{test},{backend}\n")
    write_file.close()


if __name__ == "__main__":
    main()
