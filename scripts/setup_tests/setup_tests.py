import sys
from get_all_tests import BACKENDS


def main():
    if len(sys.argv) < 2:
        return
    test = sys.argv[1]
    with open("tests_to_run", "w") as f:
        if "," in test:
            f.write(test + "\n")
        else:
            for backend in BACKENDS:
                f.write(f"{test},{backend}\n")


if __name__ == "__main__":
    main()
