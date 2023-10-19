import sys
from get_all_tests import get_all_tests

N = 40


def main():
    run_iter = int(sys.argv[1]) - 1
    test_names = get_all_tests()
    num_tests = len(test_names)
    tests_per_run = num_tests // N
    start = run_iter * tests_per_run
    end = num_tests if run_iter == N - 1 else (run_iter + 1) * tests_per_run
    with open("tests_to_run", "w") as f:
        for test in test_names[start:end]:
            f.write(test + "\n")


if __name__ == "__main__":
    main()
