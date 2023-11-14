import sys
from get_all_tests import get_all_tests


run_iter, gpu = int(sys.argv[1]), sys.argv[2]
if gpu == "true":
    from setup_priority_tests import main

    main()
    with open("tests_to_run", "r") as f:
        test_names = [line.strip().split(",")[0] for line in f.readlines()]
    test_names = list(set(test_names))
else:
    test_names = get_all_tests()
# Run 150 tests in each iteration of the cron job
num_tests = len(test_names)
tests_per_run = 150
start = run_iter * tests_per_run
end = (run_iter + 1) * tests_per_run
print("Running Tests:")
with open("tests_to_run", "w") as f:
    for i in range(start, end):
        i = i % num_tests
        test = test_names[i]
        print(test)
        f.write(test + "\n")
