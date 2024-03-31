import sys
from get_all_tests import get_all_tests


run_iter, gpu, tracer = int(sys.argv[1]), sys.argv[2], sys.argv[3]
if gpu == "true":
    from setup_priority_tests import main

    main(cron=True)
    with open("tests_to_run", "r") as f:
        test_names = [line.strip() for line in f.readlines()]
    tests_per_run = 8
else:
    test_names = get_all_tests()
    tests_per_run = 150
    if tracer == "true":
        tests_per_run = 20
num_tests = len(test_names)
start = run_iter * tests_per_run
end = (run_iter + 1) * tests_per_run
print("Running Tests:")
with open("tests_to_run", "w") as f:
    for i in range(start, end):
        i = i % num_tests
        test = test_names[i]
        print(test)
        f.write(test + "\n")
