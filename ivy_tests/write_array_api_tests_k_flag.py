import os

this_dir = os.path.dirname(os.path.realpath(__file__))
func_folder = os.path.join(this_dir, "array_api_methods_to_test")

# api function filepaths
func_fnames = os.listdir(func_folder)
func_fnames.sort()

func_fpaths = [os.path.join(func_folder, fname) for fname in func_fnames]

# all filepaths
fpaths = func_fpaths

# test lists
framework_tests_to_run = {
    "jax": list(),
    "numpy": list(),
    "torch": list(),
    "tensorflow": list(),
}
framework_tests_to_skip = {
    "jax": list(),
    "numpy": list(),
    "torch": list(),
    "tensorflow": list(),
}
# add from each filepath
for fpath in fpaths:
    # extract contents
    with open(fpath, "r") as file:
        contents = file.read()
        # update tests to run and skip
        contents = [line.replace("__", "") for line in contents.split("\n")]
        for framework in framework_tests_to_run:
            tests_to_run = list()
            tests_to_skip = list()
            for s in contents:
                if s == "":
                    continue
                if ("#" not in s) or (
                    "#" in s
                    and not (framework in s.lower())
                    and any(f in s.lower() for f in framework_tests_to_run)
                ):
                    tests_to_run += (
                        ["test_" + s]
                        if ("#" not in s)
                        else ["test_" + s.split("#")[1].split(" ")[0]]
                    )
                else:
                    tests_to_skip += ["test_" + s[1:].split(" ")[0]]
            framework_tests_to_run[framework] += tests_to_run
            framework_tests_to_skip[framework] += tests_to_skip

for framework in framework_tests_to_skip:
    # prune tests to skip
    framework_tests_to_skip[framework] = [
        tts
        for tts in framework_tests_to_skip[framework]
        if not max([tts in ttr for ttr in framework_tests_to_run[framework]])
    ]


# save to file
for framework in framework_tests_to_run:
    with open(
        os.path.join(this_dir, ".array_api_tests_k_flag_" + framework), "w+"
    ) as file:
        file.write(
            "("
            + " or ".join(framework_tests_to_run[framework])
            + ") and not ("
            + " or ".join(framework_tests_to_skip[framework])
            + ")"
        )
