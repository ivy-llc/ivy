import os

this_dir = os.path.dirname(os.path.realpath(__file__))
func_folder = os.path.join(this_dir, 'array_api_methods_to_test')

# api function filepaths
func_fnames = os.listdir(func_folder)
func_fnames.sort()
func_fpaths = [os.path.join(func_folder, fname) for fname in func_fnames]

# all filepaths
fpaths = func_fpaths

# test lists
tests_to_run = list()
tests_to_skip = list()

# add from each filepath
for fpath in fpaths:

    # extract contents
    with open(fpath, 'r') as file:
        contents = file.read()

    # update tests to run and skip
    contents = [line.replace('__', '') for line in contents.split('\n')]
    tests_to_run += ['test_' + s for s in contents if ('#' not in s and s != '')]
    tests_to_skip += ['test_' + s[1:].split(' ')[0] for s in contents if len(s.split('#')) == 2]

# temporary fix for wrongly added test, due to addition of positive method
tests_to_skip += ['test_positive_definite_matrices']

# prune tests to skip
tests_to_skip = [tts for tts in tests_to_skip if not max([tts in ttr for ttr in tests_to_run])]

# save to file
with open(os.path.join(this_dir, '.array_api_tests_k_flag'), 'w+') as file:
    file.write('(' + ' or '.join(tests_to_run) + ') and not (' + ' or '.join(tests_to_skip) + ')')
tests_to_run = ['test_irrational_numbers']
with open(os.path.join(this_dir, '.array_api_tests_k_flag_jax'), 'w+') as file:
    file.write('(' + ' or '.join(tests_to_run) + ') and not (' + ' or '.join(tests_to_skip) + ')')
