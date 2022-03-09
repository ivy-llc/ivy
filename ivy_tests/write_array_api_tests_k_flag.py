import os

this_dir = os.path.dirname(os.path.realpath(__file__))
func_folder = os.path.join(this_dir, 'array_api_methods_to_test')
array_folder = os.path.join(func_folder, 'array')

# api function filepaths
func_fnames = os.listdir(func_folder)
func_fnames.remove('array')
func_fnames.sort()
func_fpaths = [os.path.join(func_folder, fname) for fname in func_fnames]

# array function filepaths
array_fnames = os.listdir(array_folder)
array_fnames.sort()
array_fpaths = [os.path.join(array_folder, fname) for fname in array_fnames]

# all filepaths
fpaths = func_fpaths + array_fpaths

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
    tests_to_skip += ['test_' + s[1:] for s in contents if '#' in s]

# prune tests to skip
tests_to_skip = [tts for tts in tests_to_skip if not max([tts in ttr for ttr in tests_to_run])]

# save to file
with open(os.path.join(this_dir, '.array_api_tests_k_flag'), 'w+') as file:
    file.write('(' + ' or '.join(tests_to_run) + ') and not (' + ' or '.join(tests_to_skip) + ')')
