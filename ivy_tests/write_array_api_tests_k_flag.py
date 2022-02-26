import os

this_dir = os.path.dirname(os.path.realpath(__file__))
folder = os.path.join(this_dir, 'array_api_methods_to_test')
fnames = os.listdir(folder)
fnames.sort()

tests_to_run = list()
tests_to_skip = list()

for fname in fnames:

    # extract contents
    fpath = os.path.join(folder, fname)
    with open(fpath, 'r') as file:
        contents = file.read()

    # update tests to run and skip
    tests_to_run += ['test_' + s for s in contents.split('\n') if ('#' not in s and s != '')]
    tests_to_skip += ['test_' + s[1:] for s in contents.split('\n') if '#' in s]


# save to file
with open(os.path.join(this_dir, '.array_api_tests_k_flag'), 'w+') as file:
    file.write('(' + ' or '.join(tests_to_run) + ') and not (' + ' or '.join(tests_to_skip) + ')')
