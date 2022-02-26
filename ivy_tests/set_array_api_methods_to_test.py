import os

this_dir = os.path.dirname(os.path.realpath(__file__))
folder = os.path.join(this_dir, 'array_api_methods_to_test')
fnames = os.listdir(folder)
fnames.sort()

methods_to_test = list()

for fname in fnames:

    # extract contents
    fpath = os.path.join(folder, fname)
    with open(fpath, 'r') as file:
        contents = file.read()

    # update methods to test
    methods_to_test += [s for s in contents.split('\n') if ('#' not in s and s != '')]

# prepend test_
tests_to_run = ['(test_' + method + ' and not ' + 'test_' + method + '_)' for method in methods_to_test]

# save to file
with open(os.path.join(this_dir, '.array_api_methods_to_test'), 'w+') as file:
    file.write(' or '.join(tests_to_run))
