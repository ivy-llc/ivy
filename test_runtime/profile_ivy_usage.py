import ivy
import csv
import json
import argparse
import importlib
from collections import OrderedDict


def _trim_module_dict(ivy_dict):
    new_dict = dict()
    for key, val in ivy_dict.items():
        if key[0] == '_' or key.isupper():
            continue
        elif key[0].isupper():
            ret_dict = _trim_module_dict(val.__dict__)
            if ret_dict:
                new_dict[key] = ret_dict
        elif isinstance(val, dict):
            ret_dict = _trim_module_dict(val)
            if ret_dict:
                new_dict[key] = ret_dict
        elif callable(val):
            new_dict[key] = val
    return new_dict


def main(lib_strings):

    # extract code for each library function
    lib_usgae_dict = dict()
    for lib_string in lib_strings:
        lib_module = importlib.import_module(lib_string)
        lib_dict = _trim_module_dict(lib_module.__dict__)
        all_code = ''
        for lib_val in lib_dict.values():
            if isinstance(lib_val, dict):
                for class_method in lib_val.values():
                    filepath = class_method.__code__.co_filename
                    with open(filepath, 'r') as file:
                        code = file.read()
                    all_code += code
                continue
            # noinspection PyUnresolvedReferences
            filepath = lib_val.__code__.co_filename
            with open(filepath, 'r') as file:
                code = file.read()
            all_code += code
        lib_usgae_dict[lib_string] = all_code

    # count ivy method occurances
    ivy_dict = _trim_module_dict(ivy.__dict__)
    ivy_method_occurances = dict()
    for lib_string, all_code in lib_usgae_dict.items():
        ivy_method_occurances[lib_string] = dict()
        for ivy_method_name in ivy_dict.keys():
            ivy_method_occurances[lib_string][ivy_method_name] = all_code.count(
                '.' + ivy_method_name
            )

    ivy_method_occurances['all'] = dict()
    for ivy_method_name in ivy_dict.keys():
        ivy_method_occurances['all'][ivy_method_name] = sum(
            [
                ivy_method_occurances[lib_string][ivy_method_name]
                for lib_string in lib_usgae_dict.keys()
            ]
        )

    # sort values
    for lib_string, method_dict in ivy_method_occurances.items():
        ivy_method_occurances[lib_string] = OrderedDict(
            {
                k: v for k, v in sorted(
                    method_dict.items(), key=lambda item: -item[1]
                )
            }
        )

    # save to json file
    with open('library_usage.json', 'w+') as file:
        file.write(json.dumps(ivy_method_occurances, indent=4))

    # save to csv file
    with open('library_usage.csv', 'w+') as file:
        csv_writer = csv.writer(file)
        for lib_string, lib_dict in ivy_method_occurances.items():
            csv_writer.writerow([lib_string])
            for method_name, count in lib_dict.items():
                csv_writer.writerow([method_name, str(count)])
            csv_writer.writerow([''])
            csv_writer.writerow([''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--libs', nargs="+", required=True)
    main(parser.parse_args().libs)
