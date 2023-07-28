import os
import random
import ast

BACKENDS = ["numpy", "jax", "tensorflow", "torch", "paddle"]


def is_test_function(node):
    if isinstance(node, ast.FunctionDef):
        return node.name.startswith("test_")
    return False


def extract_tests_from_file(filename):
    with open(filename, "r") as file:
        try:
            module = ast.parse(file.read())
        except SyntaxError:
            print(f"Syntax error in file: {filename}")
            return []

        return [
            f"{filename}::{node.name}" for node in module.body if is_test_function(node)
        ]


def extract_tests_from_dir(directory):
    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                test_files.extend(extract_tests_from_file(full_path))

    return test_files


def get_all_tests():
    test_names_without_backend = extract_tests_from_dir("ivy_tests/test_ivy")
    test_names_without_backend = list(set(test_names_without_backend))
    test_names_without_backend.sort()
    random.Random(4).shuffle(test_names_without_backend)

    test_names = []
    for test_name in test_names_without_backend:
        for backend in BACKENDS:
            test_backend = test_name + "," + backend
            test_names.append(test_backend)

    return test_names
