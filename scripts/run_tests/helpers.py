import requests


def get_latest_package_version(package_name):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        package_info = response.json()
        return package_info["info"]["version"]
    except requests.exceptions.RequestException:
        print(f"Error: Failed to fetch package information for {package_name}.")
        return None


def get_submodule_and_function_name(test_path, is_frontend_test=False):
    submodule_test = test_path.split("/")[-1]
    submodule, test_function = submodule_test.split("::")
    submodule = submodule.replace("test_", "").replace(".py", "")

    with open(test_path.split("::")[0]) as test_file:
        test_file_content = test_file.read()
        test_function_idx = test_file_content.find(f"def {test_function}")
        test_function_block_idx = test_file_content[:test_function_idx].rfind("\n\n")
        if test_function_block_idx == -1:
            return submodule, None
        relevant_file_content = test_file_content[
            test_function_block_idx:test_function_idx
        ]
        fn_tree_idx = relevant_file_content.rfind('fn_tree="')

        # frontend test
        if is_frontend_test:
            function_name = relevant_file_content[fn_tree_idx + 9 :].split('"')[0]

            # instance method test
            if fn_tree_idx == -1:
                class_tree_idx = test_file_content.find('CLASS_TREE = "')
                method_name_idx = relevant_file_content.rfind('method_name="')
                if class_tree_idx == -1 or method_name_idx == -1:
                    return submodule, None
                class_tree = test_file_content[class_tree_idx + 14 :].split('"')[0]
                class_name = ".".join(class_tree.split(".")[3:])
                method_name = relevant_file_content[method_name_idx + 13 :].split('"')[
                    0
                ]
                function_name = f"{class_name}.{method_name}"

        # ivy test
        else:
            function_name = test_function[5:]

            # instance method test
            if fn_tree_idx == -1:
                method_name_idx = relevant_file_content.rfind('method_tree="')
                if method_name_idx != -1:
                    method_name = relevant_file_content[method_name_idx + 13 :].split(
                        '"'
                    )[0]
                    function_name = f"ivy.{method_name}"
                else:
                    return submodule, None

    return submodule, function_name
