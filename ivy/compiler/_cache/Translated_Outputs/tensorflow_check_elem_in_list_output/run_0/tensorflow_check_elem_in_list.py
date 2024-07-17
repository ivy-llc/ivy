def tensorflow_check_elem_in_list(elem, list, inverse=False, message=""):
    if inverse and elem in list:
        raise Exception(
            message if message != "" else f"{elem} must not be one of {list}"
        )
    elif not inverse and elem not in list:
        raise Exception(message if message != "" else f"{elem} must be one of {list}")
