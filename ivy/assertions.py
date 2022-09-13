import ivy


def check_elem_in_list(elem, list):
    if elem not in list:
        raise ivy.exceptions.IvyException("{} is not one of {}".format(elem, list))
