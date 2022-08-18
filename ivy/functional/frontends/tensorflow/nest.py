# global


def flatten(structure, expand_composites=False):
    if isinstance(structure, (tuple, list)):
        new_struc = []
        for child in structure:
            new_struc += flatten(child)
        return new_struc
    if isinstance(structure, dict):
        new_struc = []
        for key in sorted(structure):
            new_struc += flatten(structure[key])
        return new_struc
    return [structure]
