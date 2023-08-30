import ivy

def intersection(x, y, validate_indices=False):
    # Using Ivy's functional API to compute the intersection of sets
    intersection_set = ivy.intersection(x, y, validate_indices=validate_indices)
    return intersection_set
