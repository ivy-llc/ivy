#global
import ivy

#intersection function
def intersection(x, y):
    
    """ 
    # Example usage
    x = ivy.convert_to_constant([[1, 2, 3], [4, 5, 6]])
    y = ivy.convert_to_constant([[2, 3, 4], [5, 6, 7]])
    result = intersection(x, y)
    print(result)

    """
    # Using Ivy's functional API to compute the intersection of sets
    intersection_set = ivy.functional.frontends.tensorflow.sets.intersection(x, y, validate_indices=True)
    return intersection_set
