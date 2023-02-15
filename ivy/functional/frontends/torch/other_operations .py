import ivy
# function for bincount

def bincount(input, weights=None, minlength=0):
    # Convert the input array to a sequence of integer indices
    indices = ivy.as_index(input)

    # Create an array of weights, if necessary
    if weights is None:
        weights = ivy.ones_like(indices)
    else:
        weights = ivy.as_array(weights)

    # Create an array of zeros with the desired minimum length
    counts = ivy.zeros(minlength, dtype=weights.dtype)

    # Use the add_at function to accumulate the weights into the counts array
    ivy.add_at(counts, indices, weights)

    return counts