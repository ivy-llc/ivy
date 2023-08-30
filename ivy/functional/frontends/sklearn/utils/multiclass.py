import ivy


# reapeated utility function
def type_of_target(y, input_name='y'):
    # purely utility function
    # TODO: implement multilabel-indicator, ...-multioutput, unknown
    if y.ndim not in (1, 2):
        return "unknown"
    if ivy.is_float_dtype(y) and ivy.any(ivy.not_equal(y, y.astype('int64'))):
        return "continuous"
    else:
        vals = ivy.unique_values(y)
        if len(vals) > 2:
            return "multiclass"
        else:
            return "binary"
