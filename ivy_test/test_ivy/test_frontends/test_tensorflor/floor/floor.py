def npfloor(*params):

    x = []
    for param in params:
        param = int(param)
        x.append(param)
    return x


@handle_frontend_test(
    fn_tree="tensorflow.raw_ops.floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
    )
)
