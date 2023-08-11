@handle_frontend_test(
    fn_tree="numpy.random.standard_exponential",
    size=helpers.get_shape(allow_none=True),
    dtype=helpers.get_dtype("float"),
    method=st.sampled_from(["zig", "inv"]),
    out=st.none(),
)
def test_numpy_standard_exponential(
    size,
    dtype,
    method,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        size=size,
        dtype=dtype,
        method=method,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
    )
