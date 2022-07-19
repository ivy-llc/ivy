Ivy Frontend tests
====================

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html
.. _`ivy frontends discussion`: https://github.com/unifyai/ivy/discussions/2051
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`ivy frontends channel`: https://discord.com/channels/799879767196958751/998782045494976522
.. _`test_ivy`: https://github.com/unifyai/ivy/tree/0fc4a104e19266fb4a65f5ec52308ff816e85d78/ivy_tests/test_ivy
.. _`test_frontend_function`_: https://github.com/unifyai/ivy/blob/591ac37a664ebdf2ca50a5b0751a3a54ee9d5934/ivy_tests/test_ivy/helpers.py#L1047
.. _`hypothesis`_: https://lets-unify.ai/ivy/deep_dive/14_ivy_tests.html#id1
Ivy Frontends are the framework-specific frontend functional APIs defined to 
help with code transpilations as explained `here`_. This in turn makes it necessary 
to thoroughly test for each backend separately unlike the `test_ivy`_.

test_frontend_helper
--------------------

`test_frontend_function`_ is a helper method which checks for the given frontend framework, 
whether the ground truth matches with the result obtained from the specified 
function when used with any framework as backend. 
The ground truth for the specified function is obtained by simply setting the frontend 
framework as the backend framework. This essentially asserts that our frontend version of 
function gives the same result.

It takes the following parameters -

**input_dtypes**: Specifies the data types of the input arguments in order.
**as_variable_flags**: States whether the corresponding input argument should be treated as an ivy Variable.
**with_out**: When set to true, the function is also tested with the optional out argument.
**num_positional_args**: Specifies the number of input arguments that must be passed as positional arguments.
**native_array_flags**: States whether the corresponding input argument should be treated as a native array.
**fw**: Current backend framework.
**frontend**: Current frontend framework.
**fn_name**: Specifies the name of the function to be tested.
**rtol**: Specifies the relative tolerance value for the test to succeed/fail.
**atol**: Specifies the absolute tolerance value for the test to succeed/fail.
**test_values**: When set to true, it allows to test for the correctness of the resulting values.
**all_as_kwargs_np**: This corresponds to all the input arguments to be passed to the frontend 
function as keyword arguments.

Inorder to get a better understanding of how these tests are written for each framework,
let's begin with exploring the tan function

Examples
--------

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @given(
        dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_jax.valid_float_dtypes),
        as_variable=st.booleans(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.lax.tan"
        ),
        native_array=st.booleans(),
    )
    def test_jax_lax_tan(
        dtype_and_x,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x

        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            fn_name="lax.tan",
            x=np.asarray(x, dtype=input_dtype),
        )
    
The data required for hypothesis tests is generated with the help of hypothesis strategies and is better explained at `hypothesis`_. 
The crux of the test lies in the :code:`test_frontend_function` which takes inputs as defined above. Here, the 
:code:`all_as_kwargs_np` is composed of the last argument :code:`x=np.asarray(x, dtype=input_dtype)` and is the input
to `jax.lax.tan`

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_jax.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.add"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    )
    def test_jax_lax_add(
        dtype_and_x,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x

        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            fn_name="lax.add",
            x=np.asarray(x[0], dtype=input_dtype[0]),
            y=np.asarray(x[1], dtype=input_dtype[1]),
        )
Similarly, for :code:`add`, the :code:`all_as_kwargs_np` is composed of the last 2 arguments and are the inputs
to `jax.lax.add`.

**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_mathematical_functions/test_np_trigonometric_functions.py
    @given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.tan"
    ),
    native_array=helpers.array_bools(),
    )
    def test_numpy_tan(
        dtype_and_x,
        dtype,
        where,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        input_dtype = [input_dtype]
        where = np_frontend_helpers.handle_where_and_array_bools(
            where=where,
            input_dtype=input_dtype,
            as_variable=as_variable,
            native_array=native_array,
        )
        np_frontend_helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="numpy",
            fn_name="tan",
            x=np.asarray(x, dtype=input_dtype[0]),
            out=None,
            where=where,
            casting="same_kind",
            order="k",
            dtype=dtype,
            subok=True,
            test_values=False,
        )
    
Here, the :code:`all_as_kwargs_np` is composed of the arguments followed by :code:`fn_name` and are the input
to `numpy.tan`

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tf_functions.py
    @given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_tf.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.tan"
    ),
    native_array=st.booleans(),
    )
    def test_tensorflow_tan(
        dtype_and_x, as_variable, num_positional_args, native_array, fw
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="tensorflow",
            fn_name="tan",
            x=np.asarray(x, dtype=input_dtype),
        )
Here, the :code:`all_as_kwargs_np` is composed of the last argument :code:`x=np.asarray(x, dtype=input_dtype)` 
only and serve the input to `tensorflow.tan`.

**Torch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_pointwise_ops.py
    @given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        )
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.tan"
    ),
    native_array=st.booleans(),
    )
    def test_torch_tan(
        dtype_and_x,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            fn_name="tan",
            input=np.asarray(x, dtype=input_dtype),
            out=None,
        )
Here, the :code:`all_as_kwargs_np` is composed of the last two arguments and serve the input to `torch.tan`.
It might be intriguing to observe both the :code:`out` as well as the :code:`with_out` arguments. 
To clarify, the :code:`with_out` argument is used to specify whether the inplace update operation is 
supported by the function for the given framework and is present by design in Ivy . 
However, the :code:`out` argument is required when calling the for the framework's function implementation.

**Round Up**

This should have hopefully given you a good idea about implementing Ivy Frontend tests.

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy tests discussion`_,
or reach out on `discord`_ in the `ivy tests channel`_!