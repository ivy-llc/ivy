Ivy Frontend Tests
====================

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html
.. _`ivy frontends channel`: https://discord.com/channels/799879767196958751/998782045494976522
.. _`test_ivy`: https://github.com/unifyai/ivy/tree/0fc4a104e19266fb4a65f5ec52308ff816e85d78/ivy_tests/test_ivy
.. _`test_frontend_function`: https://github.com/unifyai/ivy/blob/591ac37a664ebdf2ca50a5b0751a3a54ee9d5934/ivy_tests/test_ivy/helpers.py#L1047
.. _`hypothesis`_: https://lets-unify.ai/ivy/deep_dive/14_ivy_tests.html#id1
.. _`ivy frontends discussion`: https://github.com/unifyai/ivy/discussions/2051
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`Function_wrapping`: https://lets-unify.ai/ivy/deep_dive/3_function_wrapping.html
.. _`open task`: https://lets-unify.ai/ivy/contributing/4_open_tasks.html#open-tasks

Introduction
------------

Just like the backend functional API, our frontend functional API has a collection of Ivy tests located in subfolder `test_ivy`_.
In this section of the deep dive we are going to jump into Ivy Frontend Tests!

**Writing Ivy Tests**

The Ivy tests in this section make use of hypothesis for performing property based testing which is documented in detail in the Ivy Tests section of the Deep Dive.
We assume knowledge of hypothesis data generation strategies and how to implement them for testing.

**Important Helper Functions**

* :func:`handle_cmd_line_args` is an important decorator that should be added to every test function. For more information, visit the `Function_wrapping`_ section of the docs.

* :func:`helpers.test_frontend_function` is an important helper function that is designed to do the heavy lifting and make testing Ivy Frontends easy! It is used to test a frontend function for the current backend by comparing the result with the function in the associated framework.

* :func:`helpers.dtype_and_values` is a convenience function that allows you to generate lists of any dimension and their associated dtype, returned as :code:`(dtype, list)`.

* :func:`helpers.num_positional_args` is a convenience function that specifies the number of positional arguments for a particular function.

* :func:`helpers.get_shape` is a convenience function that allows you to generate an array shape of type :code:`list`

* :func:`np_frontend_helpers.where` a generation strategy to generate values for NumPy's optional :code:`where` argument.

* :func:`np_frontend_helpers.test_frontend_function` behaves identical to :func:`helpers.test_frontend_function` but handles NumPy's optional :code:`where` argument

**Useful Notes**

* There are more :code:`dtype` sets than :code:`valid_float_dtypes`, there is also :code:`valid_int_dtypes`, :code:`valid_numeric_dtypes` and many more! It's important when writing Ivy Frontend tests to pick the most appropriate :code:`dtype` set to ensure thorough testing!

* The :func:`test_frontend_function` argument :code:`fn_tree` refers to the frontend function's reference in its native namespace not just the function name. For example :func:`lax.tan` is needed for some functions in Jax, :func:`nn.functional.relu` is needed for some functions in PyTorch etc.

To get a better understanding for writing frontend tests lets run through some examples!

Frontend Test Examples
-----------------------

Before you begin writing a frontend test, make sure you are placing it in the correct location. See the
'Where to place a frontend function' sub-section of the frontend APIs `open task`_ for more details.

ivy.tan()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    #tan
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float")
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.lax.tan"
        )
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
            fn_tree="lax.tan",
            x=np.asarray(x, dtype=input_dtype),
        )

* As you can see we generate almost everything we need to test a frontend function within the :code:`@given` and :code:`@handle_cmd_line_args` decorators.
* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for Jax.
* We pass :code:`fn_name` to :func:`helpers.num_positional_args` which is used to determine the number of positional arguments for :func:`jax.lax.tan`.
* We do not generate any values for :code:`fw`, these values are generated by PyTest and are only passed as an argument to :func:`test_jax_lax_tan`.
* We separate the :code:`input_dtype` and :code:`x` from :code:`dtype_and_x` using :code:`input_dtype, x = dtype_and_x` which is generated as a tuple.
* We then pass the generated values to :func:`helpers.test_frontend_function` which tests the frontend function.
* We set :code:`fn_tree` to :func:`lax.tan` which is the path to the function in the Jax namespace.
* :func:`jax.lax.tan` does not support :code:`out` arguments so we set :code:`with_out` to :code:`False`.
* We cast :code:`x` as a NumPy array with :code:`np.asarray(x, dtype=input_dtype)` whenever the frontend function requires an array as an input. This is because :code:`x` is generated as type :code:`list`.
* One last important note is that all helper functions are designed to take keyword arguments only.

**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_mathematical_functions/test_np_trigonometric_functions.py
    #tan
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric")
        ),
        dtype=helpers.get_dtypes("numeric", none=True),
        where=np_frontend_helpers.where(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.tan"
        ),
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
            fn_tree="tan",
            x=np.asarray(x, dtype=input_dtype[0]),
            out=None,
            where=where,
            casting="same_kind",
            order="k",
            dtype=dtype,
            subok=True,
            test_values=False,
        )

* Here we use :code:`helpers.get_dtypes("numeric")` to generate :code:`available_dtypes`, these are valid :code:`numeric` data types specifically for NumPy.
* NumPy has an optional argument :code:`where` which is generated using :func:`np_frontend_helpers.where`.
* :func:`numpy.tan` supports :code:`out` arguments so we set generate values for :code:`with_out`.
* Using :func:`np_frontend_helpers.handle_where_and_array_bools` we do some processing on the generated :code:`where` value.
* Instead of :func:`helpers.test_frontend_function` we use :func:`np_frontend_helpers.test_frontend_function` which behaves the same but has some extra code to handle the :code:`where` argument.
* We set :code:`fn_tree` to :code:`tan` which is the path to the function in the NumPy namespace.
* :code:`casting`, :code:`order`, :code:`subok` and :code:`test_values` are other other optional arguments for :func:`numpy.tan`.

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tf_functions.py
    #tan
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float"),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.tan"
        ),
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
            fn_tree="tan",
            x=np.asarray(x, dtype=input_dtype),
        )

* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for TensorFlow.
* We set :code:`fn_tree` to :code:`tan` which is the path to the function in the TensorFlow namespace.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_pointwise_ops.py
    #tan
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="functional.frontends.torch.tan"
        ),
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
            fn_tree="tan",
            input=np.asarray(x, dtype=input_dtype),
            out=None,
        )

* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for PyTorch.
* We set :code:`fn_tree` to :code:`tan` which is the path to the function in the PyTorch namespace.

ivy.full()
^^^^^^^^^^

Here we are going to look at an example of a function that does not consume an :code:`array`.
This is the creation function :func:`full`, which takes an array shape as an argument to create an array and filled with elements of a given value.
This function requires us to create extra functions for generating :code:`shape` and :code:`fill value`, these use the :code:`shared` hypothesis strategy.


**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    # full
    @st.composite
    def _dtypes(draw):
        return draw(
            st.shared(
                helpers.list_of_length(
                    x=st.sampled_from(ivy_jax.valid_numeric_dtypes), length=1
                ),
                key="dtype",
            )
        )


    @st.composite
    def _fill_value(draw):
        dtype = draw(_dtypes())[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        elif ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=_dtypes(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.lax.full"
        ),
    )
    def test_jax_lax_full(
        shape,
        fill_value,
        dtypes,
        num_positional_args,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=False,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=False,
            fw=fw,
            frontend="jax",
            fn_tree="lax.full",
            shape=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
        )

* The first extra function we use is :code:`_dtypes` which generates a :code:`list` of :code:`dtypes` to use for the :code:`dtype` argument. Notice how we use :code:`st.shared` to generate a dtype which is unique to that test instance.
* The second extra function we use is :code:`_fill_value` which generates a :code:`fill_value` to use for the :code:`fill_value` argument but handles the complications of :code:`int` and :code:`uint` types correctly
* We use the helper function :func:`helpers.get_shape` to generate :code:`shape`.
* We use :code:`ivy_jax.valid_numeric_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Jax. This is used to specify the data type of the output array.
* :func:`full` does not consume :code:`array`, we set :code:`as_variable_flags`, :code:`with_out` and :code:`native_array_flags` to :code:`False`.


**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/creation_routines/test_from_shape_or_value.py
    # full
    @st.composite
    def _dtypes(draw):
        return draw(
            st.shared(
                helpers.list_of_length(
                    x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
                ),
                key="dtype",
            )
        )


    @st.composite
    def _fill_value(draw):
        dtype = draw(_dtypes())[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=_dtypes(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.full"
        ),
    )
    def test_numpy_full(
        shape,
        fill_value,
        dtypes,
        num_positional_args,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=False,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=False,
            fw=fw,
            frontend="numpy",
            fn_tree="full",
            shape=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
        )

* We use :code:`ivy_np.valid_numeric_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for NumPy.
* :func:`numpy.full` does not have a :code:`where` argument so we can use :func:`helpers.test_frontend_function`

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tf_functions.py
    # full
    @st.composite
    def _dtypes(draw):
        return draw(
            st.shared(
                helpers.list_of_length(
                    x=st.sampled_from(ivy_tf.valid_numeric_dtypes), length=1
                ),
                key="dtype",
            )
        )


    @st.composite
    def _fill_value(draw):
        dtype = draw(_dtypes())[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=_dtypes(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.fill"
        ),
    )
    def test_tensorflow_full(
        shape,
        fill_value,
        dtypes,
        num_positional_args,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=False,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=False,
            fw=fw,
            frontend="tensorflow",
            fn_tree="fill",
            dims=shape,
            value=fill_value,
            rtol=1e-05,
        )

* We use :code:`ivy_tf.valid_numeric_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for TensorFlow.
* Tensorflow's version of :func:`full` is named :func:`fill` therefore we specify the :code:`fn_tree` argument to be :code:`"fill"`
* When running the test there where some small discrepancies between the values so we can use :code:`rtol` to specify the relative tolerance.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_creation_ops.py
    # full
    @st.composite
    def _dtypes(draw):
        return draw(
            st.shared(
                helpers.list_of_length(
                    x=st.sampled_from(ivy_torch.valid_numeric_dtypes), length=1
                ),
                key="dtype",
            )
        )


    @st.composite
    def _fill_value(draw):
        dtype = draw(_dtypes())[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))


    @st.composite
    def _requires_grad(draw):
        dtype = draw(_dtypes())[0]
        if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
            return draw(st.just(False))
        else:
            return draw(st.booleans())


    # full
    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=_dtypes(),
        requires_grad=_requires_grad(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.torch.full"
        ),
    )
    def test_torch_full(
        shape,
        fill_value,
        dtypes,
        requires_grad,
        device,
        num_positional_args,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=False,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=False,
            fw=fw,
            frontend="torch",
            fn_tree="full",
            size=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
            device=device,
            requires_grad=requires_grad,
        )

* Here we created another extra function, :code:`_requires_grad()`, to accommodate the :code:`requires_grad` argument. This is because when the dtype is an integer or unsigned integer the :code:`requires_grad` argument is not supported.
* We use :code:`ivy_torch.valid_numeric_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Torch.
* :func:`torch.full` supports :code:`out` so we generate :code:`with_out`.


Frontend Instance Method Tests
------------------------------

The frontend instance method tests are similar to the frontend function test, but instead 
of testing the function directly we test the instance method of the frontend class.

**Important Helper Functions**

* :func:`helpers.test_frontend_instance_method` is used to test frontend instance methods.
It is used in the same way as :func:`helpers.test_frontend_function`.

**Useful Notes**
The :func:`helpers.test_frontend_instance_method` takes an argument :code:`frontend_class`
which is the frontend class to test. This is the relevant Ivy frontend class and not the native framework class.


Frontend Instance Method Test Examples
--------------------------------------

ivy.add()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_devicearray.py
    # add
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric", full=True),
            num_arrays=2,
            shared_dtype=True,
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.DeviceArray.add",
        ),
    )
    def test_jax_instance_add(
        dtype_and_x,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            frontend_class=DeviceArray,
            fn_tree="DeviceArray.add",
            self=np.asarray(x[0], dtype=input_dtype[0]),
            other=np.asarray(x[1], dtype=input_dtype[1]),
        )

* We use :func:`test_frontend_array_instance_method` to test the instance method.
* We import the frontend class :class:`DeviceArray` from :code:`frontends.jax.DeviceArray` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`DeviceArray.add` which is the path to the function in the frontend class.
    
**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_ndarray.py
    # add
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_float_dtypes, num_arrays=2
        ),
        dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
        where=np_frontend_helpers.where(),
        as_variable=helpers.array_bools(),
        with_out=st.booleans(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.ndarray.add"
        ),
        native_array=helpers.array_bools(),
    )
    def test_numpy_instance_add(
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
        where = np_frontend_helpers.handle_where_and_array_bools(
            where=where,
            input_dtype=input_dtype,
            as_variable=as_variable,
            native_array=native_array,
        )
        np_frontend_helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="numpy",
            frontend_class=ndarray,
            fn_tree="ndarray.add",
            self=np.asarray(x[0], dtype=input_dtype[0]),
            other=np.asarray(x[1], dtype=input_dtype[1]),
            out=None,
            where=where,
            casting="same_kind",
            order="k",
            dtype=dtype,
            subok=True,
            test_values=False,
        )

* We use :func:`np_frontend_helpers.test_frontend_array_instance_method` to test the instance method. This handles the :code:`where` argument.
* We import the frontend class :class:`ndarray` from :code:`frontends.numpy.ndarray` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`ndarray.add` which is the path to the function in the frontend class.
    
**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tensor.py
    # add
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=tuple(
                set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
            ),
            num_arrays=2,
            shared_dtype=True,
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.Tensor.add",
        ),
    )
    def test_tensorflow_instance_add(
        dtype_and_x, as_variable, num_positional_args, native_array, fw
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="tensorflow",
            frontend_class=Tensor,
            fn_tree="Tensor.add",
            self=np.asarray(x[0], dtype=input_dtype[0]),
            y=np.asarray(x[1], dtype=input_dtype[1]),
        )

* We import the frontend class :class:`Tensor` from :code:`frontends.tensorflow.tensor` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`Tensor.add` which is the path to the function in the frontend class.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_tensor.py
    # add
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes)
                )
            ),
            num_arrays=2,
            min_value=-1e04,
            max_value=1e04,
            allow_inf=False,
        ),
        alpha=st.floats(min_value=-1e06, max_value=1e06, allow_infinity=False),
        num_positional_args=helpers.num_positional_args(
            fn_name="functional.frontends.torch.Tensor.add",
        ),
    )
    def test_torch_instance_add(
        dtype_and_x,
        alpha,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            frontend_class=Tensor,
            fn_tree="Tensor.add",
            rtol=1e-04,
            self=np.asarray(x[0], dtype=input_dtype[0]),
            other=np.asarray(x[1], dtype=input_dtype[1]),
            alpha=alpha,
            out=None,
        )

* We import the frontend class :class:`Tensor` from :code:`frontends.torch.tensor` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`Tensor.add` which is the path to the function in the frontend class.

ivy.reshape()
^^^^^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_devicearray.py
    # reshape
    @st.composite
    def _reshape_helper(draw):
        # generate a shape s.t len(shape) > 0
        shape = draw(helpers.get_shape(min_num_dims=1))

        reshape_shape = draw(helpers.reshape_shapes(shape=shape))

        dtype = draw(helpers.array_dtypes(num_arrays=1))[0]
        x = draw(helpers.array_values(dtype=dtype, shape=shape))

        is_dim = draw(st.booleans())
        if is_dim:
            # generate a permutation of [0, 1, 2, ... len(shape) - 1]
            permut = draw(st.permutations(list(range(len(shape)))))
            return x, dtype, reshape_shape, permut
        else:
            return x, dtype, reshape_shape, None


    @handle_cmd_line_args
    @given(
        x_reshape_permut=_reshape_helper(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.DeviceArray.reshape"
        ),
    )
    def test_jax_instance_reshape(
        x_reshape_permut,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        x, dtype, shape, dimensions = x_reshape_permut
        helpers.test_frontend_array_instance_method(
            input_dtypes=dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            frontend_class=DeviceArray,
            fn_tree="DeviceArray.reshape",
            self=np.asarray(x, dtype=dtype),
            new_sizes=shape,
            dimensions=dimensions,
        )

* For :func:`jax.reshape`, we create a helper function to generate correct data to test the function.

**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_ndarray.py
    # reshape
    @st.composite
    def dtypes_x_reshape(draw):
        dtypes, x = draw(
            helpers.dtype_and_values(
                shape=helpers.get_shape(
                    allow_none=False,
                    min_num_dims=1,
                    max_num_dims=5,
                    min_dim_size=1,
                    max_dim_size=10,
                )
            )
        )
        shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
        return dtypes, x, shape


    @handle_cmd_line_args
    @given(
        dtypes_x_shape=dtypes_x_reshape(),
        copy=st.booleans(),
        with_out=st.booleans(),
        as_variable=helpers.array_bools(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.ndarray.reshape"
        ),
        native_array=helpers.array_bools(),
    )
    def test_numpy_instance_reshape(
        dtypes_x_shape,
        copy,
        with_out,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        dtypes, x, shape = dtypes_x_shape
        helpers.test_frontend_array_instance_method(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="numpy",
            frontend_class=ndarray,
            fn_tree="ndarray.reshape",
            self=x,
            shape=shape,
            copy=copy,
        )

* For :func:`NumPy.reshape`, we create a helper function to generate correct data to test the function.

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tensor.py
    # reshape
    @st.composite
    def dtypes_x_reshape(draw):
        dtypes, x = draw(
            helpers.dtype_and_values(
                shape=helpers.get_shape(
                    allow_none=False,
                    min_num_dims=1,
                    max_num_dims=5,
                    min_dim_size=1,
                    max_dim_size=10,
                )
            )
        )
        shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
        return dtypes, x, shape


    @handle_cmd_line_args
    @given(
        dtypes_x_shape=dtypes_x_reshape(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.Tensor.Reshape",
        ),
    )
    def test_tensorflow_instance_Reshape(
        dtypes_x_shape,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        dtypes, x, shape = dtypes_x_shape
        helpers.test_frontend_array_instance_method(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="tensorflow",
            frontend_class=Tensor,
            fn_tree="Tensor.Reshape",
            self=np.asarray(x, dtype=dtypes),
            shape=shape,
        )

* For :func:`tensorflow.Reshape`, we create a helper function to generate correct data to test the function.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tensor.py
    # reshape
    @st.composite
    def dtypes_x_reshape(draw):
        dtypes, x = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float", full=True),
                shape=helpers.get_shape(
                    allow_none=False,
                    min_num_dims=1,
                    max_num_dims=5,
                    min_dim_size=1,
                    max_dim_size=10,
                )
            )
        )
        shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
        return dtypes, x, shape


    @handle_cmd_line_args
    @given(
        dtypes_x_reshape=dtypes_x_reshape(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.torch.Tensor.reshape",
        ),
    )
    def test_torch_instance_reshape(
        dtypes_x_reshape,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x, shape = dtypes_x_reshape
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            frontend_class=Tensor,
            fn_tree="Tensor.reshape",
            self=np.asarray(x, dtype=input_dtype),
            shape=shape,
        )

* For :func:`torch.reshape`, we create a helper function to generate correct data to test the function.

Hypothesis Helpers
------------------

Naturally, many of the functions in the various frontend APIs are very similar to many
of the functions in the Ivy API. Therefore, the unit tests will follow very similar
structures with regards to the data generated for testing.
There are many data generation helper functions defined in the Ivy API test files,
such as :func:`_arrays_idx_n_dtypes` defined in
:mod:`ivy/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py`.
This helper generates: a set of concatenation-compatible arrays,
the index for the concatenation, and the data types of each array.
Not surprisingly, this helper is used for testing :func:`ivy.concat`, as shown
`here <https://github.com/unifyai/ivy/blob/86287f4e45bbe581fe54e37d5081c684130cba2b/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py#L53>`_.

Clearly, this helper would also be very useful for testing the various frontend
concatenation functions, such as :code:`jax.numpy.concatenate`,
:code:`numpy.concatenate`, :code:`tensorflow.concat` and :code:`torch.cat`.
We could simply copy and paste the implementation from
:mod:`ivy/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py`
into each file
:mod:`ivy/ivy_tests/test_ivy/test_frontends/test_<framework>/test_<group>.py`,
but this would result in needless duplication.
Instead, we should simply import the helper function from the ivy test file into the
frontend test file, like so
:code:`from ivy_tests.test_ivy.test_frontends.test_manipulation import _arrays_idx_n_dtypes`.

In cases where a helper function is uniquely useful for a frontend function without
being useful for an Ivy function, then it should be implemented directly in
:mod:`ivy/ivy_tests/test_ivy/test_frontends/test_<framework>/test_<group>.py`
rather than in
:mod:`ivy/ivy_tests/test_ivy/test_functional/test_core/test_<closest_relevant_group>.py`.
However, as shown above, in many cases the same helper function can be shared between
the Ivy API tests and the frontend tests,
and we should strive for as much sharing as possible to minimize the amount of code.


**Round Up**

These examples have hopefully given you a good understanding of Ivy Frontend Tests!

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy frontends discussion`_,
or reach out on `discord`_ in the `ivy frontends channel`_!
