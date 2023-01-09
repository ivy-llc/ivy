Ivy Frontend Tests
==================

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html
.. _`ivy frontends tests channel`: https://discord.com/channels/799879767196958751/1028267758028337193
.. _`ivy frontends tests forum`: https://discord.com/channels/799879767196958751/1028297887605587998
.. _`test ivy`: https://github.com/unifyai/ivy/tree/db9a22d96efd3820fb289e9997eb41dda6570868/ivy_tests/test_ivy
.. _`test_frontend_function`: https://github.com/unifyai/ivy/blob/591ac37a664ebdf2ca50a5b0751a3a54ee9d5934/ivy_tests/test_ivy/helpers.py#L1047
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`Function Wrapping`: https://lets-unify.ai/ivy/deep_dive/3_function_wrapping.html
.. _`open task`: https://lets-unify.ai/ivy/contributing/open_tasks.html#open-tasks
.. _`Ivy Tests`: https://lets-unify.ai/ivy/deep_dive/ivy_tests.html
.. _`Function Testing Helpers`: https://github.com/unifyai/ivy/blob/bf0becd459004ae6cffeb3c38c02c94eab5b7721/ivy_tests/test_ivy/helpers/function_testing.py
.. _`CI Pipeline`: https://lets-unify.ai/ivy/deep_dive/continuous_integration.html#ci-pipeline
.. _`setting up`: https://lets-unify.ai/ivy/contributing/setting_up.html#setting-up-testing


Introduction
------------

Just like the backend functional API, our frontend functional API has a collection of Ivy tests located in subfolder `test ivy`_.
In this section of the deep dive we are going to jump into Ivy Frontend Tests!

**Writing Ivy Frontend Tests**

The Ivy tests in this section make use of hypothesis for performing property based testing which is documented in detail in the Ivy Tests section of the Deep Dive.
We assume knowledge of hypothesis data generation strategies and how to implement them for testing.

**Ivy Decorators**

Ivy provides test decorators for frontend tests to make it easier and more maintainable, currently there are two:

* :func:`@handle_frontend_test` a decorator which is used to test frontend functions, for example :func:`np.zeros` and :func:`tensorflow.tan`.
* :func:`@handle_frontend_method` a decorator which is used to test frontend methods and special methods, for example :func:`torch.Tensor.add` and :func:`numpy.ndarray.__add__`.

**Important Helper Functions**

* :func:`helpers.test_frontend_function` helper function that is designed to do the heavy lifting and make testing Ivy Frontends easy!
  One of the many `Function Testing Helpers`_.
  It is used to test a frontend function for the current backend by comparing the result with the function in the associated framework.

* :func:`helpers.get_dtypes` helper function that returns either a full list of data types or a single data type, we should **always** be using `helpers.get_dtypes` to sample data types.

* :func:`helpers.dtype_and_values` is a convenience function that allows you to generate arrays of any dimension and their associated data types, returned as :code:`([dtypes], [np.array])`.

* :func:`helpers.get_shape` is a convenience function that allows you to generate an array shape of type :code:`tuple`

* :func:`np_frontend_helpers.where` a generation strategy to generate values for NumPy's optional :code:`where` argument.

* :func:`np_frontend_helpers.test_frontend_function` behaves identical to :func:`helpers.test_frontend_function` but handles NumPy's optional :code:`where` argument

**Useful Notes**

* We should always ensure that our data type generation is complete.
  Generating float data types only for a function that accepts all numeric data types is not complete, a complete set would include **all** numeric data types.

* The :func:`test_frontend_function` argument :code:`fn_tree` refers to the frontend function's reference in its native namespace not just the function name.
  For example :func:`lax.tan` is needed for some functions in Jax, :func:`nn.functional.relu` is needed for some functions in PyTorch etc.

To get a better understanding for writing frontend tests lets run through some examples!

Frontend Test Examples
-----------------------

Before you begin writing a frontend test, make sure you are placing it in the correct location.
See the 'Where to place a frontend function' sub-section of the frontend APIs `open task`_ for more details.

ivy.tan()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @handle_frontend_test(
            fn_tree="jax.lax.tan",
            dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
            )
    def test_jax_lax_tan(
            *,
            dtype_and_x,
            as_variable,
            num_positional_args,
            native_array,
            on_device,
            fn_tree,
            frontend,
    ):
            input_dtype, x = dtype_and_x
            helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            x=x[0],
    )

* As you can see we generate almost everything we need to test a frontend function within the :code:`@handle_frontend_test` decorator.
* We set :code:`fn_tree` to :code:`jax.lax.tan` which is the path to the function in the Jax namespace.
* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid :code:`float` data types specifically for Jax.
* We do not generate any values for :code:`as_variable`, :code:`native_array`, :code:`frontend`, :code:`num_positional_args`, :code:`on_device`, these values are generated by :func:`handle_frontend_test` and are only passed as an argument to :func:`test_jax_lax_tan`.
* We unpack the :code:`dtype_and_x` to :code:`input_dtype` and :code:`x`.
* We then pass the generated values to :code:`helpers.test_frontend_function` which tests the frontend function.
* :func:`jax.lax.tan` does not support :code:`out` arguments so we set :code:`with_out` to :code:`False`.
* One last important note is that all helper functions are designed to take keyword arguments only.

**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_mathematical_functions/test_np_trigonometric_functions.py
    @handle_frontend_test(
        fn_tree="numpy.tan",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric")
        ),
        where=np_frontend_helpers.where(),
    )
    def test_numpy_tan(
        dtype_and_x,
        where,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        frontend,
        fn_tree,
        on_device,
    ):
        input_dtype, x = dtype_and_x
        dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
            dtypes=input_dtype,
            get_dtypes_kind="numeric",
        )
        where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
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
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            x=x[0],
            out=None,
            where=where,
            casting=casting,
            order="K",
            dtype=dtype,
            subok=True,
        )
    

* We set :code:`fn_tree` to :code:`numpy.tan` which is the path to the function in the NumPy namespace.
* Here we use :code:`helpers.get_dtypes("numeric")` to generate :code:`available_dtypes`, these are valid :code:`numeric` data types specifically for NumPy.
* NumPy has an optional argument :code:`where` which is generated using :func:`np_frontend_helpers.where`.
* :func:`numpy.tan` supports :code:`out` arguments so we set generate values for :code:`with_out`.
* Using :func:`np_frontend_helpers.handle_where_and_array_bools` we do some processing on the generated :code:`where` value.
* Instead of :func:`helpers.test_frontend_function` we use :func:`np_frontend_helpers.test_frontend_function` which behaves the same but has some extra code to handle the :code:`where` argument.
* :code:`casting`, :code:`order`, :code:`subok` and are other other optional arguments for :func:`numpy.tan`.

**TensorFlow**

.. code-block:: python
        
        # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_math.py
        # tan
        @handle_frontend_test(
            fn_tree="tensorflow.math.tan",
            dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
        )
        def test_tensorflow_tan(
            *,
            dtype_and_x,
            num_positional_args,
            as_variable,
            native_array,
            frontend,
            fn_tree,
            on_device,
        ):
            input_dtype, x = dtype_and_x
            helpers.test_frontend_function(
                input_dtypes=input_dtype,
                as_variable_flags=as_variable,
                with_out=False,
                num_positional_args=num_positional_args,
                native_array_flags=native_array,
                frontend=frontend,
                fn_tree=fn_tree,
                on_device=on_device,
                x=x[0],
            )
            

* We set :code:`fn_tree` to :code:`tensorflow.math.tan` which is the path to the function in the TensorFlow namespace.
* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for the function.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_pointwise_ops.py
    # tan
    @handle_frontend_test(
    fn_tree="torch.tan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
            ),
    )
        def test_torch_tan(
            *,
            dtype_and_x,
            as_variable,
            with_out,
            num_positional_args,
            native_array,
            on_device,
            fn_tree,
            frontend,
        ):
            input_dtype, x = dtype_and_x
            helpers.test_frontend_function(
                input_dtypes=input_dtype,
                as_variable_flags=as_variable,
                with_out=with_out,
                num_positional_args=num_positional_args,
                native_array_flags=native_array,
                frontend=frontend,
                fn_tree=fn_tree,
                on_device=on_device,
                input=x[0],
            )

* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for the function.

ivy.full()
^^^^^^^^^^

Here we are going to look at an example of a function that does not consume an :code:`array`.
This is the creation function :func:`full`, which takes an array shape as an argument to create an array and filled with elements of a given value.
This function requires us to create extra functions for generating :code:`shape` and :code:`fill value`, these use the :code:`shared` hypothesis strategy.


**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))


    @handle_frontend_test(
        fn_tree="jax.lax.full",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
    )
    def test_jax_lax_full(
        *,
        shape,
        fill_value,
        dtypes,
        as_variable,
        num_positional_args,
        native_array,
        on_device,
        fn_tree,
        frontend,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            shape=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
        )


* The custom function we use is :code:`_fill_value` which generates a :code:`fill_value` to use for the :code:`fill_value` argument but handles the complications of :code:`int` and :code:`uint` types correctly.
* We use the helper function :func:`helpers.get_shape` to generate :code:`shape`.
* We use :code:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Jax.
  This is used to specify the data type of the output array.
* :func:`full` does not consume :code:`array`, we set :code:`as_variable_flags`, :code:`native_array_flags` to :code:`[False]` and :code:`with_out` :code:`False`.


**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/creation_routines/test_from_shape_or_value.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_frontend_test(
        fn_tree="numpy.full",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        input_fill_dtype=_input_fill_and_dtype(),
    )
    def test_numpy_full(
        shape,
        input_fill_dtype,
        as_variable,
        num_positional_args,
        native_array,
        frontend,
        fn_tree,
        on_device,
    ):
        input_dtype, x, fill, dtype_to_cast = input_fill_dtype
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            shape=shape,
            fill_value=fill,
            dtype=dtype_to_cast,
        )

* We use :func:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for NumPy.
* :func:`numpy.full` does not have a :code:`where` argument so we can use :func:`helpers.test_frontend_function`

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tf_functions.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_frontend_test(
        fn_tree="tensorflow.raw_ops.Fill",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            min_dim_size=1,
        ),
        fill_value=_fill_value(),
        dtypes=_dtypes(),
    )
    def test_tensorflow_Fill(
        *,
        shape,
        fill_value,
        dtypes,
        as_variable,
        num_positional_args,
        native_array,
        frontend,
        fn_tree,
        on_device,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            rtol=1e-05,
            dims=shape,
            value=fill_value,
        )


* We use :func:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for this function.
* Tensorflow's version of :func:`full` is named :func:`fill` therefore we specify the :code:`fn_tree` argument to be :code:`"fill"`
* When running the test there where some small discrepancies between the values so we can use :code:`rtol` to specify the relative tolerance.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_creation_ops.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))


    @st.composite
    def _requires_grad(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
            return draw(st.just(False))
        else:
            return draw(st.booleans())


    @handle_frontend_test(
        fn_tree="torch.full",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
        requires_grad=_requires_grad(),
    )
    def test_torch_full(
        shape,
        fill_value,
        dtypes,
        requires_grad,
        on_device,
        num_positional_args,
        frontend,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=[False],
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=[False],
            frontend=frontend,
            fn_tree=fn_tree,
            size=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
            device=device,
            requires_grad=requires_grad,
        )

* Here we created another extra function, :code:`_requires_grad()`, to accommodate the :code:`requires_grad` argument.
  This is because when the dtype is an integer or unsigned integer the :code:`requires_grad` argument is not supported.
* We use :code:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Torch.
* :func:`torch.full` supports :code:`out` so we generate :code:`with_out`.

Alias functions
^^^^^^^^^^^^^^^
Let's take a quick walkthrough on testing the function alias as we know that such functions have the same behavior as original functions.
Taking an example of :func:`torch_frontend.greater` has an alias function :func:`torch_frontend.gt` which we need to make sure that it is working same as the targeted framework function :func:`torch.greater` and :func:`torch.gt`.

Code example for alias function:

.. code-block:: python

    # in ivy/functional/frontends/torch/comparison_ops.py
    @to_ivy_arrays_and_back
    def greater(input, other, *, out=None):
        input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
        return ivy.greater(input, other, out=out


    gt = greater

* As you can see the :func:`torch_frontend.gt` is an alias to :func:`torch_frontend.greater` and below is how we update the unit test of :func:`torch_frontend.greater` to test the alias function as well.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_comparison_ops.py
    @handle_frontend_test(
        fn_tree="torch.greater",
        dtype_and_inputs=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=2,
            allow_inf=False,
            shared_dtype=True,
        ),
    )
    def test_torch_greater(
        *,
        dtype_and_inputs,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        on_device,
        fn_tree,
        frontend,
    ):
        input_dtype, inputs = dtype_and_inputs
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            all_aliases=["gt"],
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            input=inputs[0],
            other=inputs[1],
        )

* We added a list of all aliases to the :code:`greater` function with a full namespace path such that when we are testing the original function we will test for the alias as well.
* During the frontend implementation, if a new alias is introduced you only need to go to the test function of the original frontend function and add that alias to :code:`all_aliases` argument in the :func:`test_frontend_function` helper with its full namespace.

Frontend Instance Method Tests
------------------------------

The frontend instance method tests are similar to the frontend function test, but instead of testing the function directly we test the instance method of the frontend class.
major difference is that we have more flags to pass now, most initialization functions take an array as an input. also some methods may take an array as input,
for example, :code:`ndarray.__add__` would expect an array as input, despite the :code:`self.array`. and to make our test is **complete** we need to generate seperate flags for each.

**Important Helper Functions**

:func:`@handle_frontend_method` requires 3 keyword only parameters:
    - :code:`class_tree` A full path to the array class in **Ivy** namespace. 
    - :code:`init_tree` A full path to initialization function.
    - :code:`method_name` The name of the method to test. 

:func:`helpers.test_frontend_method` is used to test frontend instance methods. It is used in the same way as :func:`helpers.test_frontend_function`.


Frontend Instance Method Test Examples
--------------------------------------

ivy.add()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_devicearray.py
    @handle_frontend_method(
        class_tree="ivy.functional.frontends.jax.DeviceArray"
        init_tree="jax.numpy.array",
        method_name="__add__",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric", full=True),
            num_arrays=2,
            shared_dtype=True,
        ),
    )
    def test_jax_instance_add(
        dtype_and_x,
        init_num_positional_args: pf.NumPositionalArgFn,
        method_num_positional_args: pf.NumPositionalArgMethod,
        as_variable: pf.AsVariableFlags,
        native_array: pf.NativeArrayFlags,
        frontend,
        frontend_method_data,
       ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_method(
            init_input_dtypes=input_dtype,
            init_as_variable_flags=as_variable,
            init_num_positional_args=init_num_positional_args,
            init_native_array_flags=native_array,
            init_all_as_kwargs_np={
                "object": x[0],
            },
            method_input_dtypes=input_dtype,
            method_as_variable_flags=as_variable,
            method_num_positional_args=method_num_positional_args,
            method_native_array_flags=native_array,
            method_all_as_kwargs_np={
                "other": x[1],
            },
            frontend=frontend,
            frontend_method_data=frontend_method_data,
        )

* We use :func:`test_frontend_method` to test the instance method.
* We specify the :code:`class_tree` to be :meth:`ivy.functional.frontends.jax.DeviceArray` which is the path to the class in ivy namespace.
* We specify the function that is used to initialize the array, for jax, we use :code:`jax.numpy.array` to create a :code:`DeviceArray`.
* We specify the :code:`method_name` to be :meth:`__add__` which is the path to the method in the frontend class.
* We must tell the decorator which flags to generate using type hints, as we don't want to rely on the name of the parameter only, we use the type hints
to tell the decorator that we should generate native array flags for :code:`init_as_variable_flags` by type hinting it with :code:`pf.NativeArrayFlags`.
    
**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_ndarray.py
    @handle_frontend_method(
        class_tree="numpy.ndarray"
        init_tree="numpy.array",
        method_name="__add__",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
        ),
    )
    def test_numpy_instance_add__(
        dtype_and_x,
        as_variable: pf.AsVariableFlags,
        native_array: pf.NativeArrayFlags,
        init_num_positional_args: pf.NumPositionalArgFn,
        method_num_positional_args: pf.NumPositionalArgMethod,
        frontend,
        frontend_method_data,
    ):
        input_dtype, xs = dtype_and_x
        helpers.test_frontend_method(
            init_input_dtypes=input_dtype,
            init_as_variable_flags=as_variable,
            init_num_positional_args=init_num_positional_args,
            init_native_array_flags=native_array,
            init_all_as_kwargs_np={
                "object": xs[0],
            },
            method_input_dtypes=input_dtype,
            method_as_variable_flags=as_variable,
            method_native_array_flags=native_array,
            method_num_positional_args=method_num_positional_args,
            method_all_as_kwargs_np={
                "value": xs[1],
            },
            frontend=frontend,
            frontend_method_data=frontend_method_data,
        )

* We specify the :code:`class_tree` to be :meth:`ivy.functional.frontends.numpy.array` which is the path to the class in ivy namespace.
* We specify the function that is used to initialize the array, for jax, we use :code:`numpy.array` to create a :code:`numpy.ndarray`.
* We specify the :code:`method_name` to be :meth:`__add__` which is the path to the method in the frontend class.

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tensor.py
    @handle_frontend_method(
        class_tree="tensorflow.EagerTensor"
        init_tree="tensorflow.constant",
        method_name="__add__",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=2,
            shared_dtype=True,
        ),
    )
    def test_tensorflow_instance_add(
        dtype_and_x,
        as_variable: pf.AsVariableFlags,
        native_array: pf.NativeArrayFlags,
        init_num_positional_args: pf.NumPositionalArgFn,
        method_num_positional_args: pf.NumPositionalArgMethod,
        frontend,
        frontend_method_data,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_method(
            init_input_dtypes=input_dtype,
            init_as_variable_flags=as_variable,
            init_num_positional_args=init_num_positional_args,
            init_native_array_flags=native_array,
            init_all_as_kwargs_np={
                "data": x[0],
            },
            method_input_dtypes=input_dtype,
            method_as_variable_flags=as_variable,
            method_num_positional_args=method_num_positional_args,
            method_native_array_flags=native_array,
            method_all_as_kwargs_np={
                "y": x[1],
            },
            frontend=frontend,
            frontend_method_data=frontend_method_data,
        )


* We specify the function that is used to initialize the array, for TensorFlow, we use :code:`tensorflow.constant` to create a :code:`tensorflow.EagerTensor`.
* We specify the :code:`method_tree` to be :meth:`tensorflow.EagerTensor.__add__` which is the path to the method in the frontend class.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_tensor.py
    @handle_frontend_method(
        class_tree="torch.Tensor"
        init_tree="tensor.tensor",
        method_name="add",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=2,
            min_value=-1e04,
            max_value=1e04,
            allow_inf=False,
        ),
        alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
    )
    def test_torch_instance_add(
        dtype_and_x,
        alpha,
        init_num_positional_args: pf.NumPositionalArgFn,
        method_num_positional_args: pf.NumPositionalArgMethod,
        as_variable: pf.AsVariableFlags,
        native_array: pf.NativeArrayFlags,
        frontend,
        frontend_method_data,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_method(
            init_input_dtypes=input_dtype,
            init_as_variable_flags=as_variable,
            init_num_positional_args=init_num_positional_args,
            init_native_array_flags=native_array,
            init_all_as_kwargs_np={
                "data": x[0],
            },
            method_input_dtypes=input_dtype,
            method_as_variable_flags=as_variable,
            method_num_positional_args=method_num_positional_args,
            method_native_array_flags=native_array,
            method_all_as_kwargs_np={
                "other": x[1],
                "alpha": alpha,
            },
            frontend=frontend,
            frontend_method_data=frontend_method_data,
        )

* We specify the function that is used to initialize the array, for PyTorch, we use :code:`torch.tensor` to create a :code:`torch.Tensor`.
* We specify the :code:`method_tree` to be :meth:`torch.Tensor.__add__` which is the path to the method in the frontend class.

Hypothesis Helpers
------------------

Naturally, many of the functions in the various frontend APIs are very similar to many of the functions in the Ivy API.
Therefore, the unit tests will follow very similar structures with regards to the data generated for testing.
There are many data generation helper functions defined in the Ivy API test files, such as :func:`_arrays_idx_n_dtypes` defined in :mod:`ivy/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py`.
This helper generates: a set of concatenation-compatible arrays, the index for the concatenation, and the data types of each array.
Not surprisingly, this helper is used for testing :func:`ivy.concat`, as shown `here <https://github.com/unifyai/ivy/blob/86287f4e45bbe581fe54e37d5081c684130cba2b/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py#L53>`_.

Clearly, this helper would also be very useful for testing the various frontend concatenation functions, such as :code:`jax.numpy.concatenate`, :code:`numpy.concatenate`, :code:`tensorflow.concat` and :code:`torch.cat`.
We could simply copy and paste the implementation from :mod:`/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py` into each file :mod:`/ivy_tests/test_ivy/test_frontends/test_<framework>/test_<group>.py`, but this would result in needless duplication.
Instead, we should simply import the helper function from the ivy test file into the frontend test file, like so :code:`from ivy_tests.test_ivy.test_frontends.test_manipulation import _arrays_idx_n_dtypes`.

In cases where a helper function is uniquely useful for a frontend function without being useful for an Ivy function, then it should be implemented directly in :mod:`/ivy_tests/test_ivy/test_frontends/test_<framework>/test_<group>.py` rather than in :mod:`/ivy_tests/test_ivy/test_functional/test_core/test_<closest_relevant_group>.py`.
However, as shown above, in many cases the same helper function can be shared between the Ivy API tests and the frontend tests, and we should strive for as much sharing as possible to minimize the amount of code.

**Running Ivy Frontend Tests**

The CI Pipeline runs the entire collection of Frontend Tests for the frontend that is being updated on every push to the repo.

You will need to make sure the Frontend Test is passing for each Ivy Frontend function you introduce/modify.
If a test fails on the CI, you can see details about the failure under `Details -> Run Frontend Tests` as shown in `CI Pipeline`_.

You can also run the tests locally before making a PR. See the relevant `setting up`_ section for instructions on how to do so.


**Round Up**

This should have hopefully given you a good understanding of Ivy Frontend Tests!

If you have any questions, please feel free to reach out on `discord`_ in the `ivy frontends tests channel`_ or in the `ivy frontends tests forum`_!
