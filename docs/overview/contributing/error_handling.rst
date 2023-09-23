Error Handling
==============

.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`pycharm channel`: https://discord.com/channels/799879767196958751/942114831039856730
.. _`docker channel`: https://discord.com/channels/799879767196958751/942114744691740772
.. _`pre-commit channel`: https://discord.com/channels/799879767196958751/982725464110034944
.. _`pip packages channel`: https://discord.com/channels/799879767196958751/942114789642080317
.. _`ivy tests channel`: https://discord.com/channels/799879767196958751/982738436383445073

This section, "Error Handling" aims to assist you in navigating through some common errors you might encounter while working with the Ivy's Functional API. We'll go through some common errors which you might encounter while working as a contributor or a developer.

#. This is the case where we pass in a dtype to `torch` which is not actually supported by the torch's native framework itself. The function which was

   .. code-block:: python

        E       RuntimeError: "logaddexp2_cpu" not implemented for 'Half'
        E       Falsifying example: test_logaddexp2(
        E           backend_fw='torch',
        E           on_device='cpu',
        E           dtype_and_x=(['float16', 'float16'],
        E            [array([-1.], dtype=float16), array([-1.], dtype=float16)]),
        E           test_flags=FunctionTestFlags(
        E               ground_truth_backend='tensorflow',
        E               num_positional_args=2,
        E               with_out=False,
        E               instance_method=False,
        E               test_gradients=False,
        E               test_compile=None,
        E               as_variable=[False],
        E               native_arrays=[False],
        E               container=[False],
        E           ),
        E           fn_name='logaddexp2',
        E       )
        E
        E       You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXicY2BkAAMoBaaR2WAAAACVAAY=') as a decorator on your test case

#. This is the case where the value from the ground-truth backend(tensorflow) does not match the value of the backend(jax) we are testing for this case.

   .. code-block:: python

        E       AssertionError:  the results from backend jax and ground truth framework tensorflow do not match
        E        0.25830078125!=0.258544921875
        E
        E
        E       Falsifying example: test_acosh(
        E           backend_fw='jax',
        E           on_device='cpu',
        E           dtype_and_x=(['float16'], [array(4., dtype=float16)]),
        E           test_flags=FunctionTestFlags(
        E               ground_truth_backend='tensorflow',
        E               num_positional_args=1,
        E               with_out=False,
        E               instance_method=False,
        E               test_gradients=True,
        E               test_compile=None,
        E               as_variable=[False],
        E               native_arrays=[False],
        E               container=[False],
        E           ),
        E           fn_name='acosh',
        E       )
        E
        E       You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXicY2BAABYQwQgiAABDAAY=') as a decorator on your test case

#. This is a similar assertion as stated in point 2 but with torch and ground-truth tensorflow not matching but the matrices are quite different so there should be an issue in the backends rather than a numerical instability here:

   .. code-block:: python

        E       AssertionError:  the results from backend torch and ground truth framework tensorflow do not match
        E        [[1.41421356 1.41421356 1.41421356]
        E        [1.41421356 1.41421356 1.41421356]
        E        [1.41421356        inf 1.41421356]]!=[[1.41421356e+000 1.41421356e+000 1.41421356e+000]
        E        [1.41421356e+000 1.41421356e+000 1.41421356e+000]
        E        [1.41421356e+000 1.34078079e+154 1.41421356e+000]]
        E
        E
        E       Falsifying example: test_abs(
        E           backend_fw='torch',
        E           on_device='cpu',
        E           dtype_and_x=(['complex128'],
        E            [array([[-1.-1.00000000e+000j, -1.-1.00000000e+000j, -1.-1.00000000e+000j],
        E                    [-1.-1.00000000e+000j, -1.-1.00000000e+000j, -1.-1.00000000e+000j],
        E                    [-1.-1.00000000e+000j, -1.-1.34078079e+154j, -1.-1.00000000e+000j]])]),
        E           fn_name='abs',
        E           test_flags=FunctionTestFlags(
        E               ground_truth_backend='tensorflow',
        E               num_positional_args=1,
        E               with_out=False,
        E               instance_method=False,
        E               test_gradients=False,
        E               test_compile=None,
        E               as_variable=[False],
        E               native_arrays=[False],
        E               container=[False],
        E           ),
        E       )
        E
        E       You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXicY2ZkYAIiBiBgZIAAxqHEXsAAB7jUQAAAMtEAzQ==') as a decorator on your test case


**Note**

This section is specifically targeted towards dealing with the Ivy Functional API and the Ivy Experimental API.

**Round Up**

This should have hopefully given you an understanding of how to deal with common errors while working with the the functional API.

If you have any questions, please feel free to reach out on `discord`_  in the `ivy tests channel`_, `pycharm channel`_, `docker channel`_, `pre-commit channel`_, `pip packages channel`_ depending on the question!

