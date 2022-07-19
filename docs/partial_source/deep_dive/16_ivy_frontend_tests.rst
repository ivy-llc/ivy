Ivy Frontend tests
====================

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html

Ivy Frontends are the framework-specific frontend functional APIs defined to help with code transpilations as explained `here`_. 
This in turn makes it necessary to thoroughly test each backend implementation unlike the `test_ivy`_.


As done in the `test suite`_, we also make use of `hypothesis`_ for performing property based testing.


test_frontend_helper
--------------------

The helper `test_frontend_function`_ generates the funtion's ground truth version of the specified framework

takes data, calls ground truth version of function and gets results. recreate behaviour of function which already exisits
and assert that our version of function gives same results