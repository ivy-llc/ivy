Ivy Tests
=========

.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`hypothesis`: https://hypothesis.readthedocs.io/en/latest/

On top of the tests which we have taken directly from the test suite for the Array API Standard,
we also add our own tests.

This is for two reasons:

#. Many functions in Ivy are not present in the standard, and they need to be tested somewhere
#. The standard only mandates a subset of required behaviour. Almost all Ivy functions which are in the standard have additional required behaviour, which must also be tested

As is the case for the `test suite`_, we also make use of `hypothesis`_ for performing property based testing.

# ToDo: complete this section