Ivy Tests
=========

.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`hypothesis`: https://hypothesis.readthedocs.io/en/latest/
.. _`test_array_api`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/ivy_tests
.. _`test_ivy`: https://github.com/unifyai/ivy/tree/0fc4a104e19266fb4a65f5ec52308ff816e85d78/ivy_tests/test_ivy
.. _`commit`: https://github.com/unifyai/ivy/commit/8e6074419c0b6ee27c52e8563374373c8bcff30f
.. _`uploading`: https://github.com/unifyai/ivy/blob/0fc4a104e19266fb4a65f5ec52308ff816e85d78/.github/workflows/test-array-api-torch.yml#L30
.. _`downloading`: https://github.com/unifyai/ivy/blob/0fc4a104e19266fb4a65f5ec52308ff816e85d78/.github/workflows/test-array-api-torch.yml#L14
.. _`continuous integration`: https://github.com/unifyai/ivy/tree/0fc4a104e19266fb4a65f5ec52308ff816e85d78/.github/workflows
.. _`search strategies`: https://hypothesis.readthedocs.io/en/latest/data.html
.. _`test_array_function`: https://github.com/unifyai/ivy/blob/0fc4a104e19266fb4a65f5ec52308ff816e85d78/ivy_tests/test_ivy/helpers.py#L401
.. _`artifact`: https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts
.. _`ivy tests discussion`: https://github.com/unifyai/ivy/discussions/1304
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`ivy tests channel`: https://discord.com/channels/799879767196958751/982738436383445073

On top of the Array API `test suite`_, which is included as a submodule mapped to the folder :code:`test_array_api`,
there is also a collection of Ivy tests, located in subfolder `test_ivy`_.

These tests serve two purposes:

#. test functions and classes which are *not* part of the standard
#. test additional required behaviour for functions which *are* part of the standard.
   The standard only mandates a subset of required behaviour, which the Ivy functions generally extend upon.

As done in the `test suite`_, we also make use of `hypothesis`_ for performing property based testing.

Hypothesis
----------

Using pytest fixtures (such as the ones removed in this `commit`_) cause a grid search to be performed for all
combinations of parameters. This is great when we want the test to be very thorough,
but can make the entire test suite very time consuming.
Before the changes in this commit, there were 300+ separate tests being run in total,
just for this :code:`ivy.abs` function.
If we take this approach for every function, we might hit the runtime limit permitted by GitHub actions.

A more elegant and efficient solution is to use the `hypothesis`_ module,
which intelligently samples from all of the possible combinations within user-specified ranges,
rather than grid searching all of them every single time.
The intelligent sampling is possible because hypothesis enables the results of previous test runs to be cached,
and then the new samples on subsequent runs are selected intelligently,
avoiding samples which previously passed the tests, and sampling for unexplored combinations.
Combinations which are known to have failed on previous runs are also repeatedly tested for.
With the `uploading`_ and `downloading`_ of the :code:`.hypothesis` cache as an `artifact`_,
these useful properties are also true in Ivy's GitHub Action `continuous integration`_ (CI) tests.

Rather than making use of :code:`pytest.mark.parametrize`, the Ivy tests make use of hypothesis `search strategies`_.
This reference `commit`_ outlines the difference between using pytest parametrizations and hypothesis,
for :code:`ivy.abs`.
Among other changes, all :code:`pytest.skip()` calls were replaced with return statements,
as pytest skipping does not play nicely with hypothesis testing.

Data Generation
---------------

# ToDo: write guide on best practices for generating data thoroughly, with clear examples of helper functions etc.

Self-Consistent and Explicit Testing
------------------------------------

The hypothesis data generation strategies ensure that we test for arbitrary variations in the function inputs,
but this makes it difficult to manually verify ground truth results for each input variation.
Therefore, we instead opt to test for self-consistency against the same Ivy function with a NumPy backend.
This is handled by :code:`test_array_function`, which is a helper function most unit tests defer to.
This function is explained in more detail in the following sub-section.

For *primary* functions, this approach works well.
Each backend implementation generally wraps an existing backend function,
and under the hood these implementations vary substantially.
This approach then generally suffices to correctly catch bugs for most *primary* functions.

However, for *compositional* and *mixed* functions, then it's more likely that a bug could be missed.
With such functions, it's possible that the bug exists in the shared *compositional* implementation,
and then the bug would be systematic across all backends,
including the *ground truth* NumPy which the value tests for all backends compare against.

Therefore, for all *mixed* and *compositional* functions,
the test should also be appended with known inputs and known ground truth outputs,
to safeguard against this inability for :code:`test_array_function` to catch systematic errors.
These should be added using :code:`pytest.mark.parametrize`.
However, we should still also include :code:`test_array_function` in the test,
so that we can still test for arbitrary variations in the input arguments.

test_array_function
-------------------

The helper `test_array_function`_ tests that the function:

#. can handle the :code:`out` argument correctly
#. can be called as an instance method of the ivy.Array class
#. can accept ivy.Container instances in place of any arguments for *nestable* functions,
   applying the function to the leaves of the container, and returning the resultant container
#. can be called as an instance method on the ivy.Container
#. is self-consistent with the function return values when using a NumPy backend

:code:`array` in the name :code:`test_array_function` simply refers to the fact that the function in question consumes
arrays in the arguments.

So when should :code:`test_array_function` be used?

The rule is simple, if the test should not pass any arrays in the input,
then we should not use the helper :code:`test_array_function`.
For example, :code:`ivy.num_gpus` does not receive any arrays in the input,
and so we should not make us of :code:`test_array_function` in the test implementation.

**Round Up**

This should have hopefully given you a good feel for how the tests are implemented in Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy tests discussion`_,
or reach out on `discord`_ in the `ivy tests channel`_!
