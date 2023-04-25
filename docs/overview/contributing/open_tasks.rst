Open Tasks
==========

.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`open tasks channel`: https://discord.com/channels/799879767196958751/985156466963021854
.. _`issue description`: https://github.com/unifyai/ivy/issues/1526
.. _`reference API`: https://numpy.org/doc/stable/reference/routines.linalg.html
.. _`imports`: https://github.com/unifyai/ivy/blob/38dbb607334cb32eb513630c4496ad0024f80e1c/ivy/functional/frontends/numpy/__init__.py#L27

Here, we explain all tasks which are currently open for contributions from the community!

This section of the docs will be updated frequently, whereby new tasks will be added and completed tasks will be removed.
The tasks outlined here are generally broad high-level tasks, each of which is made up of many individual sub-tasks, distributed across task-specific `ToDo List Issues <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3AToDo>`_.

Please read about `ToDo List Issues <https://lets-unify.ai/docs/ivy/overview/contributing/the_basics.html#todo-list-issues>`_ in detail before continuing.
All tasks should be selected and allocated as described in the ToDo List Issues section.
We make no mention of task selection and allocation in the explanations below, which instead focus on the steps to complete only once a sub-task has been allocated to you.

The tasks currently open are:

#. Function Formatting
#. Frontend APIs
#. Ivy Experimental API

We try to explain these tasks as clearly as possible, but in cases where things are not clear, then please feel free to reach out on `discord`_ in the `open tasks channel`_!

Please always use the latest commit on GitHub when working on any of these tasks, **DO NOT** develop your code using the latest PyPI release of :code:`ivy-core`.

Function Formatting
-------------------

Currently, we have many ToDo list issues `open <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3A%22Function+Reformatting%22+label%3AToDo>`_ for a general function formatting task, which is explained below.

Each function in each submodule should be updated to follow the implementation instructions given in the :ref:`Deep Dive` section.
The updates should be applied for the:

#. ivy API
#. all backend APIs
#. container static methods
#. array instance methods
#. container instance methods
#. array operators
#. array reverse operators
#. container operators
#. container reverse operators

The :ref:`Deep Dive` is an **essential** resource for learning how each of these functions/methods should be implemented.
Before starting any contribution task, you should go through the :ref:`Deep Dive`, and familiarize yourself with the content.

At the time of writing, many of the functions are not implemented as they should be.
You will need to make changes to the current implementations, but you do not need to address *all* sections of the :ref:`Deep Dive` in detail.
Specifically, you **do not** need to address the following:

#. Implement the hypothesis testing for the function
#. Get the tests passing for your function, if they are failing before you start

However, everything else covered in the :ref:`Deep Dive` must be addressed.
Some common important tasks are:

#. Remove all :code:`lambda` and direct bindings for the backend functions (in :code:`ivy.functional.backends`), with each function instead defined using :code:`def`.
#. Implement the following if they don't exist but should do: :class:`ivy.Array` instance method, :class:`ivy.Container` static method, :class:`ivy.Container` instance method, :class:`ivy.Array` special method, :class:`ivy.Array` reverse special method, :class:`ivy.Container` special method, :class:`ivy.Container` reverse special method.
#. Make sure that the aforementioned methods are added into the correct category-specific parent class, such as :class:`ivy.ArrayWithElementwise`, :class:`ivy.ContainerWithManipulation` etc.
#. Correct all of the :ref:`Function Arguments` and the type hints for every function **and** its *relevant methods*, including those you did not implement yourself.
#. Add the correct :ref:`Docstrings` to every function **and** its *relevant methods*, including those you did not implement yourself.
#. Add thorough :ref:`Docstring Examples` for every function **and** its *relevant methods* and ensure they pass the docstring tests.

Formatting checklist
~~~~~~~~~~~~~~~~~~~~

After creating your Pull Request on github, you should then produce the checklist for the formatting task as follows: 

1. Add a comment with the following format: :code:`add_reformatting_checklist_<category_name>` on your PR, where *<category_name>* is the name of the category that the function belongs to.
   An example of this is shown below.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/open_tasks/checklist_generator.png?raw=true
   :width: 420

Using this formatting will then trigger our github automation bots to update your comment with the proper markdown text for the checklist.
These updates might take a few moments to take effect, so please be patient 🙂.

2. After adding the checklist to your PR, you should then modify this checklist with the status of each item according to the symbols(emojis) within the LEGEND section.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/open_tasks/checklist_legend.png?raw=true
   :width: 420

3. When all check items are marked as (✅, ⏩, or 🆗), you should request a review for your PR and we will start checking your implementation and marking the items as complete using the checkboxes next to them.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/open_tasks/checklist_checked.png?raw=true
   :width: 420

4. In case you are stuck or need help with one of the checklist items, please add the 🆘 symbol next to the item on the checklist, and proceed to add a comment elaborating on your point of struggle with this item.
The PR assignee will then see this comment and address your issues.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/open_tasks/checklist_SOS.png?raw=true
   :width: 420

**Notes**: 

1. It is important that the PR author is the one to add the checklist generating comment in order to ensure they will have access to edit and update it later.
2. The checklist items' statuses should be manually updated by the PR author.
   It does not automatically run any tests to update them!
3. Do not edit the checklist text, only the emoji symbols. 😅
4. Please refrain from using the checkboxes next to checklist items.


Frontend APIs
-------------

For this task, the goal will be to implement functions for each of the frontend functional APIs (see :ref:`Ivy as a Transpiler`), with frontend APIs implemented for: :code:`JAX`, :code:`NumPy`, :code:`TensorFlow` and :code:`PyTorch`.

Currently, we have many ToDo list issues `open <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3AToDo+label%3A%22JAX+Frontend%22%2C%22TensorFlow+Frontend%22%2C%22PyTorch+Frontend%22%2C%22NumPy+Frontend%22+-label%3A%22Test+Sweep%22>`_ for this task.

The general workflow for this task is:

#. Find the correct location for the function by following the *Where to place a frontend function* subsection below
#. Implement the function by following the :ref:`Ivy Frontends` guide
#. Write tests for your function by following the :ref:`Ivy Frontend Tests` guide
#. Verify that the tests for your function are passing

If you feel as though there is an ivy function :code:`ivy.<func_name>` clearly missing, which would make your frontend function much simpler to implement, then you should first do the following:

#. Create a new issue with the title :code:`ivy.<func_name>`
#. Add the labels :code:`Suggestion`, :code:`Experimental`, :code:`Ivy API` and :code:`Next Release` to it
#. Then simply leave this issue open.

At some point, a member of our team will assess whether it should be added, and if so, they will add it to another appropriate ToDo list issue (see the open task below).
   You do not need to wait for this in order to proceed.

After this, you then have two options for how to proceed:

#. Try to implement the function as a composition of currently present ivy functions, as explained in the "Temporary Compositions" sub-section of the :ref:`Ivy Frontends` guide, and add the :code:`#ToDo` comment in the implementation as explained.
   Once the PR is merged, your sub-task issue will then be closed as normal.
#. Alternatively, if you do not want to try and implement the frontend function compositionally, or if this is not feasible, then you can simply choose another frontend function to work on.
   You could also choose to work on another open task entirely at this point if you wanted to.
   For example, you might decide to wait for a member of our team to review your suggested addition :code:`ivy.<func_name>`, and potentially add this to an Ivy Experimental ToDo list issue (see the open task below).
   In either case, you should add the label "Pending other Issue" to the frontend sub-task issue, and leave it open.
   This issue will then still show up as open in the original frontend ToDo list, helpfully preventing others from working on this problematic frontend function, which depends on the unimplemented :code:`ivy.<func_name>`.
   Finally, you should add a comment to the issue with the contents: :code:`pending <issue_link>`, which links to the :code:`ivy.<func_name>` issue, making the "Pending other Issue" label more informative.

There are a few other points to take note of when working on your chosen frontend function:

#. You should only implement **one** frontend function.
#. The frontend function is framework-specific, thus it should be implemented in its respective frontend framework only.
#. Each frontend function should be tested on all backends to ensure that conversions are working correctly.
#. Type hints, docstrings and examples are not required for frontend functions.
#. Some frontend functions shown in the ToDo list issues are aliases of other functions.
   If you detect that this is the case, then you should add all aliases in your PR, with a single implementation and then simple bindings to this implementation, such as :code:`<alias_name> = <function_name>`.
   If you notice that an alias function has already been implemented and pushed, then you can simply add this one-liner binding and get this very simple PR merged.

In the case where your chosen function exists in all frameworks by default, but is not implemented in Ivy's functional API, please convert your existing GitHub issue to request for the function to be added to Ivy.
Meanwhile, you can select another frontend function to work on from the ToDo list!
If you're stuck on a function which requires complex compositions, you're allowed to reselect a function too!

Where to place a frontend function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The placement of new frontend functions for a given backend should follow the categorisation of the backend API as faithfully as possible.
In each `issue description`_, there will be a link to the relevant `reference API`_.
Check where the function you're working on is located, e.g. :code:`numpy.inner` falls under :code:`numpy.linalg`.
Then, in the Ivy source code, check :code:`ivy/functional/frontends/[backend]` for pre-existing files which best match the function's category in the backend reference API.

Taking :code:`numpy.inner` as an example, we can see that there are a few :code:`ivy/functional/frontends/numpy` sub-directories to choose from:

.. code-block:: bash
    :emphasize-lines: 4

    creation_routines
    fft
    indexing_routines
    linalg
    logic
    ma
    manipulation_routines
    mathematical_functions
    matrix
    ndarray
    random
    sorting_searching_counting
    statistics
    ufunc

There is a :code:`linalg` sub-directory, so we choose this.
Then we need to choose from the files at this hierarchy:

.. code-block:: bash
    :emphasize-lines: 3

    __init__.py
    decompositions.py
    matrix_and_vector_products.py
    matrix_eigenvalues.py
    norms_and_other_numbers.py
    solving_equations_and_inverting_matrices.py


This may require a bit of reasoning.
:code:`inner` calculates the inner product of two arrays, so :code:`matrix_and_vector_products.py` seems like the most appropriate option.
It is important to note that some functions require the :code:`np.linalg.[func]` namespace, as can gleamed from the numpy `reference API`_.
These functions are listed out under the :code:`functional/frontends/numpy/__init__.py` `imports`_.
There are some functions which have not been implemented yet, and are therefore commented out.
Once you have finished the implementation of one of these functions, uncomment it from the list.


The location of :code:`test_numpy_inner` should mirror the location of its corresponding function, this time in :code:`ivy_tests/test_ivy/test_frontends/[backend]`.

If you're unsure about where to put the function you're working on, explore the content of these files to see if you can find a similar function.
In :code:`matrix_and_vector_products.py`, we can see other functions such as :code:`outer` that are similar to :code:`inner`.
This is confirmation that we've found the correct place!
If many of the files are empty and you're unsure where to place your function, feel free to ask the member of the Ivy team reviewing your PR.


Ivy Experimental API
--------------------

The goal of this task is to add functions to the existing Ivy API which would help with the implementation for many of the functions in the frontend.

Your task is to implement these functions in Ivy, along with their Implementation in the respective backends which are :code:`Jax`, :code:`PyTorch`, :code:`TensorFlow` and :code:`NumPy`.
You must also implement tests for these functions.

There is only one central ToDo list `issue <https://github.com/unifyai/ivy/issues/3856>`_ for this task.

A general workflow for these tasks would be:

#. Implement the functions in each of the backend files :mod:`ivy/functional/backends/backend_name/experimental/[relevant_submodule].py`, sometimes as a composition if the respective backends do not behave in a similar way.
   You may also use submodule-specific helper functions to recreate the behaviour.
   Refer the `Backend API Guide <https://lets-unify.ai/docs/ivy/deep_dive/navigating_the_code.html#backend-api>`_ on how this can be done.
#. Implement the functions in :mod:`ivy/functional/ivy/experimental/[relevant_submodule].py` simply deferring to their backend-specific implementation.
   Refer the `Ivy API Guide <https://lets-unify.ai/docs/ivy/deep_dive/navigating_the_code.html#ivy-api>`_ to get a clearer picture of how this must be done.
#. Implement the container instance method in :mod:`ivy/container/experimental/[relevant_submodule].py` and the array instance method 
   in :mod:`ivy/array/experimental/[relevant_submodule].py`
#. Write tests for the function using the :ref:`Ivy Tests` guide, and make sure they are passing.

A few points to keep in mind while doing this:

#. Make sure all the positional arguments are positional-only and optional arguments are keyword-only.
#. In case some tests require function-specific parameters, you can create composite hypothesis strategies using the :code:`draw` function in the hypothesis library.

If you’re stuck on a function which requires complex compositions, feel free to reselect a function 🙂.


**Round Up**

This should have hopefully given you a good understanding of the basics for contributing.

If you have any questions, please feel free to reach out on `discord`_ in the `open tasks channel`_!
