Open Tasks
==========

.. _`open tasks discussion`: https://github.com/unifyai/ivy/discussions/1403
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`open tasks channel`: https://discord.com/channels/799879767196958751/985156466963021854

Here, we explain all tasks which are currently open for
contributions from the community!

This section of the docs will be updated frequently, whereby new tasks will be added and
completed tasks will be removed. The tasks outlined here are generally broad high-level
tasks, each of which is made up of many individual sub-tasks,
distributed across task-specific
`ToDo list issues <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3AToDo>`_.

The tasks currently open are:

#. Function Formatting
#. Frontend APIs

We try to explain these tasks are clearly as possible, but in cases where things are not
clear, then please feel free to engage with the `open tasks discussion`_,
or reach out on `discord`_ in the `open tasks channel`_!

Please always use the latest commit on GitHub when working on any of these tasks,
**DO NOT** develop your code using the latest PyPI release of :code:`ivy-core`.

Function Formatting
-------------------

Currently, we have many ToDo list issues
`open <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3A%22Function+Reformatting%22+label%3AToDo>`_
for a general function reformatting task,
which is explained below.

Each function in each submodule should be updated to follow the implementation
instructions given in the :ref:`Deep Dive` section.
The updates should be applied for both the Ivy API and Backend API,
as explained in the :ref:`Navigating the Code` section.

The :ref:`Deep Dive` is the best general resource for learning in detail how functions
should be implemented.
At the time of writing, many of the functions are not implemented as they should be.
You will need to make changes to the current implementations,
but you do not need to address *all* sections of the :ref:`Deep Dive` in detail.
Specifically, you should make the following changes, where appropriate:

#. remove all :code:`lambda` and direct bindings for backend functions,
   with each function instead defined using :code:`def`.
#. update the :ref:`Function Arguments` and the type hints.
#. add the correct :ref:`Docstrings`
#. add thorough :ref:`Docstring Examples`

Frontend APIs
-------------

For this task, the goal will be to implement functions for each of the
frontend functional APIs (see :ref:`Ivy as a Transpiler`),
with frontend APIs implemented for:
:code:`JAX`, :code:`MXNet`, :code:`NumPy`, :code:`TensorFlow` and :code:`PyTorch`.

This task is not *quite* ready to be engaged with yet,
more details coming in the next few weeks! 🗓️

**Round Up**

This should have hopefully given you a good understanding of the basics for contributing.

If you're ever unsure of how best to proceed,
please feel free to engage with the `open tasks discussion`_,
or reach out on `discord`_ in the `open tasks channel`_!
