Formatting
==========

.. _`flake8`: https://flake8.pycqa.org/en/latest/index.html
.. _`black`: https://black.readthedocs.io/en/stable/index.html
.. _`formatting thread`: https://discord.com/channels/799879767196958751/1190247322626572408
.. _`discord`: https://discord.gg/sXyFF8tDtm

Currently, Ivy follows the `black`_ code style, and `flake8`_ formatter in order to ensure that our code is consistent,
readable, and bug free. This deep-dive will explain how to use these tools to ensure that your code is formatted
correctly.

Please ensure to conform to the formatting rules before submitting a pull request. You are encouraged to take a look
these coding style guides before you start contributing to Ivy.

Lint Checks
-----------

In addition to `black`_ and `flake8`_, Ivy uses other linters to help automate the formatting process, especially for
issues `flake8`_ detects but doesn't fix automatically. In addition to that, we validate docstring as part of our
linting process. You can learn more about our docstring formatting in the `Docstrings <docstrings.rst>`_ section.

We use the following linters:

* `black`_
* `flake8`_
* `autoflake <https://github.com/PyCQA/autoflake>`_
* `docformatter <https://github.com/PyCQA/docformatter>`_
* `pydocstyle <https://github.com/pycqa/pydocstyle>`_
* `ivy-lint <https://github.com/unifyai/lint-hook>`_

You can also take a look at our configuration for linting in `setup.cfg <https://github.com/unifyai/ivy/blob/main/setup.cfg>`_
file.

Setup Formatting Locally
------------------------

Pre-commit
~~~~~~~~~~

To centralize the formatting process, we use `pre-commit <https://pre-commit.com/>`_. This tool allows us to run all
the checks written in the `.pre-commit-config.yaml <https://github.com/unifyai/ivy/blob/main/.pre-commit-config.yaml>`_
file.

Pre-commit can run alone or as a git hook. To install it, you can run the following command:

.. code-block:: bash

    pip install pre-commit

Once you have installed pre-commit, you can install the git hook by running the following command:

.. code-block:: bash

    pre-commit install

This will install the git hook and will run the checks before you commit your code. If you want to run the checks
manually, you can run the following command:

.. code-block:: bash

    pre-commit run --all-files

This will run all the required checks and will show you the output of each check.

Also when you make a commit, pre-commit will run the required checks and will show you the output of each check. If
there are any errors, it will not allow you to commit your code. You can fix the errors and commit again.

You should expect to see something similar to the following output when you run the checks:

.. code-block:: text

    [INFO] Stashing unstaged files to ~/.cache/pre-commit/patch1687898304-8072.
    black....................................................................Passed
    autoflake................................................................Passed
    flake8...................................................................Passed
    docformatter.............................................................Passed
    pydocstyle...............................................................Passed
    ivy-lint.................................................................Passed
    [INFO] Restored changes from ~/.cache/pre-commit/patch1687898304-8072.
    [formatting-docs 3516aed563] Test commit
    1 file changed, 1 insertion(+)

If something goes wrong, you will see the following output:

.. code-block:: text

    [INFO] Stashing unstaged files to ~/.cache/pre-commit/patch1687898304-8072.
    black....................................................................Failed
    - hook id: black
    - files were modified by this hook

    reformatted ivy/stateful/activations.py

    All done! ✨ 🍰 ✨
    1 file reformatted.

    autoflake................................................................Passed
    flake8...................................................................Passed
    docformatter.............................................................Passed
    pydocstyle...............................................................Passed
    ivy-lint.................................................................Passed
    [INFO] Restored changes from ~/.cache/pre-commit/patch1687898304-8072.

You will notice that some files have changed if you checked ``git status``, you'll need to add them and commit again.

VS Code
~~~~~~~

There are some helpful extensions for VS Code that can detect and format your code according to our style guide. Here
is the list of extensions that we recommend:

* `Black Formatter <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_
* `Flake8 Extension <https://marketplace.visualstudio.com/items?itemName=ms-python.flake8>`_

PyCharm
~~~~~~~

Unfortunately, PyCharm doesn't have formatting extensions like VS Code. We don't have specific instructions for PyCharm
but you can use the following links to set up the formatting:

* `Akshay Jain's article on Pycharm + Black with Formatting on Auto-save
  <https://akshay-jain.medium.com/pycharm-black-with-formatting-on-auto-save-4797972cf5de>`_

Common Issues with Pre-Commit
-----------------------------

As the pre-commit hook runs before each commit, when it fails it provides an error message that's readable on terminals
but not on IDE GUIs. So you might see a cryptic error message like one of the following:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/deep_dive/formatting/vscode_error.png?raw=true
   :alt: git commit error in VS Code

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/deep_dive/formatting/pycharm_error.png?raw=true
   :alt: git commit error in PyCharm

We recommend you commit your code from the terminal when you contribute to Ivy. But if you want to commit from your IDE,
you can always either click on "Show Command Output" or "Show details in console" to see the error message.

And be aware that some of the linters we use format your code automatically like ``black`` and ``autoflake``. So you
will need to add the changes to your commit and commit again.

Continuous Integration
----------------------

We have multiple GitHub actions to check and fix the formatting of the code. They can be divided into lint checks and
lint formatting (or lint-bot).

All the checks we do are made by pre-commit, you don't need to worry about lint errors arising from the CI checks that
are not caught by pre-commit.

Lint Checks
~~~~~~~~~~~

We have a GitHub action that runs:

1. Every commit
2. Every pull request

The important check is the one that runs on every pull request. You should expect this check to pass if you have
pre-commit correctly set up. Note that you can also reformat your code directly from GitHub by making a comment with
``ivy-gardener``, we will go through more details about it in the next section.

Lint Formatting
~~~~~~~~~~~~~~~

We have a GitHub action that runs:

1. Every day at 08:00 UTC
2. Manually invoked by making a comment with ``ivy-gardener`` on a PR

The first action is to ensure that the code in the whole codebase is always formatted correctly. The second action
is to reformat the files you changed in your PR directly on GitHub. This is useful in case if you didn't setup
pre-commit correctly or if you or one of our maintainers want to reformat your code remotely.

Under the hood, when ``ivy-gardener`` is found in a comment, an ivy bot will trigger the same set of lint checks
as in the pre-commit process. Then the suggested changes produced in the checks will be applied automatically as
a new commit if there is any.

However, it is possible for the linters run in the ``ivy-gardener`` and the GitHub action every day to face
formatting errors that need human intervention like typos and uninitialized arguments. In this case, errors will
be thrown by the linters and by the lint checks that runs later, while fixes to other simpler errors will still
be applied by the ``ivy-gardener`` properly.

On the other hand, ``ivy-gardener`` itself can fail if the bot handling it (ivy-branch) can not apply the changes
suggested by the linters, for example, when it does not have access to edit the target branch. In this case, you
should try to give the maintainer bot the access to your branch (which is an option shown in GitHub UI) and give it
another try, or manually resolve the formatting errors by committing the changes yourself.

**Round Up**

This should have hopefully given you a good feel for what is our coding style and how to format your code to contribute
to Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `formatting thread`_!

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/JXQ8aI8vJ_8" class="video">
    </iframe>
