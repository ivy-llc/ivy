Formatting
==========

.. _`flake8`: https://flake8.pycqa.org/en/latest/index.html
.. _`pre-commit guide`: https://lets-unify.ai/ivy/contributing/0_setting_up.html#pre-commit

Lint Checks
-----------

To ensure that Ivy is always formatted correctly, `flake8`_ is used to run
lint checks on all Python files in the
`CI <https://github.com/unifyai/ivy/blob/ff7d40e7f1e6ea5b48b7b460013c011cd7752a0e/.github/workflows/lint.yml>`_.
Some of the main things which `flake8`_ checks for are listed below.

**Imports**

Module imports are checked to detect:

* unused imports
* module imported but not used
* module used without imports
* duplicate imports
* undefined import names

**Syntax**

Flake8 is useful in detecting syntax errors, which are one of the most common mistakes.
Some examples are:

* :code:`break` or :code:`continue` statement outside a :code:`for` or :code:`while` loop
* :code:`continue` statement in the :code:`finally` block of a loop
* :code:`yield` or :code:`yield from` statement outside a function
* :code:`return` statement used with arguments in a generator, or outside a function or method
* :code:`except:` block not being the last exception handler
* syntax or length errors in docstrings, comments, or annotations

**Literals**

Literals formatting are often used in a string statement; some common checks related to
this are:

* invalid :code:`%` format
* :code:`%` format with missing arguments or unsupported character
* :code:`.format(...)` with invalid format, missing or unused arguments
* f-string without placeholders

**Others**

There are many more types of checks which :code:`flake8` can perform.
These include but are not limited to:

* repeated :code:`dict` key or variable assigned to different values
* star-unpacking assignment with too many expressions
* assertion test is a :code:`tuple`, which is always :code:`true`
* use of :code:`==` or :code:`!=` to compare :code:`str`, :code:`bytes` or :code:`int` literals
* :code:`raise NotImplemented` should be :code:`raise NotImplementedError`

Pre-Commit Hook
---------------

In Ivy, we try our best to avoid committing code with lint errors. To achieve this,
we make use of the pre-commit package. The installation is explained in
the `pre-commit guide`_.

The pre-commit hook runs the :code:`flake8` lint checks before each commit. This is
efficient and useful in preventing errors being pushed to the repository.

In the case where errors are found, error messages will be raised and committing will
be unsuccessful until the mistake is corrected. If the errors are related to argument
formatting, it will be reformatted automatically.

For example, the line length limit might be exceeded if arguments are all added in a
single line of code like so:

.. code-block:: python

    def indices_where(
        x: Union[ivy.Array, ivy.NativeArray], *, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:

When a commit is attempted, `pre-commit` would detect this error by running the lint
check, and it would then reformat the arguments automatically.

.. code-block:: none

    black....................................................................Failed
    - hook id: black
    - files were modified by this hook

    reformatted ivy/functional/ivy/general.py

    All done! ✨ 🍰 ✨
    1 file reformatted.

    flake8...................................................................Passed

The above message indicates that a file disobeying the formatting rules is detected
and reformatting has taken place successfully. The correctly formatted code, with each
argument added on a new line, has been saved and the related file(s) can now be staged
and committed accordingly.

.. code-block:: python

    def indices_where(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:


If the code is all formatted correctly, then in this case `pre-commit` will not modify
the code. For example, when the line limit is not exceeded by the function arguments,
then the arguments should indeed be listed on the same line, together with the function
:code:`def(...)` syntax, as shown below.

.. code-block:: python

    def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray]) -> Iinfo:

This would pass the lint checks, and :code:`pre-commit` would allow the code to be
committed without error.