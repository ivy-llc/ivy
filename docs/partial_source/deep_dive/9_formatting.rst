Formatting
==========

.. _`flake8`: https://flake8.pycqa.org/en/latest/index.html
.. _`pre-commit guide`: https://lets-unify.ai/ivy/contributing/0_setting_up.html#pre-commit

Lint Checks
-----------

To ensure that Ivy is always clean and correctly formatted, `flake8`_ is used to run
lint checks on all Python files. Some examples are listed below for an idea of what
is being checked for.

**Imports**

Module imports are being checked to detect:

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

Literals formatting are often used in a string statement, which some related common
checks are:

* invalid :code:`%` format
* :code:`%` format with missing arguments or unsupported character
* :code:`.format(...)` with invalid format, missing or unused arguments
* f-string without placeholders

**Others**

There are many more types of checking which `flake8` is able to carry out. They
include but are not limited to:

* repeated :code:`dict` key or variable assigned to different values
* star-unpacking assignment with too many expressions
* assertion test is a :code:`tuple`, which is always :code:`true`
* use of :code:`==` or :code:`!=` to compare :code:`str`, :code:`bytes` or :code:`int` literals
* :code:`raise NotImplemented` should be :code:`raise NotImplementedError`

Pre-Commit Hook
---------------

In Ivy, we try our best to avoid committing code with lint errors. To achieve this,
we make use of the pre-commit package, where its installation is explained in
the `pre-commit guide`_.

The pre-commit hook runs the `flake8` lint checks before each commit. This is
efficient and useful in preventing errors being pushed to the repository.

In the case where errors are found, error messages will be raised and committing will
be unsuccessful until the mistake is corrected. On the other hand, if the errors are
related to arguments formatting, it will be reformatted automatically. More
information can be found in the next section.

Automatic Reformatting Before Committing
----------------------------------------

If the arguments in a function are formatted incorrectly, for example, each argument
is not formatted in its own line, which caused a line to exceed the length limit:

.. code-block:: python

    def indices_where(
        x: Union[ivy.Array, ivy.NativeArray], *, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:

When a commit is attempted, `pre-commit` detects this through running the lint check,
then reformat the arguments automatically.

.. code-block:: none

    black....................................................................Failed
    - hook id: black
    - files were modified by this hook

    reformatted ivy/functional/ivy/general.py

    All done! ✨ 🍰 ✨
    1 file reformatted.

    flake8...................................................................Passed

The above message indicates that a file disobeying the formatting rules is detected
and reformatting has taken place successfully. The correctly formatted code has been
saved and the related file(s) can now be staged and committed accordingly.

.. code-block:: python

    def indices_where(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:


Examples
--------

To ensure a better understanding of the formatting rules, examples are shown below
for visualizing a better comparison.

When a function has few arguments which will not exceed the length limit, arguments
should be listed on the same line, together with the function :code:`def(...)`
syntax.

.. code-block:: python

    def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray]) -> Iinfo:

When there are many arguments in a function, each argument and its respective type
hints should be placed in separate lines as shown below:

.. code-block:: python

    def all(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
