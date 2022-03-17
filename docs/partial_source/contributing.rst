Contributing to Ivy
===================

.. _`Array API`: https://data-apis.org/array-api/latest/
.. _`tutorial series`: https://www.youtube.com/channel/UCGlkr-YCs3TjMVeOhbbULsw
.. _`source files`: https://github.com/data-apis/array-api/tree/main/spec/API_specification/signatures

Array API Standardization
-------------------------

One of the key tasks currently underway is to update Ivy to conform to the `Array API`_ standard. The YouTube `tutorial series`_
is the best starting reference for this task.

A few points are made here. You should first check what methods are still available to work on. Methods are reserved by
contributors in the community by creating an issue, and then commenting with a link to this issue on one of the ToDo list issues,
which has labels "ToDo" and "Array API".

As explained in the YouTube `tutorial series`_, you can test for each backend seperately by replacing
:code:`array_module = None` with lines :code:`import ivy as array_module` and :code:`array_module.set_framework('torch')`
for example.

Once you have got the unit tests passing for each backend for your particular method, you should then make sure your changes have not
inadvertently caused other unit tests to fail. First, uncomment your method in the folder
:code:`ivy_tests/array_api_methods_to_test`, add back the line :code:`array_module = None`, and then verify all other flagged methods are passing without error as explained below.

This is required for each backend framework :code:`numpy`, :code:`jax`,
:code:`torch` and :code:`tensorflow`. You can test all of these tests locally by running
:code:`./test_array_api.sh backend_name`, for example :code:`./test_array_api.sh torch` or
:code:`./test_array_api.sh jax`.

The docstrings for all methods in the Ivy API should be taken directly from the associated docstring in the Array API
standard. Use the `source files`_ in the Array API repository rather than the website for copying, so that the
formatting can be copied correctly. Many Ivy methods still use the Sphinx documentation format, but these should be
updated to now use the NumPy style, which is the same format used by all methods in the Array API Standard.
However, when defining our method in Ivy, we should remove the following arguments which appear in the standard to
denote positional-only and keyword-only arguments :code:`*, \,`.

Additionally, we should remove all argument types from the docstrings. These are all defined using type-hints in the
arguments already, and adding these also to the docstrings would create unecessary duplication. Our documentation
builder adds the correct types to the online documentation dynamically using the type hints directly.


Keeping Your Fork Updated
-------------------------

There is a script in the root repo :code:`merge_with_upstream.sh`. To update your local fork to the upstream master
branch, simply run :code:`./merge_with_upstream.sh name_of_your_branch`. If you are simply developing in master branch
(the default if you haven't explicitly created any new branches), then simply run :code:`./merge_with_upstream.sh master`.

If you are developing for pull requests (PRs), then it is common to create PR-specific branches. In this case, you would
run :code:`./merge_with_upstream.sh name_of_your_pr_branch`.


ToDo List Issues
----------------

We make extensive use of ToDo list issues, which act as placeholders for tracking many related sub-tasks in a ToDo list.

We have a clear process for contributors to engage with such ToDo lists:

(a) Find a task to work on which (i) is not marked as completed with a tick (ii) does not have an issue created and
(iii) is not mentioned in the comments.

(b) Create a new issue with the title being just the name of the task you would like to work on.

(c) comment on the ToDo list issue with a reference to this issue like so:

- [ ] #Issue_number

Your issue will then automatically be added to the ToDo list at some point, and the comment will be deleted.
No need to wait for this to happen before progressing to stage d.

(d) Start working on the task, and create a PR as soon as you have a full or partial solution, and then directly
reference the issue in the pull request. If you have a partial solution, the Ivy team can help to guide you through
the process of getting it working :)

(e) Wait for us to review your PR. Once we have reviewed your PR we will either merge or request changes. Every time you
respond to our requested changes you must re-request a review in order for us to re-engage with the PR.

(e) Once the PR is in good shape, we will merge into master, and you then become and Ivy contributor!

In order to keep our ToDo lists moving quickly, if your PR is not created within 7 days of creating the issue, then
the issue will be closed and the method will be made free for others in the community. Likewise, if we have requested
changes on your PR, and you do not respond and request a new code review within 7 days, then the PR and the associated
issue will be closed, and the method will be freed for others in the community. Even if you do not make code changes,
you should request a new code review to flag to us that our attention is again needed to further the discussion.

The purpose of this is to ensure our ToDo lists remain accessible for all in the community to engage with, where
priority is given to those who can engage on a more short-term basis. We want to avoid the situation where tasks are
allocated but then are not acted upon for long periods of time, whilst preveting others in the community from working
on these instead.

Starting an issue and then being unable to complete it is not a problem from our side at all, we automatically close
these just so we can keep our communuty engaged with these tasks :)

Our automatic closing is obviously never a reflection on the quality of the PR or the developer who made it, or any
reflection of hypothetical frustration we have for more delayed response times etc. Developers are of course very busy
people, and sometimes there is not as much free time available as initially thought. Please don't take it personally
if your issue or PR gets closed because of these time limits.

Reach out to me on discord if at any point you believe this happened to you unfairly, and we will definitely
investigate!

Creating Pull Requests
----------------------

Our process for responding to pull requests is simple. All newly created PRs will be reviewed by a member of the team,
and then the PR will either be merged or changes will be requested. In order for us to look at the changes you have made,
you will then need to request a code review once you have addressed our requested changes.
We will then take another look, and either merge the PR or request further changes.
This process then will repeat until either the PR is closed by us or yourslef, or the PR is merged.

If we request changes, you make those changes, but you do not request a code review, then we will not check the changes.
This is the case even if you comment on the PR. This simple process makes it much simpler for us to track where and when
attention is needed.

Note that you cannot request a code review until you have already received at least one review from us. All new PRs will
receive a code review, so just wait and we will check out and review your newly created PR as soon as possible!

Finally, all PRs must give write access to Ivy maintainers of the branch. This can be done by checking a tickbox in the
lower right corner of the PR. This will enable us to quickly fix conflicts, merge with upstream, and get things moving
much more quickly without us needing to request very simple fixes from yourself.


Submodule Design
----------------

Many already-implemented methods will need to be moved into new locations during various stages of refactoring.
The package is designed so all methods are called directly from the :code:`ivy` namespace, such as :code:`ivy.matmul`,
and not :code:`ivy.some_namespace.matmul`. Therefore, inside any of the folders :code:`ivy.functional.ivy`,
:code:`ivy.functional.backends.some_backend`, :code:`ivy.functional.backends.another_backend` the functions can be moved
to different files or folders without breaking anything. This makes it very simple to continually refactor and re-organize
the code structure in an ongoing manner.

Currently, we are in the process of refactoring things to more closely follow the Array API standard. Many methods will
need to be moved to new locations. Again, this is not a problem, provided the :code:`__init__` files have the correct
imports. Generally, these files look like the following, so that both the submodule namespace is imported but also
all methods.

.. code-block:: python

    from . import dtype
    from .dtype import *
    from . import general
    from .general import *
    # etc.

Ivy API
-------

All function signatures for the Ivy API are defined in the :code:`ivy.functional.ivy` submodule. Functions written here
should adhere to the following type hint format:


.. code-block:: python


    def my_func(x: Union[ivy.Array, ivy.NativeArray],
                axes: Union[int, Tuple[int], List[int]],
                dtype: Optional[Union[ivy.Dtype, str]] = None,
                dev: Optional[Union[ivy.Dev, str]] = None) \
            -> ivy.Array:
        """
        My function does something cool.

        .. note::
            This is an important note.

        **Special Cases**

        For this particular case,

        - If ``x`` is ``NaN``, do something
        - If ``y`` is ``-0``, do something else
        - etc.

        Parameters
        ----------
        x:
            input array. Should have a numeric data type.
        axes:
            the axes along which to perform the op.
        dtype:
            array data type.
        dev:
            the device on which to place the new array.

        Returns
        -------
        out:
            a cooler array.
        """
        return _cur_framework(x).my_func(x, dtype, dev)

Note that the input array has type :code:`Union[ivy.Array, ivy.NativeArray]` whereas the output array has type
:code:`ivy.Array`. This is the case for all functions in the ivy API.
We always return an :code:`ivy.Array` instance to ensure that any subsequent Ivy code is fully framework-agnostic, with
all operators performed on the array being handled by Ivy, and not the backend framework. However, there is no need to
prevent native arrays from being permitted in the input. For Ivy methods which wrap backend-specific implementations, the
input would need to be converted to a native array (such as :code:`torch.Tensor`) anyway before calling the backend method,
and for Ivy methods implemented as a composition of other Ivy methods such as :code:`ivy.lstm_update`, the native inputs can
just be converted to :code:`ivy.Array` instances before executing the Ivy implementation.

As for the :code:`axes` arg, generally the `Array API`_ standard dictates that shapes, axes and other similar args should be
of type :code:`Tuple[int]` when representing a sequence, not :code:`List[int]`. However, in order to make Ivy code
less brittle, we accept both tuples and lists for such arguments. This does not break the standard, as the standard is only
intended to define a subset of required function behaviour. The standard can be freely extended, as we are doing here.

As for the other arguments in the example above, :code:`dtype` and :code:`dev` do not need to be added to all methods,
these are just examples. These should be added to all creation methods though. Note that for both of these, the type is a
:code:`Union` including :code:`str`. This is because, in order to remain fully framework agnostic, Ivy accepts string
representations of devices and data types, such as :code:`"int32"`, :code:`"float32"`, :code:`"bool"`, :code:`"cpu"`,
:code:`"gpu0"`, :code:`"gpu2"` etc.

All functions which adhere to the `Array API`_ standard should be placed in the correct file in alignment with the
categories used in the standard.


Backend API
-----------

Code in the backend submodules such as :code:`ivy.functional.backends.torch` should then look something like:

.. code-block:: python


    def my_func(x: torch.Tensor,
                dtype: Optional[Union[torch.dtype, str]] = None,
                dev: Optional[Union[torch.device, str]] = None) \
            -> torch.Tensor:
        dtype = ivy.dtype_from_str(ivy.default_dtype(dtype, x))
        dev = ivy.dev_from_str(ivy.default_dev(dev, x))
        return torch.something_cool(x, dtype, dev)

Specifically, we should use type hints for all arguments in the Ivy API and also the backend APIs. These type hints
should be identical apart from all :code:`ivy.Array`, :code:`ivy.Dtype` and :code:`ivy.Dev` types replaced by
framework-specific types, in this case :code:`torch.Tensor`, :code:`torch.dtype` and :code:`torch.device`.

The backend methods should not add a docstring, as this would be identical to the docstring provided in the Ivy API.

All backend functions which adhere to the `Array API`_ standard should also be placed in the correct file in alignment with the
categories used in the standard.