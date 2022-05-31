The Basics
==========


Managing Your Fork
------------------

When contributing to Ivy, the first step is create a fork of the repository.
Then, it's best practice to create a separate branch for each new pull request you create.
The master branch then simply has the role of being kept up to date with upstream.
You can create PRs based on the master branch of your fork, but this will make things more complicated if you would then like to create additional PRs in future.

For keeping any branch on your fork up to date, there is a script in the root folder of the repo :code:`merge_with_upstream.sh`.
To update your fork's branch to the upstream master branch, simply run :code:`./merge_with_upstream.sh name_of_your_branch`.
To update the master branch, this would then be: :code:`./merge_with_upstream.sh master`.


ToDo List Issues
----------------

We make extensive use of ToDo list issues, which act as placeholders for tracking many related sub-tasks in a ToDo list.

We have a clear process for contributors to engage with such ToDo lists:

<a> Find a task to work on which (i) is not marked as completed with a tick (ii) does not have an issue created and (iii) is not mentioned in the comments.

<b> Create a new issue with the title being just the name of the task you would like to work on.

<c> comment on the ToDo list issue with a reference to this issue like so:

- [ ] #Issue_number

Your issue will then automatically be added to the ToDo list at some point, and the comment will be deleted.
No need to wait for this to happen before progressing to stage.

<d> Start working on the task, and create a PR as soon as you have a full or partial solution, and then directly
reference the issue in the pull request. If you have a partial solution, the Ivy team can help to guide you through
the process of getting it working 🙂

<e> Wait for us to review your PR. Once we have reviewed your PR we will either merge or request changes. Every time you
respond to our requested changes you must re-request a review in order for us to re-engage with the PR.

<f> Once the PR is in good shape, we will merge into master, and you then become and Ivy contributor!

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


Small Commits Often
-------------------

Sometimes, you might want to try any make substantial improvements that span many files,
with the intention of then creating one very large PR at the end in order to merge all of your changes.

While this is generally an acceptable approach, we discourage this approach for contributions to Ivy.

We adopt a philosophy where small, incremental, frequent commits are **much** more valuable to us and the entire
Ivy developer community, than infrequent and very large commits.

This is for a few reasons:

#. It keeps everyone up to date and on the same page as early as possible.
#. It avoids the case where multiple people waste time fixing the same problem!
#. It enables others to spot mistakes or conflicts in proposed changes much earlier.
#. It also means you avoid having a mountain of conflicts to resolve when you do get around to merging.

This is one of the reaons why we advocate using on pull-request per issue in the ToDo list issues.
This keeps each of the commits very contained and incremental, which is the style we're going for.

Sometimes, you've already dived very deep into some substantial changes in your fork,
and it might be that only some of the problems you were trying to fix are currently fixed.

In this hypothetical situation, you should aim to get the working parts merged into master **as soon as possible**.
Adding subsections of your local changes to git is easy. You can add individual files using:

.. code-block:: none

    git add filepath

You can also enter an interactive session for adding individual lines of code:

.. code-block:: none

    git add -p filepath  # choose lines to add from the file
    get add -p           # choose lines to add from all changes

When in the interactive session, you can split code blocks into smaller code blocks using :code:`s`.
You can also manually edit the exact lines added if further splitting is not possible, using :code:`e`.
Check the `git documentation <https://git-scm.com/doc>`_ for more details.

As a final note, a beautiful commit hisotry is not something we particularly care about.
We're much more concerned that the code is good, and things are updated as quickly as possible.
If a mistake is commited into the history, it's generally not too difficult to simply undo this in future commits,
so don't stress about this too much 🙂