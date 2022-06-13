The Basics
==========

.. _`contributing discussion`: https://github.com/unifyai/ivy/discussions/1309
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`todo list issues channel`: https://discord.com/channels/799879767196958751/982728618469912627
.. _`fork management channel`: https://discord.com/channels/799879767196958751/982728689408167956
.. _`pull requests channel`: https://discord.com/channels/799879767196958751/982728733859414056
.. _`commit frequency channel`: https://discord.com/channels/799879767196958751/982728822317256712
.. _`other channel`: https://discord.com/channels/799879767196958751/933380219832762430

Getting Help
------------

There are a few different communication channels that you can make use of in order to ask for help:

#. `discord server <https://discord.gg/ZVQdvbzNQJ>`_
#. `discussions <https://github.com/unifyai/ivy/discussions>`_
#. `issues <https://github.com/unifyai/ivy/issues>`_

We'll quickly outline how each of these should be used, and also which question is most appropriate for which context.

**Discord Server**

The `discord server <https://discord.gg/ZVQdvbzNQJ>`_ is most suitable for very quick and simple questions.
These questions should **always** be asked in the correct channel.
There is a tendency to use the *general* landing channel for everything.
This isn't the end of the world, but if many unrelated messages come flying into the *general* channel,
then it does make it very hard to keep track of the different discussions,
and it makes it less likely that you will receive a response.
For example, if you are applying for an internship, then you should make use of the **internship** channels,
and **not** the general channel for your questions.

**Discussions**

Almost all questions are best asked in the
`discussions <https://github.com/unifyai/ivy/discussions>`_ section on GitHub.
Even for very simple questions, there are benefits in starting a discussion for this.
This means that the question can be asked once and then easily found by others in future,
avoiding the case where different people ask the same question multiple times on the discord server.

I or someone else in the team will strive to respond to your newly created discussion as quickly as possible!

**Issues**

As the name suggests, the `issues <https://github.com/unifyai/ivy/issues>`_ section on GitHub is the best place to
raise issues or general bugs that you find with the project.
It can also serve as a useful place to ask questions, but only if you suspect that the behaviour you are observing
*might* be a bug. If you are confident there is nothing wrong with the code, but you just don't understand something,
then the `discussions <https://github.com/unifyai/ivy/discussions>`_ section is your best bet.

**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/T5vQP1pCXS8" class="video">
    </iframe>

|

For questions, please reach out on the `contributing discussion`_!

ToDo List Issues
----------------

We make extensive use of
`ToDo list issues <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3AToDo>`_,
which act as placeholders for tracking many related sub-tasks in a ToDo list.

We have a clear process for contributors to engage with such ToDo lists:

a. Find a task to work on which (i) is not marked as completed with a tick (ii) does not have an
   issue created and (iii) is not mentioned in the comments.

b. Create a new issue with the title being just the name of the task you would like to work on.

c. comment on the ToDo list issue with a reference to this issue like so:

   :code:`- [ ] #Issue_number`

   Your issue will then automatically be added to the ToDo list at some point, and the comment will be deleted.
   No need to wait for this to happen before progressing to stage.

d. Start working on the task, and create a PR as soon as you have a full or partial solution, and then directly
   reference the issue in the pull request. If you have a partial solution, the Ivy team can help to guide you through
   the process of getting it working 🙂

e. Wait for us to review your PR. Once we have reviewed your PR we will either merge or request changes. Every time you
   respond to our requested changes you must re-request a review in order for us to re-engage with the PR.

f. Once the PR is in good shape, we will merge into master, and you then become and Ivy contributor!

In order to keep our ToDo lists moving quickly, if your PR is not created within 7 days of creating the issue, then
the issue will be closed and the task will be made free for others in the community. Likewise, if we have requested
changes on your PR, and you do not respond and request a new code review within 7 days, then the PR and the associated
issue will be closed, and the task will be freed for others in the community. Even if you do not make code changes,
you should request a new code review to flag to us that our attention is again needed to further the discussion.

The purpose of this is to ensure our ToDo lists remain accessible for all in the community to engage with, where
priority is given to those who can engage on a more short-term basis. We want to avoid the situation where tasks are
allocated but then are not acted upon for long periods of time, while preventing others in the community from working
on these instead.

Starting an issue and then being unable to complete it is not a problem from our side at all, we automatically close
these just so we can keep our community engaged with these tasks :)

Our automatic closing is obviously never a reflection on the quality of the PR or the developer who made it, or any
reflection of hypothetical frustration we have for more delayed response times etc. Developers are of course very busy
people, and sometimes there is not as much free time available as initially thought. That's totally fine.
Please don't take it personally if your issue or PR gets closed because of this 7-day inactivity time limit.

Reach out to me on discord if at any point you believe this happened to you unfairly, and we will definitely
investigate!

**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/wBKTOGmwfbo" class="video">
    </iframe>

|

For questions, please reach out on the `contributing discussion`_
or on `discord`_ in the `todo list issues channel`_!

Managing Your Fork
------------------

When contributing to Ivy, the first step is create a fork of the repository.
Then, it's best practice to create a separate branch for each new pull request (PR) you create.
The master branch then simply has the role of being kept up to date with upstream.
You *can* create PRs based on the master branch of your fork,
but this will make things more complicated if you would then like to create additional PRs in future.

For keeping any branch on your fork up to date,
there is a script in the root folder of the repo
`merge_with_upstream.sh <https://github.com/unifyai/ivy/blob/2994da4f7347b0b3fdd81b91c83bcbaa5580e7fb/merge_with_upstream.sh>`_.
To update your fork's branch to the upstream master branch,
simply run :code:`./merge_with_upstream.sh name_of_your_branch`.
To update the master branch, this would then be: :code:`./merge_with_upstream.sh master`.

When making a PR (explained in the next sub-section),
sometimes you will see that changes to upstream have caused conflicts with your PR.
In this case, you will need to either resolve these conflicts in the browser,
or clone your fork and make changes locally in the terminal and push once resolved.
Both of these cases are explained in the following video.

**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/TFMPihytg9U" class="video">
    </iframe>

|

For questions, please reach out on the `contributing discussion`_
or on `discord`_ in the `fork management channel`_!

Pull Requests
-------------

Our process for responding to pull requests is quite simple.
All newly created PRs will be reviewed by a member of the team,
and then the PR will either be merged or changes will be requested.
In order for us to look at the changes you have made,
you will then need to request a code review once you have addressed our requested changes.
We will then take another look, and either merge the PR or request further changes.
This process then will repeat until either the PR is closed by us or yourself,
or the PR is merged.

If we request changes, you make those changes, but you do not request a code review,
then we will likely not check the changes.
This is the case even if you comment on the PR.
This simple process makes it much simpler for us to track where and when
our attention is needed.

Note that you cannot request a code review until you have already received at least one
review from us. Therefore, all new PRs will receive a code review,
so please just wait and we will check out and review your newly created PR
as soon as possible!
Your PR will never be closed until we have provided at least code review on it.

Finally, all PRs must give write access to Ivy maintainers of the branch.
This can be done by checking a tickbox in the lower right corner of the PR.
This will enable us to quickly fix conflicts, merge with upstream, and get things moving
much more quickly without us needing to request very simple fixes from yourself.

The code review process is explained in more detail in the following video.

**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/9G4d-CvlT2g" class="video">
    </iframe>

|

For questions, please reach out on the `contributing discussion`_
or on `discord`_ in the `pull requests channel`_!

Small Commits Often
-------------------

Sometimes, you might want to try any make substantial improvements that span many files,
with the intention of then creating one very large PR at the end in order to merge all of your changes.

While this is generally an acceptable approach when working on software projects,
we discourage this approach for contributions to Ivy.

We adopt a philosophy where small, incremental, frequent commits are **much** more valuable to us and the entire
Ivy developer community, than infrequent large commits.

This is for a few reasons:

#. It keeps everyone up to date and on the same page as early as possible.
#. It avoids the case where multiple people waste time fixing the same problem.
#. It enables others to spot mistakes or conflicts in proposals much earlier.
#. It means you avoid the mountain of conflicts to resolve when you do get around to merging.

This is also why we advocate using individual pull-requests per issue in the ToDo list issues.
This keeps each of the commits on master very contained and incremental, which is the style we're going for.

Sometimes, you've already dived very deep into some substantial changes in your fork,
and it might be that only some of the problems you were trying to fix are actually fixed by your local changes.

In this hypothetical situation, you should aim to get the working parts merged into master **as soon as possible**.
Adding subsections of your local changes with :code:`git` is easy. You can add individual files using:

.. code-block:: none

    git add filepath

You can also enter an interactive session for adding individual lines of code:

.. code-block:: none

    git add -p filepath  # choose lines to add from the file
    get add -p           # choose lines to add from all changes

When in the interactive session, you can split code blocks into smaller code blocks using :code:`s`.
You can also manually edit the exact lines added if further splitting is not possible, using :code:`e`.
Check the `git documentation <https://git-scm.com/doc>`_ for more details.

As a final note, a beautiful commit history is not something we particularly care about.
We're much more concerned that the code itself is good, that things are updated as quickly as possible,
and that all developers are able to work efficiently.
If a mistake is commited into the history, it's generally not too difficult to simply undo this in future commits,
so don't stress about this too much 🙂

For questions, please reach out on the `contributing discussion`_
or on `discord`_ in the `commit frequency channel`_!

Running Tests Locally
---------------------

# ToDo: write this section

**Round Up**

This should have hopefully given you a good understanding of the basics for contributing.

If you're ever unsure of how best to proceed,
please feel free to engage with the `contributing discussion`_,
or reach out on `discord`_ in the `todo list issues channel`_,
`fork management channel`_, `pull requests channel`_, `commit frequency channel`_
or `other channel`_, depending on the question!
