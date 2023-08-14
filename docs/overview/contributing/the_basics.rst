The Basics
==========

.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`todo list issues channel`: https://discord.com/channels/799879767196958751/982728618469912627
.. _`Atlassian tutorial`: https://www.atlassian.com/git/tutorials/saving-changes/git-stash
.. _`fork management channel`: https://discord.com/channels/799879767196958751/982728689408167956
.. _`pull requests channel`: https://discord.com/channels/799879767196958751/982728733859414056
.. _`commit frequency channel`: https://discord.com/channels/799879767196958751/982728822317256712
.. _`PyCharm blog`: https://www.jetbrains.com/help/pycharm/finding-and-replacing-text-in-file.html
.. _`Debugging`: https://www.jetbrains.com/help/pycharm/debugging-code.html
.. _`Ivy Experimental API Open Task`: https://unify.ai/docs/ivy/overview/contributing/open_tasks.html#ivy-experimental-api

Getting Help
------------

There are a few different communication channels that you can make use of in order to ask for help:

#. `Discord server <https://discord.gg/sXyFF8tDtm>`_
#. `Issues <https://github.com/unifyai/ivy/issues>`_

We'll quickly outline how each of these should be used, and also which question is most appropriate for which context.

**Discord Server**

The `discord server <https://discord.gg/sXyFF8tDtm>`_ is most suitable for very quick and simple questions.
These questions should **always** be asked in the correct channel.
There is a tendency to use the *general* landing channel for everything.
This isn't the end of the world, but if many unrelated messages come flying into the *general* channel, then it does make it very hard to keep track of the different discussions, and it makes it less likely that you will receive a response.
For example, if you are applying for an internship, then you should make use of the **internship** channels, and **not** the general channel for your questions.


**Issues**

As the name suggests, the `issues <https://github.com/unifyai/ivy/issues>`_ section on GitHub is the best place to raise issues or general bugs that you find with the project.
It can also serve as a useful place to ask questions, but only if you suspect that the behaviour you are observing *might* be a bug.

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/T5vQP1pCXS8" class="video" allowfullscreen="true">
    </iframe>

|


ToDo List Issues
----------------

We make extensive use of `ToDo list issues <https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3AToDo>`_, which act as placeholders for tracking many related sub-tasks in a ToDo list.

We have a clear process for contributors to engage with such ToDo lists:

a. Find a task to work on which (i) is not marked as completed with a tick (ii) does not have an issue created and (iii) is not mentioned in the comments. Currently, there are three open tasks: `function reformatting <https://unify.ai/docs/ivy/overview/contributing/open_tasks.html#function-formatting>`_, `frontend APIs <https://unify.ai/docs/ivy/overview/contributing/open_tasks.html#frontend-apis>`_ and `ivy experimental API <https://unify.ai/docs/ivy/overview/contributing/open_tasks.html#ivy-experimental-api>`_.

b. Create a new issue with the title being just the name of the sub-task you would like to work on.

c. Comment on the ToDo list issue with a reference to this issue like so:

   :code:`- [ ] #Issue_number`

   Your issue will then automatically be added to the ToDo list at some point, and the comment will be deleted.
   No need to wait for this to happen before progressing to the next stage. Don’t comment anything else on these ToDo issues, which should    be kept clean with comments only as described above. 

d. Start working on the task, and create a PR as soon as you have a full or partial solution, and then directly reference the issue in the pull request by adding the following content to the description of the PR:

   :code:`Close #Issue_number`

   This is important, so that the merging of your PR will automatically close the associated issue. Make sure this is in the 
   description of the PR, otherwise it might not link correctly. If you have a partial solution, the Ivy team can help to guide you through the process of getting it working 🙂
   Also remember to make the PR name well described and if there are some details that can support your changes add them to the description of the PR.

e. Wait for us to review your PR.
   Once we have reviewed your PR we will either merge or request changes.
   Every time you respond to our requested changes you must re-request a review in order for us to re-engage with the PR.

f. Once the PR is in good shape, we will merge into master, and then you become an Ivy contributor!

In order to keep our ToDo lists moving quickly, if your PR is not created within 7 days of creating the issue, then a warning message will appear on the issue.
If another 7 days pass without any changes, the issue will be closed and the task will be made free for others in the community.
Likewise, if we have requested changes on your PR, and you do not respond and request a new code review within 7 days, then a warning message will appear on the PR.
If another 7 days pass without any changes, then the PR and the associated issue will be closed, and the task will be freed for others in the community.
Even if you do not make code changes, you should request a new code review to flag to us that our attention is again needed to further the discussion.

The purpose of this is to ensure our ToDo lists remain accessible for all in the community to engage with, where priority is given to those who can engage on a more short-term basis.
We want to avoid the situation where tasks are allocated but then are not acted upon for long periods of time, while preventing others in the community from working on these instead.

Starting an issue and then being unable to complete it is not a problem from our side at all, we automatically close these just so we can keep our community engaged with these tasks 🙂

Our automatic closing is obviously never a reflection on the quality of the PR or the developer who made it, or any reflection of hypothetical frustration we have for more delayed response times etc.
Developers are of course very busy people, and sometimes there is not as much free time available as initially thought.
That's totally fine.
Please don't take it personally if your issue or PR gets closed because of this 7-day inactivity time limit.

Reach out to me on discord if at any point you believe this happened to you unfairly, and we will definitely investigate!

Finally, we limit the maximum number of *open* and *incomplete* sub-task issues at *three* per person.
This is to prevent anyone from self-allocating many sub-tasks, preventing others in the community from engaging, and then not being able to complete them.
Even though the limit is three, sub-tasks should only be self-assigned using **one comment per sub-task**.
For example, a sequence of comments like this :code:`- [ ] #Issue_number` will register correctly whereas a single comment like this :code:`- [ ] #Issue_number, - [ ] #Issue_number, - [ ] #Issue_number` or this :code:`- [ ] #Issue_number #Issue_number #Issue_number` etc. will not.

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/wBKTOGmwfbo" class="video" allowfullscreen="true">
    </iframe>

|

For questions, please reach out on `discord`_ in the `todo list issues channel`_!

Managing Your Fork
------------------

When contributing to Ivy, the first step is create a fork of the repository.
Then, it's best practice to create a separate branch for each new pull request (PR) you create.
This can be done using:

.. code-block:: bash

   git checkout -b name_of_your_branch

The master branch then simply has the role of being kept up to date with upstream.
You *can* create PRs based on the master branch of your fork, but this will make things more complicated if you would then like to create additional PRs in future.

For keeping any branch on your fork up to date, there is a script in the root folder of the repo `merge_with_upstream.sh <https://github.com/unifyai/ivy/blob/2994da4f7347b0b3fdd81b91c83bcbaa5580e7fb/merge_with_upstream.sh>`_.
To update your fork's branch to the upstream master branch, simply run :code:`./merge_with_upstream.sh name_of_your_branch`.
To update the master branch, this would then be: :code:`./merge_with_upstream.sh master`.

When making a PR (explained in the next sub-section), sometimes you will see that changes to upstream have caused conflicts with your PR.
In this case, you will need to either resolve these conflicts in the browser, or clone your fork and make changes locally in the terminal and push once resolved.
Both of these cases are explained in the following video.

You may find that once you have made changes locally and try pulling from master, the pull request is aborted as there are merge conflicts.
In order to avoid tedious merge conflict resolution, you can try 'stashing' your local changes, then pulling from master.
Once your branch is up-to-date with master, you can reinstate the most recently stashed changes, commit and push to master with no conflicts.
The corresponding commands are :code:`git stash` -> :code:`git fetch` -> :code:`git pull` -> :code:`git stash apply stash@{0}`.
Note that this only works for uncommitted changes (staged and unstaged) and untracked files won't be stashed.
For a comprehensive explanation of git stashing, check out this `Atlassian tutorial`_.

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/TFMPihytg9U" class="video" allowfullscreen="true">
    </iframe>

|

For questions, please reach out on `discord`_ in the `fork management channel`_!

Who To Ask
----------

When raising issues on the Ivy repo, it can be useful to know who in the team wrote which piece of code.
Armed with this information, you can then for example directly tag (using @) the member of the team who worked on a particular piece of code, which you are trying to understand, or you would like to ask questions about.

Here we describe a workflow to help navigate this question of "who to ask".

With Command Line:
******************

**git blame** - Show what revision and author last modified each line of a file

**git log**   - Show commit logs

.. code-block:: none

    # Eg: From line 16 to next 5 lines since past 2 weeks
    git blame --since=2.weeks -L 16,+5 <filepath> | grep -v "^\^"
    # Deeper look at what each author changed in files retrieved from the above step
    git log <commit_id> -p

With Browser:
*************

**Git Blame View** is a handy tool to view the line-by-line revision history for an entire file, or view the revision history of a single line within a file.

    .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/the_basics/git_blame/git_blame_1.png?raw=true
       :width: 420

This view can be toggled from the option in left vertical pane, or from the "blame" icon in top-right, as highlighted above.

    .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/the_basics/git_blame/git_blame_2.png?raw=true
       :width: 420

Each time you click the highlighted icon, the previous revision information for that line is shown, including who committed the change and when this happened.

    .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/the_basics/git_blame/git_blame_3.png?raw=true
       :width: 420

Whenever starting a discussion or creating an issue, you are very welcome to tag members of the Ivy team using "@", selecting the person you think would most suitable to interact with, based on the information gained from the above steps.

Pull Requests
-------------

Our process for responding to pull requests is quite simple.
All newly created PRs will be reviewed by a member of the team, and then the PR will either be merged or changes will be requested.
In order for us to look at the changes you have made, you will then need to request a code review once you have addressed our requested changes.
We will then take another look, and either merge the PR or request further changes.
This process then will repeat until either the PR is closed by us or yourself, or the PR is merged.

If we request changes, you make those changes, but you do not request a code review, then we will likely not check the changes.
This is the case even if you comment on the PR.
This simple process makes it much simpler for us to track where and when our attention is needed.

Note that you cannot request a code review until you have already received at least one review from us.
Therefore, all new PRs will receive a code review, so please just wait and we will check out and review your newly created PR as soon as possible!
Your PR will never be closed until we have provided at least code review on it.

After a new PR is made, for the tests to run, it needs an approval of someone from the ivy team for the workflows to start running.
Once approved, you can see the failing and passing checks for a commit relevant to your PR by clicking on the ❌ or ✔️ or 🟤 (each for: one or more tests are failing, all tests are passing, the check has just started, respectively) icon next to the commit hash.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/pull_requests/PR_checks.png?raw=true
       :width: 420

Further, if you click on the details next to a check then you can see the logs for that particular test.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/pull_requests/pr_logs.png?raw=true
       :width: 420

Also, if you have pushed multiple commits to a PR in a relatively short time, you may want to cancel the checks for a previous commit to speedup the process, you can do that by going to the log page as described above and clicking on the `Cancel Workflow` button.

Note that this option might be unavailable depending on the level of access that you have.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/pull_requests/cancel_workflow.png?raw=true
       :width: 420

Finally, all PRs must give write access to Ivy maintainers of the branch.
This can be done by checking a tickbox in the lower right corner of the PR.
This will enable us to quickly fix conflicts, merge with upstream, and get things moving much more quickly without us needing to request very simple fixes from yourself.

The code review process is explained in more detail in the following video.

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/9G4d-CvlT2g" class="video" allowfullscreen="true">
    </iframe>

|

For questions, please reach out on `discord`_ in the `pull requests channel`_!

Small Commits Often
-------------------

Sometimes, you might want to try any make substantial improvements that span many files, with the intention of then creating one very large PR at the end in order to merge all of your changes.

While this is generally an acceptable approach when working on software projects, we discourage this approach for contributions to Ivy.

We adopt a philosophy where small, incremental, frequent commits are **much** more valuable to us and the entire Ivy developer community, than infrequent large commits.

This is for a few reasons:

#. It keeps everyone up to date and on the same page as early as possible.
#. It avoids the case where multiple people waste time fixing the same problem.
#. It enables others to spot mistakes or conflicts in proposals much earlier.
#. It means you avoid the mountain of conflicts to resolve when you do get around to merging.

This is also why we advocate using individual pull-requests per issue in the ToDo list issues.
This keeps each of the commits on master very contained and incremental, which is the style we're going for.

Sometimes, you've already dived very deep into some substantial changes in your fork, and it might be that only some of the problems you were trying to fix are actually fixed by your local changes.

In this hypothetical situation, you should aim to get the working parts merged into master **as soon as possible**.
Adding subsections of your local changes with :code:`git` is easy.
You can add individual files using:

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
We're much more concerned that the code itself is good, that things are updated as quickly as possible, and that all developers are able to work efficiently.
If a mistake is committed into the history, it's generally not too difficult to simply undo this in future commits, so don't stress about this too much 🙂

For questions, please reach out on the on `discord`_ in the `commit frequency channel`_!

Interactive Ivy Docker Container
--------------------------------

The advantage of Docker interactive mode is that it allows us to execute commands at the time of running the container.
It's quite a nifty tool which can be used to reassure that the functions are working as expected in an isolated environment.

An interactive bash shell in ivy's docker container can be created by using the following command,

.. code-block:: none

    docker run --rm -it unifyai/ivy bash

The project structure and file-system can be explored.
This can be very useful when you want to test out the bash scripts in ivy, run the tests from the command line etc,.
In fact, if you only want to quickly test things in an interactive python shell run the following command,

.. code-block:: none

    docker run --rm -it unifyai/ivy python3

In both cases, the ivy version at the time when the container was built will be used.
If you want to try out your local version of ivy, with all of the local changes you have made, you should add the following mount:

.. code-block:: none

    docker run --rm -it -v /local_path_to_ivy/ivy/ivy:/ivy/ivy unifyai/ivy bash

* This will overwrite the *ivy* subfolder inside the ivy repo in the container with the *ivy* subfolder inside your local ivy repo.
* Ivy is installed system-wide inside the container via the command :code:`python3 setup.py develop --no-deps`
* The :code:`develop` command means that the system-wide installation will still depend on the original source files, rather than creating a fresh copy.
* Therefore, ivy can be imported into an interactive python shell from any directory inside the container, and it will still use the latest updates made to the source code.

Clearly, running a container in interactive mode can be a helpful tool in a developer’s arsenal.

Running Tests Locally
---------------------

With Docker
***********

#. With PyCharm (With or without docker):
    1. PyCharm enables users to run pytest using the green button present near every function declaration inside the :code:`ivy_tests` folder.
        
    .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/the_basics/pytest_with_pycharm/pytest_button_pycharm.png?raw=true
        :width: 420
        
    2. Testing can be done for the entire project, individual submodules, individual files and individual tests.
       This can be done by selecting the appropriate configuration from the top pane in PyCharm.
        
    .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/the_basics/pytest_with_pycharm/pytest_with_pycharm.png?raw=true
        :width: 420
        

#. Through the command line (With docker):
    1. We need to replace the folder inside the container with the current local ivy directory to run tests on the current local code.

    .. code-block:: none

        docker exec <container-name> rm -rf ivy
        docker cp ivy <container-name>:/ 

    2. We need to then enter inside the docker container and change into the :code:`ivy` directory using the following command.

    .. code-block:: none

        docker exec -it ivy_container bash 
        cd ivy

    3. Run the test using the pytest command.

        1. Ivy Tests:

            1. For a single function: 

            .. code-block:: none
            
                pytest ivy_tests/test_ivy/test_functional/test_core/test_image.py::test_random_crop --no-header --no-summary -q
            
            2. For a single file:

            .. code-block:: none
            
                pytest ivy_tests/test_ivy/test_functional/test_core/test_image.py --no-header --no-summary -q

            3. For all tests:

            .. code-block:: none

                pytest ivy_tests/test_ivy/ --no-header --no-summary -q

        2.  Array API Tests:

            1. For a single function: 

            .. code-block:: none
            
                pytest ivy_tests/array_api_testing/test_array_api/array_api_tests/test_creation_functions.py::test_arange --no-header --no-summary -q
            
            2. For a single file:

            .. code-block:: none
            
                pytest ivy_tests/array_api_testing/test_array_api/array_api_tests/test_creation_functions.py --no-header --no-summary -q
            
            3. For all tests:

            .. code-block:: none

                pytest ivy_tests/array_api_testing/test_array_api/ --no-header --no-summary -q
        
        3. For the entire project:

        .. code-block:: none
            
            pytest ivy_tests/ --no-header --no-summary -q

#. Through the command line (Without docker):
    1. We need to first enter inside the virtual environment.

    .. code-block:: none

        ivy_dev\Scripts\activate.bat

    (on Windows)

    OR

    .. code-block:: none

        source ivy_dev/bin/activate

    (on Mac/Linux)

    2. Run the test using the pytest command.

        1. Ivy Tests:

            1. For a single function: 

            .. code-block:: none
            
                python -m pytest ivy_tests/test_ivy/test_functional/test_core/test_image.py::test_random_crop --no-header --no-summary -q
            
            2. For a single file:

            .. code-block:: none
            
                python -m pytest ivy_tests/test_ivy/test_functional/test_core/test_image.py --no-header --no-summary -q

            3. For all tests:

            .. code-block:: none

                python -m pytest ivy_tests/test_ivy/ --no-header --no-summary -q

        2.  Array API Tests 

            1. For a single function: 

                .. code-block:: none
                
                    python -m pytest ivy_tests/array_api_testing/test_array_api/array_api_tests/test_creation_functions.py::test_arange --no-header --no-summary -q
            
            2. For a single file:

            .. code-block:: none
            
                python -m pytest ivy_tests/array_api_testing/test_array_api/array_api_tests/test_creation_functions.py --no-header --no-summary -q
            
            3. For all tests:

            .. code-block:: none

                python -m pytest ivy_tests/array_api_testing/test_array_api/ --no-header --no-summary -q
        
        3. For the entire project

        .. code-block:: none
            
            python -m pytest ivy_tests/ --no-header --no-summary -q

#. Optional Flags: Various optional flags are available for running the tests such as :code:`device`, :code:`backend`, etc.
    1. :code:`device`: 
        1. This flag enables setting of the device where the tests would be run.
        2. Possible values being :code:`cpu` and :code:`gpu`.
        3. Default value is :code:`cpu`
    2. :code:`backend`:
        1. This flag enables running the tests for particular backends.
        2. The values of this flag could be any possible combination of JAX, numpy, tensorflow and torch.
        3. Default value is :code:`jax,numpy,tensorflow,torch`.
    3. :code:`num-examples`:
        1. Set the maximum number for examples to be generated by Hypothesis.
        2. The value of this flag could be any positive integer value that is greater than 1.
        3. Default value is :code:`5`.

Getting the most out of IDE
---------------------------
with PyCharm
************
#. Find a text:
    1. :code:`Ctrl+F` will prompt you to type in the text to be found, if not already selected, and then find all the instances of text within current file.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/find_file.png?raw=true
        :align: center

    2. :code:`Ctrl+Shift+F` will find all the instances of text within the project.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/find_project_wide.png?raw=true
        :align: center

#. Find+Replace a text:
    1. :code:`Ctrl+R` will prompt you to type in the text to be found and the text to be replaced, if not already selected, within current file.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/find_n_replace_file.png?raw=true
        :align: center

    2. :code:`Ctrl+Shift+R` will prompt you to type in the text to be found and the text to be replaced, if not already selected, within the whole project.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/find_and_replace_project_wide.png?raw=true
        :align: center

#. Find and multiply the cursor:
    1. :code:`Ctrl+Shift+Alt+J` will find all the instances of selected text and multiply the cursor to all these locations.

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/multiple_cursor.png?raw=true
        :align: center

    You can visit `Pycharm Blog`_ for more details on efficient coding!

#. Debugging:
    1. add breakpoints:
        1. Click the gutter at the executable line of code where you want to set the breakpoint or place the caret at the line and press :code:`Ctrl+F8`

        .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/adding_breakpoint.png?raw=true
           :align: center

    2. Enter into the debug mode:
        1. Click on Run icon and Select **Debug test** or press :code:`Shift+F9`.
        This will open up a Debug Window Toolbar:

        .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/open_in_debug_mode.png?raw=true
           :align: center

    3. Stepping through the code:
        1. Step over: 
            Steps over the current line of code and takes you to the next line even if the highlighted line has method calls in it.

            1. Click the Step Over button or press :code:`F8`

            .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/step_over.png?raw=true
               :align: center

        2. Step into:
            Steps into the method to show what happens inside it.
            Use this option when you are not sure the method is returning a correct result.

            Click the Step Into button or press :code:`F7`

            1. Smart step into:
                Smart step into is helpful when there are several method calls on a line, and you want to be specific about which method to enter.
                This feature allows you to select the method call you are interested in.

                1. Press :code:`Shift+F7`.
                   This will prompt you to select the method you want to step into:

                .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/smart_step_into.png?raw=true
                   :align: center

                2. Click the desired method.

    4. Python Console: 
        1. Click the Console option on Debug Tool Window:
            This currently stores variables and their values upto which the code has been executed.
            You can print outputs and debug the code further on.

        2. If you want to open console at certain breakpoint:
            1. Select the breakpoint-fragment of code, press :code:`Alt+shift+E` Start debugging!

            .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/console_coding.png?raw=true
               :align: center

    5. Using **try-except**:
        1. PyCharm is great at pointing the lines of code which are causing tests to fail.
           Navigating to that line, you can add Try-Except block with breakpoints to get in depth understanding of the errors.

        .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/try_except.png?raw=true
           :align: center

    6. Dummy **test** file:
        1. Create a separate dummy :code:`test.py` file wherein you can evaluate a particular test failure.
           Make sure you don't add or commit this dummy file while pushing your changes.

        .. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/the_basics/getting_most_out_of_IDE/dummy_test.png?raw=true
           :align: center

    PyCharm has a detailed blog on efficient `Debugging`_ which is quite useful.

**Round Up**

This should have hopefully given you a good understanding of the basics for contributing.

If you have any questions, please feel free to reach out on `discord`_ in the `todo list issues channel`_, `fork management channel`_, `pull requests channel`_, `commit frequency channel`_ depending on the question!
