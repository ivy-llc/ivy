Building the Docs Pipeline
==========================

.. _Sphinx: http://sphinx-doc.org/
.. _Sphinx configuration file: https://www.sphinx-doc.org/en/master/usage/configuration.html
.. _autosummary: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

To build our docs, we use `Sphinx`_. Sphinx is an extendable documentation generator
for Python. As our building pipeline is complex, we heavily customize Sphinx using 
custom and third party extensions. As well as having a convenience script to build
the docs.

How the doc-builder is being run
--------------------------------

There are 2 ways to build the docs:

1. Through a convenience script, which is useful for local development.
2. Through a Docker image, which is the recommended way.

We will go through how they work in the following sections.

The convenience script
~~~~~~~~~~~~~~~~~~~~~~

``make_docs_without_docker.sh`` is a convenience script to build the docs. It takes 
one argument, the path to a project to document. The project should have the following
characteristics:

1. It should have a ``requirements.txt``, or alternatively a ``requirements`` folder,
   which includes a ``requirements.txt`` and an optional ``optional.txt`` file.

2. It can have an optional ``optional.txt`` file, if not the script will
   simply ignore it.

3. It should have a ``docs`` folder, which contains an ``index.rst`` file. This file
   is the root of the documentation.

4. It can contain an optional ``docs/prebuild.sh`` file, which will be executed before
   the docs are built. This is useful if you need to install some dependencies for the
   docs to build.

5. It can contain an optional ``docs/partial_conf.py`` which is a partial `Sphinx
   configuration file`_.
   This file will be imported with the default ``conf.py`` file located in the 
   ``doc-builder`` repo.

Running the script:

.. code-block:: bash

    ./make_docs_without_docker.sh /path/to/project

will result in the creation of documentation for the project in the directory 
``doc/build``.

Options
"""""""

-h, --help                       Show this help
-C, --no-cleanup                 Disable the backup/cleanup procedure
-g, --git-add                    Stage changed files before generating the docs
-s, --skip-dependencies-install  Skip installing dependencies using pip
-j, --jobs N                     Build in parallel with N processes where possible 
                                 (special value ``auto`` will set N to cpu-count)
-D setting                       Override a setting in ``conf.py``

The Docker image
~~~~~~~~~~~~~~~~

The Docker image `unifyai/doc-builder <https://hub.docker.com/r/unifyai/doc-builder>`_
works as a wrapper around the ``make_docs_without_docker.sh`` script. It runs the script
on the ``/project`` directory, located in the container `as shown here 
<https://github.com/unifyai/doc-builder/blob/master/Dockerfile#L21>`_:

.. code-block:: bash

    ./make_docs_without_docker.sh /project

To build the docs through docker you use this command:

.. code-block:: bash

    docker run -v /path/to/project:/project unifyai/doc-builder

You can as well add options described in the :ref:`The convenience script` section.

.. code-block:: bash
    
    docker run -v /path/to/project:/project unifyai/doc-builder --no-cleanup

How Ivy's docs is structured
-----------------------------

Looking at `Ivy docs <https://github.com/unifyai/ivy/tree/master/docs>`_, we can see 
that it structured like this:

.. code-block:: bash

    docs
    ├── index.rst
    ├── partial_conf.py
    ├── prebuild.sh
    ├── overview
    │   ├── background.rst
    │   ├── ...
    │   └── ...
    └── ...

Let's go through each of these files and folders.

``index.rst``
~~~~~~~~~~~~~

This is the root of the documentation. It is the first file that Sphinx will read when
building the docs. It is also the file that will be displayed when you open the docs
in a browser.

Here is a segment of the file:

.. code-block:: rst

    .. include:: ../README.rst

    .. toctree::
        :hidden:
        :maxdepth: -1
        :caption: Overview

        overview/background.rst
        overview/design.rst
        overview/related_work.rst
        overview/extensions.rst
        overview/contributing.rst
        overview/deep_dive.rst
        overview/faq.rst
        overview/glossary.rst

    .. autosummary::
        :toctree: docs/functional
        :template: top_functional_toc.rst
        :caption: API Reference
        :recursive:
        :hide-table:

        ivy.functional.ivy

You can see here different reStructuredText directives. The first one is ``include``,
which simply includes the main README file of the project, this is a good place if you
want to make the rendered docs looks different from the README, or simply include it as
is.

The second directive is ``toctree``, which is used to create a table of contents. The
``:hidden:`` option hides the table of contents from the rendered docs, only keeping it
on the left side of the docs, not inline in the page itself. The ``:maxdepth:`` option
is used to specify how deep the table of contents should go. The ``:caption:`` option
is used to specify the title of the table of contents. The rest of the arguments are
the files that should be included in the table of contents. Which in recursively points
to every page in this documentation, for example this page is included in the
``toctree`` of ``overview/deep_dive.rst``, which is included in the ``toctree`` of
``index.rst``. You can read more about the ``toctree`` directive in `sphinx docs
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-toctree>`_, from 
now on we'll only explain the directives that are custom to Ivy's doc-builder.

The last directive is ``autosummary``, which is used to automatically generate a table
of contents for a module, as well as the documentation itself automatically by
discovering the docstrings of the module. This is a custom directive, built on the original
`autosummary`_
extension. We will explain in details how did we change it, in :ref:`Custom Extensions`.

``partial_conf.py``
~~~~~~~~~~~~~~~~~~~

This is a partial `Sphinx configuration file`_. Which is being imported in the 
`conf.py <https://github.com/unifyai/doc-builder/blob/master/docs/conf.py#L150>`_,
it's used to customize options that are specific to the project being documented.
While importing common configuration such as the theme, the extensions, etc in the 
original ``conf.py``

This is a part of ``partial_conf.py``:

.. code-block:: python

    ivy_toctree_caption_map = {
        "ivy.functional.ivy": "Functions",
        "ivy.stateful": "Framework classes",
        "ivy.nested_array": "Nested array",
        "ivy.utils": "Utils",
        "ivy_tests.test_ivy.helpers": "Testing",
    }

Here we are overriding the ``ivy_toctree_caption_map`` configuration, which is used to 
customize the title of the table of contents for each module. 
``ivy_toctree_caption_map`` is one of the configuration options we have in our
``custom_autosummary`` extension, which will be covered extensively in 
:ref:`Custom Extensions`.

``prebuild.sh``
~~~~~~~~~~~~~~~

This is an optional file, which is executed before the docs are built. This is useful
if you need to install some dependencies for the docs to build. In Ivy's case, we 
install ``torch`` then ``torch-scatter`` sequentially to avoid a bug in 
``torch-scatter``'s setup. And if we want to do any changes to the docker container
before building the docs.

Custom Extensions
-----------------

As of writing this documentation, Ivy's doc-builder is using 4 custom extensions:

#. ``custom_autosummary``
#. ``discussion_linker``
#. ``skippable_function``
#. ``ivy_data``

``custom_autosummary``
~~~~~~~~~~~~~~~~~~~~~~

This extension is a modified version of the original `autosummary`_, which is used to
discover and automatically document the docstrings of a module. This is done by
generating "stub" rst files for each module listed in the ``autosummary`` directive,
you can add a template for these stub files using the ``:template:`` option. Which can
inturn include the ``autosummary`` directive again, recursing on the whole module.

Unfortunately, the original ``autosummary`` extension is very limited, forcing you to
have a table of contents for each modules.

We'll go through each option or configuration value added to the original ``autosummary``

``:hide-table:``
""""""""""""""""

As the name suggests, the original behavior of ``autosummary`` is to generate a table
of contents for each module. And it generate stub files only if ``:toctree:`` option is
specified. As we only need the ``toctree`` this option hides the table of contents, but
it require the ``:toctree:`` option to be specified.

``discussion_linker``
~~~~~~~~~~~~~~~~~~~~~

Discussion linker is a simple extension that adds a link to our discord server, as well
as specific discussion boards for each modules.

The directive is included like this:

.. code-block:: rst

    .. discussion-links:: module.foo


First it will look for ``discussion_channel_map`` configuration, in Ivy it looks like 
this:

.. code-block:: python

    discussion_channel_map = {
        ...,
        "ivy.functional.ivy.creation": ["1000043690254946374", "1028298816526499912"],
        "ivy.functional.ivy.data_type": ["1000043749088436315", "1028298847950225519"],
        ...,
    }

The key is the module name, if it's not found the ``discussion-link`` directive will
render an empty node. The first value in the list is the channel id of the module, and
the second is forum id of the module.

The output string is generated by a series of replaces on template strings, which are
customizable using the config. To understand how it works, let's look at the default
configurations and their values:

- ``discussion_paragraph``: ``"This should have hopefully given you an overview of the 
  {{submodule}} submodule, if you have any questions, please feel free to reach out on 
  our [discord]({{discord_link}}) in the [{{submodule}} channel]({{channel_link}}) or in
  the [{{submodule}} forum]({{forum_link}})!"``
- ``discord_link``: ``"https://discord.gg/ZVQdvbzNQJ"``
- ``channel_link``: ``"https://discord.com/channels/799879767196958751/{{channel_id}}"``
- ``forum_link``: ``"https://discord.com/channels/799879767196958751/{{forum_id}}"``

Here is an example of how it works for ``ivy.functional.ivy.creation``:

1. First we resolve the ``{{submodule}}`` template string, which is the last part of the
   module name, in this case it's ``creation``.

   The result will be like this:

    This should have hopefully given you an overview of the 
    **creation** submodule, if you have any questions, please feel free to reach out on 
    our [discord]({{discord_link}}) in the [**creation** channel]({{channel_link}}) or in
    the [**creation** forum]({{forum_link}})!

2. Then we resolve the ``{{discord_link}}`` template string.

   The result will be like this:
    
    This should have hopefully given you an overview of the 
    creation submodule, if you have any questions, please feel free to reach out on 
    our [discord](**https://discord.gg/ZVQdvbzNQJ**) in the [creation channel]({{channel_link}}) or in
    the [creation forum]({{forum_link}})!

3. Then we resolve the ``{{channel_link}}`` template string.

   The result will be like this:
    
    This should have hopefully given you an overview of the 
    creation submodule, if you have any questions, please feel free to reach out on 
    our [discord](\https://discord.gg/ZVQdvbzNQJ) in the [creation channel](**https://discord.com/channels/799879767196958751/{{channel_id}}**) or in
    the [creation forum]({{forum_link}})!

4. Then we resolve the ``{{forum_link}}`` template string.

   The result will be like this:
    
    This should have hopefully given you an overview of the 
    creation submodule, if you have any questions, please feel free to reach out on 
    our [discord](\https://discord.gg/ZVQdvbzNQJ) in the [creation channel](\https://discord.com/channels/799879767196958751/{{channel_id}}) or in
    the [creation forum](**https://discord.com/channels/799879767196958751/{{forum_id}}**)!

5. We finally resolve ``{{channel_id}}`` and ``{{forum_id}}`` template strings.

   The result will be like this:
    
    This should have hopefully given you an overview of the 
    creation submodule, if you have any questions, please feel free to reach out on 
    our [discord](\https://discord.gg/ZVQdvbzNQJ) in the [creation channel](\https://discord.com/channels/799879767196958751/**1000043690254946374**) or in
    the [creation forum](\https://discord.com/channels/799879767196958751/**1028298816526499912**)!

6. After that we render the node paragraph as if it's a Markdown text resulting this:

    This should have hopefully given you an overview of the 
    creation submodule, if you have any questions, please feel free to reach out on 
    our `discord <https://discord.gg/ZVQdvbzNQJ>`_ in the `creation channel 
    <https://discord.com/channels/799879767196958751/1000043690254946374>`_ or in the
    `creation forum <https://discord.com/channels/799879767196958751/1028298816526499912>`_!

All of the above template strings can be customized using the configuration, so feel free
to change them to your liking.

``skippable_function``
~~~~~~~~~~~~~~~~~~~~~~

This extension provides a custom auto documenter ``autoskippablemethod`` that skip 
functions that match values in ``skippable_method_attributes`` configuration.

This is an example of ``skippable_method_attributes`` configuration in
``partial_conf.py``:

.. code-block:: python

    skippable_method_attributes = [
        {
            "__qualname__": "_wrap_function.<locals>.new_function"
        }
    ]

This will remove any function that has ``__qualname__`` attribute equal to 
``_wrap_function.<locals>.new_function``.

``ivy_data``
~~~~~~~~~~~~

This is a custom documenter for ``autodoc`` that document Ivy data attributes that live
in ``ivy.functional.ivy``, it will replace the module to ``ivy.`` instead of 
``ivy.functional.ivy.<submodule>``.

It's used instead of simply using ``ivy.<data atribute>`` because data attributes have
no ``__doc__`` atribute, instead docs are discovered by parsing the source code itself.
So for Sphinx to find the required docs, it need to be supplied the full module name,
then using ``autoivydata`` directive will replace the module name to ``ivy.``.

Please refer to the `auto documenter guide in sphinx documentation 
<https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html>`_ for more
info.
