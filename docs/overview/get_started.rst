Get Started
===========

..

   If you want to use **Ivy's tracer and transpiler**, make sure to follow the
   :ref:`setting up instructions for the API key <overview/get_started:Ivy's tracer and transpiler>`
   after installing Ivy!


Depending on your preferred environment you can install Ivy in various ways:

Installing using pip
--------------------

The easiest way to set up Ivy is to install it using pip with the following command:

.. code-block:: bash

    pip install ivy

Keep in mind that this **won't** install any framework other than NumPy!

Docker
------

If you prefer to use containers, we also have pre-built Docker images with all the
supported frameworks and some relevant packages already installed, which you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest

If you are working on a GPU device, you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest-gpu

Installing from source
----------------------

You can also install Ivy from source if you want to take advantage of the latest
changes, but we can't ensure everything will work as expected!

.. code-block:: bash

    git clone https://github.com/unifyai/ivy.git
    cd ivy
    pip install --user -e .


If you are planning to contribute, you want to run the tests, or you are looking
for more in-depth instructions, it's probably best to check out
the `Contributing - Setting Up <contributing/setting_up.rst>`_ page,
where OS-specific and IDE-specific instructions and video tutorials to install Ivy are available!


Ivy's tracer and transpiler
-----------------------------

To use Ivy's tracer and transpiler, you'll need an **API key**. If you don't have one yet, you can
register in `the console <https://console.unify.ai/>`_ to get it!

Ivy Folder
~~~~~~~~~~

When importing Ivy for the first time, a ``.ivy`` folder will be created in your
working directory. If you want to keep this folder in a different location,
you can set an ``IVY_ROOT`` environment variable with the path of your ``.ivy`` folder.

Setting Up the API key
~~~~~~~~~~~~~~~~~~~~~~

Once the ``.ivy`` folder has been created (either manually or automatically by
importing Ivy), you will have to paste your API key as the content of the ``key.pem`` file.
For reference, this would be equivalent to:

.. code-block:: bash

    echo -n API_KEY > .ivy/key.pem

Issues and Questions
~~~~~~~~~~~~~~~~~~~~

If you find any issue or bug while using the tracer and/or the transpiler, please
raise an `issue in GitHub <https://github.com/unifyai/ivy/issues>`_ and add the ``tracer``
or the ``transpiler`` label accordingly. A member of the team will get back to you ASAP!

Otherwise, if you haven't found a bug but want to ask a question, suggest something, or get help
from the team directly, feel free to open a new post at the ``pilot-access`` forum in
`Ivy's discord server! <https://discord.com/invite/sXyFF8tDtm>`_
