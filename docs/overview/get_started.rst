Get Started
===========

Depending on your preferred environment you can install Ivy in various ways:

Installing using pip
--------------------

The easiest way to set up Ivy is to install it using pip with the following command:

.. code-block:: bash

    pip install ivy

Keep in mind that this **won't** install any framework other than NumPy! (PyTorch, TensorFlow, etc...)

Installing from source
----------------------

You can also install Ivy from source if you want to take advantage of the latest
changes:

.. code-block:: bash

    git clone https://github.com/ivy-llc/ivy.git
    cd ivy
    pip install --user -e .

When installing from source, we recommend installing ivy's dev dependencies with the commands:

.. code-block:: bash

    pip install -r requirements/requirements.txt
    pip install -r requirements/optional.txt

There are also other 'requirements/optional...' files in the 'requirements' folder that
can be install the dependencies for specific hardware, such as GPU machines or Apple silicon.

If you are planning to contribute, you want to run the tests, or you are looking
for more in-depth instructions, it's probably best to check out
the `Contributing - Setting Up <contributing/setting_up.rst>`_ page,
where OS-specific and IDE-specific instructions and video tutorials to install Ivy are available!

Docker
------

If you prefer to use containers, we also have pre-built Docker images with all the
supported frameworks and some relevant packages already installed, which you can pull from:

.. code-block:: bash

    docker pull ivyllc/ivy:latest

Ivy Folder
~~~~~~~~~~

When importing Ivy for the first time, a ``.ivy`` folder will be created in your
working directory. If you want to keep this folder in a different location,
you can set an ``IVY_ROOT`` environment variable with the path of your ``.ivy`` folder.

Issues and Questions
~~~~~~~~~~~~~~~~~~~~

If you find any issue or bug while using the tracer and/or the transpiler, please
raise an `issue in GitHub <https://github.com/ivy-llc/ivy/issues>`_ and add the ``tracer``
or the ``transpiler`` label accordingly. A member of the team will get back to you ASAP!

`Join ivy's discord server <https://discord.gg/huQXz3XN>`_
