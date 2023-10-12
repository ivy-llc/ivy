Setting Up
==========

To use Ivy's compiler and transpiler, you'll need an **API key**. We are starting to 
grant pilot access to certain users, so you can `join the waitlist <https://console.unify.ai/>`_ 
if you want to get one! 

Ivy Folder
----------

When importing Ivy for the first time, a ``.ivy`` folder will be created in your 
working directory. If you want to keep this folder in a different location, 
you can set an ``IVY_ROOT`` environment variable with the path of your ``.ivy`` folder.

Setting Up the API key
----------------------

Once the ``.ivy`` folder has been created (either manually or automatically by 
importing Ivy), you will have to paste your API key as the content of the ``key.pem`` file.
For reference, this would be equivalent to:

.. code-block:: console

    echo -n API_KEY > .ivy/key.pem

Issues and Questions
--------------------

If you find any issue or bug while using the compiler and/or the transpiler, please
raise an `issue in GitHub <https://github.com/unifyai/ivy/issues>`_ and add the ``compiler`` 
or the ``transpiler`` label accordingly. A member of the team will get back to you ASAP!

Otherwise, if you haven't found a bug but want to ask a question, suggest something, or get help 
from the team directly, feel free to open a new post at the ``pilot-access`` forum in 
`Ivy's discord server! <https://discord.com/invite/sXyFF8tDtm>`_ 