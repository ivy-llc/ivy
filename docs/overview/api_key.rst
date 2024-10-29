Setting up your API key
=======================

You can generate a personal api key for Ivy by signing up for free at https://ivy.dev/signin.
This will grant you additional transpilations that can be used for completely free on any code! Find out more at https://ivy.dev/.

Using your API key
------------------

There are two ways to use your api key in your local environment. The easiest is to export it as the IVY_KEY environment variable, like so:

.. code-block:: bash

    export IVY_KEY=<my_key>

The other way is to create a 'ivy-key.pem' file in your local directory, containing just your api key. Ivy should automatically detect this file and use your key.

If you run into any problems with your api key, feel free to raise an issue at https://github.com/ivy-llc/ivy/issues, or get in touch at contact@ivy.dev.
