Building the Docs
=================

.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`doc building channel`: https://discord.com/channels/799879767196958751/982733967998484520
.. _`miniconda`: https://docs.conda.io/en/latest/miniconda.html
.. _`venv`: https://docs.python.org/3/library/venv.html

Building the Docs with Docker
-----------------------------

#. Build the docker image using the doc-builder Dockerfile
#. Clone or pull the :code:`unifyai/doc-builder` repository
#. Change to the doc-builder folder

    .. code-block:: none

        cd doc-builder

#. Build the docker image using the following command

    #. For Windows and MacOS users:

        .. code-block:: none

            docker build -t unifyai/doc-builder:latest -f Dockerfile .
        
    #. For Ubuntu users: 

        .. code-block:: none

            docker build --build-arg user=<your-user-name> --build-arg uid=<your-uid>  -t unifyai/doc-builder:latest -f Dockerfile .

#. Build the docs using the image:

    #. Clone or pull the :code:`unifyai/ivy` repository.
    #. Change to the docs folder

        .. code-block:: none

            cd ivy\docs

    #. Build the docs using :code:`make_docs.sh`
        
        #. For Windows users

            .. code-block:: none
                
                sh make_docs.sh
        
        #. For MacOS users
            
            .. code-block:: none

                bash make_docs.sh root
        
        #. For Ubuntu users

            .. code-block:: none

                ./make_docs.sh

Dockerless Doc-building
-----------------------

To build the Ivy documentation locally, we will need to clone both the :code:`unifyai/doc-builder` and :code:`unifyai/ivy` repositories.
After cloning repos, we need to create a virtual environment using either `miniconda`_ or `venv`_:

Using miniconda
***************

#. Install `miniconda`_
#. Open conda terminal
#. Create the environment by running the command (:code:`ivy_dev` is the name of the environment)

    .. code-block:: none

        conda create --name ivy_dev python=3.8.10

#. Activate the environment by:

    .. code-block:: none

        conda activate ivy_dev

Using venv
**********

This is a built-in package and doesn't require explicit installation.

#. Open your terminal/cmd in the directory where you would like to have the folder with the environment files
#. Create the environment by running the command below with a new environment name. We named it :code:`ivy_dev` like above.

    .. code-block:: none

        python -m venv ivy_dev

    Try :code:`python3` if :code:`python` doesn't work.

#. Activate the created environment by running (in the same working directory as the environment folder)

    .. code-block:: none

        ivy_dev\Scripts\activate.bat

    (on Windows)

    OR

    .. code-block:: none

        source ivy_dev/bin/activate

    (on Mac/Linux)

Building the docs
*****************

After setting up the virtual environment, we can build the documentation locally.
Move to the :code:`ivy/docs` folder in a terminal and execute the following command to build the docs from inside the virtual environment.

1. For Windows users

    .. code-block:: none
                    
        sh make_docs_without_docker.sh 
        
2. For MacOS users
            
    .. code-block:: none

        bash make_docs_without_docker.sh <path to doc-builder folder relative to current folder i.e. ivy/docs>
            
3. For Ubuntu users

    .. code-block:: none

        bash make_docs_without_docker.sh <path to doc-builder folder relative to current folder i.e. ivy/docs>

Assuming that we cloned the :code:`doc-builder` and :code:`ivy` repositories in the same folder, we can run :code:`make_docs_without_docker.sh` as follows:

.. code-block:: none

        bash make_docs_without_docker.sh ../../doc-builder/

You will now see the `build` folder where you can find generated `html` files.
You can open them using any preferred browser and see how your changes affected the documentation pages.

Working of the Documentation Building Pipeline
----------------------------------------------

Following are the files involved in the documentation building pipeline:

make_docs.sh:
*************

This file exists in the docs folder of the project for which documentation is to be generated.
It uses the unifyai/doc-builder:latest image and creates a container for documentation building.

entrypoint.sh:
**************

This is the first file involved in the documentation building pipeline. 
It installs the requirements, synchronizes the container's folder with the project's build folder.
It is responsible for executing the rest of the pipeline as it runs the :code:`_make_docs.sh` and :code:`remove_docs.sh` files.

_make_docs.sh:
**************

| This file contains commands for executing files involved in generation of the documentation. _ at the start is to indicate that this is a privately accessed file.
| This file fulfils the following purpose
| 1. It deletes any previously generated documentation in the :code:`autogenerated_source` folder and the :code:`build` folder.
| 2. Then it executes the :code:`generate_src_rst_files.py`` file, followed by the :code:`sphinx-build.py` file.
| 3. Then it deletes all files of format :code:`X.png` from the :code:`build/_images`` folder.
| 4. Lastly, it executes the :code:`correct_built_html_files.py` file.

generate_src_rst_files.py:
**************************

| This file is the first stage of documentation generation. It involves the use of code docstrings and project structure to generate rst files for the project.
| 
| The :code:`main` function does the following:
| 1. It looks for the submodules to be skipped, stepped into and processed out of order.
| 2. It reads the configuration from the :code:`partial_source/conf.py` file.
| 3. It deletes any previously generated documentation in the :code:`autogenerated_source` folder.
| 4. It copies files from the :code:`partial_source/images` folder to the :code:`build/_images` folder.
| 5. It calls the :code:`create_rst_files` function.
| 
| The :code:`create_rst_files` function does the following:
| 1. Lists all files in the current folder.
| 2. Filters out the submodules to be skipped.
| 3. Recursively accesses all submodules.
| 4. For every submodule, it creates a directory structure.
| 5. Every python module will be represented with a folder which will contain rst files for all its functions and a rst file which will use these files to generate the overall markup for the module.
| 6. Writing the rst files involves extracting function and class names using the :code:`get_functions_and_classes` function, followed by their doctrings.
| 7. A README.rst file is generated for every module and is named as module_name.rst using the :code:`copy_readme_to_rst` function.
| 8. The Table of Contents(TOC) tree is generated for the module according to the order followed and is appended to the rst file using the :code:`append_toctree_to_rst`.
| 9. An index.rst file is generated for the root directory using the :code:`create_index_rst` function.
| 10. If a module is to be stepped into, then this folder structure is not generated for it.

sphinx-build.py
***************

This file is used to generate the HTML files for the documentation using the rst files generated previously.

correct_built_html_files.py
***************************

This file is used for further processing on the HTML files generated by Sphinx.
This involves replacing 3.14 with π, updating paths according to current folder, and updating namespaces in the code in the documentation.

remove_files.sh
***************

All files involved in the generation of documentation are removed by executing these commands.

**Round Up**

This should have hopefully given you a good understanding of the basics for
building the docs locally.

If you have any questions, please feel free to reach out on `discord`_ in the `doc building channel`_!
