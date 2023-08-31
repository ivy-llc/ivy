Tutorials And Examples
======================

Welcome to Ivy's tutorials webpage! Our goal is to provide you with a comprehensive 
learning experience on a variety of topics. We have organized our tutorials into three 
main sections to help you find exactly what you need. 

If you are in a rush, you can jump straight into the `Quickstart <quickstart.ipynb>`_,
a quick and general introduction to Ivy's features and capabilities!

- In the **Learn the basics** section, you will find basic and to the point tutorials to
  help you get started with Ivy.

- The **Guides** section includes more involved tutorials for those who want to dive 
  deeper into the framework. 

- Finally, in the **Examples and Demos** section, you will find start-to-finish projects
  and applications that showcase real-world applications of Ivy. Whether you're a 
  beginner or an advanced user, we've got you covered!

.. note::

    Want to use Ivy locally? Check out the `Get Started section of the docs 
    <https://unify.ai/docs/ivy/overview/get_started.html>`_!

Learn the basics
----------------

.. grid:: 1 1 3 3
    :gutter: 4

    .. grid-item-card:: Write Ivy Code
        :link: learn_the_basics/01_write_ivy_code.ipynb

        Get familiar with Ivy’s basic concepts and start writing framework-agnostic code.

    .. grid-item-card:: Unify Code
        :link: learn_the_basics/02_unify_code.ipynb

        Unify a simple ``torch`` function and use it alongside any ML framework!

    .. grid-item-card:: Compile Code
        :link: learn_the_basics/03_compile_code.ipynb

        Turn your Ivy code into an efficient fully-functional graph, removing wrappers and unused parts of the code.

    .. grid-item-card:: Transpile Code
        :link: learn_the_basics/04_transpile_code.ipynb

        Convert a ``torch`` function to ``jax`` with just one line of code.

    .. grid-item-card:: Lazy vs Eager
        :link: learn_the_basics/05_lazy_vs_eager.ipynb

        Understand the difference between eager and lazy compilation and transpilation.

    .. grid-item-card:: How to use decorators
        :link: learn_the_basics/06_how_to_use_decorators.ipynb

        Learn about the different ways to use compilation and transpilation functions.

    .. grid-item-card:: Transpile any library
        :link: learn_the_basics/07_transpile_any_library.ipynb

        Transpile the ``kornia`` library to ``jax`` with just one line of code.

    .. grid-item-card:: Transpile any model
        :link: learn_the_basics/08_transpile_any_model.ipynb

        Transpile a ``Keras`` model into a ``PyTorch`` module.

Guides
------

.. grid:: 1 1 3 3
    :gutter: 4

    .. grid-item-card:: Transpiling a PyTorch model to build on top
        :link: guides/01_transpiling_a_torch_model.ipynb

        Transpile a ``timm`` model to ``tensorflow`` and build a new model around it.

    .. grid-item-card:: Transpiling a Tensorflow model to build on top
        :link: guides/03_transpiling_a_tf_model.ipynb

        Transpile a ``keras`` model to ``torch`` and build a new model around it.

Examples and Demos
------------------

.. grid:: 1 1 3 3
    :gutter: 4

    .. grid-item-card:: Using Ivy ResNet
        :link: examples_and_demos/resnet_demo.ipynb

        Use the Ivy ``ResNet`` model for image classification.

    .. grid-item-card:: Accelerating PyTorch models with JAX
        :link: examples_and_demos/torch_to_jax.ipynb

        Accelerate your Pytorch models by converting them to JAX for faster inference.

    .. grid-item-card:: Accelerating MMPreTrain models with JAX
        :link: examples_and_demos/mmpretrain_to_jax.ipynb

        Accelerate your MMPreTrain models by converting them to JAX for faster inference.

    .. grid-item-card:: Image Segmentation with Ivy UNet
        :link: examples_and_demos/image_segmentation_with_ivy_unet.ipynb

        Use the Ivy ``UNet`` model for image segmentation.

    .. grid-item-card:: Ivy AlexNet demo
        :link: examples_and_demos/alexnet_demo.ipynb

        In this demo, we show how an AlexNet model written in…

.. toctree::
    :hidden:
    :maxdepth: 1

    self
    quickstart.ipynb

.. toctree::
    :hidden:
    :maxdepth: -1
    :caption: Learn the basics

    learn_the_basics/01_write_ivy_code.ipynb
    learn_the_basics/02_unify_code.ipynb
    learn_the_basics/03_compile_code.ipynb
    learn_the_basics/04_transpile_code.ipynb
    learn_the_basics/05_lazy_vs_eager.ipynb
    learn_the_basics/06_how_to_use_decorators.ipynb
    learn_the_basics/07_transpile_any_library.ipynb
    learn_the_basics/08_transpile_any_model.ipynb

.. toctree::
    :hidden:
    :maxdepth: -1
    :caption: Guides

    guides/01_transpiling_a_torch_model.ipynb
    guides/03_transpiling_a_tf_model.ipynb

.. toctree::
    :hidden:
    :maxdepth: -1
    :caption: Examples and Demos

    examples_and_demos/resnet_demo.ipynb
    examples_and_demos/torch_to_jax.ipynb
    examples_and_demos/mmpretrain_to_jax.ipynb
    examples_and_demos/image_segmentation_with_ivy_unet.ipynb
    examples_and_demos/alexnet_demo.ipynb
