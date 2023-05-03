Transpiler
==========

..

   ⚠️ **Warning**: The compiler and the transpiler are not publicly available yet, so certain parts of this doc won't work as expected as of now!


Ivy's Transpiler converts a function written in any framework into your framework of 
choice, preserving all the logic between frameworks. 
As the output of transpilation is native code in the target framework, it
can be used as if it was originally developed in that framework, 
allowing you to apply and use framework-specific optimizations or tools.

This makes all ML-related projects available to you, independently of the framework you 
want to use to research, develop, or deploy systems. So if you want to:

- Use functions and building blocks like neural networks, layers, activations, and 
  training pipelines published in other frameworks. Ex: Using Haiku modules in PyTorch to 
  get access to the latest model.
- Integrate code developed in other frameworks into your code. Ex: Use the Kornia 
  library in Jax for extra performance.
- Take advantage of specific features in other frameworks. Ex: Convert Jax code to Tensorflow for deployment. 

Ivy's Transpiler is definitely the tool for the job 🔧

To convert the code, it traces a computational graph using the Graph Compiler and 
leverages Ivy's frontends and backends to link one framework to another. After swapping 
each function node in the computational graph with their equivalent Ivy frontend 
functions, the compiler removes all the wrapping in the frontends and replaces them with the native
functions of the target framework.


Transpiler API
--------------

.. py:function:: ivy.transpile(*objs, source = None, to = None, debug_mode = False, args = None, kwargs = None, params_v = None,)

  Transpiles a ``Callable`` or set of them from a ``source`` framework to another framework. If ``args`` or ``kwargs`` are specified, 
  transpilation is performed eagerly, otherwise, transpilation will happen lazily.

  :param objs: Native callable(s) to transpile.
  :type objs: ``Callable``
  :param source: The framework that ``obj`` is from. This must be provided unless ``obj`` is a framework-specific module.
  :type source: ``Optional[str]``
  :param to: The target framework to transpile ``obj`` to.
  :type to: ``Optional[str]``
  :param debug_mode: Whether to transpile to ivy first, before the final compilation to
                     the target framework. If the target is ivy, then this flag makes no
                     difference.
  :type debug_mode: ``bool``
  :param args: If specified, arguments that will be used to transpile eagerly.
  :type args: ``Optional[Tuple]``
  :param kwargs: If specified, keyword arguments that will be used to transpile eagerly.
  :type kwargs: ``Optional[dict]``
  :param params_v: Parameters of a haiku model, as when transpiling these, the parameters
                   need to be passed explicitly to the function call.
  :rtype: ``Union[Graph, LazyGraph, ModuleType, ivy.Module, torch.nn.Module, tf.keras.Model, hk.Module]``
  :return: A transpiled ``Graph`` or a non-initialized ``LazyGraph``. If the object is an native trainable module, the corresponding module in the target framework will be returned. If the object is a ``ModuleType``, the function will return a copy of the module with every method lazily transpiled.

.. py:function:: ivy.unify(*objs, source = None, args = None, kwargs = None, **transpile_kwargs,)

  Transpiles an object into Ivy code. It's an alias to 
  ``ivy.transpile(..., to=”ivy”, ...)``

  :param objs: Native callable(s) to transpile.
  :type objs: ``Callable``
  :param source: The framework that ``obj`` is from. This must be provided unless ``obj`` is a framework-specific module.
  :type source: ``Optional[str]``
  :param args: If specified, arguments that will be used to unify eagerly.
  :type args: ``Optional[Tuple]``
  :param kwargs: If specified, keyword arguments that will be used to unify eagerly.
  :type kwargs: ``Optional[dict]``
  :param transpile_kwargs: Arbitrary keyword arguments that will be passed to ``ivy.transpile``.

  :rtype: ``Union[Graph, LazyGraph, ModuleType, ivy.Module]``
  :return: A transpiled ``Graph`` or a non-initialized ``LazyGraph``. If the object is an native trainable module, the corresponding module in the target framework will be returned. If the object is a ``ModuleType``, the function will return a copy of the module with every method lazily transpiled.

Using the transpiler
--------------------

Similar to the ``ivy.compile`` function, ``ivy.unify`` and ``ivy.transpile`` can be used
eagerly and lazily. If you pass the necessary arguments, the function will be called 
instantly, otherwise, transpilation will happen the first time you invoke the function 
with the proper arguments. 

In both cases, arguments or keyword arguments can be arrays from 
either the ``source`` framework or the target (``to``) framework.

Transpiling functions
~~~~~~~~~~~~~~~~~~~~~

First, let's start transpiling some simple functions. In the snippet below, we transpile
a small JAX function to Torch both eagerly and lazily.

.. code-block:: python

  import ivy
  ivy.set_backend("jax")

  # Simple JAX function to transpile
  def test_fn(x):
      return jax.numpy.sum(x)

  x1 = ivy.array([1., 2.])

  # Arguments are available -> transpilation happens eagerly
  eager_graph = ivy.transpile(test_fn, source="jax", to="torch", args=(x1,))

  # eager_graph is now torch code and runs efficiently
  ret = eager_graph(x1)

  # Arguments are not available -> transpilation happens lazily
  lazy_graph = ivy.transpile(test_fn, source="jax", to="torch") 

  # The transpiled graph is initialized, transpilation will happen here
  ret = lazy_graph(x1)

  # lazy_graph is now torch code and runs efficiently
  ret = lazy_graph(x1)

Transpiling Libraries
~~~~~~~~~~~~~~~~~~~~~

Likewise, you can use ``ivy.transpile`` to convert entire libraries and modules with just one line of 
code!

.. code-block:: python

  import ivy
  import kornia
  import requests
  import jax.numpy as jnp
  from PIL import Image

  # transpile kornia from torch to jax
  jax_kornia = ivy.transpile(kornia, source="torch", to="jax")

  # get an image
  url = "http://images.cocodataset.org/train2017/000000000034.jpg"
  raw_img = Image.open(requests.get(url, stream=True).raw)

  # convert it to the format expected by kornia
  img = jnp.transpose(jnp.array(raw_img), (2, 0, 1))
  img = jnp.expand_dims(img, 0) / 255

  # and use the transpiled version of any function from the library!
  out = jax_kornia.enhance.sharpness(img, 5)

Transpiling Modules
~~~~~~~~~~~~~~~~~~~

Last but not least, Ivy can also transpile trainable modules from one framework to 
another, at the moment we support ``torch.nn.Module`` when ``to=”torch”``, 
``tf.keras.Model`` when ``to=”tensorflow”``, and haiku models when ``to=”jax”``.

.. code-block::

  import ivy
  import timm
  import torch
  import jax
  import haiku as hk

  # Get a pretrained pytorch model
  mlp_encoder = timm.create_model("mixer_b16_224", pretrained=True, num_classes=0)

  # Transpile it into a hk.Module with the corresponding parameters
  noise = torch.randn(1, 3, 224, 224)
  mlp_encoder = ivy.transpile(mlp_encoder, to="jax", args=(noise,))

  # Build a classifier using the transpiled encoder
  class Classifier(hk.Module):
      def __init__(self, num_classes=1000):
          super(Classifier, self).__init__()
              self.encoder = mlp_encoder()
              self.fc = hk.Linear(output_size=num_classes, with_bias=True)

      def __call__(self, x):
          x = self.encoder(x)
          x = self.fc(x)
          return x

      def _forward_classifier(x):
          module = Classifier()
          return module(x)

  # Transform the classifier and use it as a standard hk.Module
  rng_key = jax.random.PRNGKey(42)
  x = jax.random.uniform(key=rng_key, shape=(1, 3, 224, 224), dtype=jax.numpy.float32)
  forward_classifier = hk.transform(_forward_classifier)
  params = forward_classifier.init(rng=rng_key, x=x)

  ret = forward_classifier.apply(params, None, x)

Ivy.unify
~~~~~~~~~

As mentioned above, ``ivy.unify`` is an alias to transpilation to Ivy, so you can use it
exactly in the same way to convert framework specific code to Ivy.

.. code-block:: python

  import ivy
  ivy.set_backend("jax")

  def test_fn(x):
      return jax.numpy.sum(x)

  x1 = ivy.array([1., 2.])

  # transpiled_func and unified_func will have the same result
  transpiled_func = ivy.transpile(test_fn, to="ivy", args=(x1,))
  unified_func = ivy.unify(test_fn, args=(x1,))

Sharp bits
----------

In a similar fashion to the compiler, the transpiler is under development and we are 
still working on some rough edges. These include:

1. **Keras model subclassing**: If a model is transpiled to keras, the resulting 
   ``tf.keras.Model`` can not be used within a keras sequential model at the moment. If 
   you want to use the transpiled model as part of a more complex keras model, you can 
   `create a Model subclass 
   <https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class>`_. 
   Due to this, any training of a keras model should be done using a TensorFlow training
   pipeline instead of the keras utils.
2. **Keras arguments**: Keras models require at least an argument to be passed, so if a 
   model from another framework that only takes ``kwargs`` is transpiled to keras, 
   you'll need to pass a ``None`` argument to the transpiled model before the 
   corresponding ``kwargs``.
3. **Haiku transform with state**: As of now, we only support transpilation of 
   transformed Haiku modules, this means that ``transformed_with_state`` objects will 
   not be correctly transpiled.
4. **Array format between frameworks**: As the compiler outputs a 1-to-1 mapping of the 
   compiled function, the format of the tensors is preserved when transpiling from a 
   framework to another. As an example, if you transpile a convolutional block from 
   PyTorch (which uses ``N, C, H, W``) to TensorFlow (which uses ``N, H, W, C``) and want
   to use it as part of a bigger (TensorFlow) model, you'll need to include a permute statement for 
   the inference to be correct. 

Keep in mind that the transpiler uses the graph compiler under the hood, so the 
`sharp bits of the compiler <https://unify.ai/docs/ivy/compiler/compiler.html#sharp-bits>`_ 
apply here as well!

Examples
--------

Here, we are transpiling a HF model from torch to tensorflow and then using the 
resulting model with tensorflow tensors directly:

.. code-block:: python

  import ivy
  from transformers import AutoImageProcessor, ResNetForImageClassification
  from datasets import load_dataset

  # Set backend to torch
  ivy.set_backend("torch")

  # Download the input image
  dataset = load_dataset("huggingface/cats-image")
  image = dataset["test"]["image"][0]

  # Setting the model
  image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
  model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

  # Transpiling the model to tensorflow
  tf_model = ivy.transpile(model, source="torch", to="tensorflow", kwargs=inputs)

  # Using the transpiled model
  tf_inputs = image_processor(image, return_tensors="tf")
  ret = tf_model(None, **tf_inputs)
