Graph Compiler
==============

..

   ⚠️ **Warning**: The compiler and the transpiler are not publicly available yet, so certain parts of this doc won't work as expected as of now!


When we call an Ivy function, there is always a small performance hit due to added 
Python wrapping. This overhead becomes increasingly noticeable when we use large 
models with multiple function calls. The Graph Compiler improves the performance of 
Ivy by removing the extra wrapping around each function call. 

The Graph Compiler takes in any Ivy function, framework-specific (backend) function, 
or composition of both, and produces a simplified executable computation graph composed 
of functions from the backend functional API only, which results in:

- Simplified code: The Graph Compiler simplifies the code by removing all the wrapping 
  and functions that don't contribute to the output: print statements, loggers, etc.
- Improved performance: The compiled graph has no performance overhead due to Ivy's 
  function wrapping, likewise, redundant operations from the original function are also 
  removed, increasing its overall performance.

Compiler API
------------

.. py:function:: ivy.compile(*objs, stateful = None, arg_stateful_idxs = None, kwarg_stateful_idxs = None, to = None, include_generators = True, array_caching = True, return_backend_compiled_fn = False, static_argnums = None, static_argnames = None, args = None, kwargs = None,)
    
    Compiles a ``Callable`` or set of them into an Ivy graph. If ``args`` or ``kwargs`` are specified, 
    compilation is performed eagerly, otherwise, compilation will happen lazily.
    
    :param objs: Callable(s) to compile and create a graph of.
    :type objs: ``Callable``
    :param stateful: List of instances to be considered stateful during the graph compilation.
    :type stateful: ``Optional[List]``
    :param arg_stateful_idxs: Positional arguments to be considered stateful during the graph compilation.
    :type arg_stateful_idxs: ``Optional[List]``
    :param kwarg_stateful_idxs: Keyword arguments to be considered stateful during the graph compilation.
    :type kwarg_stateful_idxs: ``Optional[List]``
    :param to: Backend that the graph will be compiled to. If not specified, the current backend will be used.
    :type to: ``Optional[str]``
    :param include_generators: Include array creation/generation functions as part of the graph.
    :type include_generators: ``bool``
    :param array_caching: Cache the constant arrays that appear as arguments to the functions in the graph.
    :type array_caching: ``bool``
    :param return_backend_compiled_fn: Whether to apply the native compilers, i.e. tf.function, after ivy's compilation.
    :type return_backend_compiled_fn: ``bool``
    :param static_argnums: For jax's jit compilation.
    :type static_argnums: ``Optional[Union[int, Iterable[int]]]``
    :param static_argnames: For jax's jit compilation.
    :type static_argnames: ``Optional[Union[str, Iterable[str]]]``
    :param args: Positional arguments for obj.
    :type args: ``Optional[Tuple]``
    :param kwargs: Keyword arguments for obj.
    :type kwargs: ``Optional[dict]``
    :rtype: ``Union[Graph, LazyGraph, ivy.Module, ModuleType]``
    :return: A compiled ``Graph`` or a non-initialized ``LazyGraph``. If the object is an ``ivy.Module``, the forward pass will be compiled and the same module will be returned. If the object is a ``ModuleType``, the function will return a copy of the module with every method lazily compiled.

Using the compiler
------------------

To use the ``ivy.compile()`` function, you need to pass a callable object and the corresponding inputs
to the function.

Let's start with a simple function:

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(x, y):
        z = x**y
        print(z)
        k = x * y
        j = ivy.concat([x, z, y])
        sum_j = ivy.sum(j)
        return z

    x = ivy.array([1, 2, 3])
    y = ivy.array([2, 3, 4])

    # Compile the function
    compiled_fn = ivy.compile(fn, args=(x, y))

In this case, the compiled graph would be:

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/compiler/figure1.png

From the graph, we can observe that:

1. As ``x`` and ``y`` are the only variables used when calculating the returned value ``z``,
   the non-contributing variable(s), ``k`` was not included in the graph. Function calls that 
   don't contribute to the output like the ``print`` function were also excluded.
2. As we set the backend to ``torch`` during the compilation process, the compiled 
   functions are torch functions, and the input and output types are torch tensors.
3. The tensor shape in the graph only indicates the shape of the inputs the graph was 
   traced with. The compiler doesn't impose additional restrictions on the shape or 
   datatype of the input array(s).

.. code-block:: python

    # Original set of inputs
    out = compiled_fn(x, y)

    # Inputs of different shape
    a = ivy.array([[1., 2.]])
    b = ivy.array([[2., 3.]])

    # New set of inputs
    out = compiled_fn(x, y)

Eager vs lazy Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

The graph compiler runs the original function under the hood and tracks its computation 
to create the compiled graph. The **eager compilation** method traces the graph in the 
corresponding function call with the specified inputs before we use the compiled 
function.

Instead of compiling functions before using them, Ivy also allows you to compile the 
function dynamically. This can be done by passing only the function to the 
compile method and not including the function arguments. In this case, the output will be a 
``LazyGraph`` instead of a ``Graph`` instance. When this ``LazyGraph`` object is first invoked with 
function arguments, it compiles the function and returns the output of the compiled 
function. Once the graph has been initialized, calls to the ``LazyGraph`` object will 
use the compiled function to compute the outputs directly.

.. code-block:: python

    # Compile the function eagerly (compilation happens here)
    eager_graph = ivy.compile(fn, args=(x, y))

    # Compile the function lazily (compilation does not happen here)
    lazy_graph = ivy.compile(fn)

    # Compile and return the output
    out = lazy_graph(x, y)

To sum up, lazy compilation enables you to delay the compilation process until you have 
the necessary inputs during execution. This is particularly useful in cases like 
compiling libraries, where it’s not feasible to provide valid arguments for every 
function call.

Now let's look at additional functionalities that you can find in the 
compiler.

Array caching
~~~~~~~~~~~~~

The compiler is able to cache constant arrays and their operations through the 
``array_caching`` flag, reducing computation time after compilation.

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(x):
        b = ivy.array([2])
        a = ivy.array([2])
        z = x ** (a + b)
        return z

    comp_func = ivy.compile(fn, args=(x,))

When calling ``ivy.compile()``, the ``array_caching`` argument is set to ``True`` by 
default, which returns the following graph.

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/compiler/figure2.png

This shows that by caching the constant operation in the graph, a simpler graph can be 
obtained. However, if desired, this argument can be set to ``False``, which results in the 
graph below. This ultimately results in a trade-off between time and memory, as 
cached results need to be stored in memory but if they are not cached these operations 
need to be performed.

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/compiler/figure3.png

Generators
~~~~~~~~~~

By using the ``include_generators`` argument, you can choose whether generator functions
are included as nodes or "baked" into the graph.

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(x):
        a = torch.randint(0, 100, size=[1])
        z = x ** a
        return z + torch.rand([1])
        
    comp_func = ivy.compile(fn, include_generators=True, args=(x,))

Returns:

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/compiler/figure4.png

And instead,

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(x):
        a = torch.randint(0, 100, size=[1])
        z = x * a
        return z + torch.rand([1])

    comp_func = ivy.compile(fn, include_generators=False, args=(x,))

Returns:

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/compiler/figure5.png

Stateful
~~~~~~~~

Finally, you can also track ``__setattr__`` and ``__getattr__`` methods of 
arbitrary classes using the ``stateful`` parameters.

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(cont, x):
        cont.new_attribute = x
        return x + 1

    x = torch.tensor([0])
    cont = ivy.Container(x=x)

    args = (cont.cont_deep_copy(), x)
    comp_func = ivy.compile(fn, arg_stateful_idxs=[[0]], args=args)

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/compiler/figure6.png

Sharp bits
----------

As some parts of the graph compiler are still under development, there are some sharp 
bits to take into account when using it. All of these points are WIP, so they'll be 
removed soon!

1. **Dynamic control flow**: The compiled graph is built using function tracing at the 
   moment, so dynamic control flow such as conditional branches or conditional loops 
   will not be registered correctly. As an example, if there is a while loop in your 
   code that depends on a changing value, the number of iterations in the final graph 
   will be the same as the number of iterations performed with the input passed to the 
   compile function.
2. **Non-framework-specific code**: As the compiler traces the function using the 
   functional API of the underlying framework, any piece of code inside the model that 
   is not from said framework will not be correctly registered, this includes other 
   frameworks code (such as NumPy statements inside a torch model) or python statements 
   such as len().
3. **Incorrectly cached parts of the graph**: There are certain cases where compilation 
   can succeed but hide some cached parts of the graph which shouldn't really be cached.
   To check this, it's recommended to compile with a noise array of the same shape and 
   then check if the output of the original function and the compiled graph with another
   input is the same. If you find out that the graph is not right, feel free to open an 
   `issue <https://github.com/unifyai/ivy/issues>`_ with a minimal example and we'll look 
   into it!

Examples
--------

Below, we compile a ResNet50 model from 
`Hugging Face <https://huggingface.co/microsoft/resnet-50>`_ and use it to classify the 
breed of a cat.

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

    # Preprocessing the input image
    inputs = image_processor(image, return_tensors="pt")

Normally, we would then feed these inputs to the model itself without compiling it

.. code-block:: python

    # Normal flow using pytorch
    with torch.no_grad():
    logits = model(**inputs).logits

With ivy, you can compile your model to a computation graph for increased performance.

.. code-block:: python

    # Compiling the model
    compiled_graph = ivy.compile(model, args=(**inputs,))

    # Using the compiled function
    logits = compiled_graph(**inputs).logits

Time for the final output of our computation graph.

.. code-block:: python

    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
