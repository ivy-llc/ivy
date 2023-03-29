Graph Compiler
==============

When we call an Ivy function, there is always a small performance hit due to added 
Python wrapping. This performance gap becomes increasingly visible when we use large 
models with multiple function calls. The Ivy graph compiler improves the performance of 
Ivy by removing the extra wrapping around each function call. The compiler takes in any 
Ivy function, framework-specific (backend) function, or composition of both, and returns
a simplified executable computation graph composed of the backend functional API only.

While the graph compiler is also used during transpilation, it's a stand-alone module 
that can be used independently.

Benefits of using the Graph Compiler
------------------------------------

- Simplified code: The graph compiler simplifies the code by removing all the wrapping 
  and functions that don't contribute to the output: print statements, logger, etc.
- Improved performance: The compiled function has no performance overhead due to Ivy's 
  function wrapping. The compiler also removes redundant operations from the computation
  graph, increasing the overall performance of the original function.

Compiler API
------------

.. py:function:: ivy.compile(*objs, stateful = None, arg_stateful_idxs = None, kwarg_stateful_idxs = None, to_ivy  = False, include_generators = True, array_caching = True, time_chronological = True, return_backend_compiled_fn = False, static_argnums = None, static_argnames = None, args = None, kwargs = None,)
    
    Returns either a compiled Graph or a non-initialized LazyGraph.

    If args and kwargs are specified, compilation is performed eagerly, otherwise, 
    compilation will happen lazily.
    
    :param objs: Callable(s) to compile and create a graph of.
    :type objs: ``Callable``
    :param stateful: List of instances to be considered stateful during the graph compilation.
    :type stateful: ``Optional[List]``
    :param arg_stateful_idxs: Positional arguments to be considered stateful during the graph compilation.
    :type arg_stateful_idxs: ``Optional[List]``
    :param kwarg_stateful_idxs: Keyword arguments to be considered stateful during the graph compilation.
    :type kwarg_stateful_idxs: ``Optional[List]``
    :param to_ivy: Whether to compile the code into a graph composed by Ivy functions.
    :type to_ivy: ``bool``
    :param include_generators: Include array creation/generation functions as part of the graph.
    :type include_generators: ``bool``
    :param array_caching: Cache the constant arrays that appear as arguments to the functions in the graph.
    :type array_caching: ``bool``
    :param time_chronological: Whether to run the operations in the compiled function, in the same order as they are run in fn.
    :type time_chronological: ``bool``
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
    :return: A compiled Graph or a non-initialized LazyGraph.
    :rtype: ``Union[Graph, LazyGraph]``

.. rubric:: Example

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

Using the compiler
------------------

As mentioned previously, the graph compiler removes all the wrapping and functions that 
don't contribute to the output and builds a graph of functions from the underlying 
framework making the compiled graph faster than the original function.

To use the ``ivy.compile()`` function, you need to pass a callable object and its inputs
to the function.

Let's start by compiling a simple function:

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

    # View the graph
    compiled_fn.show()

The compiler generates the following graph:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler/figure1.png

From the graph, we can observe that:

1. As ``x`` and ``y`` are the only variables used when calculating the returned value z,
   the non-contributing variable(s), k was not included in the graph. Function calls that 
   don't contribute to the output like the print function were also excluded.
2. Torch was set as the backend during the compilation process. So the compiled 
   functions are torch functions, the input and output types are torch tensors.
3. The tensor shape in the graph only indicates the shape of the inputs the graph was 
   traced with. The compiler doesn't impose additional restrictions on the shape or 
   datatype of the input array(s).

.. code-block:: python

    out = compiled_fn(x, y) # orginial set of inputs

.. code-block:: python

    # Try with inputs of different shape
    a = ivy.array([[1., 2.]])
    b = ivy.array([[2., 3.]])

    out = compiled_fn(x, y) # check with different set of inputs

Eager vs lazy Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

The graph compiler runs the original function under the hood and tracks its computation 
to create the compiled graph. The **eager compilation** method traces the graph in the 
corresponding function call with the specified inputs before we use the compiled 
function.

Instead of compiling functions before using them, Ivy also allows you to compile the 
function dynamically on the fly. This can be done by passing only the function to the 
compile method and not including the function arguments. In this case, the output is a 
LazyGraph instead of a Graph object. When this LazyGraph object is first invoked with 
function arguments, it compiles the function and returns the output of the compiled 
function. Next time, when the LazyGraph is invoked, it uses the compiled function to 
compute the outputs.

.. code-block:: python

    # Compile the function eagerly (compilation happens here)
    eager_graph = ivy.compile(fn, args=(x, y))

    # Compile the function lazily (compilation does not happen here)
    lazy_graph = ivy.compile(fn)

    # Compile and return the output
    out = lazy_graph(x, y)

To sum up, lazy compilation enables you to delay the compilation process until you have 
the necessary inputs during execution. This is particularly useful in cases like 
transpiling libraries, where it’s not feasible to provide valid arguments for every 
function call.

Now let's look at examples of additional functionalities you can specify in the 
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

    comp_func = ivy.compile(fn, x)

When calling ``ivy.compile()``, the ``array_caching`` argument is set to True by 
default, which returns the following graph.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler/figure2.png

This shows that by catching the constant operation in the graph, a simpler graph can be 
obtained. However, if desired, this argument can be set to false, resulting in the 
following graph. This ultimately results in a trade-off between time and memory, as 
cached results need to be stored in memory but if they are not cached these operations 
need to be performed.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler/figure3.png

Generators
~~~~~~~~~~

By using the ``include_generators`` argument, you can choose whether generator functions
can be included or "baked" into the graph.

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(x):
        a = torch.randint(0, 100, size=[1])
        
    comp_func = ivy.compile(fn, x, include_generators=True)

Returns:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler/figure4.png

And instead,

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(x):
        a = torch.randint(0, 100, size=[1])
        z = x * a
        return z + torch.rand([1])

    comp_func = ivy.compile(fn, x, include_generators=False)

Returns:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler/figure5.png

Stateful
~~~~~~~~

Finally, you can choose to track ``__setattr__`` and ``__getattr__`` methods of 
arbitrary classes.

.. code-block:: python

    import ivy

    ivy.set_backend("torch")

    def fn(cont, x):
        cont.new_attribute = x
        return x + 1

    x = torch.tensor([0])
    cont = ivy.Container(x=x)

    comp_func = ivy.compile(fn, cont.cont_deep_copy(), x, arg_stateful_idxs=[[0]])

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/compiler/figure6.png

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
   input is the same.

Examples
--------

.. code-block:: python

    import ivy, time

    ivy.set_backend("torch")
    x = ivy.array([1.])

    def fn(x):
        y = ivy.sum(x)
        z = ivy.prod(x)
        a = ivy.sin(y)
        b = ivy.cos(z)
        c = ivy.tan(z)
        i = ivy.round(a)
        j = ivy.floor(b)
        k = ivy.ceil(c)
        return i, j, k

    graph = ivy.compile(fn, args=(x,))

    start = time.time()
    fn(x)
    print(time.time() - start)
    # 0.0003559589385986328

    start = time.time()
    graph(x)
    print(time.time() - start)
    # 0.0001785755157470703
