Glossary
========

All of these new words can get confusing! We've created a glossary to help nail down some Ivy terms that you might find tricky.

.. glossary::
    :sorted:

    Pipeline
        A pipeline is a means of automating the machine learning workflow by enabling data to be transformed and correlated into a model that can then be analyzed to achieve outputs.
    
    Ivy Backends
        Ivy Backends are supported frameworks that Ivy can convert code to. The default is NumPy.
    
    Ivy Frontends
        Ivy Frontends are supported frameworks that Ivy can convert code from.
    
    Framework
         Frameworks are interfaces that allow scientists and developers to build and deploy machine learning models faster and easier. E.g. Tensorflow and PyTorch.
    
    Ivy Transpiler
        The transpiler allows framework to framework code conversions for supported frameworks.
    
    Ivy Container
        An Ivy class which inherits from :code:`dict` allows for storing nested data.
    
    Ivy Compiler
        A wrapper function around native compiler functions, which uses lower level compilers such as XLA to compile to lower level languages such as C++, CUDA, TorchScript etc.
    
    Graph Compiler
        Graph compilers map the high-level computational graph coming from frameworks to operations that are executable on a specific device.

    Ivy Graph Compiler
        Ivy's Graph Compiler, which traces the graph as a composition of functions in the functional API in Python.
    
    Ivy Functional API
        Is used for defining complex models, the Ivy functional API does not implement its own backend but wraps around other frameworks functional APIs and brings them into alignment.
    
    Framework Handler
    Backend Handler
        Used to control which framework Ivy is converting code to.
    
    Automatic Code Conversions
        Allows code to be converted from framework to another whilst retaining its funtional assets.
    
    Applied Libraries
        Suite of various machine learning libraries that have been built using the Ivy framework.
     
    Ivy Builder
         Helpful classes and functions for creating training workflows.
     
    Primary Functions
        Primary functions are the lowest level building blocks in Ivy and are generally implemented as light wrapping around an existing function in the backend framework, which serves a near-identical purpose.
     
    Compositional Functions
        Compositional functiona are functions that are implemented as a composition of other Ivy functions,
     
    Mixed Functions
        Mixed functions are functions that have some backend-specific implementations but not for all backends. 
     
    Standalone Functions
        Standalone functions are functions which do not reference any other primary, compositional or mixed functions whatsoever. These are mainly convenience functions.
     
    Nestable Functions
        Nestable functions are functions which can accept :code:`ivy.Container` instances in place of any of the arguments. 
     
    Convenience Functions
        Convenience functions can be used to organize and improve the code for other functions.
    
    Native Array
        The :code:`ivy.NativeArray` is simply a placeholder class for a backend-specific array class, such as :code:`np.ndarray`, :code:`tf.Tensor` or :code:`torch.Tensor`.
    
    Ivy Array
        The :code:`ivy.Array` is a simple wrapper class, which wraps around the :code:`ivy.NativeArray`.
    
    Submodule Helper Functions
        These are standalone/convenience functions that are specific to a submodule.

