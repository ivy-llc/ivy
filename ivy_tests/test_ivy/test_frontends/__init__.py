class NativeClass:
    """An empty class to represent a class that only exist in a specific
    framework.

    Attributes
    ----------
    _native_class : class reference
        A reference to the framework-specific class.
    """

    def __init__(self, native_class):
        """Construct the native class object.

        Parameters
        ----------
        native_class : class reference
            A reperence to the framework-specific class being represented.
        """
        self._native_class = native_class
