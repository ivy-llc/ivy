class DefaultDevice:
    """"""

    # noinspection PyShadowingNames
    
    @handle_out_argument
    @infer_device
    def __init__(
        self,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> None:

        """Initialises the DefaultDevice class

        Parameters
        ----------
        self
            first input array. An instance of the same class
        device
            second input array. The device string is an ivy device or nativedevice class
        
        Returns
        -------
        ret
 				    None
        
	      This function conforms to the `Array API Standard <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
        `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_
        in the standard.

        Examples
        --------
        >>> x = ivy.DefaultDevice("cpu")
            print(x)
        >>> y = ivy.DefaultDevice("gpu:0")
            print(y)
        >>> z = ivy.DefaultDevice("tpu:0")
            print(z)
        >>> print(self._dev)

        """
        self._dev = device
