import warnings
import ivy


# helpers


def _ensure_tuple(value):
    """Returns a tuple if `value` isn't one already."""
    if isinstance(value, int):
        if value == 1:
            return ()
        else:
            return (value,)
    elif isinstance(value, tuple):
        if value == (1,):
            return ()
        return tuple(value)
    else:
        return tuple(value)


class MetaFactorizedTensor(type):
    """
    Meta class for tensor factorizations.

    .. info::

        1. Calls __new__ normally.
        2. Removes the keyword argument 'factorization' if present
        3. Calls __init__ with the remaining *args and **kwargs
    """

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        kwargs.pop("factorization", None)

        instance.__init__(*args, **kwargs)
        return instance


def _format_factorization(factorization):
    """
    Small utility function to make sure factorization names are dealt with the same
    whether using capital letters or not.

    factorization=None is remapped to 'Dense'.
    """
    if factorization is None:
        factorization = "Dense"
    return factorization.lower()


class FactorizedTensor(metaclass=MetaFactorizedTensor):
    """
    Tensor in Factorized form.

    .. important::

       All tensor factorization must have an `order` parameter
    """

    _factorizations = dict()

    def __init_subclass__(cls, name, **kwargs):
        """When a subclass is created, register it in _factorizations."""
        super().__init_subclass__(**kwargs)

        if name != "":
            cls._factorizations[_format_factorization(name)] = cls
            cls._name = name
        else:
            if (
                cls.__name__ != "TensorizedTensor"
            ):  # Don't display warning when instantiating the TensorizedTensor class
                warnings.warn(
                    f"Creating a subclass of FactorizedTensor {cls.__name__} with no"
                    " name."
                )

    def __new__(cls, *args, **kwargs):
        """
        Customize the creation of a factorized convolution.

        Takes a parameter `factorization`, a string that specifies with subclass to use

        Returns
        -------
        FactorizedTensor._factorizations[_format_factorization(factorization)]
            subclass implementing the specified tensor factorization
        """
        if cls is FactorizedTensor:
            factorization = kwargs.get("factorization")
            try:
                cls = cls._factorizations[_format_factorization(factorization)]
            except KeyError:
                raise ValueError(
                    f"Got factorization={factorization} but expected"
                    f"one of {cls._factorizations.keys()}"
                )

        instance = super().__new__(cls)

        return instance

    def __getitem__(indices):
        """
        Returns raw indexed factorization, not class.

        Parameters
        ----------
        indices : int or tuple
        """
        raise NotImplementedError

    @classmethod
    def new(cls, shape, rank="same", factorization="Tucker", **kwargs):
        """
        Main way to create a factorized tensor.

        Parameters
        ----------
        shape : tuple[int]
            shape of the factorized tensor to create
        rank : int, 'same' or float, default is 'same'
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'Tucker'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Examples
        --------
        Create a Tucker tensor of shape `(3, 4, 2)`
        with half the parameters as a dense tensor would:

        >>> tucker_tensor = FactorizedTensor.new((3, 4, 2)), rank=0.5, factorization='tucker')

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        try:
            cls = cls._factorizations[_format_factorization(factorization)]
        except KeyError:
            raise ValueError(
                f"Got factorization={factorization} but expected"
                f"one of {cls._factorizations.keys()}"
            )

        return cls.new(shape, rank, **kwargs)

    @classmethod
    def from_tensor(cls, tensor, rank, factorization="CP", **kwargs):
        """
        Create a factorized tensor by decomposing a dense tensor.

        Parameters
        ----------
        tensor : torch.tensor
            tensor to factorize
        rank : int, 'same' or float
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        try:
            cls = cls._factorizations[_format_factorization(factorization)]
        except KeyError:
            raise ValueError(
                f"Got factorization={factorization} but expected"
                f"one of {cls._factorizations.keys()}"
            )

        return cls.from_tensor(tensor, rank, **kwargs)

    @property
    def decomposition(self):
        """Returns the factors and parameters composing the tensor in factorized
        form."""
        raise NotImplementedError

    @property
    def _factorization(self, indices=None, **kwargs):
        """
        Returns the raw, unprocessed indexed tensor.

        Parameters
        ----------
        indices : int, or tuple of int
            use to index the tensor

        Returns
        -------
        TensorFactorization
            tensor[indices] but without any forward hook applied
        """
        if indices is None:
            return self
        else:
            return self[indices]

    def to_tensor(self):
        """Reconstruct the full tensor from its factorized form."""
        raise NotImplementedError

    def dim(self):
        """
        Order of the tensor.

        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also
        --------
        ndim
        """
        return len(self.shape)

    def numel(self):
        return int(ivy.prod(self.shape))

    @property
    def ndim(self):
        """
        Order of the tensor.

        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also
        --------
        dim
        """
        return len(self.shape)

    def size(self, index=None):
        """
        Shape of the tensor.

        Parameters
        ----------
        index : int, or tuple, default is None
            if not None, returns tensor.shape[index]

        See Also
        --------
        shape
        """
        if index is None:
            return self.shape
        else:
            return self.shape[index]

    def normal_(self, mean=0, std=1):
        """
        Inialize the factors of the factorization such that the **reconstruction**
        follows a Gaussian distribution.

        Parameters
        ----------
        mean : float, currently only 0 is supported
        std : float
            standard deviation

        Returns
        -------
        self
        """
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, rank={self.rank})"

    def to_unfolded(self, mode):
        return NotImplementedError

    def to_vec(self):
        return NotImplementedError

    def norm(self):
        """Norm l2 of the tensor."""
        return ivy.l2_normalize(self.to_tensor())

    def mode_dot(self, matrix_or_tensor, mode):
        return ivy.mode_dot(self.to_tensor(), matrix_or_tensor, mode)

    @property
    def name(self):
        """Factorization name ('tucker', 'tt', 'cp', ...)"""
        return self._name

    @property
    def tensor_shape(self):
        return self.shape


class TensorizedTensor(FactorizedTensor, metaclass=MetaFactorizedTensor, name=""):
    """
    Matrix in Tensorized Format.

    .. important::

       `order` and `tensorized_shape` correspond to the underlying tensor

       `shape`, `dim` and `ndim` correspond to the matrix
    """

    _factorizations = dict()

    def __init_subclass__(cls, name, **kwargs):
        """When a subclass is created, register it in _factorizations."""
        cls._factorizations[_format_factorization(name)] = cls
        cls._name = name

    def __new__(cls, *args, **kwargs):
        """
        Customize the creation of a matrix in tensorized form.

        Returns
        -------
        TensorizedMatrix._factorizations[_format_factorization(factorization)]
            subclass implementing the specified tensorized matrix
        """
        if cls is TensorizedTensor:
            factorization = kwargs.get("factorization")
            try:
                cls = cls._factorizations[_format_factorization(factorization)]
            except KeyError:
                raise ValueError(
                    f"Got factorization={factorization} but expected"
                    f"one of {cls._factorizations.keys()}"
                )

        instance = super().__new__(cls)

        return instance

    @classmethod
    def new(cls, tensorized_shape, rank, factorization="CP", **kwargs):
        """
        Main way to create a Tensorized Matrix.

        Parameters
        ----------
        tensorized_shape : tuple[int]
        rank : int, 'same' or float
            rank of the decomposition
        n_matrices : tuple or int, default is ()
            if not (), indicates how many matrices have to be jointly factorized
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorizedTensor
            Tensor in Tensorized and Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        try:
            cls = cls._factorizations[_format_factorization(factorization)]
        except KeyError:
            raise ValueError(
                f"Got factorization={factorization} but expected"
                f"one of {cls._factorizations.keys()}"
            )

        return cls.new(tensorized_shape, rank, **kwargs)

    @classmethod
    def from_tensor(cls, tensor, shape, rank, factorization="CP", **kwargs):
        """
        Create a factorized tensor by decomposing a full tensor.

        Parameters
        ----------
        tensor : torch.tensor
            tensor to factorize
        shape : tuple[int]
            shape of the factorized tensor to create
        rank : int, 'same' or float
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        try:
            cls = cls._factorizations[_format_factorization(factorization)]
        except KeyError:
            raise ValueError(
                f"Got factorization={factorization} but expected"
                f"one of {cls._factorizations.keys()}"
            )

        return cls.from_tensor(tensor, shape, rank, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix,
        tensorized_row_shape,
        tensorized_column_shape,
        rank,
        factorization="CP",
        **kwargs,
    ):
        """
        Create a Tensorized Matrix by tensorizing and decomposing an existing matrix.

        Parameters
        ----------
        matrix : torch.tensor of order 2
            matrix to decompose
        tensorized_row_shape : tuple[int]
            The first dimension (rows) of the matrix will be tensorized to that shape
        tensorized_column_shape : tuple[int]
            The second dimension (columns) of the matrix will be tensorized to that shape
        rank : int, 'same' or float
            rank of the decomposition
        n_matrices : tuple or int, default is ()
            if not (), indicates how many matrices have to be jointly factorized
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorizedMatrix
            Matrix in Tensorized and Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        if matrix.ndim > 2:
            batch_dims = _ensure_tuple(ivy.shape(matrix)[:-2])
        else:
            batch_dims = ()
        tensor = matrix.reshape(
            (*batch_dims, *tensorized_row_shape, *tensorized_column_shape)
        )
        return cls.from_tensor(
            tensor,
            batch_dims + (tensorized_row_shape, tensorized_column_shape),
            rank,
            factorization=factorization,
            **kwargs,
        )

    def to_matrix(self):
        """
        Reconstruct the full matrix from the factorized tensorization.

        If several matrices are parametrized, a batch of matrices is
        returned
        """
        # warnings.warn(f'{self} is being reconstructed into a matrix, consider operating on the decomposed form.')

        return self.to_tensor().reshape(self.shape)

    @property
    def tensor_shape(self):
        return sum(
            [(e,) if isinstance(e, int) else tuple(e) for e in self.tensorized_shape],
            (),
        )

    def init_from_matrix(self, matrix, **kwargs):
        tensor = matrix.reshape(self.tensor_shape)
        return self.init_from_tensor(tensor, **kwargs)

    def __repr__(self):
        msg = (
            f"{self.__class__.__name__}(shape={self.shape},"
            f" tensorized_shape={self.tensorized_shape}, "
        )
        msg += f"rank={self.rank})"
        return msg

    def __getitem__(self, indices):
        """
        Outer indexing of a factorized tensor.

        .. important::

            We use outer indexing,  not vectorized indexing!
            See e.g. https://numpy.org/neps/nep-0021-advanced-indexing.html
        """
        raise NotImplementedError
