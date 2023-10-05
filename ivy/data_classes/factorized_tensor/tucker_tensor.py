# local
from .base import FactorizedTensor
import ivy

# global
from copy import deepcopy
import warnings
import numpy as np


def _bisection_root_finder(fun, a, b, tol=1e-6, max_iter=100):
    if fun(a) * fun(b) >= 0:
        raise ValueError(
            "Function values at the interval endpoints must have opposite signs"
        )

    for _ in range(max_iter):
        c = (a + b) / 2
        if fun(c) == 0 or (b - a) / 2 < tol:
            return c

        if fun(c) * fun(a) < 0:
            b = c
        else:
            a = c

    raise RuntimeError("Bisection algorithm did not converge")


class TuckerTensor(FactorizedTensor, name="Tucker"):
    """
    Tucker Factorization.

    Parameters
    ----------
    core
    factors
    shape
    rank
    """

    def __init__(self, core, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = TuckerTensor.validate_tucker_tensor((core, factors))

        self.order = len(self.shape)
        # self.core = core

        self.factors = factors

    # Built-ins #
    # ----------#

    def __setitem__(self, index, value):
        if index == 0:
            self.core = value
        elif index == 1:
            self.factors = value
        else:
            raise IndexError(
                f"You tried to set index {index} of a Tucker tensor.\n"
                "You can only set index 0 and 1 of a Tucker tensor"
                "(corresponding respectively to core and factors)"
            )

    def __iter__(self):
        yield self.core
        yield self.factors

    def __len__(self):
        return 2

    def __repr__(self):
        message = f"Decomposed rank-{self.rank} TuckerTensor of shape {self.shape} "
        return message

    # Public Methods #
    # ---------------#

    def to_unfolded(self, mode):
        return TuckerTensor.tucker_to_unfolded(self, mode)

    def tucker_copy(self):
        return TuckerTensor(
            (
                deepcopy(self.core),
                [deepcopy(self.factors[i]) for i in range(len(self.factors))],
            )
        )

    def to_vec(self):
        return TuckerTensor.tucker_to_vec(self)

    def mode_dot(
        self,
        matrix_or_vector,
        mode,
        keep_dim,
        copy,
    ):
        return TuckerTensor.tucker_mode_dot(
            self, matrix_or_vector, mode, keep_dim=keep_dim, copy=copy
        )

    # Properties #
    # ---------------#
    @property
    def n_param(self):
        core, factors = self.core, self.factors
        total_params = sum(int(ivy.prod(tensor.shape)) for tensor in [core] + factors)
        return total_params

    @property
    def decomposition(self):
        return self.core, self.factors

    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, device=None, dtype=None, **kwargs):
        rank = TuckerTensor.validate_tucker_rank(
            shape, rank, fixed_modes=fixed_rank_modes
        )

        # Register the parameters
        core = ivy.empty(rank, device=device, dtype=dtype)
        # Avoid the issues with ParameterList
        factors = [
            ivy.empty((s, r), device=device, dtype=dtype) for (s, r) in zip(shape, rank)
        ]

        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = TuckerTensor.validate_tucker_rank(
            shape, rank, fixed_modes=fixed_rank_modes
        )

        core, factors = ivy.tucker(tensor, rank, **kwargs)

        return cls(core.contiguous(), [f.contiguous() for f in factors])

    def init_from_tensor(
        self, tensor, unsqueezed_modes=None, unsqueezed_init="average", **kwargs
    ):
        """
        Initialize the tensor factorization from a tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            full tensor to decompose
        unsqueezed_modes : int list
            list of modes for which the rank is 1 that don't correspond to a mode in the full tensor
            essentially we are adding a new dimension for which the core has dim 1,
            and that is not initialized through decomposition.
            Instead first `tensor` is decomposed into the other factors.
            The `unsqueezed factors` are then added and  initialized e.g. with 1/dim[i]
        unsqueezed_init : 'average' or float
            if unsqueezed_modes, this is how the added "unsqueezed" factors will be initialized
            if 'average', then unsqueezed_factor[i] will have value 1/tensor.shape[i]
        """
        if unsqueezed_modes is not None:
            unsqueezed_modes = sorted(unsqueezed_modes)
            for mode in unsqueezed_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = (
                        "It is only possible to initialize by averagig over mode for"
                        " which rank=1."
                    )
                    msg += (
                        f"However, got unsqueezed_modes={unsqueezed_modes} but"
                        f" rank[{mode}]={self.rank[mode]} != 1."
                    )
                    raise ValueError(msg)

            rank = tuple(
                r for (i, r) in enumerate(self.rank) if i not in unsqueezed_modes
            )
        else:
            rank = self.rank

        core, factors = ivy.tucker(tensor, rank, **kwargs)

        if unsqueezed_modes is not None:
            # Initialise with 1/shape[mode] or given value
            for mode in unsqueezed_modes:
                size = self.shape[mode]
                factor = ivy.ones(size, 1)
                if unsqueezed_init == "average":
                    factor /= size
                else:
                    factor *= unsqueezed_init
                factors.insert(mode, factor)
                core = core.unsqueeze(mode)

        self.core = core.contiguous()
        self.factors = [f.contiguous() for f in factors]
        return self

    def to_tensor(self):
        return TuckerTensor.tucker_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod([ivy.sqrt(r) for r in self.rank])
        std_factors = (std / r) ** (1 / (self.order + 1))

        self.core.data.normal_(0, std_factors)
        for factor in self.factors:
            factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = self.factors
            core = ivy.mode_dot(self.core, mixing_factor[indices, :], 0)
            return self.__class__(core, factors)

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            return self.__class__(self.core, factors)

        else:
            # Index multiple dimensions
            modes = []
            factors = []
            factors_contract = []
            for i, (index, factor) in enumerate(zip(indices, self.factors)):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices},"
                        f" indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    modes.append(i)
                    factors_contract.append(factor[index, :])
                else:
                    factors.append(factor[index, :])

            if modes:
                core = ivy.multi_mode_dot(self.core, factors_contract, modes=modes)
            else:
                core = self.core
            factors = factors + self.factors[i + 1 :]

            if factors:
                return self.__class__(core, factors)

            # Fully contracted tensor
            return core

    @staticmethod
    def validate_tucker_tensor(tucker_tensor):
        core, factors = tucker_tensor

        if len(factors) < 2:
            raise ValueError(
                "A Tucker tensor should be composed of at least two factors and a core."
                f"However, {len(factors)} factor was given."
            )

        if len(factors) != len(core.shape):
            raise ValueError(
                "Tucker decompositions should have one factor per mode of the core"
                f" tensor.However, core has {len(core.shape)} modes but"
                f" {len(factors)} factors have been provided"
            )

        shape = []
        rank = []
        for i, factor in enumerate(factors):
            current_shape, current_rank = factor.shape
            if current_rank != ivy.shape(core)[i]:
                raise ValueError(
                    "Factor `n` of Tucker decomposition should"
                    " verify:\nfactors[n].shape[1] = core.shape[n].However,"
                    f" factors[{i}].shape[1]={ivy.shape(factor)[1]} but"
                    f" core.shape[{i}]={ivy.shape(core)[i]}."
                )
            shape.append(current_shape)
            rank.append(current_rank)

        return tuple(shape), tuple(rank)

    @staticmethod
    def tucker_to_tensor(
        tucker_tensor,
        skip_factor=None,
        transpose_factors=False,
    ):
        core, factors = tucker_tensor
        return ivy.multi_mode_dot(
            core, factors, skip=skip_factor, transpose=transpose_factors
        )

    @staticmethod
    def tucker_normalize(tucker_tensor):
        core, factors = tucker_tensor
        normalized_factors = []
        for i, factor in enumerate(factors):
            scales = ivy.sqrt(ivy.sum(ivy.abs(factor) ** 2, axis=0))
            scales_non_zero = ivy.where(
                scales == 0, ivy.ones(ivy.shape(scales), dtype=factor[0].dtype), scales
            )
            core = core * ivy.reshape(
                scales, (1,) * i + (-1,) + (1,) * (len(core.shape) - i - 1)
            )
            normalized_factors.append(factor / ivy.reshape(scales_non_zero, (1, -1)))
        return TuckerTensor((core, normalized_factors))

    @staticmethod
    def tucker_to_unfolded(
        tucker_tensor,
        mode=0,
        skip_factor=None,
        transpose_factors=False,
    ):
        return ivy.unfold(
            TuckerTensor.tucker_to_tensor(
                tucker_tensor,
                skip_factor=skip_factor,
                transpose_factors=transpose_factors,
            ),
            mode,
        )

    @staticmethod
    def tucker_to_vec(
        tucker_tensor,
        skip_factor=None,
        transpose_factors=False,
    ):
        return ivy.reshape(
            TuckerTensor.tucker_to_tensor(
                tucker_tensor,
                skip_factor=skip_factor,
                transpose_factors=transpose_factors,
            ),
            (-1,),
        )

    @staticmethod
    def tucker_mode_dot(
        tucker_tensor,
        matrix_or_vector,
        mode,
        keep_dim=False,
        copy=False,
    ):
        shape, _ = TuckerTensor.validate_tucker_tensor(tucker_tensor)
        core, factors = tucker_tensor
        contract = False

        if len(matrix_or_vector.shape) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != shape[mode]:
                raise ValueError(
                    f"shapes {shape} and {matrix_or_vector.shape} not aligned in"
                    f" mode-{mode} multiplication: {shape[mode]} (mode = {mode}) !="
                    f" {matrix_or_vector.shape[1]} (dim 1 of matrix)"
                )

        elif len(matrix_or_vector.shape) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != shape[mode]:
                raise ValueError(
                    f"shapes {shape} and {matrix_or_vector.shape} not aligned for"
                    f" mode-{mode} multiplication: {shape[mode]} (mode = {mode}) !="
                    f" {matrix_or_vector.shape[0]} (vector size)"
                )
            if not keep_dim:
                contract = True  # Contract over that mode
        else:
            raise ValueError("Can only take n_mode_product with a vector or a matrix.")

        if copy:
            factors = [deepcopy(f) for f in factors]
            core = deepcopy(core)

        if contract:
            print("contracting mode")
            f = factors.pop(mode)
            core = ivy.mode_dot(core, ivy.dot(matrix_or_vector, f), mode=mode)
        else:
            factors[mode] = ivy.dot(matrix_or_vector, factors[mode])

        return TuckerTensor((core, factors))

    @staticmethod
    def validate_tucker_rank(
        tensor_shape, rank="same", rounding="round", fixed_modes=None
    ):
        if rounding == "ceil":
            rounding_fun = ivy.ceil
        elif rounding == "floor":
            rounding_fun = ivy.floor
        elif rounding == "round":
            rounding_fun = ivy.round
        else:
            raise ValueError(
                f"Rounding should be round, floor or ceil, but got {rounding}"
            )

        # rank is 'same' or float: choose rank so as to
        #  preserve a fraction of the original #parameters
        if rank == "same":
            rank = float(1)

        if isinstance(rank, float):
            n_modes_compressed = len(tensor_shape)
            n_param_tensor = int(ivy.prod(tensor_shape))

            if fixed_modes is not None:
                tensor_shape = list(tensor_shape)

                # sorted to be careful with the order when popping
                #  and reinserting to not remove/add at wrong index.
                # list (mode, shape) that we removed as they will
                #  be kept the same, rank[i] =
                fixed_modes = [
                    (mode, tensor_shape.pop(mode))
                    for mode in sorted(fixed_modes, reverse=True)
                ][::-1]

                # number of parameters coming from the fixed modes
                #  (these don't have a variable size as a fun of fraction_param)
                n_fixed_params = ivy.sum(
                    [s**2 for _, s in fixed_modes]
                )  # size of the factors
                n_modes_compressed -= len(fixed_modes)
            else:
                n_fixed_params = 0

            # Doesn't contain fixed_modes,
            #  those factors are accounted for in fixed_params
            squared_dims = ivy.sum([s**2 for s in tensor_shape])

            def fun(x):
                return (
                    n_param_tensor * x**n_modes_compressed
                    + squared_dims * x
                    + n_fixed_params * x
                    - rank * n_param_tensor
                )

            # fraction_param = brentq(fun, 0.0, max(rank, 1.0))
            fraction_param = _bisection_root_finder(fun, 0.0, max(rank, 1.0))
            rank = [max(int(rounding_fun(s * fraction_param)), 1) for s in tensor_shape]

            if fixed_modes is not None:
                for mode, size in fixed_modes:
                    rank.insert(mode, size)

        elif isinstance(rank, int):
            n_modes = len(tensor_shape)
            warnings.warn(
                "Given only one int for 'rank' for decomposition a tensor of order"
                f" {n_modes}. Using this rank for all modes."
            )
            if fixed_modes is None:
                rank = [rank] * n_modes
            else:
                rank = [
                    rank if i not in fixed_modes else s
                    for (i, s) in enumerate(tensor_shape)
                ]  # *n_mode

        return rank

    @staticmethod
    def tucker_n_param(shape, rank):
        core_params = ivy.prod(rank)
        factors_params = ivy.sum([r * s for (r, s) in zip(rank, shape)])
        return int(core_params + factors_params)
