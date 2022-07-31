__author__ = 'Alex Rogozhnikov'

import functools

from einops.einops import _apply_recipe

from ..einops import TransformRecipe, _prepare_transformation_recipe
from .. import EinopsError


class RearrangeMixin:
    """
    Rearrange layer behaves identically to einops.rearrange operation.

    :param pattern: str, rearrangement pattern
    :param axes_lengths: any additional specification of dimensions

    See einops.rearrange for source_examples.
    """

    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()  # checking parameters

    def __repr__(self):
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) -> TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, operation='rearrange', axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        return _apply_recipe(self._recipe, x, reduction_type='rearrange')


class ReduceMixin:
    """
    Reduce layer behaves identically to einops.reduce operation.

    :param pattern: str, rearrangement pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
    :param axes_lengths: any additional specification of dimensions

    See einops.reduce for source_examples.
    """

    def __init__(self, pattern, reduction, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()  # checking parameters

    def __repr__(self):
        params = '{!r}, {!r}'.format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) -> TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(
                self.pattern, operation=self.reduction, axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        return _apply_recipe(self._recipe, x, reduction_type=self.reduction)
