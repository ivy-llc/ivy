# global
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Literal,
    Iterable,
    Any,
)
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithExtensions(ContainerBase):
    @staticmethod
    def static_pad(
        input: ivy.Container,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return ContainerBase.multi_map_in_static_method(
            "pad",
            input,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **kwargs,
        )

    def pad(
        self: ivy.Container,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return self.static_pad(
            self,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **kwargs,
        )
