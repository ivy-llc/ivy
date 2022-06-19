# local
from typing import Optional, List, Union, Tuple, Dict

from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithImage(ContainerBase):

    @staticmethod
    def static_stack_images(
            images: List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            desired_aspect_ratio: Tuple[int, int] = (1, 1),
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "stack_images",
            images,
            desired_aspect_ratio,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def stack_images(
        self: ivy.Container,
        images: List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        desired_aspect_ratio: Tuple[int, int] = (1, 1),
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_stack_images(
            self,
            images,
            desired_aspect_ratio,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
        )

    @staticmethod
    def static_bilinear_resample(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            warp: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "bilinear_resample",
            x,
            warp,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def bilinear_resample(
        self: ivy.Container,
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        warp: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_bilinear_resample(
            self,
            x,
            warp,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
        )

    @staticmethod
    def static_gradient_image(
            x: Union[ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "gradient_image",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def gradient_image(
            self: ivy.Container,
            x: Union[ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:

        return self.static_gradient_image(
            self,
            x,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
        )

    @staticmethod
    def static_float_img_to_uint8_img(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "float_img_to_uint8_img",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    def float_img_to_uint8_img(
            self: ivy.Container,
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return self.static_float_img_to_uint8_img(
            self,
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    @staticmethod
    def static_uint8_img_to_float_img(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "uint8_img_to_float_img",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    def uint8_img_to_float_img(
            self: ivy.Container,
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return self.static_uint8_img_to_float_img(
            self,
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    @staticmethod
    def static_random_crop(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            crop_size: List[int],
            batch_shape: Optional[List[int]] = None,
            image_dims: Optional[List[int]] = None,
            seed: int = None,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "random_crop",
            x,
            crop_size,
            batch_shape,
            image_dims,
            seed,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    def random_crop(
            self: ivy.Container,
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            crop_size: List[int],
            batch_shape: Optional[List[int]] = None,
            image_dims: Optional[List[int]] = None,
            seed: int = None,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return self.static_random_crop(
            self,
            x,
            crop_size,
            batch_shape,
            image_dims,
            seed,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    @staticmethod
    def static_linear_resample(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            num_samples: int,
            axis: int = -1,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "linear_resample",
            x,
            num_samples,
            axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def linear_resample(
            self: ivy.Container,
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            num_samples: int,
            axis: int = -1,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_bilinear_resample(
            self,
            x,
            num_samples,
            axis,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
        )
