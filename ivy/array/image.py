# global
import abc
from typing import Optional, List, Union, Tuple
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithImage(abc.ABC):
    def stack_images(
        self: ivy.Array,
        images: List[Union[ivy.Array, ivy.NativeArray]],
        desired_aspect_ratio: Tuple[int, int] = (1, 1),
    ) -> ivy.Array:

        return ivy.stack_images(
            self, images, desired_aspect_ratio
        )

    def bilinear_resample(
        self: ivy.Array,
        x: Union[ivy.Array, ivy.NativeArray],
        warp: Union[ivy.Array, ivy.NativeArray],
    ) -> ivy.Array:

        return ivy.bilinear_resample(
            self, x, warp
        )

    def gradient_image(
            self: ivy.Array,
            x: Union[ivy.Array, ivy.NativeArray],
    ) -> ivy.Array:

        return ivy.gradient_image(
            self, x
        )

    def float_img_to_uint8_img(
            self: ivy.Array,
            x: Union[ivy.Array, ivy.NativeArray],
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:

        return ivy.float_img_to_uint8_img(
            self, x, out=out
        )

    def uint8_img_to_float_img(
            self: ivy.Array,
            x: Union[ivy.Array, ivy.NativeArray],
            out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

        return ivy.uint8_img_to_float_img(
            self, x, out=out
        )

    def random_crop(
            self: ivy.Array,
            x: Union[ivy.Array, ivy.NativeArray],
            crop_size: List[int],
            batch_shape: Optional[List[int]] = None,
            image_dims: Optional[List[int]] = None,
            seed: int = None,
            out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

        return ivy.random_crop(
            self, x, crop_size, batch_shape, image_dims, seed, out=out
        )

    def linear_resample(
            self: ivy.Array,
            x: Union[ivy.Array, ivy.NativeArray],
            num_samples: int,
            axis: int = -1
    ) -> Union[ivy.Array, ivy.NativeArray]:

        return ivy.linear_resample(
            self, x, num_samples, axis
        )


