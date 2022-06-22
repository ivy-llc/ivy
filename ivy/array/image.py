# global
import abc
from typing import Optional, List, Union, Tuple
import ivy

# ToDo: implement all methods here as public instance methods

class ArrayWithImage(abc.ABC):
    def bilinear_resample(
        self: ivy.Array,
        warp: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:

        return ivy.bilinear_resample(
            self._data, warp, out=out
        )

    def gradient_image(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:

        return ivy.gradient_image(
            self._data, out=out
        )

    def float_img_to_uint8_img(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:

        return ivy.float_img_to_uint8_img(
            self._data, out=out
        )

    def uint8_img_to_float_img(
            self: ivy.Array,
            out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

        return ivy.uint8_img_to_float_img(
            self._data, out=out
        )

    def random_crop(
            self: ivy.Array,
            crop_size: List[int],
            batch_shape: Optional[List[int]] = None,
            image_dims: Optional[List[int]] = None,
            seed: int = None,
            out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

        return ivy.random_crop(
            self._data, crop_size, batch_shape, image_dims, seed, out=out
        )

    def linear_resample(
            self: ivy.Array,
            num_samples: int,
            axis: int = -1,
            out: Optional[ivy.Array] = None
    ) -> Union[ivy.Array, ivy.NativeArray]:

        return ivy.linear_resample(
            self._data, num_samples, axis, out=out
        )


