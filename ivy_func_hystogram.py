import tensorflow as tf


# ivy/container/experimental/statistical.py

class ContainerWithStatisticalExperimental(ContainerBase):
    @staticmethod
    def using_file_hystogram(self: ivy.Array, 
        /, 
        *,
        name: Optional[ivy.Array, ivy.NativeArray],
        data:Optional[Union[Tensorfloat64]],
        step:Optional[Union[ivyArray,ivy.NativeArray]] = None, 
        buckets:Optional[ivy.Array[int]], 
        description:Optional[ivy.Array[str]]
        ) -> ivy.Array:


        a = tf.summary.create_file_writer('test/logs')
        with a.as_default():
            for step in range(100):
            
            # Generate fake "activations".

                activations = [
                    tf.random.normal([1000], mean=step, stddev=1),
                    tf.random.normal([1000], mean=step, stddev=10),
                    tf.random.normal([1000], mean=step, stddev=100),
                ]

                tf.summary.histogram("layer1/activate", activations[0], step=step)
                tf.summary.histogram("layer2/activate", activations[1], step=step)
                tf.summary.histogram("layer3/activate", activations[2], step=step)

        return using_file_hystogram


    def using_file_hystogram(self: ivy.Array, 
        /, 
        *,
        name: Optional[ivy.Array, ivy.NativeArray],
        data:Optional[Union[Tensorfloat64]],
        step:Optional[Union[ivyArray,ivy.NativeArray]] = None, 
        buckets:Optional[ivy.Array[int]], 
        description:Optional[ivy.Array[str]]
        ) -> ivy.Array:

    @staticmethod
    def static_median(
        input: ivy.Container,
        /, 
        *,
        name: Optional[ivy.Array, ivy.NativeArray],
        data:Optional[Union[Tensorfloat64]],
        step:Optional[Union[ivyArray,ivy.NativeArray]] = None, 
        buckets:Optional[ivy.Array[int]], 
        description:Optional[ivy.Array[str]]
        ) -> ivy.Container:





"""
        ivy.Container static method variant of ivy.median. This method simply wraps
        the function, and so the docstring for ivy.median also applies to this method
        with minimal changes.
        Parameters
        ----------
        input
            Input container including arrays.
        axis
            Axis or axes along which the medians are computed. The default is to compute
            the median along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one.
        out
            optional output array, for writing the result to.
        Returns
        -------
        ret
            The median of the array elements.
        Examples
        --------
        With one :class:`ivy.Container` input:

"""


                # ivy/functional/backends/tensorflow/experimental/statistical.py


from . import backend_version

def using_file_hystogram(self: ivy.Array, 
                    /, 
                    *,
                    name: Optional[ivy.Array, ivy.NativeArray],
                    data:Optional[Union[Tensorfloat64]],
                    step:Optional[Union[ivyArray,ivy.NativeArray]] = None, 
                    buckets:Optional[ivy.Array[int]], 
                    description:Optional[ivy.Array[str]]):

    return using_file hystogram

def median(
    input: torch.tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    if hasattr(axis, "__iter__"):
        for dim in axis:
            input = torch.median(
                input,
                dim=dim,
                keepdim=keepdims,
                out=out,
            )[0]
        return input
    else:
        return tf.Tensor.median(
            input,
            dim=axis,
            keepdim=keepdims,
            out=out,
        )
