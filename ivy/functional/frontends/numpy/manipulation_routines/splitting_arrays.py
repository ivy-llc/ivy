# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def split(ary, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, (list, tuple)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[axis]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=False
    )


@to_ivy_arrays_and_back
def array_split(ary, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, (list, tuple)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[axis]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=True
    )


@to_ivy_arrays_and_back
def dsplit(ary, indices_or_sections):
    return ivy.dsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def vsplit(ary, indices_or_sections):
    return ivy.vsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def hsplit(ary, indices_or_sections):
    return ivy.hsplit(ary, indices_or_sections)
