import ivy
import jax
import gradslam as gs
import torch
from gradslam.geometry.projutils import homogenize_points
import pytest


@pytest.fixture
def pts():
    return jax.numpy.array(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    # return torch.Tensor(
    #     [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    # )


def test_homogenize_points(pts):
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Got {} instead".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    return torch.nn.functional.pad(pts, (0, 1), "constant", 1.0)


if __name__ == "__main__":
    # GradSLAM
    print(gs.__version__)
    print(gs.geometry)

    # Run original PyTorch GradSLAM homogenize_points function
    pts_torch = torch.Tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    print("Original GradSLAM homogenize_points() output:", homogenize_points(pts_torch))

    homogenized_pts_jax_transpiled = ivy.transpile(
        test_homogenize_points, source="torch", to="jax"
    )

    """ Uncomment this to transpile the directly imported gradslam function
    pts = jax.numpy.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [-1.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0]])
    print(pts)
    homogenized_pts_jax_transpiled = ivy.transpile(homogenize_points, \
        source="torch", to='jax')
    """

    homogenized_pts = homogenized_pts_jax_transpiled(pts)
    print(homogenized_pts)
