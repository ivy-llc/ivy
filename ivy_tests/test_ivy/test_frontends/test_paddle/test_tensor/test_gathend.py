import sys
import pytest
import ivy


def test_gather_nd():
    # create an array
    x = ivy.array(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    )  # ([0., 1., 2., 3., 4., 5., 6.])
    y = ivy.array([[0, 1]])

    # use the function
    result = ivy.gather_nd(x, y)

    #  output = paddle.gather_nd(x, index)

    # check the result
    assert result == ivy.array(1.0)


sys.path.append(r"C:\Users\xiggy\electionWeb\Unify-Ivy\ivy_dev\lib\site-packages")


if __name__ == "__main__":
    pytest.main()
