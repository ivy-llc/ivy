import ivy
import unittest


class TestUnion(unittest.TestCase):
    def test_union_containers(self):
        x = ivy.Container([1, 2, 3])
        y = ivy.Container([4, 5, 6])
        z = ivy.union_container(x, y)
        self.assertEqual(z, ivy.Container([1, 2, 3, 4, 5, 6]))

    def test_union_arrays(self):
        x = ivy.array([1, 2, 3])
        y = ivy.array([4, 5, 6])
        z = ivy.union_array(x, y)
        self.assertEqual(z, ivy.array([1, 2, 3, 4, 5, 6]))


if __name__ == "__main__":
    unittest.main()
