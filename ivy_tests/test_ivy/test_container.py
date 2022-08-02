# global
import os
import queue
import pytest
import random
import numpy as np
import multiprocessing
import pickle

# local
import ivy
from ivy.container import Container
import ivy_tests.test_ivy.helpers as helpers


def test_container_list_join(device, call):
    container_0 = Container(
        {
            "a": [ivy.array([1], device=device)],
            "b": {
                "c": [ivy.array([2], device=device)],
                "d": [ivy.array([3], device=device)],
            },
        }
    )
    container_1 = Container(
        {
            "a": [ivy.array([4], device=device)],
            "b": {
                "c": [ivy.array([5], device=device)],
                "d": [ivy.array([6], device=device)],
            },
        }
    )
    container_list_joined = ivy.Container.list_join([container_0, container_1])
    assert np.allclose(ivy.to_numpy(container_list_joined["a"][0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_joined.a[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_joined["b"]["c"][0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.c[0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_joined["b"]["d"][0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.d[0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_joined["a"][1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_joined.a[1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_joined["b"]["c"][1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.c[1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_joined["b"]["d"][1]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.d[1]), np.array([6]))


def test_container_list_stack(device, call):
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container_list_stacked = ivy.Container.list_stack([container_0, container_1], 0)
    assert np.allclose(ivy.to_numpy(container_list_stacked["a"][0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.a[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_stacked["b"]["c"][0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.c[0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_stacked["b"]["d"][0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.d[0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_stacked["a"][1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.a[1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_stacked["b"]["c"][1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.c[1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_stacked["b"]["d"][1]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.d[1]), np.array([6]))


def test_container_unify(device, call):

    # devices and containers
    devices = list()
    dev0 = device
    devices.append(dev0)
    conts = dict()
    conts[dev0] = Container(
        {
            "a": ivy.array([1], device=dev0),
            "b": {"c": ivy.array([2], device=dev0), "d": ivy.array([3], device=dev0)},
        }
    )
    if "gpu" in device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = device[:-1] + str(idx)
        devices.append(dev1)
        conts[dev1] = Container(
            {
                "a": ivy.array([4], device=dev1),
                "b": {
                    "c": ivy.array([5], device=dev1),
                    "d": ivy.array([6], device=dev1),
                },
            }
        )

    # test
    container_unified = ivy.Container.unify(ivy.MultiDevItem(conts), dev0, "concat", 0)
    assert np.allclose(ivy.to_numpy(container_unified.a[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_unified.b.c[0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_unified.b.d[0]), np.array([3]))
    if len(devices) > 1:
        assert np.allclose(ivy.to_numpy(container_unified.a[1]), np.array([4]))
        assert np.allclose(ivy.to_numpy(container_unified.b.c[1]), np.array([5]))
        assert np.allclose(ivy.to_numpy(container_unified.b.d[1]), np.array([6]))


def test_container_concat(device, call):
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container_concatenated = ivy.concat([container_0, container_1], 0)
    assert np.allclose(ivy.to_numpy(container_concatenated["a"]), np.array([1, 4]))
    assert np.allclose(ivy.to_numpy(container_concatenated.a), np.array([1, 4]))
    assert np.allclose(ivy.to_numpy(container_concatenated["b"]["c"]), np.array([2, 5]))
    assert np.allclose(ivy.to_numpy(container_concatenated.b.c), np.array([2, 5]))
    assert np.allclose(ivy.to_numpy(container_concatenated["b"]["d"]), np.array([3, 6]))
    assert np.allclose(ivy.to_numpy(container_concatenated.b.d), np.array([3, 6]))


def test_container_combine(device, call):
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "e": ivy.array([6], device=device),
            },
        }
    )
    container_comb = ivy.Container.combine(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_comb.a), np.array([4]))
    assert np.equal(ivy.to_numpy(container_comb.b.c), np.array([5]))
    assert np.equal(ivy.to_numpy(container_comb.b.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_comb.b.e), np.array([6]))


def test_container_diff(device, call):
    # all different arrays
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.a.diff_1), np.array([4]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_1), np.array([6]))
    container_diff_diff_only = ivy.Container.diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == {}

    # some different arrays
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" in container_diff_diff_only["b"]
    assert "d" not in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.diff(
        container_0, container_1, mode="same_only"
    )
    assert "a" in container_diff_same_only
    assert "b" in container_diff_same_only
    assert "c" not in container_diff_same_only["b"]
    assert "d" in container_diff_same_only["b"]

    # all different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "e": ivy.array([1], device=device),
            "f": {
                "g": ivy.array([2], device=device),
                "h": ivy.array([3], device=device),
            },
        }
    )
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.e.diff_1), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.g), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.h), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == {}

    # some different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "e": ivy.array([3], device=device),
            },
        }
    )
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" not in container_diff_diff_only["b"]
    assert "d" in container_diff_diff_only["b"]
    assert "e" in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.diff(
        container_0, container_1, mode="same_only"
    )
    assert "a" in container_diff_same_only
    assert "b" in container_diff_same_only
    assert "c" in container_diff_same_only["b"]
    assert "d" not in container_diff_same_only["b"]
    assert "e" not in container_diff_same_only["b"]

    # same containers
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == {}
    container_diff_same_only = ivy.Container.diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == container_diff.to_dict()

    # all different strings
    container_0 = Container({"a": "1", "b": {"c": "2", "d": "3"}})
    container_1 = Container({"a": "4", "b": {"c": "5", "d": "6"}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert container_diff.a.diff_0 == "1"
    assert container_diff.a.diff_1 == "4"
    assert container_diff.b.c.diff_0 == "2"
    assert container_diff.b.c.diff_1 == "5"
    assert container_diff.b.d.diff_0 == "3"
    assert container_diff.b.d.diff_1 == "6"
    container_diff_diff_only = ivy.Container.diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == {}


def test_container_structural_diff(device, call):
    # all different keys or shapes
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([[4]], device=device),
            "b": {
                "c": ivy.array([[[5]]], device=device),
                "e": ivy.array([3], device=device),
            },
        }
    )
    container_diff = ivy.Container.structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.a.diff_1), np.array([[4]]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([[[5]]]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([3]))
    container_diff_diff_only = ivy.Container.structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == {}

    # some different shapes
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([[5]], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container_diff = ivy.Container.structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" in container_diff_diff_only["b"]
    assert "d" not in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert "a" in container_diff_same_only
    assert "b" in container_diff_same_only
    assert "c" not in container_diff_same_only["b"]
    assert "d" in container_diff_same_only["b"]

    # all different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "e": ivy.array([4], device=device),
            "f": {
                "g": ivy.array([5], device=device),
                "h": ivy.array([6], device=device),
            },
        }
    )
    container_diff = ivy.Container.structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.e.diff_1), np.array([4]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.g), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.h), np.array([6]))
    container_diff_diff_only = ivy.Container.structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == {}

    # some different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "e": ivy.array([6], device=device),
            },
        }
    )
    container_diff = ivy.Container.structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([6]))
    container_diff_diff_only = ivy.Container.structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" not in container_diff_diff_only["b"]
    assert "d" in container_diff_diff_only["b"]
    assert "e" in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert "a" in container_diff_same_only
    assert "b" in container_diff_same_only
    assert "c" in container_diff_same_only["b"]
    assert "d" not in container_diff_same_only["b"]
    assert "e" not in container_diff_same_only["b"]

    # all same
    container_0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container_diff = ivy.Container.structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.to_dict() == {}
    container_diff_same_only = ivy.Container.structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.to_dict() == container_diff.to_dict()


def test_container_from_dict(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_depth(device, call):
    cont_depth1 = Container(
        {"a": ivy.array([1], device=device), "b": ivy.array([2], device=device)}
    )
    assert cont_depth1.max_depth == 1
    cont_depth2 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    assert cont_depth2.max_depth == 2
    cont_depth3 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": {"d": ivy.array([2], device=device)},
                "e": ivy.array([3], device=device),
            },
        }
    )
    assert cont_depth3.max_depth == 3
    cont_depth4 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {"c": {"d": {"e": ivy.array([2], device=device)}}},
        }
    )
    assert cont_depth4.max_depth == 4


@pytest.mark.parametrize("inplace", [True, False])
def test_container_cutoff_at_depth(inplace, device, call):

    # values
    a_val = ivy.array([1], device=device)
    bcde_val = ivy.array([2], device=device)

    # depth 1
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cutoff_at_depth(1, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert not cont_cutoff.b

    # depth 2
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cutoff_at_depth(2, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert not cont_cutoff.b.c

    # depth 3
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cutoff_at_depth(3, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert not cont_cutoff.b.c.d

    # depth 4
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cutoff_at_depth(4, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert np.allclose(ivy.to_numpy(cont_cutoff.b.c.d.e), ivy.to_numpy(bcde_val))


@pytest.mark.parametrize("inplace", [True, False])
def test_container_cutoff_at_height(inplace, device, call):

    # values
    d_val = ivy.array([2], device=device)
    e_val = ivy.array([3], device=device)

    # height 0
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cutoff_at_height(0, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a.c.d), ivy.to_numpy(d_val))
    assert np.allclose(ivy.to_numpy(cont_cutoff.b.c.d.e), ivy.to_numpy(e_val))

    # height 1
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cutoff_at_height(1, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff.a.c
    assert not cont_cutoff.b.c.d

    # height 2
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cutoff_at_height(2, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff.a
    assert not cont_cutoff.b.c

    # height 3
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cutoff_at_height(3, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff.a
    assert not cont_cutoff.b

    # height 4
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cutoff_at_height(4, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff


@pytest.mark.parametrize("str_slice", [True, False])
def test_container_slice_keys(str_slice, device, call):

    # values
    a_val = ivy.array([1], device=device)
    b_val = ivy.array([2], device=device)
    c_val = ivy.array([3], device=device)
    d_val = ivy.array([4], device=device)
    e_val = ivy.array([5], device=device)

    # slice
    if str_slice:
        slc = "b:d"
    else:
        slc = slice(1, 4, 1)

    # without dict
    cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    cont_sliced = cont.slice_keys(slc)
    assert "a" not in cont_sliced
    assert np.allclose(ivy.to_numpy(cont_sliced.b), ivy.to_numpy(b_val))
    assert np.allclose(ivy.to_numpy(cont_sliced.c), ivy.to_numpy(c_val))
    assert np.allclose(ivy.to_numpy(cont_sliced.d), ivy.to_numpy(d_val))
    assert "e" not in cont_sliced

    # with dict, depth 0
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.slice_keys({0: slc})
    assert "a" not in cont_sliced
    assert Container.identical([cont_sliced.b, sub_cont])
    assert Container.identical([cont_sliced.c, sub_cont])
    assert Container.identical([cont_sliced.d, sub_cont])
    assert "e" not in cont_sliced

    # with dict, depth 1
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    sub_sub_cont = Container({"b": b_val, "c": c_val, "d": d_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.slice_keys({1: slc})
    assert Container.identical([cont_sliced.a, sub_sub_cont])
    assert Container.identical([cont_sliced.b, sub_sub_cont])
    assert Container.identical([cont_sliced.c, sub_sub_cont])
    assert Container.identical([cont_sliced.d, sub_sub_cont])
    assert Container.identical([cont_sliced.e, sub_sub_cont])

    # with dict, depth 0, 1
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    sub_sub_cont = Container({"b": b_val, "c": c_val, "d": d_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.slice_keys({0: slc, 1: slc})
    assert "a" not in cont_sliced
    assert Container.identical([cont_sliced.b, sub_sub_cont])
    assert Container.identical([cont_sliced.c, sub_sub_cont])
    assert Container.identical([cont_sliced.d, sub_sub_cont])
    assert "e" not in cont_sliced

    # all depths
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    sub_sub_cont = Container({"b": b_val, "c": c_val, "d": d_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.slice_keys(slc, all_depths=True)
    assert "a" not in cont_sliced
    assert Container.identical([cont_sliced.b, sub_sub_cont])
    assert Container.identical([cont_sliced.c, sub_sub_cont])
    assert Container.identical([cont_sliced.d, sub_sub_cont])
    assert "e" not in cont_sliced


def test_container_show(device, call):
    if call is helpers.mx_call:
        # ToDo: get this working for mxnet again, recent version update caused errors.
        pytest.skip()
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    cont = Container(dict_in)
    print(cont)
    cont.show()


def test_container_find_sub_container(device, call):
    arr1 = ivy.array([1], device=device)
    arr2 = ivy.array([2], device=device)
    arr3 = ivy.array([3], device=device)
    dict_in = {"a": arr1, "b": {"c": arr2, "d": arr3}}
    top_cont = Container(dict_in)

    # full
    sub_cont = Container(dict_in["b"])
    assert sub_cont in top_cont
    found_kc = top_cont.find_sub_container(sub_cont)
    assert found_kc == "b"
    found_kc = top_cont.find_sub_container(top_cont)
    assert found_kc == ""

    # partial
    partial_sub_cont = Container({"d": arr3})
    found_kc = top_cont.find_sub_container(partial_sub_cont, partial=True)
    assert found_kc == "b"
    assert partial_sub_cont.find_sub_container(top_cont, partial=True) is False
    partial_sub_cont = Container({"b": {"d": arr3}})
    found_kc = top_cont.find_sub_container(partial_sub_cont, partial=True)
    assert found_kc == ""
    assert partial_sub_cont.find_sub_container(top_cont, partial=True) is False


def test_container_find_sub_structure(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    top_cont = Container(dict_in)

    # full
    sub_cont = Container(
        {"c": ivy.array([4], device=device), "d": ivy.array([5], device=device)}
    )
    assert not top_cont.find_sub_container(sub_cont)
    found_kc = top_cont.find_sub_structure(sub_cont)
    assert found_kc == "b"
    found_kc = top_cont.find_sub_structure(top_cont)
    assert found_kc == ""

    # partial
    partial_sub_cont = Container({"d": ivy.array([5], device=device)})
    found_kc = top_cont.find_sub_structure(partial_sub_cont, partial=True)
    assert found_kc == "b"
    partial_sub_cont = Container({"b": {"d": ivy.array([5], device=device)}})
    found_kc = top_cont.find_sub_structure(partial_sub_cont, partial=True)
    assert found_kc == ""


def test_container_show_sub_container(device, call):
    if call is helpers.mx_call:
        # ToDo: get this working for mxnet again, recent version update caused errors.
        pytest.skip()
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    top_cont = Container(dict_in)
    sub_cont = Container(dict_in["b"])
    top_cont.show_sub_container("b")
    top_cont.show_sub_container(sub_cont)


def test_container_from_dict_w_cont_types(device, call):
    # ToDo: add tests for backends other than jax
    if call is not helpers.jnp_call:
        pytest.skip()
    from haiku._src.data_structures import FlatMapping

    dict_in = {
        "a": ivy.array([1], device=device),
        "b": FlatMapping(
            {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)}
        ),
    }
    container = Container(dict_in)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_from_kwargs(device, call):
    container = Container(
        a=ivy.array([1], device=device),
        b={"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_from_list(device, call):
    list_in = [
        ivy.array([1], device=device),
        [ivy.array([2], device=device), ivy.array([3], device=device)],
    ]
    container = Container(list_in, types_to_iteratively_nest=[list])
    assert np.allclose(ivy.to_numpy(container["it_0"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.it_0), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_0"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_0), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_1"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_1), np.array([3]))


def test_container_from_tuple(device, call):
    tuple_in = (
        ivy.array([1], device=device),
        (ivy.array([2], device=device), ivy.array([3], device=device)),
    )
    container = Container(tuple_in, types_to_iteratively_nest=[tuple])
    assert np.allclose(ivy.to_numpy(container["it_0"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.it_0), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_0"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_0), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_1"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_1), np.array([3]))


def test_container_to_raw(device, call):
    tuple_in = (
        ivy.array([1], device=device),
        (ivy.array([2], device=device), ivy.array([3], device=device)),
    )
    container = Container(tuple_in, types_to_iteratively_nest=[tuple])
    raw = container.to_raw()
    assert np.allclose(ivy.to_numpy(raw[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(raw[1][0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(raw[1][1]), np.array([3]))


def test_container_clip_vector_norm(device, call):
    container = Container({"a": ivy.array([[0.8, 2.2], [1.5, 0.2]], device=device)})
    container_clipped = container.clip_vector_norm(2.5, 2.0)
    assert np.allclose(
        ivy.to_numpy(container_clipped["a"]),
        np.array([[0.71749604, 1.9731141], [1.345305, 0.17937401]]),
    )
    assert np.allclose(
        ivy.to_numpy(container_clipped.a),
        np.array([[0.71749604, 1.9731141], [1.345305, 0.17937401]]),
    )


def test_container_einsum(device, call):
    dict_in = {
        "a": ivy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device),
        "b": {
            "c": ivy.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], device=device),
            "d": ivy.array([[-2.0, -4.0], [-6.0, -8.0], [-10.0, -12.0]], device=device),
        },
    }
    container = Container(dict_in)
    container_einsummed = container.einsum("ij->i")
    assert np.allclose(
        ivy.to_numpy(container_einsummed["a"]), np.array([3.0, 7.0, 11.0])
    )
    assert np.allclose(ivy.to_numpy(container_einsummed.a), np.array([3.0, 7.0, 11.0]))
    assert np.allclose(
        ivy.to_numpy(container_einsummed["b"]["c"]), np.array([6.0, 14.0, 22.0])
    )
    assert np.allclose(
        ivy.to_numpy(container_einsummed.b.c), np.array([6.0, 14.0, 22.0])
    )
    assert np.allclose(
        ivy.to_numpy(container_einsummed["b"]["d"]), np.array([-6.0, -14.0, -22.0])
    )
    assert np.allclose(
        ivy.to_numpy(container_einsummed.b.d), np.array([-6.0, -14.0, -22.0])
    )


# def test_container_vector_norm(device, call):
#     dict_in = {
#         "a": ivy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device),
#         "b": {
#             "c": ivy.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], device=device),
#             "d": ivy.array([[3.0, 6.0], [9.0, 12.0], [15.0, 18.0]], device=device),
#         },
#     }
#     container = Container(dict_in)
#     container_normed = container.vector_norm(axis=(-1, -2))
#     assert np.allclose(ivy.to_numpy(container_normed["a"]), 9.5394)
#     assert np.allclose(ivy.to_numpy(container_normed.a), 9.5394)
#     assert np.allclose(ivy.to_numpy(container_normed["b"]["c"]), 19.0788)
#     assert np.allclose(ivy.to_numpy(container_normed.b.c), 19.0788)
#     assert np.allclose(ivy.to_numpy(container_normed["b"]["d"]), 28.6182)
#     assert np.allclose(ivy.to_numpy(container_normed.b.d), 28.6182)


def test_container_matrix_norm(device, call):
    if call is helpers.mx_call:
        # MXNet does not support matrix norm
        pytest.skip()
    dict_in = {
        "a": ivy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device),
        "b": {
            "c": ivy.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], device=device),
            "d": ivy.array([[3.0, 6.0], [9.0, 12.0], [15.0, 18.0]], device=device),
        },
    }
    container = Container(dict_in)
    container_normed = container.matrix_norm()
    assert np.allclose(ivy.to_numpy(container_normed["a"]), 9.52551809)
    assert np.allclose(ivy.to_numpy(container_normed.a), 9.52551809)
    assert np.allclose(ivy.to_numpy(container_normed["b"]["c"]), 19.05103618)
    assert np.allclose(ivy.to_numpy(container_normed.b.c), 19.05103618)
    assert np.allclose(ivy.to_numpy(container_normed["b"]["d"]), 28.57655427)
    assert np.allclose(ivy.to_numpy(container_normed.b.d), 28.57655427)


def test_container_flip(device, call):
    dict_in = {
        "a": ivy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device),
        "b": {
            "c": ivy.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], device=device),
            "d": ivy.array([[-2.0, -4.0], [-6.0, -8.0], [-10.0, -12.0]], device=device),
        },
    }
    container = Container(dict_in)
    container_flipped = container.flip(-1)
    assert np.allclose(
        ivy.to_numpy(container_flipped["a"]),
        np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]),
    )
    assert np.allclose(
        ivy.to_numpy(container_flipped.a),
        np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]),
    )
    assert np.allclose(
        ivy.to_numpy(container_flipped["b"]["c"]),
        np.array([[4.0, 2.0], [8.0, 6.0], [12.0, 10.0]]),
    )
    assert np.allclose(
        ivy.to_numpy(container_flipped.b.c),
        np.array([[4.0, 2.0], [8.0, 6.0], [12.0, 10.0]]),
    )
    assert np.allclose(
        ivy.to_numpy(container_flipped["b"]["d"]),
        np.array([[-4.0, -2.0], [-8.0, -6.0], [-12.0, -10.0]]),
    )
    assert np.allclose(
        ivy.to_numpy(container_flipped.b.d),
        np.array([[-4.0, -2.0], [-8.0, -6.0], [-12.0, -10.0]]),
    )


def test_container_as_bools(device, call):
    dict_in = {"a": ivy.array([1], device=device), "b": {"c": [], "d": True}}
    container = Container(dict_in)

    container_bools = container.as_bools()
    assert container_bools["a"] is True
    assert container_bools.a is True
    assert container_bools["b"]["c"] is False
    assert container_bools.b.c is False
    assert container_bools["b"]["d"] is True
    assert container_bools.b.d is True


def test_container_all_true(device, call):
    assert not Container(
        {"a": ivy.array([1], device=device), "b": {"c": [], "d": True}}
    ).all_true()
    assert Container(
        {"a": ivy.array([1], device=device), "b": {"c": [1], "d": True}}
    ).all_true()
    # noinspection PyBroadException
    try:
        assert Container(
            {"a": ivy.array([1], device=device), "b": {"c": [1], "d": True}}
        ).all_true(assert_is_bool=True)
        error_raised = False
    except AssertionError:
        error_raised = True
    assert error_raised


def test_container_all_false(device, call):
    assert Container({"a": False, "b": {"c": [], "d": 0}}).all_false()
    assert not Container({"a": False, "b": {"c": [1], "d": 0}}).all_false()
    # noinspection PyBroadException
    try:
        assert Container(
            {"a": ivy.array([1], device=device), "b": {"c": [1], "d": True}}
        ).all_false(assert_is_bool=True)
        error_raised = False
    except AssertionError:
        error_raised = True
    assert error_raised


def test_container_unstack_conts(device, call):
    dict_in = {
        "a": ivy.array([[1], [2], [3]], device=device),
        "b": {
            "c": ivy.array([[2], [3], [4]], device=device),
            "d": ivy.array([[3], [4], [5]], device=device),
        },
    }
    container = Container(dict_in)

    # without key_chains specification
    container_unstacked = container.unstack_conts(0)
    for cont, a, bc, bd in zip(container_unstacked, [1, 2, 3], [2, 3, 4], [3, 4, 5]):
        assert np.array_equal(ivy.to_numpy(cont["a"]), np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont.a), np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont["b"]["c"]), np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont.b.c), np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont["b"]["d"]), np.array([bd]))
        assert np.array_equal(ivy.to_numpy(cont.b.d), np.array([bd]))


def test_container_split_conts(device, call):
    dict_in = {
        "a": ivy.array([[1], [2], [3]], device=device),
        "b": {
            "c": ivy.array([[2], [3], [4]], device=device),
            "d": ivy.array([[3], [4], [5]], device=device),
        },
    }
    container = Container(dict_in)

    # without key_chains specification
    container_split = container.split_conts(1, -1)
    for cont, a, bc, bd in zip(container_split, [1, 2, 3], [2, 3, 4], [3, 4, 5]):
        assert np.array_equal(ivy.to_numpy(cont["a"])[0], np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont.a)[0], np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont["b"]["c"])[0], np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont.b.c)[0], np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont["b"]["d"])[0], np.array([bd]))
        assert np.array_equal(ivy.to_numpy(cont.b.d)[0], np.array([bd]))


def test_container_num_arrays(device, call):
    dict_in = {
        "a": ivy.array([[0.0, 1.0, 2.0, 3.0]], device=device),
        "b": {
            "c": ivy.array([[5.0, 10.0, 15.0, 20.0]], device=device),
            "d": ivy.array([[10.0, 9.0, 8.0, 7.0]], device=device),
        },
    }
    container = Container(dict_in)
    assert container.num_arrays() == 3
    dict_in = {
        "a": ivy.array([[0.0, 1.0, 2.0, 3.0]], device=device),
        "b": {
            "c": ivy.variable(ivy.array([[5.0, 10.0, 15.0, 20.0]], device=device)),
            "d": ivy.array([[10.0, 9.0, 8.0, 7.0]], device=device),
        },
    }
    container = Container(dict_in)
    assert (
        container.num_arrays() == 3
        if call in [helpers.np_call, helpers.jnp_call]
        else 2
    )


def test_container_size_ordered_arrays(device, call):
    dict_in = {
        "a": ivy.array([[0.0, 1.0, 2.0, 3.0]], device=device),
        "b": {
            "c": ivy.array([[5.0, 10.0]], device=device),
            "d": ivy.array([[10.0, 9.0, 8.0]], device=device),
        },
    }
    container = Container(dict_in)
    size_ordered = container.size_ordered_arrays()
    assert np.allclose(ivy.to_numpy(size_ordered.a), np.array([[0.0, 1.0, 2.0, 3.0]]))
    assert np.allclose(ivy.to_numpy(size_ordered.b__c), np.array([[5.0, 10.0]]))
    assert np.allclose(ivy.to_numpy(size_ordered.b__d), np.array([[10.0, 9.0, 8.0]]))
    for v, arr in zip(
        size_ordered.values(),
        [
            np.array([[5.0, 10.0]]),
            np.array([[10.0, 9.0, 8.0]]),
            np.array([[0.0, 1.0, 2.0, 3.0]]),
        ],
    ):
        assert np.allclose(ivy.to_numpy(v), arr)


def test_container_has_key(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    assert container.has_key("a")  # noqa
    assert container.has_key("b")  # noqa
    assert container.has_key("c")  # noqa
    assert container.has_key("d")  # noqa
    assert not container.has_key("e")  # noqa
    assert not container.has_key("f")  # noqa


def test_container_has_key_chain(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    assert container.has_key_chain("a")
    assert container.has_key_chain("b")
    assert container.has_key_chain("b/c")
    assert container.has_key_chain("b/d")
    assert not container.has_key_chain("b/e")
    assert not container.has_key_chain("c")


def test_container_has_nans(device, call):
    container = Container(
        {
            "a": ivy.array([1.0, 2.0], device=device),
            "b": {
                "c": ivy.array([2.0, 3.0], device=device),
                "d": ivy.array([3.0, 4.0], device=device),
            },
        }
    )
    container_nan = Container(
        {
            "a": ivy.array([1.0, 2.0], device=device),
            "b": {
                "c": ivy.array([float("nan"), 3.0], device=device),
                "d": ivy.array([3.0, 4.0], device=device),
            },
        }
    )
    container_inf = Container(
        {
            "a": ivy.array([1.0, 2.0], device=device),
            "b": {
                "c": ivy.array([2.0, 3.0], device=device),
                "d": ivy.array([3.0, float("inf")], device=device),
            },
        }
    )
    container_nan_n_inf = Container(
        {
            "a": ivy.array([1.0, 2.0], device=device),
            "b": {
                "c": ivy.array([float("nan"), 3.0], device=device),
                "d": ivy.array([3.0, float("inf")], device=device),
            },
        }
    )

    # global

    # with inf check
    assert not container.has_nans()
    assert container_nan.has_nans()
    assert container_inf.has_nans()
    assert container_nan_n_inf.has_nans()

    # without inf check
    assert not container.has_nans(include_infs=False)
    assert container_nan.has_nans(include_infs=False)
    assert not container_inf.has_nans(include_infs=False)
    assert container_nan_n_inf.has_nans(include_infs=False)

    # leafwise

    # with inf check
    container_hn = container.has_nans(leafwise=True)
    assert container_hn.a is False
    assert container_hn.b.c is False
    assert container_hn.b.d is False

    container_nan_hn = container_nan.has_nans(leafwise=True)
    assert container_nan_hn.a is False
    assert container_nan_hn.b.c is True
    assert container_nan_hn.b.d is False

    container_inf_hn = container_inf.has_nans(leafwise=True)
    assert container_inf_hn.a is False
    assert container_inf_hn.b.c is False
    assert container_inf_hn.b.d is True

    container_nan_n_inf_hn = container_nan_n_inf.has_nans(leafwise=True)
    assert container_nan_n_inf_hn.a is False
    assert container_nan_n_inf_hn.b.c is True
    assert container_nan_n_inf_hn.b.d is True

    # without inf check
    container_hn = container.has_nans(leafwise=True, include_infs=False)
    assert container_hn.a is False
    assert container_hn.b.c is False
    assert container_hn.b.d is False

    container_nan_hn = container_nan.has_nans(leafwise=True, include_infs=False)
    assert container_nan_hn.a is False
    assert container_nan_hn.b.c is True
    assert container_nan_hn.b.d is False

    container_inf_hn = container_inf.has_nans(leafwise=True, include_infs=False)
    assert container_inf_hn.a is False
    assert container_inf_hn.b.c is False
    assert container_inf_hn.b.d is False

    container_nan_n_inf_hn = container_nan_n_inf.has_nans(
        leafwise=True, include_infs=False
    )
    assert container_nan_n_inf_hn.a is False
    assert container_nan_n_inf_hn.b.c is True
    assert container_nan_n_inf_hn.b.d is False


def test_container_at_keys(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    new_container = container.at_keys(["a", "c"])
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]
    new_container = container.at_keys("c")
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]
    new_container = container.at_keys(["b"])
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))


def test_container_at_key_chain(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)

    # explicit function call
    sub_container = container.at_key_chain("b")
    assert np.allclose(ivy.to_numpy(sub_container["c"]), np.array([2]))
    sub_container = container.at_key_chain("b/c")
    assert np.allclose(ivy.to_numpy(sub_container), np.array([2]))

    # overridden built-in function call
    sub_container = container["b"]
    assert np.allclose(ivy.to_numpy(sub_container["c"]), np.array([2]))
    sub_container = container["b/c"]
    assert np.allclose(ivy.to_numpy(sub_container), np.array([2]))


def test_container_at_key_chains(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    target_cont = Container({"a": True, "b": {"c": True}})
    new_container = container.at_key_chains(target_cont)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]
    new_container = container.at_key_chains(["b/c", "b/d"])
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))
    new_container = container.at_key_chains("b/c")
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_all_key_chains(include_empty, device, call):
    a_val = Container() if include_empty else ivy.array([1], device=device)
    bc_val = Container() if include_empty else ivy.array([2], device=device)
    bd_val = Container() if include_empty else ivy.array([3], device=device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)
    kcs = container.all_key_chains(include_empty)
    assert kcs[0] == "a"
    assert kcs[1] == "b/c"
    assert kcs[2] == "b/d"


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_key_chains_containing(include_empty, device, call):
    a_val = Container() if include_empty else ivy.array([1], device=device)
    bc_val = Container() if include_empty else ivy.array([2], device=device)
    bd_val = Container() if include_empty else ivy.array([3], device=device)
    dict_in = {"a_sub": a_val, "b": {"c": bc_val, "d_sub": bd_val}}
    container = Container(dict_in)
    kcs = container.key_chains_containing("sub", include_empty)
    assert kcs[0] == "a_sub"
    assert kcs[1] == "b/d_sub"


# noinspection PyUnresolvedReferences
def test_container_set_at_keys(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container_orig = Container(dict_in)

    # explicit function call
    orig_container = container_orig.copy()
    container = orig_container.set_at_keys({"b": ivy.array([4], device=device)})
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]), np.array([4]))
    assert not container.has_key("c")  # noqa
    assert not container.has_key("d")  # noqa
    container = orig_container.set_at_keys(
        {"a": ivy.array([5], device=device), "c": ivy.array([6], device=device)}
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))


# noinspection PyUnresolvedReferences
def test_container_set_at_key_chain(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.copy()
    container = container.set_at_key_chain("b/e", ivy.array([4], device=device))
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    container = container.set_at_key_chain("f", ivy.array([5], device=device))
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["f"]), np.array([5]))

    # overridden built-in function call
    container = container_orig.copy()
    assert "b/e" not in container
    container["b/e"] = ivy.array([4], device=device)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    assert "f" not in container
    container["f"] = ivy.array([5], device=device)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["f"]), np.array([5]))


# noinspection PyUnresolvedReferences
def test_container_overwrite_at_key_chain(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.copy()
    # noinspection PyBroadException
    try:
        container.overwrite_at_key_chain("b/e", ivy.array([4], device=device))
        exception_raised = False
    except Exception:
        exception_raised = True
    assert exception_raised
    container = container.overwrite_at_key_chain("b/d", ivy.array([4], device=device))
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([4]))


def test_container_set_at_key_chains(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    target_container = Container(
        {"a": ivy.array([4], device=device), "b": {"d": ivy.array([5], device=device)}}
    )
    new_container = container.set_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([5]))
    target_container = Container({"b": {"c": ivy.array([7], device=device)}})
    new_container = container.set_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([7]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))


def test_container_overwrite_at_key_chains(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    target_container = Container(
        {"a": ivy.array([4], device=device), "b": {"d": ivy.array([5], device=device)}}
    )
    new_container = container.overwrite_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([5]))
    target_container = Container({"b": {"c": ivy.array([7], device=device)}})
    new_container = container.overwrite_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([7]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))
    # noinspection PyBroadException
    try:
        container.overwrite_at_key_chains(
            Container({"b": {"e": ivy.array([5], device=device)}})
        )
        exception_raised = False
    except Exception:
        exception_raised = True
    assert exception_raised


def test_container_prune_keys(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    container_pruned = container.prune_keys(["a", "c"])
    assert "a" not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert "c" not in container_pruned["b"]

    def _test_a_exception(container_in):
        try:
            _ = container_in.a
            return False
        except AttributeError:
            return True

    def _test_bc_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    def _test_bd_exception(container_in):
        try:
            _ = container_in.b.d
            return False
        except AttributeError:
            return True

    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)

    container_pruned = container.prune_keys(["a", "d"])
    assert "a" not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["c"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.c), np.array([[2]]))
    assert "d" not in container_pruned["b"]
    assert _test_a_exception(container_pruned)
    assert _test_bd_exception(container_pruned)


def test_container_prune_key_chain(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": None},
    }
    container = Container(dict_in)
    container_pruned = container.prune_key_chain("b/c")
    assert np.allclose(ivy.to_numpy(container_pruned["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.a), np.array([[1]]))
    assert container_pruned["b"]["d"] is None
    assert container_pruned.b.d is None
    assert "c" not in container_pruned["b"].keys()

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)

    container_pruned = container.prune_key_chain("b")
    assert np.allclose(ivy.to_numpy(container_pruned["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.a), np.array([[1]]))
    assert "b" not in container_pruned.keys()

    def _test_exception(container_in):
        try:
            _ = container_in.b
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_prune_key_chains(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    container_pruned = container.prune_key_chains(["a", "b/c"])
    assert "a" not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert "c" not in container_pruned["b"]

    def _test_a_exception(container_in):
        try:
            _ = container_in.a
            return False
        except AttributeError:
            return True

    def _test_bc_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)

    container_pruned = container.prune_key_chains(
        Container({"a": True, "b": {"c": True}})
    )
    assert "a" not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert "c" not in container_pruned["b"]
    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)


def test_container_format_key_chains(device, call):
    dict_in = {
        "_a": ivy.array([1], device=device),
        "b ": {"c": ivy.array([2], device=device), "d-": ivy.array([3], device=device)},
    }
    cont = Container(dict_in)
    cont_formatted = cont.format_key_chains(
        lambda s: s.replace("_", "").replace(" ", "").replace("-", "")
    )
    assert np.allclose(ivy.to_numpy(cont_formatted["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(cont_formatted.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(cont_formatted["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(cont_formatted.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(cont_formatted["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(cont_formatted.b.d), np.array([3]))


def test_container_sort_by_key(device, call):
    dict_in = {
        "b": ivy.array([1], device=device),
        "a": {"d": ivy.array([2], device=device), "c": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    container_sorted = container.sort_by_key()
    for k, k_true in zip(container_sorted.keys(), ["a", "b"]):
        assert k == k_true
    for k, k_true in zip(container_sorted.a.keys(), ["c", "d"]):
        assert k == k_true


def test_container_prune_empty(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": {}, "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    container_pruned = container.prune_empty()
    assert np.allclose(ivy.to_numpy(container_pruned["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert "c" not in container_pruned["b"]

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_prune_key_from_key_chains(device, call):
    container = Container(
        {
            "Ayy": ivy.array([1], device=device),
            "Bee": {
                "Cee": ivy.array([2], device=device),
                "Dee": ivy.array([3], device=device),
            },
            "Beh": {
                "Ceh": ivy.array([4], device=device),
                "Deh": ivy.array([5], device=device),
            },
        }
    )

    # absolute
    container_pruned = container.prune_key_from_key_chains("Bee")
    assert np.allclose(ivy.to_numpy(container_pruned["Ayy"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Cee"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Dee"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert "Bee" not in container_pruned

    # containing
    container_pruned = container.prune_key_from_key_chains(containing="B")
    assert np.allclose(ivy.to_numpy(container_pruned["Ayy"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Cee"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Dee"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Ceh"]), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ceh), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Deh"]), np.array([[5]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Deh), np.array([[5]]))
    assert "Bee" not in container_pruned
    assert "Beh" not in container_pruned


def test_container_prune_keys_from_key_chains(device, call):
    container = Container(
        {
            "Ayy": ivy.array([1], device=device),
            "Bee": {
                "Cee": ivy.array([2], device=device),
                "Dee": ivy.array([3], device=device),
            },
            "Eee": {"Fff": ivy.array([4], device=device)},
        }
    )

    # absolute
    container_pruned = container.prune_keys_from_key_chains(["Bee", "Eee"])
    assert np.allclose(ivy.to_numpy(container_pruned["Ayy"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Cee"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Dee"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Fff"]), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Fff), np.array([[4]]))
    assert "Bee" not in container_pruned
    assert "Eee" not in container_pruned

    # containing
    container_pruned = container.prune_keys_from_key_chains(containing=["B", "E"])
    assert np.allclose(ivy.to_numpy(container_pruned["Ayy"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Cee"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Dee"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Fff"]), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Fff), np.array([[4]]))
    assert "Bee" not in container_pruned
    assert "Eee" not in container_pruned


def test_container_restructure_key_chains(device, call):

    # single
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_restructured = container.restructure_key_chains({"a": "A"})
    assert np.allclose(ivy.to_numpy(container_restructured["A"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured.A), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured["b/c"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured.b.c), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured["b/d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_restructured.b.d), np.array([[3]]))

    # full
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_restructured = container.restructure_key_chains(
        {"a": "A", "b/c": "B/C", "b/d": "B/D"}
    )
    assert np.allclose(ivy.to_numpy(container_restructured["A"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured.A), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured["B/C"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured.B.C), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured["B/D"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_restructured.B.D), np.array([[3]]))


def test_container_restructure(device, call):
    container = Container(
        {
            "a": ivy.array([[1, 2], [3, 4]], device=device),
            "b": {
                "c": ivy.array([[2, 4], [6, 8]], device=device),
                "d": ivy.array([3, 6, 9, 12], device=device),
            },
        }
    )
    container_restructured = container.restructure(
        {
            "a": {"key_chain": "A", "pattern": "a b -> b a"},
            "b/c": {"key_chain": "B/C", "pattern": "a b -> (a b)"},
            "b/d": {
                "key_chain": "B/D",
                "pattern": "(a b) -> a b",
                "axes_lengths": {"a": 2, "b": 2},
            },
        },
        keep_orig=False,
    )
    assert np.allclose(
        ivy.to_numpy(container_restructured["A"]), np.array([[1, 3], [2, 4]])
    )
    assert np.allclose(
        ivy.to_numpy(container_restructured.A), np.array([[1, 3], [2, 4]])
    )
    assert np.allclose(
        ivy.to_numpy(container_restructured["B/C"]), np.array([2, 4, 6, 8])
    )
    assert np.allclose(ivy.to_numpy(container_restructured.B.C), np.array([2, 4, 6, 8]))
    assert np.allclose(
        ivy.to_numpy(container_restructured["B/D"]), np.array([[3, 6], [9, 12]])
    )
    assert np.allclose(
        ivy.to_numpy(container_restructured.B.D), np.array([[3, 6], [9, 12]])
    )


def test_container_flatten_key_chains(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": {"d": ivy.array([2], device=device)},
                "e": {"f": {"g": ivy.array([3], device=device)}},
            },
        }
    )

    # full
    container_flat = container.flatten_key_chains()
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__c__d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__c__d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__e__f__g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__e__f__g), np.array([[3]]))

    # above height 1
    container_flat = container.flatten_key_chains(above_height=1)
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__c"]["d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__c.d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__e__f"]["g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__e__f.g), np.array([[3]]))

    # below depth 1
    container_flat = container.flatten_key_chains(below_depth=1)
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["c__d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.c__d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["e__f__g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.e__f__g), np.array([[3]]))

    # above height 1, below depth 1
    container_flat = container.flatten_key_chains(above_height=1, below_depth=1)
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["c"]["d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.c.d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["e__f"]["g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.e__f.g), np.array([[3]]))


def test_container_deep_copy(device, call):
    dict_in = {
        "a": ivy.array([0.0], device=device),
        "b": {
            "c": ivy.array([1.0], device=device),
            "d": ivy.array([2.0], device=device),
        },
    }
    cont = Container(dict_in)
    cont_deepcopy = cont.deep_copy()
    assert np.allclose(ivy.to_numpy(cont.a), ivy.to_numpy(cont_deepcopy.a))
    assert np.allclose(ivy.to_numpy(cont.b.c), ivy.to_numpy(cont_deepcopy.b.c))
    assert np.allclose(ivy.to_numpy(cont.b.d), ivy.to_numpy(cont_deepcopy.b.d))
    assert id(cont.a) != id(cont_deepcopy.a)
    assert id(cont.b.c) != id(cont_deepcopy.b.c)
    assert id(cont.b.d) != id(cont_deepcopy.b.d)


def test_container_contains(device, call):
    arr0 = ivy.array([0.0], device=device)
    arr1 = ivy.array([1.0], device=device)
    arr2 = ivy.array([2.0], device=device)
    sub_cont = Container({"c": arr1, "d": arr2})
    container = Container({"a": arr0, "b": sub_cont})

    # keys
    assert "a" in container
    assert "b" in container
    assert "c" not in container
    assert "b/c" in container
    assert "d" not in container
    assert "b/d" in container

    # sub-container
    assert container.contains_sub_container(container)
    assert container.contains_sub_container(sub_cont)
    assert sub_cont in container

    # partial sub-container
    partial_sub_cont = Container({"b": {"d": arr2}})
    assert container.contains_sub_container(container, partial=True)
    assert container.contains_sub_container(partial_sub_cont, partial=True)
    assert not partial_sub_cont.contains_sub_container(container, partial=True)

    # sub-structure
    sub_struc = Container(
        {"c": ivy.array([3.0], device=device), "d": ivy.array([4.0], device=device)}
    )
    assert not container.contains_sub_container(sub_struc)
    assert sub_struc not in container
    assert container.contains_sub_structure(sub_struc)
    assert container.contains_sub_structure(container)

    # partial sub-structure
    partial_sub_struc = Container({"b": {"d": ivy.array([4.0], device=device)}})
    assert container.contains_sub_structure(container, partial=True)
    assert container.contains_sub_structure(partial_sub_struc, partial=True)
    assert not partial_sub_struc.contains_sub_structure(container, partial=True)


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_to_iterator(include_empty, device, call):
    a_val = Container() if include_empty else ivy.array([1], device=device)
    bc_val = Container() if include_empty else ivy.array([2], device=device)
    bd_val = Container() if include_empty else ivy.array([3], device=device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.to_iterator(include_empty=include_empty)
    for (key_chain, value), expected in zip(
        container_iterator, [("a", a_val), ("b/c", bc_val), ("b/d", bd_val)]
    ):
        expected_key_chain = expected[0]
        expected_value = expected[1]
        assert key_chain == expected_key_chain
        assert value is expected_value

    # with leaf keys
    container_iterator = container.to_iterator(
        leaf_keys_only=True, include_empty=include_empty
    )
    for (key_chain, value), expected in zip(
        container_iterator, [("a", a_val), ("c", bc_val), ("d", bd_val)]
    ):
        expected_key_chain = expected[0]
        expected_value = expected[1]
        assert key_chain == expected_key_chain
        assert value is expected_value


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_to_iterator_values(include_empty, device, call):
    a_val = Container() if include_empty else ivy.array([1], device=device)
    bc_val = Container() if include_empty else ivy.array([2], device=device)
    bd_val = Container() if include_empty else ivy.array([3], device=device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.to_iterator_values(include_empty=include_empty)
    for value, expected_value in zip(container_iterator, [a_val, bc_val, bd_val]):
        assert value is expected_value


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_to_iterator_keys(include_empty, device, call):
    a_val = Container() if include_empty else ivy.array([1], device=device)
    bc_val = Container() if include_empty else ivy.array([2], device=device)
    bd_val = Container() if include_empty else ivy.array([3], device=device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.to_iterator_keys(include_empty=include_empty)
    for key_chain, expected_key_chain in zip(container_iterator, ["a", "b/c", "b/d"]):
        assert key_chain == expected_key_chain

    # with leaf keys
    container_iterator = container.to_iterator_keys(
        leaf_keys_only=True, include_empty=include_empty
    )
    for key, expected_key in zip(container_iterator, ["a", "c", "d"]):
        assert key == expected_key


def test_container_to_flat_list(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    container_flat_list = container.to_flat_list()
    for value, expected_value in zip(
        container_flat_list,
        [
            ivy.array([1], device=device),
            ivy.array([2], device=device),
            ivy.array([3], device=device),
        ],
    ):
        assert value == expected_value


def test_container_from_flat_list(device, call):
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container = Container(dict_in)
    flat_list = [4, 5, 6]
    container = container.from_flat_list(flat_list)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


@pytest.mark.parametrize("inplace", [True, False])
def test_container_map(inplace, device, call):
    # without key_chains specification
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }
    container_orig = Container(dict_in)
    container = container_orig.deep_copy()
    container_mapped = container.map(lambda x, _: x + 1, inplace=inplace)
    if inplace:
        container_iterator = container.to_iterator()
    else:
        container_iterator = container_mapped.to_iterator()
    for (key, value), expected_value in zip(
        container_iterator,
        [
            ivy.array([2], device=device),
            ivy.array([3], device=device),
            ivy.array([4], device=device),
        ],
    ):
        assert call(lambda x: x, value) == call(lambda x: x, expected_value)

    # with key_chains to apply
    container = container_orig.deep_copy()
    container_mapped = container.map(lambda x, _: x + 1, ["a", "b/c"], inplace=inplace)
    if inplace:
        container_mapped = container
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["c"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[3]]))

    # with key_chains to apply pruned
    container = container_orig.deep_copy()
    container_mapped = container.map(
        lambda x, _: x + 1, ["a", "b/c"], prune_unapplied=True, inplace=inplace
    )
    if inplace:
        container_mapped = container
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["c"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    if not inplace:
        assert "b/d" not in container_mapped

    # with key_chains to not apply
    container = container_orig.deep_copy()
    container_mapped = container.map(
        lambda x, _: x + 1,
        Container({"a": None, "b": {"d": None}}),
        to_apply=False,
        inplace=inplace,
    )
    if inplace:
        container_mapped = container
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["c"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[3]]))

    # with key_chains to not apply pruned
    container = container_orig.deep_copy()
    container_mapped = container.map(
        lambda x, _: x + 1,
        Container({"a": None, "b": {"d": None}}),
        to_apply=False,
        prune_unapplied=True,
        inplace=inplace,
    )
    if inplace:
        container_mapped = container
    if not inplace:
        assert "a" not in container_mapped
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["c"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    if not inplace:
        assert "b/d" not in container_mapped

    # with sequences
    container_orig = Container(
        {
            "a": ivy.array([1], device=device),
            "b": [ivy.array([2], device=device), ivy.array([3], device=device)],
        }
    )
    container = container_orig.deep_copy()
    container_mapped = container.map(
        lambda x, _: x + 1, inplace=inplace, map_sequences=True
    )
    if inplace:
        container_mapped = container
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"][0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"][1]), np.array([4]))


@pytest.mark.parametrize("inplace", [True, False])
def test_container_map_conts(inplace, device, call):
    # without key_chains specification
    container_orig = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )

    def _add_e_attr(cont_in):
        cont_in.e = ivy.array([4], device=device)
        return cont_in

    # with self
    container = container_orig.deep_copy()
    container_mapped = container.map_conts(lambda c, _: _add_e_attr(c), inplace=inplace)
    if inplace:
        container_mapped = container
    assert "e" in container_mapped
    assert np.array_equal(ivy.to_numpy(container_mapped.e), np.array([4]))
    assert "e" in container_mapped.b
    assert np.array_equal(ivy.to_numpy(container_mapped.b.e), np.array([4]))

    # without self
    container = container_orig.deep_copy()
    container_mapped = container.map_conts(
        lambda c, _: _add_e_attr(c), include_self=False, inplace=inplace
    )
    if inplace:
        container_mapped = container
    assert "e" not in container_mapped
    assert "e" in container_mapped.b
    assert np.array_equal(ivy.to_numpy(container_mapped.b.e), np.array([4]))


def test_container_multi_map(device, call):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )

    # with key_chains to apply
    container_mapped = ivy.Container.multi_map(
        lambda x, _: x[0] + x[1], [container0, container1], assert_identical=True
    )
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["c"]), np.array([[6]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[6]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["d"]), np.array([[8]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[8]]))

    # with sequences
    container0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": [
                ivy.array([2], device=device),
                ivy.array([3], device=device),
            ],
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=device),
            "b": [
                ivy.array([4], device=device),
                ivy.array([5], device=device),
            ],
        }
    )

    container_mapped = ivy.Container.multi_map(
        lambda x, _: x[0] + x[1],
        [container0, container1],
        map_nests=True,
        assert_identical=True,
    )

    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"][0]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"][1]), np.array([8]))

    # Non identical containers
    a = ivy.Container(a={"b": 2, "c": 4}, d={"e": 6, "f": 9})
    b = ivy.Container(a=2, d=3)
    container_mapped = ivy.Container.multi_map(lambda xs, _: xs[0] / xs[1], [a, b])

    assert np.allclose(ivy.to_numpy(container_mapped["a"].b), 1)
    assert np.allclose(ivy.to_numpy(container_mapped["a"]["c"]), 2)
    assert np.allclose(ivy.to_numpy(container_mapped.d.e), 2)
    assert np.allclose(ivy.to_numpy(container_mapped["d"].f), 3)


def test_container_common_key_chains(device, call):
    arr1 = ivy.array([1], device=device)
    arr2 = ivy.array([2], device=device)
    arr3 = ivy.array([3], device=device)
    cont0 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    cont1 = Container({"b": {"c": arr2, "d": arr3, "e": arr1}})
    cont2 = Container({"a": arr1, "b": {"d": arr3, "e": arr1}})

    # 0
    common_kcs = Container.common_key_chains([cont0])
    assert len(common_kcs) == 3
    assert "a" in common_kcs
    assert "b/c" in common_kcs
    assert "b/d" in common_kcs

    # 0-1
    common_kcs = Container.common_key_chains([cont0, cont1])
    assert len(common_kcs) == 2
    assert "b/c" in common_kcs
    assert "b/d" in common_kcs

    # 0-2
    common_kcs = Container.common_key_chains([cont0, cont2])
    assert len(common_kcs) == 2
    assert "a" in common_kcs
    assert "b/d" in common_kcs

    # 1-2
    common_kcs = Container.common_key_chains([cont1, cont2])
    assert len(common_kcs) == 2
    assert "b/d" in common_kcs
    assert "b/e" in common_kcs

    # all
    common_kcs = Container.common_key_chains([cont0, cont1, cont2])
    assert len(common_kcs) == 1
    assert "b/d" in common_kcs


def test_container_identical(device, call):
    # without key_chains specification
    arr1 = ivy.array([1], device=device)
    arr2 = ivy.array([2], device=device)
    arr3 = ivy.array([3], device=device)
    container0 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    container1 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    container2 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container3 = Container({"b": {"d": arr3}})
    container4 = Container({"d": arr3})

    # the same
    assert ivy.Container.identical([container0, container1])
    assert ivy.Container.identical([container1, container0])

    # not the same
    assert not ivy.Container.identical([container0, container2])
    assert not ivy.Container.identical([container2, container0])
    assert not ivy.Container.identical([container1, container2])
    assert not ivy.Container.identical([container2, container1])

    # partial
    assert ivy.Container.identical([container0, container3], partial=True)
    assert ivy.Container.identical([container3, container0], partial=True)
    assert not ivy.Container.identical([container0, container4], partial=True)
    assert not ivy.Container.identical([container4, container0], partial=True)


def test_container_identical_structure(device, call):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container2 = Container(
        {
            "a": ivy.array([3], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([5], device=device),
                "e": ivy.array([6], device=device),
            },
        }
    )
    container3 = Container(
        {
            "a": ivy.array([3], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([5], device=device),
            },
            "e": ivy.array([6], device=device),
        }
    )
    container4 = Container({"b": {"d": ivy.array([4], device=device)}})
    container5 = Container({"d": ivy.array([4], device=device)})

    # with identical
    assert ivy.Container.identical_structure([container0, container1])
    assert ivy.Container.identical_structure([container1, container0])
    assert ivy.Container.identical_structure([container1, container0, container1])

    # without identical
    assert not ivy.Container.identical_structure([container2, container3])
    assert not ivy.Container.identical_structure([container0, container3])
    assert not ivy.Container.identical_structure([container1, container2])
    assert not ivy.Container.identical_structure([container1, container0, container2])

    # partial
    assert ivy.Container.identical_structure([container0, container4], partial=True)
    assert ivy.Container.identical_structure([container1, container4], partial=True)
    assert ivy.Container.identical_structure([container2, container4], partial=True)
    assert ivy.Container.identical_structure([container3, container4], partial=True)
    assert ivy.Container.identical_structure([container4, container4], partial=True)
    assert not ivy.Container.identical_structure([container0, container5], partial=True)
    assert not ivy.Container.identical_structure([container1, container5], partial=True)
    assert not ivy.Container.identical_structure([container2, container5], partial=True)
    assert not ivy.Container.identical_structure([container3, container5], partial=True)
    assert not ivy.Container.identical_structure([container4, container5], partial=True)


def test_container_identical_configs(device, call):
    container0 = Container({"a": ivy.array([1], device=device)}, print_limit=5)
    container1 = Container({"a": ivy.array([1], device=device)}, print_limit=5)
    container2 = Container({"a": ivy.array([1], device=device)}, print_limit=10)

    # with identical
    assert ivy.Container.identical_configs([container0, container1])
    assert ivy.Container.identical_configs([container1, container0])
    assert ivy.Container.identical_configs([container1, container0, container1])

    # without identical
    assert not ivy.Container.identical_configs([container1, container2])
    assert not ivy.Container.identical_configs([container1, container0, container2])


def test_container_identical_array_shapes(device, call):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1, 2], device=device),
            "b": {
                "c": ivy.array([2, 3, 4], device=device),
                "d": ivy.array([3, 4, 5, 6], device=device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([1, 2, 3, 4], device=device),
            "b": {
                "c": ivy.array([3, 4], device=device),
                "d": ivy.array([3, 4, 5], device=device),
            },
        }
    )
    container2 = Container(
        {
            "a": ivy.array([1, 2, 3, 4], device=device),
            "b": {
                "c": ivy.array([3, 4], device=device),
                "d": ivy.array([3, 4, 5, 6], device=device),
            },
        }
    )

    # with identical
    assert ivy.Container.identical_array_shapes([container0, container1])
    assert ivy.Container.identical_array_shapes([container1, container0])
    assert ivy.Container.identical_array_shapes([container1, container0, container1])
    assert not ivy.Container.identical([container0, container2])
    assert not ivy.Container.identical([container1, container2])
    assert not ivy.Container.identical([container0, container1, container2])


def test_container_with_entries_as_lists(device, call):
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2.0], device=device), "d": "some string"},
    }
    container = Container(dict_in)
    container_w_list_entries = container.with_entries_as_lists()
    for (key, value), expected_value in zip(
        container_w_list_entries.to_iterator(), [[1], [2.0], "some string"]
    ):
        assert value == expected_value


def test_container_reshape_like(device, call):
    container = Container(
        {
            "a": ivy.array([[1.0]], device=device),
            "b": {
                "c": ivy.array([[3.0], [4.0]], device=device),
                "d": ivy.array([[5.0], [6.0], [7.0]], device=device),
            },
        }
    )
    new_shapes = Container({"a": (1,), "b": {"c": (1, 2, 1), "d": (3, 1, 1)}})

    # without leading shape
    container_reshaped = container.reshape_like(new_shapes)
    assert list(container_reshaped["a"].shape) == [1]
    assert list(container_reshaped.a.shape) == [1]
    assert list(container_reshaped["b"]["c"].shape) == [1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [1, 2, 1]
    assert list(container_reshaped["b"]["d"].shape) == [3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 1, 1]

    # with leading shape
    container = Container(
        {
            "a": ivy.array([[[1.0]], [[1.0]], [[1.0]]], device=device),
            "b": {
                "c": ivy.array(
                    [[[3.0], [4.0]], [[3.0], [4.0]], [[3.0], [4.0]]], device=device
                ),
                "d": ivy.array(
                    [
                        [[5.0], [6.0], [7.0]],
                        [[5.0], [6.0], [7.0]],
                        [[5.0], [6.0], [7.0]],
                    ],
                    device=device,
                ),
            },
        }
    )
    container_reshaped = container.reshape_like(new_shapes, leading_shape=[3])
    assert list(container_reshaped["a"].shape) == [3, 1]
    assert list(container_reshaped.a.shape) == [3, 1]
    assert list(container_reshaped["b"]["c"].shape) == [3, 1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [3, 1, 2, 1]
    assert list(container_reshaped["b"]["d"].shape) == [3, 3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 3, 1, 1]


def test_container_slice(device, call):
    dict_in = {
        "a": ivy.array([[0.0], [1.0]], device=device),
        "b": {
            "c": ivy.array([[1.0], [2.0]], device=device),
            "d": ivy.array([[2.0], [3.0]], device=device),
        },
    }
    container = Container(dict_in)
    container0 = container[0]
    container1 = container[1]
    assert np.array_equal(ivy.to_numpy(container0["a"]), np.array([0.0]))
    assert np.array_equal(ivy.to_numpy(container0.a), np.array([0.0]))
    assert np.array_equal(ivy.to_numpy(container0["b"]["c"]), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(container0.b.c), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(container0["b"]["d"]), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(container0.b.d), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(container1["a"]), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(container1.a), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(container1["b"]["c"]), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(container1.b.c), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(container1["b"]["d"]), np.array([3.0]))
    assert np.array_equal(ivy.to_numpy(container1.b.d), np.array([3.0]))


def test_container_slice_via_key(device, call):
    dict_in = {
        "a": {
            "x": ivy.array([0.0], device=device),
            "y": ivy.array([1.0], device=device),
        },
        "b": {
            "c": {
                "x": ivy.array([1.0], device=device),
                "y": ivy.array([2.0], device=device),
            },
            "d": {
                "x": ivy.array([2.0], device=device),
                "y": ivy.array([3.0], device=device),
            },
        },
    }
    container = Container(dict_in)
    containerx = container.slice_via_key("x")
    containery = container.slice_via_key("y")
    assert np.array_equal(ivy.to_numpy(containerx["a"]), np.array([0.0]))
    assert np.array_equal(ivy.to_numpy(containerx.a), np.array([0.0]))
    assert np.array_equal(ivy.to_numpy(containerx["b"]["c"]), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(containerx.b.c), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(containerx["b"]["d"]), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(containerx.b.d), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(containery["a"]), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(containery.a), np.array([1.0]))
    assert np.array_equal(ivy.to_numpy(containery["b"]["c"]), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(containery.b.c), np.array([2.0]))
    assert np.array_equal(ivy.to_numpy(containery["b"]["d"]), np.array([3.0]))
    assert np.array_equal(ivy.to_numpy(containery.b.d), np.array([3.0]))


def test_container_to_and_from_disk_as_hdf5(device, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = "container_on_disk.hdf5"
    dict_in_1 = {
        "a": ivy.array([np.float32(1.0)], device=device),
        "b": {
            "c": ivy.array([np.float32(2.0)], device=device),
            "d": ivy.array([np.float32(3.0)], device=device),
        },
    }
    container1 = Container(dict_in_1)
    dict_in_2 = {
        "a": ivy.array([np.float32(1.0), np.float32(1.0)], device=device),
        "b": {
            "c": ivy.array([np.float32(2.0), np.float32(2.0)], device=device),
            "d": ivy.array([np.float32(3.0), np.float32(3.0)], device=device),
        },
    }
    container2 = Container(dict_in_2)

    # saving
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_hdf5(save_filepath, slice(1))
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container1.a))
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container1.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container1.b.d)
    )

    # appending
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2, starting_index=1)
    assert os.path.exists(save_filepath)

    # loading after append
    loaded_container = Container.from_disk_as_hdf5(save_filepath)
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container2.a))
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container2.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container2.b.d)
    )

    # load slice
    loaded_sliced_container = Container.from_disk_as_hdf5(save_filepath, slice(1, 2))
    assert np.array_equal(
        ivy.to_numpy(loaded_sliced_container.a), ivy.to_numpy(container1.a)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_sliced_container.b.c), ivy.to_numpy(container1.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_sliced_container.b.d), ivy.to_numpy(container1.b.d)
    )

    # file size
    file_size, batch_size = Container.h5_file_size(save_filepath)
    assert file_size == 6 * np.dtype(np.float32).itemsize
    assert batch_size == 2

    os.remove(save_filepath)


def test_container_to_disk_shuffle_and_from_disk_as_hdf5(device, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = "container_on_disk.hdf5"
    dict_in = {
        "a": ivy.array([1, 2, 3], device=device),
        "b": {
            "c": ivy.array([1, 2, 3], device=device),
            "d": ivy.array([1, 2, 3], device=device),
        },
    }
    container = Container(dict_in)

    # saving
    container.to_disk_as_hdf5(save_filepath, max_batch_size=3)
    assert os.path.exists(save_filepath)

    # shuffling
    Container.shuffle_h5_file(save_filepath)

    # loading
    container_shuffled = Container.from_disk_as_hdf5(save_filepath, slice(3))

    # testing
    data = np.array([1, 2, 3])
    random.seed(0)
    random.shuffle(data)

    assert (ivy.to_numpy(container_shuffled["a"]) == data).all()
    assert (ivy.to_numpy(container_shuffled.a) == data).all()
    assert (ivy.to_numpy(container_shuffled["b"]["c"]) == data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == data).all()
    assert (ivy.to_numpy(container_shuffled["b"]["d"]) == data).all()
    assert (ivy.to_numpy(container_shuffled.b.d) == data).all()

    os.remove(save_filepath)


def test_container_pickle(device, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    dict_in = {
        "a": ivy.array([np.float32(1.0)], device=device),
        "b": {
            "c": ivy.array([np.float32(2.0)], device=device),
            "d": ivy.array([np.float32(3.0)], device=device),
        },
    }

    # without module attribute
    cont = Container(dict_in)
    assert cont._local_ivy is None
    pickled = pickle.dumps(cont)
    cont_again = pickle.loads(pickled)
    assert cont_again._local_ivy is None
    ivy.Container.identical_structure([cont, cont_again])
    ivy.Container.identical_configs([cont, cont_again])

    # with module attribute
    cont = Container(dict_in, ivyh=ivy)
    assert cont._local_ivy is ivy
    pickled = pickle.dumps(cont)
    cont_again = pickle.loads(pickled)
    # noinspection PyUnresolvedReferences
    assert cont_again._local_ivy.current_backend_str() is ivy.current_backend_str()
    ivy.Container.identical_structure([cont, cont_again])
    ivy.Container.identical_configs([cont, cont_again])


def test_container_to_and_from_disk_as_pickled(device, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = "container_on_disk.pickled"
    dict_in = {
        "a": ivy.array([np.float32(1.0)], device=device),
        "b": {
            "c": ivy.array([np.float32(2.0)], device=device),
            "d": ivy.array([np.float32(3.0)], device=device),
        },
    }
    container = Container(dict_in)

    # saving
    container.to_disk_as_pickled(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_pickled(save_filepath)
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container.a))
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container.b.d)
    )

    os.remove(save_filepath)


def test_container_to_and_from_disk_as_json(device, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = "container_on_disk.json"
    dict_in = {
        "a": 1.274e-7,
        "b": {"c": True, "d": ivy.array([np.float32(3.0)], device=device)},
    }
    container = Container(dict_in)

    # saving
    container.to_disk_as_json(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_json(save_filepath)
    assert np.array_equal(loaded_container.a, container.a)
    assert np.array_equal(loaded_container.b.c, container.b.c)
    assert isinstance(loaded_container.b.d, str)

    os.remove(save_filepath)


def test_container_positive(device, call):
    container = +Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([-2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([-2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([-2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_negative(device, call):
    container = -Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([-2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([-3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([-3]))


def test_container_pow(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container = container_a**container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([16]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([16]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([729]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([729]))


def test_container_scalar_pow(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container_a**2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([9]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([9]))


def test_container_reverse_scalar_pow(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2**container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([8]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([8]))


def test_container_scalar_addition(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container += 3
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


def test_container_reverse_scalar_addition(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 3 + container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


def test_container_addition(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container = container_a + container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([6]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([9]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([9]))


def test_container_scalar_subtraction(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container -= 1
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2]))


def test_container_reverse_scalar_subtraction(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 1 - container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([-2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([-2]))


def test_container_subtraction(device, call):
    container_a = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([1], device=device),
                "d": ivy.array([4], device=device),
            },
        }
    )
    container = container_a - container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2]))


def test_container_scalar_multiplication(device, call):
    container = Container(
        {
            "a": ivy.array([1.0], device=device),
            "b": {
                "c": ivy.array([2.0], device=device),
                "d": ivy.array([3.0], device=device),
            },
        }
    )
    container *= 2.5
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5.0]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5.0]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([7.5]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([7.5]))


def test_container_reverse_scalar_multiplication(device, call):
    container = Container(
        {
            "a": ivy.array([1.0], device=device),
            "b": {
                "c": ivy.array([2.0], device=device),
                "d": ivy.array([3.0], device=device),
            },
        }
    )
    container = 2.5 * container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5.0]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5.0]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([7.5]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([7.5]))


def test_container_multiplication(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([4], device=device),
                "d": ivy.array([6], device=device),
            },
        }
    )
    container = container_a * container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([8]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([8]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([18]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([18]))


def test_container_scalar_truediv(device, call):
    container = Container(
        {
            "a": ivy.array([1.0], device=device),
            "b": {
                "c": ivy.array([5.0], device=device),
                "d": ivy.array([5.0], device=device),
            },
        }
    )
    container /= 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2.5]))


def test_container_reverse_scalar_truediv(device, call):
    container = Container(
        {
            "a": ivy.array([1.0], device=device),
            "b": {
                "c": ivy.array([5.0], device=device),
                "d": ivy.array([5.0], device=device),
            },
        }
    )
    container = 2 / container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([2.0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2.0]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([0.4]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([0.4]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([0.4]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([0.4]))


def test_container_truediv(device, call):
    container_a = Container(
        {
            "a": ivy.array([1.0], device=device),
            "b": {
                "c": ivy.array([5.0], device=device),
                "d": ivy.array([5.0], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2.0], device=device),
            "b": {
                "c": ivy.array([2.0], device=device),
                "d": ivy.array([4.0], device=device),
            },
        }
    )
    container = container_a / container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([1.25]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([1.25]))


def test_container_scalar_floordiv(device, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit
        # ivy.floordiv is implemented at some point
        pytest.skip()
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container //= 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2]))


def test_container_reverse_scalar_floordiv(device, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit
        # ivy.floordiv is implemented at some point
        pytest.skip()
    container = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([1], device=device),
                "d": ivy.array([7], device=device),
            },
        }
    )
    container = 5 // container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([0]))


def test_container_floordiv(device, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit
        # ivy.floordiv is implemented at some point
        pytest.skip()
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([4], device=device),
            },
        }
    )
    container = container_a // container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([1]))


def test_container_abs(device, call):
    container = abs(
        Container(
            {
                "a": ivy.array([1], device=device),
                "b": {
                    "c": ivy.array([-2], device=device),
                    "d": ivy.array([3], device=device),
                },
            }
        )
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_scalar_less_than(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container < 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_less_than(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2 < container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_less_than(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container = container_a < container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_less_than_or_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container <= 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_less_than_or_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2 <= container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_less_than_or_equal_to(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container = container_a <= container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container == 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2 == container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_equal_to(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container = container_a == container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_not_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container != 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_not_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2 != container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_not_equal_to(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container = container_a != container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_greater_than(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container > 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_greater_than(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2 > container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_greater_than(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container = container_a > container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_greater_than_or_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = container >= 2
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_greater_than_or_equal_to(device, call):
    container = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([3], device=device),
            },
        }
    )
    container = 2 >= container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_greater_than_or_equal_to(device, call):
    container_a = Container(
        {
            "a": ivy.array([1], device=device),
            "b": {
                "c": ivy.array([5], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([2], device=device),
            "b": {
                "c": ivy.array([2], device=device),
                "d": ivy.array([5], device=device),
            },
        }
    )
    container = container_a >= container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_and(device, call):
    container = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container & True
    # ToDo: work out why "container and True" does not work. Perhaps bool(container)
    #  is called first implicitly?
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_and(device, call):
    container = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = True and container
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_and(device, call):
    container_a = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([False], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container_a and container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_or(device, call):
    container = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container or False
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_or(device, call):
    container = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container or False
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_or(device, call):
    container_a = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([False], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container_a or container_b
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_not(device, call):
    container = ~Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_xor(device, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit
        # ivy.logical_xor is implemented at some point
        pytest.skip()
    container = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container != True  # noqa
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_xor(device, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit
        # ivy.logical_xor is implemented at some point
        pytest.skip()
    container = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = False != container  # noqa
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_xor(device, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit
        # ivy.logical_xor is implemented at some point
        pytest.skip()
    container_a = Container(
        {
            "a": ivy.array([True], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container_b = Container(
        {
            "a": ivy.array([False], device=device),
            "b": {
                "c": ivy.array([True], device=device),
                "d": ivy.array([False], device=device),
            },
        }
    )
    container = container_a != container_b  # noqa
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_shape(device, call):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=device),
        },
    }
    container = Container(dict_in)
    assert container.shape == [1, 3, 1]
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]], device=device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=device),
        },
    }
    container = Container(dict_in)
    assert container.shape == [1, 3, None]
    dict_in = {
        "a": ivy.array([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]], device=device),
            "d": ivy.array([[[3.0, 4.0], [6.0, 7.0], [9.0, 10.0]]], device=device),
        },
    }
    container = Container(dict_in)
    assert container.shape == [1, 3, 2]


def test_container_shapes(device, call):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0], [4.0]]], device=device),
            "d": ivy.array([[9.0]], device=device),
        },
    }
    container_shapes = Container(dict_in).shapes
    assert list(container_shapes["a"]) == [1, 3, 1]
    assert list(container_shapes.a) == [1, 3, 1]
    assert list(container_shapes["b"]["c"]) == [1, 2, 1]
    assert list(container_shapes.b.c) == [1, 2, 1]
    assert list(container_shapes["b"]["d"]) == [1, 1]
    assert list(container_shapes.b.d) == [1, 1]


def test_container_dev_str(device, call):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=device),
        },
    }
    container = Container(dict_in)
    assert container.dev_str == device


def test_container_create_if_absent(device, call):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=device),
        },
    }

    # depth 1
    container = Container(dict_in)
    container.create_if_absent("a", None, True)
    assert np.allclose(ivy.to_numpy(container.a), np.array([[[1.0], [2.0], [3.0]]]))
    container.create_if_absent("e", ivy.array([[[4.0], [8.0], [12.0]]]), True)
    assert np.allclose(ivy.to_numpy(container.e), np.array([[[4.0], [8.0], [12.0]]]))

    # depth 2
    container.create_if_absent("f/g", np.array([[[5.0], [10.0], [15.0]]]), True)
    assert np.allclose(ivy.to_numpy(container.f.g), np.array([[[5.0], [10.0], [15.0]]]))


def test_container_if_exists(device, call):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=device),
        },
    }
    container = Container(dict_in)
    assert np.allclose(
        ivy.to_numpy(container.if_exists("a")), np.array([[[1.0], [2.0], [3.0]]])
    )
    assert "c" not in container
    assert container.if_exists("c") is None
    container["c"] = ivy.array([[[1.0], [2.0], [3.0]]], device=device)
    assert np.allclose(
        ivy.to_numpy(container.if_exists("c")), np.array([[[1.0], [2.0], [3.0]]])
    )
    assert container.if_exists("d") is None
    container.d = ivy.array([[[1.0], [2.0], [3.0]]], device=device)
    assert np.allclose(
        ivy.to_numpy(container.if_exists("d")), np.array([[[1.0], [2.0], [3.0]]])
    )


def test_jax_pytree_compatibility(device, call):

    if call is not helpers.jnp_call:
        pytest.skip()

    # import
    from jax.tree_util import tree_flatten

    # dict in
    dict_in = {
        "a": ivy.array([1], device=device),
        "b": {"c": ivy.array([2], device=device), "d": ivy.array([3], device=device)},
    }

    # container
    container = Container(dict_in)

    # container flattened
    cont_values = tree_flatten(container)[0]

    # dict flattened
    true_values = tree_flatten(dict_in)[0]

    # assertion
    for i, true_val in enumerate(true_values):
        assert np.array_equal(ivy.to_numpy(cont_values[i]), ivy.to_numpy(true_val))


def test_container_from_queues(device, call):

    if "gpu" in device:
        # Cannot re-initialize CUDA in forked subprocess. 'spawn'
        # start method must be used.
        pytest.skip()

    if ivy.gpu_is_available() and call is helpers.jnp_call:
        # Not found a way to set default device for JAX, and this causes
        # issues with multiprocessing and CUDA, even when device=cpu
        # ToDo: find a fix for this problem ^^
        pytest.skip()

    def worker_fn(in_queue, out_queue, load_size, worker_id):
        keep_going = True
        while keep_going:
            try:
                keep_going = in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            out_queue.put(
                {
                    "a": [
                        ivy.to_native(ivy.array([1.0, 2.0, 3.0], device=device))
                        * worker_id
                    ]
                    * load_size
                }
            )

    workers = list()
    in_queues = list()
    out_queues = list()
    queue_load_sizes = [1, 2, 1]
    for i, queue_load_size in enumerate(queue_load_sizes):
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=worker_fn, args=(input_queue, output_queue, queue_load_size, i + 1)
        )
        worker.start()
        in_queues.append(input_queue)
        out_queues.append(output_queue)
        workers.append(worker)

    container = Container(
        queues=out_queues, queue_load_sizes=queue_load_sizes, queue_timeout=0.25
    )

    # queue 0
    queue_was_empty = False
    try:
        container[0]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    in_queues[0].put(True)
    assert np.allclose(ivy.to_numpy(container[0].a), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(ivy.to_numpy(container[0].a), np.array([1.0, 2.0, 3.0]))

    # queue 1
    queue_was_empty = False
    try:
        container[1]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    queue_was_empty = False
    try:
        container[2]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    in_queues[1].put(True)
    assert np.allclose(ivy.to_numpy(container[1].a), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(ivy.to_numpy(container[1].a), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(ivy.to_numpy(container[2].a), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(ivy.to_numpy(container[2].a), np.array([2.0, 4.0, 6.0]))

    # queue 2
    queue_was_empty = False
    try:
        container[3]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    in_queues[2].put(True)
    assert np.allclose(ivy.to_numpy(container[3].a), np.array([3.0, 6.0, 9.0]))
    assert np.allclose(ivy.to_numpy(container[3].a), np.array([3.0, 6.0, 9.0]))

    # stop workers
    in_queues[0].put(False)
    in_queues[1].put(False)
    in_queues[2].put(False)
    in_queues[0].close()
    in_queues[1].close()
    in_queues[2].close()

    # join workers
    for worker in workers:
        worker.join()

    del container
