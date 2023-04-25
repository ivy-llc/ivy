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
from ivy.functional.ivy.gradients import _variable
from ivy.data_classes.container import Container
from ivy.utils.exceptions import IvyException


def test_container_list_join(on_device):
    container_0 = Container(
        {
            "a": [ivy.array([1], device=on_device)],
            "b": {
                "c": [ivy.array([2], device=on_device)],
                "d": [ivy.array([3], device=on_device)],
            },
        }
    )
    container_1 = Container(
        {
            "a": [ivy.array([4], device=on_device)],
            "b": {
                "c": [ivy.array([5], device=on_device)],
                "d": [ivy.array([6], device=on_device)],
            },
        }
    )
    container_list_joined = ivy.Container.cont_list_join([container_0, container_1])
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


def test_container_list_stack(on_device):
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {
                "c": ivy.array([5], device=on_device),
                "d": ivy.array([6], device=on_device),
            },
        }
    )
    container_list_stacked = ivy.Container.cont_list_stack(
        [container_0, container_1], 0
    )
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


def test_container_unify(on_device):
    # on_devices and containers
    on_devices = list()
    dev0 = on_device
    on_devices.append(dev0)
    conts = dict()
    conts[dev0] = Container(
        {
            "a": ivy.array([1], device=dev0),
            "b": {"c": ivy.array([2], device=dev0), "d": ivy.array([3], device=dev0)},
        }
    )
    if "gpu" in on_device and ivy.num_gpus() > 1:
        idx = ivy.num_gpus() - 1
        dev1 = on_device[:-1] + str(idx)
        on_devices.append(dev1)
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
    container_unified = ivy.Container.cont_unify(conts, dev0, "concat", 0)
    assert np.allclose(ivy.to_numpy(container_unified.a[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_unified.b.c[0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_unified.b.d[0]), np.array([3]))
    if len(on_devices) > 1:
        assert np.allclose(ivy.to_numpy(container_unified.a[1]), np.array([4]))
        assert np.allclose(ivy.to_numpy(container_unified.b.c[1]), np.array([5]))
        assert np.allclose(ivy.to_numpy(container_unified.b.d[1]), np.array([6]))


def test_container_combine(on_device):
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {
                "c": ivy.array([5], device=on_device),
                "e": ivy.array([6], device=on_device),
            },
        }
    )
    container_comb = ivy.Container.cont_combine(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_comb.a), np.array([4]))
    assert np.equal(ivy.to_numpy(container_comb.b.c), np.array([5]))
    assert np.equal(ivy.to_numpy(container_comb.b.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_comb.b.e), np.array([6]))


def test_container_diff(on_device):
    # all different arrays
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {
                "c": ivy.array([5], device=on_device),
                "d": ivy.array([6], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.a.diff_1), np.array([4]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_1), np.array([6]))
    container_diff_diff_only = ivy.Container.cont_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == container_diff.cont_to_dict()
    container_diff_same_only = ivy.Container.cont_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == {}

    # some different arrays
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([5], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" in container_diff_diff_only["b"]
    assert "d" not in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.cont_diff(
        container_0, container_1, mode="same_only"
    )
    assert "a" in container_diff_same_only
    assert "b" in container_diff_same_only
    assert "c" not in container_diff_same_only["b"]
    assert "d" in container_diff_same_only["b"]

    # all different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "e": ivy.array([1], device=on_device),
            "f": {
                "g": ivy.array([2], device=on_device),
                "h": ivy.array([3], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.e.diff_1), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.g), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.h), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == container_diff.cont_to_dict()
    container_diff_same_only = ivy.Container.cont_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == {}

    # some different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "e": ivy.array([3], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" not in container_diff_diff_only["b"]
    assert "d" in container_diff_diff_only["b"]
    assert "e" in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.cont_diff(
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
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == {}
    container_diff_same_only = ivy.Container.cont_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == container_diff.cont_to_dict()

    # all different strings
    container_0 = Container({"a": "1", "b": {"c": "2", "d": "3"}})
    container_1 = Container({"a": "4", "b": {"c": "5", "d": "6"}})
    container_diff = ivy.Container.cont_diff(container_0, container_1)
    assert container_diff.a.diff_0 == "1"
    assert container_diff.a.diff_1 == "4"
    assert container_diff.b.c.diff_0 == "2"
    assert container_diff.b.c.diff_1 == "5"
    assert container_diff.b.d.diff_0 == "3"
    assert container_diff.b.d.diff_1 == "6"
    container_diff_diff_only = ivy.Container.cont_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == container_diff.cont_to_dict()
    container_diff_same_only = ivy.Container.cont_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == {}


def test_container_structural_diff(on_device):
    # all different keys or shapes
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([[4]], device=on_device),
            "b": {
                "c": ivy.array([[[5]]], device=on_device),
                "e": ivy.array([3], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.a.diff_1), np.array([[4]]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([[[5]]]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == container_diff.cont_to_dict()
    container_diff_same_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == {}

    # some different shapes
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {
                "c": ivy.array([[5]], device=on_device),
                "d": ivy.array([6], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" in container_diff_diff_only["b"]
    assert "d" not in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert "a" in container_diff_same_only
    assert "b" in container_diff_same_only
    assert "c" not in container_diff_same_only["b"]
    assert "d" in container_diff_same_only["b"]

    # all different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "e": ivy.array([4], device=on_device),
            "f": {
                "g": ivy.array([5], device=on_device),
                "h": ivy.array([6], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.e.diff_1), np.array([4]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.g), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.h), np.array([6]))
    container_diff_diff_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == container_diff.cont_to_dict()
    container_diff_same_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == {}

    # some different keys
    container_0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {
                "c": ivy.array([5], device=on_device),
                "e": ivy.array([6], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([6]))
    container_diff_diff_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert "a" not in container_diff_diff_only
    assert "b" in container_diff_diff_only
    assert "c" not in container_diff_diff_only["b"]
    assert "d" in container_diff_diff_only["b"]
    assert "e" in container_diff_diff_only["b"]
    container_diff_same_only = ivy.Container.cont_structural_diff(
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
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_1 = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {
                "c": ivy.array([5], device=on_device),
                "d": ivy.array([6], device=on_device),
            },
        }
    )
    container_diff = ivy.Container.cont_structural_diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="diff_only"
    )
    assert container_diff_diff_only.cont_to_dict() == {}
    container_diff_same_only = ivy.Container.cont_structural_diff(
        container_0, container_1, mode="same_only"
    )
    assert container_diff_same_only.cont_to_dict() == container_diff.cont_to_dict()


def test_container_from_dict(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_depth(on_device):
    cont_depth1 = Container(
        {"a": ivy.array([1], device=on_device), "b": ivy.array([2], device=on_device)}
    )
    assert cont_depth1.cont_max_depth == 1
    cont_depth2 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    assert cont_depth2.cont_max_depth == 2
    cont_depth3 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": {"d": ivy.array([2], device=on_device)},
                "e": ivy.array([3], device=on_device),
            },
        }
    )
    assert cont_depth3.cont_max_depth == 3
    cont_depth4 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {"c": {"d": {"e": ivy.array([2], device=on_device)}}},
        }
    )
    assert cont_depth4.cont_max_depth == 4


@pytest.mark.parametrize("inplace", [True, False])
def test_container_cutoff_at_depth(inplace, on_device):
    # values
    a_val = ivy.array([1], device=on_device)
    bcde_val = ivy.array([2], device=on_device)

    # depth 1
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cont_cutoff_at_depth(1, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert not cont_cutoff.b

    # depth 2
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cont_cutoff_at_depth(2, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert not cont_cutoff.b.c

    # depth 3
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cont_cutoff_at_depth(3, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert not cont_cutoff.b.c.d

    # depth 4
    cont = Container({"a": a_val, "b": {"c": {"d": {"e": bcde_val}}}})
    cont_cutoff = cont.cont_cutoff_at_depth(4, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a), ivy.to_numpy(a_val))
    assert np.allclose(ivy.to_numpy(cont_cutoff.b.c.d.e), ivy.to_numpy(bcde_val))


@pytest.mark.parametrize("inplace", [True, False])
def test_container_cutoff_at_height(inplace, on_device):
    # values
    d_val = ivy.array([2], device=on_device)
    e_val = ivy.array([3], device=on_device)

    # height 0
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cont_cutoff_at_height(0, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert np.allclose(ivy.to_numpy(cont_cutoff.a.c.d), ivy.to_numpy(d_val))
    assert np.allclose(ivy.to_numpy(cont_cutoff.b.c.d.e), ivy.to_numpy(e_val))

    # height 1
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cont_cutoff_at_height(1, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff.a.c
    assert not cont_cutoff.b.c.d

    # height 2
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cont_cutoff_at_height(2, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff.a
    assert not cont_cutoff.b.c

    # height 3
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cont_cutoff_at_height(3, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff.a
    assert not cont_cutoff.b

    # height 4
    cont = Container({"a": {"c": {"d": d_val}}, "b": {"c": {"d": {"e": e_val}}}})
    cont_cutoff = cont.cont_cutoff_at_height(4, inplace=inplace)
    if inplace:
        cont_cutoff = cont
    assert not cont_cutoff


@pytest.mark.parametrize("str_slice", [True, False])
def test_container_slice_keys(str_slice, on_device):
    # values
    a_val = ivy.array([1], device=on_device)
    b_val = ivy.array([2], device=on_device)
    c_val = ivy.array([3], device=on_device)
    d_val = ivy.array([4], device=on_device)
    e_val = ivy.array([5], device=on_device)

    # slice
    if str_slice:
        slc = "b:d"
    else:
        slc = slice(1, 4, 1)

    # without dict
    cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    cont_sliced = cont.cont_slice_keys(slc)
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
    cont_sliced = cont.cont_slice_keys({0: slc})
    assert "a" not in cont_sliced
    assert Container.cont_identical([cont_sliced.b, sub_cont])
    assert Container.cont_identical([cont_sliced.c, sub_cont])
    assert Container.cont_identical([cont_sliced.d, sub_cont])
    assert "e" not in cont_sliced

    # with dict, depth 1
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    sub_sub_cont = Container({"b": b_val, "c": c_val, "d": d_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.cont_slice_keys({1: slc})
    assert Container.cont_identical([cont_sliced.a, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.b, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.c, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.d, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.e, sub_sub_cont])

    # with dict, depth 0, 1
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    sub_sub_cont = Container({"b": b_val, "c": c_val, "d": d_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.cont_slice_keys({0: slc, 1: slc})
    assert "a" not in cont_sliced
    assert Container.cont_identical([cont_sliced.b, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.c, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.d, sub_sub_cont])
    assert "e" not in cont_sliced

    # all depths
    sub_cont = Container({"a": a_val, "b": b_val, "c": c_val, "d": d_val, "e": e_val})
    sub_sub_cont = Container({"b": b_val, "c": c_val, "d": d_val})
    cont = Container(
        {"a": sub_cont, "b": sub_cont, "c": sub_cont, "d": sub_cont, "e": sub_cont}
    )
    cont_sliced = cont.cont_slice_keys(slc, all_depths=True)
    assert "a" not in cont_sliced
    assert Container.cont_identical([cont_sliced.b, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.c, sub_sub_cont])
    assert Container.cont_identical([cont_sliced.d, sub_sub_cont])
    assert "e" not in cont_sliced


def test_container_show(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    cont = Container(dict_in)
    print(cont)
    cont.cont_show()


def test_container_find_sub_container(on_device):
    arr1 = ivy.array([1], device=on_device)
    arr2 = ivy.array([2], device=on_device)
    arr3 = ivy.array([3], device=on_device)
    dict_in = {"a": arr1, "b": {"c": arr2, "d": arr3}}
    top_cont = Container(dict_in)

    # full
    sub_cont = Container(dict_in["b"])
    assert sub_cont in top_cont
    found_kc = top_cont.cont_find_sub_container(sub_cont)
    assert found_kc == "b"
    found_kc = top_cont.cont_find_sub_container(top_cont)
    assert found_kc == ""

    # partial
    partial_sub_cont = Container({"d": arr3})
    found_kc = top_cont.cont_find_sub_container(partial_sub_cont, partial=True)
    assert found_kc == "b"
    assert partial_sub_cont.cont_find_sub_container(top_cont, partial=True) is False
    partial_sub_cont = Container({"b": {"d": arr3}})
    found_kc = top_cont.cont_find_sub_container(partial_sub_cont, partial=True)
    assert found_kc == ""
    assert partial_sub_cont.cont_find_sub_container(top_cont, partial=True) is False


def test_container_find_sub_structure(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    top_cont = Container(dict_in)

    # full
    sub_cont = Container(
        {"c": ivy.array([4], device=on_device), "d": ivy.array([5], device=on_device)}
    )
    assert not top_cont.cont_find_sub_container(sub_cont)
    found_kc = top_cont.cont_find_sub_structure(sub_cont)
    assert found_kc == "b"
    found_kc = top_cont.cont_find_sub_structure(top_cont)
    assert found_kc == ""

    # partial
    partial_sub_cont = Container({"d": ivy.array([5], device=on_device)})
    found_kc = top_cont.cont_find_sub_structure(partial_sub_cont, partial=True)
    assert found_kc == "b"
    partial_sub_cont = Container({"b": {"d": ivy.array([5], device=on_device)}})
    found_kc = top_cont.cont_find_sub_structure(partial_sub_cont, partial=True)
    assert found_kc == ""


def test_container_show_sub_container(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    top_cont = Container(dict_in)
    sub_cont = Container(dict_in["b"])
    top_cont.cont_show_sub_container("b")
    top_cont.cont_show_sub_container(sub_cont)


def test_container_from_dict_w_cont_types(on_device):
    # ToDo: add tests for backends other than jax
    if ivy.current_backend_str() == "jax":
        pytest.skip()
    from haiku._src.data_structures import FlatMapping

    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": FlatMapping(
            {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            }
        ),
    }
    container = Container(dict_in)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_from_kwargs(on_device):
    container = Container(
        a=ivy.array([1], device=on_device),
        b={
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_from_list(on_device):
    list_in = [
        ivy.array([1], device=on_device),
        [ivy.array([2], device=on_device), ivy.array([3], device=on_device)],
    ]
    container = Container(list_in, types_to_iteratively_nest=[list])
    assert np.allclose(ivy.to_numpy(container["it_0"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.it_0), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_0"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_0), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_1"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_1), np.array([3]))


def test_container_from_tuple(on_device):
    tuple_in = (
        ivy.array([1], device=on_device),
        (ivy.array([2], device=on_device), ivy.array([3], device=on_device)),
    )
    container = Container(tuple_in, types_to_iteratively_nest=[tuple])
    assert np.allclose(ivy.to_numpy(container["it_0"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.it_0), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_0"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_0), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["it_1"]["it_1"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.it_1.it_1), np.array([3]))


def test_container_to_raw(on_device):
    tuple_in = (
        ivy.array([1], device=on_device),
        (ivy.array([2], device=on_device), ivy.array([3], device=on_device)),
    )
    container = Container(tuple_in, types_to_iteratively_nest=[tuple])
    raw = container.cont_to_raw()
    assert np.allclose(ivy.to_numpy(raw[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(raw[1][0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(raw[1][1]), np.array([3]))


def test_container_as_bools(on_device):
    dict_in = {"a": ivy.array([1], device=on_device), "b": {"c": [], "d": True}}
    container = Container(dict_in)

    container_bools = container.cont_as_bools()
    assert container_bools["a"] is True
    assert container_bools.a is True
    assert container_bools["b"]["c"] is False
    assert container_bools.b.c is False
    assert container_bools["b"]["d"] is True
    assert container_bools.b.d is True


def test_container_all_true(on_device):
    assert not Container(
        {"a": ivy.array([1], device=on_device), "b": {"c": [], "d": True}}
    ).cont_all_true()
    assert Container(
        {"a": ivy.array([1], device=on_device), "b": {"c": [1], "d": True}}
    ).cont_all_true()
    # noinspection PyBroadException
    try:
        assert Container(
            {"a": ivy.array([1], device=on_device), "b": {"c": [1], "d": True}}
        ).cont_all_true(assert_is_bool=True)
        error_raised = False
    except IvyException:
        error_raised = True
    assert error_raised


def test_container_all_false(on_device):
    assert Container({"a": False, "b": {"c": [], "d": 0}}).cont_all_false()
    assert not Container({"a": False, "b": {"c": [1], "d": 0}}).cont_all_false()
    # noinspection PyBroadException
    try:
        assert Container(
            {"a": ivy.array([1], device=on_device), "b": {"c": [1], "d": True}}
        ).cont_all_false(assert_is_bool=True)
        error_raised = False
    except IvyException:
        error_raised = True
    assert error_raised


def test_container_unstack_conts(on_device):
    dict_in = {
        "a": ivy.array([[1], [2], [3]], device=on_device),
        "b": {
            "c": ivy.array([[2], [3], [4]], device=on_device),
            "d": ivy.array([[3], [4], [5]], device=on_device),
        },
    }
    container = Container(dict_in)

    # without key_chains specification
    container_unstacked = container.cont_unstack_conts(0)
    for cont, a, bc, bd in zip(container_unstacked, [1, 2, 3], [2, 3, 4], [3, 4, 5]):
        assert np.array_equal(ivy.to_numpy(cont["a"]), np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont.a), np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont["b"]["c"]), np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont.b.c), np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont["b"]["d"]), np.array([bd]))
        assert np.array_equal(ivy.to_numpy(cont.b.d), np.array([bd]))


def test_container_split_conts(on_device):
    dict_in = {
        "a": ivy.array([[1], [2], [3]], device=on_device),
        "b": {
            "c": ivy.array([[2], [3], [4]], device=on_device),
            "d": ivy.array([[3], [4], [5]], device=on_device),
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


def test_container_num_arrays(on_device):
    dict_in = {
        "a": ivy.array([[0.0, 1.0, 2.0, 3.0]], device=on_device),
        "b": {
            "c": ivy.array([[5.0, 10.0, 15.0, 20.0]], device=on_device),
            "d": ivy.array([[10.0, 9.0, 8.0, 7.0]], device=on_device),
        },
    }
    container = Container(dict_in)
    assert container.cont_num_arrays() == 3
    dict_in = {
        "a": ivy.array([[0.0, 1.0, 2.0, 3.0]], device=on_device),
        "b": {
            "c": _variable(ivy.array([[5.0, 10.0, 15.0, 20.0]], device=on_device)),
            "d": ivy.array([[10.0, 9.0, 8.0, 7.0]], device=on_device),
        },
    }
    container = Container(dict_in)
    assert (
        container.cont_num_arrays() == 3
        if ivy.current_backend_str() in ("numpy", "jax")
        else 2
    )


def test_container_size_ordered_arrays(on_device):
    dict_in = {
        "a": ivy.array([[0.0, 1.0, 2.0, 3.0]], device=on_device),
        "b": {
            "c": ivy.array([[5.0, 10.0]], device=on_device),
            "d": ivy.array([[10.0, 9.0, 8.0]], device=on_device),
        },
    }
    container = Container(dict_in)
    size_ordered = container.cont_size_ordered_arrays()
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


def test_container_has_key(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    assert container.cont_has_key("a")  # noqa
    assert container.cont_has_key("b")  # noqa
    assert container.cont_has_key("c")  # noqa
    assert container.cont_has_key("d")  # noqa
    assert not container.cont_has_key("e")  # noqa
    assert not container.cont_has_key("f")  # noqa


def test_container_has_key_chain(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    assert container.cont_has_key_chain("a")
    assert container.cont_has_key_chain("b")
    assert container.cont_has_key_chain("b/c")
    assert container.cont_has_key_chain("b/d")
    assert not container.cont_has_key_chain("b/e")
    assert not container.cont_has_key_chain("c")


def test_container_at_keys(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    new_container = container.cont_at_keys(["a", "c"])
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]
    new_container = container.cont_at_keys("c")
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]
    new_container = container.cont_at_keys(["b"])
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))


def test_container_at_key_chain(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)

    # explicit function call
    sub_container = container.cont_at_key_chain("b")
    assert np.allclose(ivy.to_numpy(sub_container["c"]), np.array([2]))
    sub_container = container.cont_at_key_chain("b/c")
    assert np.allclose(ivy.to_numpy(sub_container), np.array([2]))

    # overridden built-in function call
    sub_container = container["b"]
    assert np.allclose(ivy.to_numpy(sub_container["c"]), np.array([2]))
    sub_container = container["b/c"]
    assert np.allclose(ivy.to_numpy(sub_container), np.array([2]))


def test_container_at_key_chains(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    target_cont = Container({"a": True, "b": {"c": True}})
    new_container = container.cont_at_key_chains(target_cont)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]
    new_container = container.cont_at_key_chains(["b/c", "b/d"])
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))
    new_container = container.cont_at_key_chains("b/c")
    assert "a" not in new_container
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert "d" not in new_container["b"]


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_all_key_chains(include_empty, on_device):
    a_val = Container() if include_empty else ivy.array([1], device=on_device)
    bc_val = Container() if include_empty else ivy.array([2], device=on_device)
    bd_val = Container() if include_empty else ivy.array([3], device=on_device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)
    kcs = container.cont_all_key_chains(include_empty)
    assert kcs[0] == "a"
    assert kcs[1] == "b/c"
    assert kcs[2] == "b/d"


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_key_chains_containing(include_empty, on_device):
    a_val = Container() if include_empty else ivy.array([1], device=on_device)
    bc_val = Container() if include_empty else ivy.array([2], device=on_device)
    bd_val = Container() if include_empty else ivy.array([3], device=on_device)
    dict_in = {"a_sub": a_val, "b": {"c": bc_val, "d_sub": bd_val}}
    container = Container(dict_in)
    kcs = container.cont_key_chains_containing("sub", include_empty)
    assert kcs[0] == "a_sub"
    assert kcs[1] == "b/d_sub"


# noinspection PyUnresolvedReferences
def test_container_set_at_keys(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container_orig = Container(dict_in)

    # explicit function call
    orig_container = container_orig.cont_copy()
    container = orig_container.cont_set_at_keys({"b": ivy.array([4], device=on_device)})
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]), np.array([4]))
    assert not container.cont_has_key("c")  # noqa
    assert not container.cont_has_key("d")  # noqa
    container = orig_container.cont_set_at_keys(
        {"a": ivy.array([5], device=on_device), "c": ivy.array([6], device=on_device)}
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))


# noinspection PyUnresolvedReferences
def test_container_set_at_key_chain(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.cont_copy()
    container = container.cont_set_at_key_chain("b/e", ivy.array([4], device=on_device))
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    container = container.cont_set_at_key_chain("f", ivy.array([5], device=on_device))
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["f"]), np.array([5]))

    # overridden built-in function call
    container = container_orig.cont_copy()
    assert "b/e" not in container
    container["b/e"] = ivy.array([4], device=on_device)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    assert "f" not in container
    container["f"] = ivy.array([5], device=on_device)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container["b"]["e"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["f"]), np.array([5]))


# noinspection PyUnresolvedReferences
def test_container_overwrite_at_key_chain(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.cont_copy()
    # noinspection PyBroadException
    try:
        container.cont_overwrite_at_key_chain("b/e", ivy.array([4], device=on_device))
        exception_raised = False
    except Exception:
        exception_raised = True
    assert exception_raised
    container = container.cont_overwrite_at_key_chain(
        "b/d", ivy.array([4], device=on_device)
    )
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([4]))


def test_container_set_at_key_chains(on_device):
    container = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    target_container = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {"d": ivy.array([5], device=on_device)},
        }
    )
    new_container = container.cont_set_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([5]))
    target_container = Container({"b": {"c": ivy.array([7], device=on_device)}})
    new_container = container.cont_set_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([7]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))


def test_container_overwrite_at_key_chains(on_device):
    container = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    target_container = Container(
        {
            "a": ivy.array([4], device=on_device),
            "b": {"d": ivy.array([5], device=on_device)},
        }
    )
    new_container = container.cont_overwrite_at_key_chains(
        target_container, inplace=False
    )
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([5]))
    target_container = Container({"b": {"c": ivy.array([7], device=on_device)}})
    new_container = container.cont_overwrite_at_key_chains(
        target_container, inplace=False
    )
    assert np.allclose(ivy.to_numpy(new_container["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["c"]), np.array([7]))
    assert np.allclose(ivy.to_numpy(new_container["b"]["d"]), np.array([3]))
    # noinspection PyBroadException
    try:
        container.cont_overwrite_at_key_chains(
            Container({"b": {"e": ivy.array([5], device=on_device)}})
        )
        exception_raised = False
    except Exception:
        exception_raised = True
    assert exception_raised


def test_container_prune_keys(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    container_pruned = container.cont_prune_keys(["a", "c"])
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

    container_pruned = container.cont_prune_keys(["a", "d"])
    assert "a" not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["c"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.c), np.array([[2]]))
    assert "d" not in container_pruned["b"]
    assert _test_a_exception(container_pruned)
    assert _test_bd_exception(container_pruned)


def test_container_prune_key_chain(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {"c": ivy.array([2], device=on_device), "d": None},
    }
    container = Container(dict_in)
    container_pruned = container.cont_prune_key_chain("b/c")
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

    container_pruned = container.cont_prune_key_chain("b")
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


def test_container_prune_key_chains(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    container_pruned = container.cont_prune_key_chains(["a", "b/c"])
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

    container_pruned = container.cont_prune_key_chains(
        Container({"a": True, "b": {"c": True}})
    )
    assert "a" not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert "c" not in container_pruned["b"]
    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)


def test_container_format_key_chains(on_device):
    dict_in = {
        "_a": ivy.array([1], device=on_device),
        "b ": {
            "c": ivy.array([2], device=on_device),
            "d-": ivy.array([3], device=on_device),
        },
    }
    cont = Container(dict_in)
    cont_formatted = cont.cont_format_key_chains(
        lambda s: s.replace("_", "").replace(" ", "").replace("-", "")
    )
    assert np.allclose(ivy.to_numpy(cont_formatted["a"]), np.array([1]))
    assert np.allclose(ivy.to_numpy(cont_formatted.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(cont_formatted["b"]["c"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(cont_formatted.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(cont_formatted["b"]["d"]), np.array([3]))
    assert np.allclose(ivy.to_numpy(cont_formatted.b.d), np.array([3]))


def test_container_sort_by_key(on_device):
    dict_in = {
        "b": ivy.array([1], device=on_device),
        "a": {
            "d": ivy.array([2], device=on_device),
            "c": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    container_sorted = container.cont_sort_by_key()
    for k, k_true in zip(container_sorted.keys(), ["a", "b"]):
        assert k == k_true
    for k, k_true in zip(container_sorted.a.keys(), ["c", "d"]):
        assert k == k_true


def test_container_prune_empty(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {"c": {}, "d": ivy.array([3], device=on_device)},
    }
    container = Container(dict_in)
    container_pruned = container.cont_prune_empty()
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


def test_container_prune_key_from_key_chains(on_device):
    container = Container(
        {
            "Ayy": ivy.array([1], device=on_device),
            "Bee": {
                "Cee": ivy.array([2], device=on_device),
                "Dee": ivy.array([3], device=on_device),
            },
            "Beh": {
                "Ceh": ivy.array([4], device=on_device),
                "Deh": ivy.array([5], device=on_device),
            },
        }
    )

    # absolute
    container_pruned = container.cont_prune_key_from_key_chains("Bee")
    assert np.allclose(ivy.to_numpy(container_pruned["Ayy"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Cee"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned["Dee"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert "Bee" not in container_pruned

    # containing
    container_pruned = container.cont_prune_key_from_key_chains(containing="B")
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


def test_container_prune_keys_from_key_chains(on_device):
    container = Container(
        {
            "Ayy": ivy.array([1], device=on_device),
            "Bee": {
                "Cee": ivy.array([2], device=on_device),
                "Dee": ivy.array([3], device=on_device),
            },
            "Eee": {"Fff": ivy.array([4], device=on_device)},
        }
    )

    # absolute
    container_pruned = container.cont_prune_keys_from_key_chains(["Bee", "Eee"])
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
    container_pruned = container.cont_prune_keys_from_key_chains(containing=["B", "E"])
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


def test_container_restructure_key_chains(on_device):
    # single
    container = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_restructured = container.cont_restructure_key_chains({"a": "A"})
    assert np.allclose(ivy.to_numpy(container_restructured["A"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured.A), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured["b/c"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured.b.c), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured["b/d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_restructured.b.d), np.array([[3]]))

    # full
    container = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_restructured = container.cont_restructure_key_chains(
        {"a": "A", "b/c": "B/C", "b/d": "B/D"}
    )
    assert np.allclose(ivy.to_numpy(container_restructured["A"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured.A), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_restructured["B/C"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured.B.C), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_restructured["B/D"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_restructured.B.D), np.array([[3]]))


def test_container_restructure(on_device):
    container = Container(
        {
            "a": ivy.array([[1, 2], [3, 4]], device=on_device),
            "b": {
                "c": ivy.array([[2, 4], [6, 8]], device=on_device),
                "d": ivy.array([3, 6, 9, 12], device=on_device),
            },
        }
    )
    container_restructured = container.cont_restructure(
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


def test_container_flatten_key_chains(on_device):
    container = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": {"d": ivy.array([2], device=on_device)},
                "e": {"f": {"g": ivy.array([3], device=on_device)}},
            },
        }
    )

    # full
    container_flat = container.cont_flatten_key_chains()
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__c__d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__c__d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__e__f__g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__e__f__g), np.array([[3]]))

    # above height 1
    container_flat = container.cont_flatten_key_chains(above_height=1)
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__c"]["d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__c.d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b__e__f"]["g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b__e__f.g), np.array([[3]]))

    # below depth 1
    container_flat = container.cont_flatten_key_chains(below_depth=1)
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["c__d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.c__d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["e__f__g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.e__f__g), np.array([[3]]))

    # above height 1, below depth 1
    container_flat = container.cont_flatten_key_chains(above_height=1, below_depth=1)
    assert np.allclose(ivy.to_numpy(container_flat["a"]), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["c"]["d"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.c.d), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_flat["b"]["e__f"]["g"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_flat.b.e__f.g), np.array([[3]]))


def test_container_deep_copy(on_device):
    dict_in = {
        "a": ivy.array([0.0], device=on_device),
        "b": {
            "c": ivy.array([1.0], device=on_device),
            "d": ivy.array([2.0], device=on_device),
        },
    }
    cont = Container(dict_in)
    cont_deepcopy = cont.cont_deep_copy()
    assert np.allclose(ivy.to_numpy(cont.a), ivy.to_numpy(cont_deepcopy.a))
    assert np.allclose(ivy.to_numpy(cont.b.c), ivy.to_numpy(cont_deepcopy.b.c))
    assert np.allclose(ivy.to_numpy(cont.b.d), ivy.to_numpy(cont_deepcopy.b.d))
    assert id(cont.a) != id(cont_deepcopy.a)
    assert id(cont.b.c) != id(cont_deepcopy.b.c)
    assert id(cont.b.d) != id(cont_deepcopy.b.d)


def test_container_contains(on_device):
    arr0 = ivy.array([0.0], device=on_device)
    arr1 = ivy.array([1.0], device=on_device)
    arr2 = ivy.array([2.0], device=on_device)
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
    assert container.cont_contains_sub_container(container)
    assert container.cont_contains_sub_container(sub_cont)
    assert sub_cont in container

    # partial sub-container
    partial_sub_cont = Container({"b": {"d": arr2}})
    assert container.cont_contains_sub_container(container, partial=True)
    assert container.cont_contains_sub_container(partial_sub_cont, partial=True)
    assert not partial_sub_cont.cont_contains_sub_container(container, partial=True)

    # sub-structure
    sub_struc = Container(
        {
            "c": ivy.array([3.0], device=on_device),
            "d": ivy.array([4.0], device=on_device),
        }
    )
    assert not container.cont_contains_sub_container(sub_struc)
    assert sub_struc not in container
    assert container.cont_contains_sub_structure(sub_struc)
    assert container.cont_contains_sub_structure(container)

    # partial sub-structure
    partial_sub_struc = Container({"b": {"d": ivy.array([4.0], device=on_device)}})
    assert container.cont_contains_sub_structure(container, partial=True)
    assert container.cont_contains_sub_structure(partial_sub_struc, partial=True)
    assert not partial_sub_struc.cont_contains_sub_structure(container, partial=True)


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_to_iterator(include_empty, on_device):
    a_val = Container() if include_empty else ivy.array([1], device=on_device)
    bc_val = Container() if include_empty else ivy.array([2], device=on_device)
    bd_val = Container() if include_empty else ivy.array([3], device=on_device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.cont_to_iterator(include_empty=include_empty)
    for (key_chain, value), expected in zip(
        container_iterator, [("a", a_val), ("b/c", bc_val), ("b/d", bd_val)]
    ):
        expected_key_chain = expected[0]
        expected_value = expected[1]
        assert key_chain == expected_key_chain
        assert value is expected_value

    # with leaf keys
    container_iterator = container.cont_to_iterator(
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
def test_container_to_iterator_values(include_empty, on_device):
    a_val = Container() if include_empty else ivy.array([1], device=on_device)
    bc_val = Container() if include_empty else ivy.array([2], device=on_device)
    bd_val = Container() if include_empty else ivy.array([3], device=on_device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.cont_to_iterator_values(include_empty=include_empty)
    for value, expected_value in zip(container_iterator, [a_val, bc_val, bd_val]):
        assert value is expected_value


@pytest.mark.parametrize("include_empty", [True, False])
def test_container_to_iterator_keys(include_empty, on_device):
    a_val = Container() if include_empty else ivy.array([1], device=on_device)
    bc_val = Container() if include_empty else ivy.array([2], device=on_device)
    bd_val = Container() if include_empty else ivy.array([3], device=on_device)
    dict_in = {"a": a_val, "b": {"c": bc_val, "d": bd_val}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.cont_to_iterator_keys(include_empty=include_empty)
    for key_chain, expected_key_chain in zip(container_iterator, ["a", "b/c", "b/d"]):
        assert key_chain == expected_key_chain

    # with leaf keys
    container_iterator = container.cont_to_iterator_keys(
        leaf_keys_only=True, include_empty=include_empty
    )
    for key, expected_key in zip(container_iterator, ["a", "c", "d"]):
        assert key == expected_key


def test_container_to_flat_list(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    container_flat_list = container.cont_to_flat_list()
    for value, expected_value in zip(
        container_flat_list,
        [
            ivy.array([1], device=on_device),
            ivy.array([2], device=on_device),
            ivy.array([3], device=on_device),
        ],
    ):
        assert value == expected_value


def test_container_from_flat_list(on_device):
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container = Container(dict_in)
    flat_list = [4, 5, 6]
    container = container.cont_from_flat_list(flat_list)
    assert np.allclose(ivy.to_numpy(container["a"]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container["b"]["c"]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container["b"]["d"]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


@pytest.mark.parametrize("inplace", [True, False])
def test_container_map(inplace, on_device):
    # without key_chains specification
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
    }
    container_orig = Container(dict_in)
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map(lambda x, _: x + 1, inplace=inplace)
    if inplace:
        container_iterator = container.cont_to_iterator()
    else:
        container_iterator = container_mapped.cont_to_iterator()
    for (key, value), expected_value in zip(
        container_iterator,
        [
            ivy.array([2], device=on_device),
            ivy.array([3], device=on_device),
            ivy.array([4], device=on_device),
        ],
    ):
        assert ivy.to_numpy(value) == ivy.to_numpy(expected_value)

    # with key_chains to apply
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map(
        lambda x, _: x + 1, ["a", "b/c"], inplace=inplace
    )
    if inplace:
        container_mapped = container
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["c"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"]["d"]), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[3]]))

    # with key_chains to apply pruned
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map(
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
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map(
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
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map(
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
            "a": ivy.array([1], device=on_device),
            "b": [ivy.array([2], device=on_device), ivy.array([3], device=on_device)],
        }
    )
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map(
        lambda x, _: x + 1, inplace=inplace, map_sequences=True
    )
    if inplace:
        container_mapped = container
    assert np.allclose(ivy.to_numpy(container_mapped["a"]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"][0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_mapped["b"][1]), np.array([4]))


@pytest.mark.parametrize("inplace", [True, False])
def test_container_map_sub_conts(inplace, on_device):
    # without key_chains specification
    container_orig = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )

    def _add_e_attr(cont_in):
        cont_in.e = ivy.array([4], device=on_device)
        return cont_in

    # with self
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map_sub_conts(
        lambda c, _: _add_e_attr(c), inplace=inplace
    )
    if inplace:
        container_mapped = container
    assert "e" in container_mapped
    assert np.array_equal(ivy.to_numpy(container_mapped.e), np.array([4]))
    assert "e" in container_mapped.b
    assert np.array_equal(ivy.to_numpy(container_mapped.b.e), np.array([4]))

    # without self
    container = container_orig.cont_deep_copy()
    container_mapped = container.cont_map_sub_conts(
        lambda c, _: _add_e_attr(c), include_self=False, inplace=inplace
    )
    if inplace:
        container_mapped = container
    assert "e" not in container_mapped
    assert "e" in container_mapped.b
    assert np.array_equal(ivy.to_numpy(container_mapped.b.e), np.array([4]))


def test_container_multi_map(on_device):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
            },
        }
    )

    # with key_chains to apply
    container_mapped = ivy.Container.cont_multi_map(
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
            "a": ivy.array([1], device=on_device),
            "b": [
                ivy.array([2], device=on_device),
                ivy.array([3], device=on_device),
            ],
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": [
                ivy.array([4], device=on_device),
                ivy.array([5], device=on_device),
            ],
        }
    )

    container_mapped = ivy.Container.cont_multi_map(
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
    container_mapped = ivy.Container.cont_multi_map(lambda xs, _: xs[0] / xs[1], [a, b])

    assert np.allclose(ivy.to_numpy(container_mapped["a"].b), 1)
    assert np.allclose(ivy.to_numpy(container_mapped["a"]["c"]), 2)
    assert np.allclose(ivy.to_numpy(container_mapped.d.e), 2)
    assert np.allclose(ivy.to_numpy(container_mapped["d"].f), 3)


def test_container_common_key_chains(on_device):
    arr1 = ivy.array([1], device=on_device)
    arr2 = ivy.array([2], device=on_device)
    arr3 = ivy.array([3], device=on_device)
    cont0 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    cont1 = Container({"b": {"c": arr2, "d": arr3, "e": arr1}})
    cont2 = Container({"a": arr1, "b": {"d": arr3, "e": arr1}})

    # 0
    common_kcs = Container.cont_common_key_chains([cont0])
    assert len(common_kcs) == 3
    assert "a" in common_kcs
    assert "b/c" in common_kcs
    assert "b/d" in common_kcs

    # 0-1
    common_kcs = Container.cont_common_key_chains([cont0, cont1])
    assert len(common_kcs) == 2
    assert "b/c" in common_kcs
    assert "b/d" in common_kcs

    # 0-2
    common_kcs = Container.cont_common_key_chains([cont0, cont2])
    assert len(common_kcs) == 2
    assert "a" in common_kcs
    assert "b/d" in common_kcs

    # 1-2
    common_kcs = Container.cont_common_key_chains([cont1, cont2])
    assert len(common_kcs) == 2
    assert "b/d" in common_kcs
    assert "b/e" in common_kcs

    # all
    common_kcs = Container.cont_common_key_chains([cont0, cont1, cont2])
    assert len(common_kcs) == 1
    assert "b/d" in common_kcs


def test_container_identical(on_device):
    # without key_chains specification
    arr1 = ivy.array([1], device=on_device)
    arr2 = ivy.array([2], device=on_device)
    arr3 = ivy.array([3], device=on_device)
    container0 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    container1 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    container2 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container3 = Container({"b": {"d": arr3}})
    container4 = Container({"d": arr3})

    # the same
    assert ivy.Container.cont_identical([container0, container1])
    assert ivy.Container.cont_identical([container1, container0])

    # not the same
    assert not ivy.Container.cont_identical([container0, container2])
    assert not ivy.Container.cont_identical([container2, container0])
    assert not ivy.Container.cont_identical([container1, container2])
    assert not ivy.Container.cont_identical([container2, container1])

    # partial
    assert ivy.Container.cont_identical([container0, container3], partial=True)
    assert ivy.Container.cont_identical([container3, container0], partial=True)
    assert not ivy.Container.cont_identical([container0, container4], partial=True)
    assert not ivy.Container.cont_identical([container4, container0], partial=True)


def test_container_identical_structure(on_device):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
            },
        }
    )
    container2 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
                "e": ivy.array([6], device=on_device),
            },
        }
    )
    container3 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
            },
            "e": ivy.array([6], device=on_device),
        }
    )
    container4 = Container({"b": {"d": ivy.array([4], device=on_device)}})
    container5 = Container({"d": ivy.array([4], device=on_device)})

    # with identical
    assert ivy.Container.cont_identical_structure([container0, container1])
    assert ivy.Container.cont_identical_structure([container1, container0])
    assert ivy.Container.cont_identical_structure([container1, container0, container1])

    # without identical
    assert not ivy.Container.cont_identical_structure([container2, container3])
    assert not ivy.Container.cont_identical_structure([container0, container3])
    assert not ivy.Container.cont_identical_structure([container1, container2])
    assert not ivy.Container.cont_identical_structure(
        [container1, container0, container2]
    )

    # partial
    assert ivy.Container.cont_identical_structure(
        [container0, container4], partial=True
    )
    assert ivy.Container.cont_identical_structure(
        [container1, container4], partial=True
    )
    assert ivy.Container.cont_identical_structure(
        [container2, container4], partial=True
    )
    assert ivy.Container.cont_identical_structure(
        [container3, container4], partial=True
    )
    assert ivy.Container.cont_identical_structure(
        [container4, container4], partial=True
    )
    assert not ivy.Container.cont_identical_structure(
        [container0, container5], partial=True
    )
    assert not ivy.Container.cont_identical_structure(
        [container1, container5], partial=True
    )
    assert not ivy.Container.cont_identical_structure(
        [container2, container5], partial=True
    )
    assert not ivy.Container.cont_identical_structure(
        [container3, container5], partial=True
    )
    assert not ivy.Container.cont_identical_structure(
        [container4, container5], partial=True
    )


def test_container_identical_configs(on_device):
    container0 = Container({"a": ivy.array([1], device=on_device)}, print_limit=5)
    container1 = Container({"a": ivy.array([1], device=on_device)}, print_limit=5)
    container2 = Container({"a": ivy.array([1], device=on_device)}, print_limit=10)

    # with identical
    assert ivy.Container.cont_identical_configs([container0, container1])
    assert ivy.Container.cont_identical_configs([container1, container0])
    assert ivy.Container.cont_identical_configs([container1, container0, container1])

    # without identical
    assert not ivy.Container.cont_identical_configs([container1, container2])
    assert not ivy.Container.cont_identical_configs(
        [container1, container0, container2]
    )


def test_container_identical_array_shapes(on_device):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1, 2], device=on_device),
            "b": {
                "c": ivy.array([2, 3, 4], device=on_device),
                "d": ivy.array([3, 4, 5, 6], device=on_device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([1, 2, 3, 4], device=on_device),
            "b": {
                "c": ivy.array([3, 4], device=on_device),
                "d": ivy.array([3, 4, 5], device=on_device),
            },
        }
    )
    container2 = Container(
        {
            "a": ivy.array([1, 2, 3, 4], device=on_device),
            "b": {
                "c": ivy.array([3, 4], device=on_device),
                "d": ivy.array([3, 4, 5, 6], device=on_device),
            },
        }
    )

    # with identical
    assert ivy.Container.cont_identical_array_shapes([container0, container1])
    assert ivy.Container.cont_identical_array_shapes([container1, container0])
    assert ivy.Container.cont_identical_array_shapes(
        [container1, container0, container1]
    )
    assert not ivy.Container.cont_identical([container0, container2])
    assert not ivy.Container.cont_identical([container1, container2])
    assert not ivy.Container.cont_identical([container0, container1, container2])


def test_container_with_entries_as_lists(on_device):
    if ivy.current_backend_str() == "tensorflow":
        # to_list() requires eager execution
        pytest.skip()
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {"c": ivy.array([2.0], device=on_device), "d": "some string"},
    }
    container = Container(dict_in)
    container_w_list_entries = container.cont_with_entries_as_lists()
    for (key, value), expected_value in zip(
        container_w_list_entries.cont_to_iterator(), [[1], [2.0], "some string"]
    ):
        assert value == expected_value


def test_container_reshape_like(on_device):
    container = Container(
        {
            "a": ivy.array([[1.0]], device=on_device),
            "b": {
                "c": ivy.array([[3.0], [4.0]], device=on_device),
                "d": ivy.array([[5.0], [6.0], [7.0]], device=on_device),
            },
        }
    )
    new_shapes = Container({"a": (1,), "b": {"c": (1, 2, 1), "d": (3, 1, 1)}})

    # without leading shape
    container_reshaped = container.cont_reshape_like(new_shapes)
    assert list(container_reshaped["a"].shape) == [1]
    assert list(container_reshaped.a.shape) == [1]
    assert list(container_reshaped["b"]["c"].shape) == [1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [1, 2, 1]
    assert list(container_reshaped["b"]["d"].shape) == [3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 1, 1]

    # with leading shape
    container = Container(
        {
            "a": ivy.array([[[1.0]], [[1.0]], [[1.0]]], device=on_device),
            "b": {
                "c": ivy.array(
                    [[[3.0], [4.0]], [[3.0], [4.0]], [[3.0], [4.0]]], device=on_device
                ),
                "d": ivy.array(
                    [
                        [[5.0], [6.0], [7.0]],
                        [[5.0], [6.0], [7.0]],
                        [[5.0], [6.0], [7.0]],
                    ],
                    device=on_device,
                ),
            },
        }
    )
    container_reshaped = container.cont_reshape_like(new_shapes, leading_shape=[3])
    assert list(container_reshaped["a"].shape) == [3, 1]
    assert list(container_reshaped.a.shape) == [3, 1]
    assert list(container_reshaped["b"]["c"].shape) == [3, 1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [3, 1, 2, 1]
    assert list(container_reshaped["b"]["d"].shape) == [3, 3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 3, 1, 1]


def test_container_slice(on_device):
    dict_in = {
        "a": ivy.array([[0.0], [1.0]], device=on_device),
        "b": {
            "c": ivy.array([[1.0], [2.0]], device=on_device),
            "d": ivy.array([[2.0], [3.0]], device=on_device),
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


def test_container_slice_via_key(on_device):
    dict_in = {
        "a": {
            "x": ivy.array([0.0], device=on_device),
            "y": ivy.array([1.0], device=on_device),
        },
        "b": {
            "c": {
                "x": ivy.array([1.0], device=on_device),
                "y": ivy.array([2.0], device=on_device),
            },
            "d": {
                "x": ivy.array([2.0], device=on_device),
                "y": ivy.array([3.0], device=on_device),
            },
        },
    }
    container = Container(dict_in)
    containerx = container.cont_slice_via_key("x")
    containery = container.cont_slice_via_key("y")
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


def test_container_to_and_from_disk_as_hdf5(on_device):
    if ivy.current_backend_str() == "tensorflow":
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = "container_on_disk.hdf5"
    dict_in_1 = {
        "a": ivy.array([np.float32(1.0)], device=on_device),
        "b": {
            "c": ivy.array([np.float32(2.0)], device=on_device),
            "d": ivy.array([np.float32(3.0)], device=on_device),
        },
    }
    container1 = Container(dict_in_1)
    dict_in_2 = {
        "a": ivy.array([np.float32(1.0), np.float32(1.0)], device=on_device),
        "b": {
            "c": ivy.array([np.float32(2.0), np.float32(2.0)], device=on_device),
            "d": ivy.array([np.float32(3.0), np.float32(3.0)], device=on_device),
        },
    }
    container2 = Container(dict_in_2)

    # saving
    container1.cont_to_disk_as_hdf5(save_filepath, max_batch_size=2)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.cont_from_disk_as_hdf5(save_filepath, slice(1))
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container1.a))
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container1.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container1.b.d)
    )

    # appending
    container1.cont_to_disk_as_hdf5(save_filepath, max_batch_size=2, starting_index=1)
    assert os.path.exists(save_filepath)

    # loading after append
    loaded_container = Container.cont_from_disk_as_hdf5(save_filepath)
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container2.a))
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container2.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container2.b.d)
    )

    # load slice
    loaded_sliced_container = Container.cont_from_disk_as_hdf5(
        save_filepath, slice(1, 2)
    )
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


def test_container_to_disk_shuffle_and_from_disk_as_hdf5(on_device):
    if ivy.current_backend_str() == "tensorflow":
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = "container_on_disk.hdf5"
    dict_in = {
        "a": ivy.array([1, 2, 3], device=on_device),
        "b": {
            "c": ivy.array([1, 2, 3], device=on_device),
            "d": ivy.array([1, 2, 3], device=on_device),
        },
    }
    container = Container(dict_in)

    # saving
    container.cont_to_disk_as_hdf5(save_filepath, max_batch_size=3)
    assert os.path.exists(save_filepath)

    # shuffling
    Container.shuffle_h5_file(save_filepath)

    # loading
    container_shuffled = Container.cont_from_disk_as_hdf5(save_filepath, slice(3))

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


def test_container_pickle(on_device):
    dict_in = {
        "a": ivy.array([np.float32(1.0)], device=on_device),
        "b": {
            "c": ivy.array([np.float32(2.0)], device=on_device),
            "d": ivy.array([np.float32(3.0)], device=on_device),
        },
    }

    # without module attribute
    cont = Container(dict_in)
    assert cont._local_ivy is None
    pickled = pickle.dumps(cont)
    cont_again = pickle.loads(pickled)
    assert cont_again._local_ivy is None
    ivy.Container.cont_identical_structure([cont, cont_again])
    ivy.Container.cont_identical_configs([cont, cont_again])

    # with module attribute
    cont = Container(dict_in, ivyh=ivy)
    assert cont._local_ivy is ivy
    pickled = pickle.dumps(cont)
    cont_again = pickle.loads(pickled)
    # noinspection PyUnresolvedReferences
    assert cont_again._local_ivy.current_backend_str() is ivy.current_backend_str()
    ivy.Container.cont_identical_structure([cont, cont_again])
    ivy.Container.cont_identical_configs([cont, cont_again])


def test_container_to_and_from_disk_as_pickled(on_device):
    save_filepath = "container_on_disk.pickled"
    dict_in = {
        "a": ivy.array([np.float32(1.0)], device=on_device),
        "b": {
            "c": ivy.array([np.float32(2.0)], device=on_device),
            "d": ivy.array([np.float32(3.0)], device=on_device),
        },
    }
    container = Container(dict_in)

    # saving
    container.cont_to_disk_as_pickled(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.cont_from_disk_as_pickled(save_filepath)
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container.a))
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container.b.c)
    )
    assert np.array_equal(
        ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container.b.d)
    )

    os.remove(save_filepath)


def test_container_to_and_from_disk_as_json(on_device):
    save_filepath = "container_on_disk.json"
    dict_in = {
        "a": 1.274e-7,
        "b": {"c": True, "d": ivy.array([np.float32(3.0)], device=on_device)},
    }
    container = Container(dict_in)

    # saving
    container.cont_to_disk_as_json(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.cont_from_disk_as_json(save_filepath)
    assert np.array_equal(loaded_container.a, container.a)
    assert np.array_equal(loaded_container.b.c, container.b.c)
    assert isinstance(loaded_container.b.d, str)

    os.remove(save_filepath)


def test_container_shapes(on_device):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=on_device),
        "b": {
            "c": ivy.array([[[2.0], [4.0]]], device=on_device),
            "d": ivy.array([[9.0]], device=on_device),
        },
    }
    container_shapes = Container(dict_in).cont_shapes
    assert list(container_shapes["a"]) == [1, 3, 1]
    assert list(container_shapes.a) == [1, 3, 1]
    assert list(container_shapes["b"]["c"]) == [1, 2, 1]
    assert list(container_shapes.b.c) == [1, 2, 1]
    assert list(container_shapes["b"]["d"]) == [1, 1]
    assert list(container_shapes.b.d) == [1, 1]


def test_container_dev_str(on_device):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=on_device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=on_device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=on_device),
        },
    }
    container = Container(dict_in)
    assert container.cont_dev_str == on_device


def test_container_create_if_absent(on_device):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=on_device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=on_device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=on_device),
        },
    }

    # depth 1
    container = Container(dict_in)
    container.cont_create_if_absent("a", None, True)
    assert np.allclose(ivy.to_numpy(container.a), np.array([[[1.0], [2.0], [3.0]]]))
    container.cont_create_if_absent("e", ivy.array([[[4.0], [8.0], [12.0]]]), True)
    assert np.allclose(ivy.to_numpy(container.e), np.array([[[4.0], [8.0], [12.0]]]))

    # depth 2
    container.cont_create_if_absent("f/g", np.array([[[5.0], [10.0], [15.0]]]), True)
    assert np.allclose(ivy.to_numpy(container.f.g), np.array([[[5.0], [10.0], [15.0]]]))


def test_container_if_exists(on_device):
    dict_in = {
        "a": ivy.array([[[1.0], [2.0], [3.0]]], device=on_device),
        "b": {
            "c": ivy.array([[[2.0], [4.0], [6.0]]], device=on_device),
            "d": ivy.array([[[3.0], [6.0], [9.0]]], device=on_device),
        },
    }
    container = Container(dict_in)
    assert np.allclose(
        ivy.to_numpy(container.cont_if_exists("a")), np.array([[[1.0], [2.0], [3.0]]])
    )
    assert "c" not in container
    assert container.cont_if_exists("c") is None
    container["c"] = ivy.array([[[1.0], [2.0], [3.0]]], device=on_device)
    assert np.allclose(
        ivy.to_numpy(container.cont_if_exists("c")), np.array([[[1.0], [2.0], [3.0]]])
    )
    assert container.cont_if_exists("d") is None
    container.d = ivy.array([[[1.0], [2.0], [3.0]]], device=on_device)
    assert np.allclose(
        ivy.to_numpy(container.cont_if_exists("d")), np.array([[[1.0], [2.0], [3.0]]])
    )


def test_jax_pytree_compatibility(on_device):
    if ivy.current_backend_str() != "jax":
        pytest.skip()

    # import
    from jax.tree_util import tree_flatten

    # dict in
    dict_in = {
        "a": ivy.array([1], device=on_device),
        "b": {
            "c": ivy.array([2], device=on_device),
            "d": ivy.array([3], device=on_device),
        },
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


def test_container_from_queues(on_device):
    if "gpu" in on_device:
        # Cannot re-initialize CUDA in forked subprocess. 'spawn'
        # start method must be used.
        pytest.skip()

    if ivy.gpu_is_available() and ivy.current_backend_str() == "jax":
        # Not found a way to set default on_device for JAX, and this causes
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
                        ivy.to_native(ivy.array([1.0, 2.0, 3.0], device=on_device))
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


def test_container_reduce(on_device):
    container_a = ivy.Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container_b = ivy.Container(
        {
            "a": ivy.array([2], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([6], device=on_device),
            },
        }
    )
    res = ivy.Container.cont_reduce([container_a, container_b], lambda x: x[0] + x[1])
    assert np.allclose(ivy.to_numpy(res.a), np.array([3.0]))
    assert np.allclose(ivy.to_numpy(res.b.c), np.array([6]))
    assert np.allclose(ivy.to_numpy(res.b.d), np.array([9]))


def test_container_assert_identical(on_device):
    # without key_chains specification
    arr1 = ivy.array([1], device=on_device)
    arr2 = ivy.array([2], device=on_device)
    arr3 = ivy.array([3], device=on_device)
    container0 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    container1 = Container({"a": arr1, "b": {"c": arr2, "d": arr3}})
    container2 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container3 = Container({"b": {"d": arr3}})
    container4 = Container({"d": arr3})

    # the same
    ivy.Container.cont_assert_identical([container0, container1])
    ivy.Container.cont_assert_identical([container1, container0])

    # not the same
    try:
        ivy.Container.cont_assert_identical([container0, container2])
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught
    try:
        ivy.Container.cont_assert_identical([container1, container2])
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught

    # partial
    ivy.Container.cont_assert_identical([container0, container3], partial=True)
    ivy.Container.cont_assert_identical([container3, container0], partial=True)
    try:
        ivy.Container.cont_assert_identical([container4, container0], partial=True)
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught


def test_container_assert_identical_structure(on_device):
    # without key_chains specification
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([2], device=on_device),
                "d": ivy.array([3], device=on_device),
            },
        }
    )
    container1 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
            },
        }
    )
    container2 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
                "e": ivy.array([6], device=on_device),
            },
        }
    )
    container3 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
            },
            "e": ivy.array([6], device=on_device),
        }
    )
    container4 = Container({"b": {"d": ivy.array([4], device=on_device)}})
    container5 = Container({"d": ivy.array([4], device=on_device)})

    # with identical
    ivy.Container.cont_assert_identical_structure([container0, container1])
    ivy.Container.cont_assert_identical_structure([container1, container0])
    ivy.Container.cont_assert_identical_structure([container1, container0, container1])

    # without identical
    try:
        ivy.Container.cont_assert_identical_structure(
            [container0, container1, container2, container3]
        )
        error_caught = False
    except IvyException:
        error_caught = True
    # partial
    try:
        ivy.Container.cont_assert_identical_structure(
            [container0, container1, container2, container3, container4, container5],
            partial=True,
        )
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught
    try:
        ivy.Container.cont_assert_identical_structure(
            [container0, container5], partial=True
        )
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught


def test_container_duplicate_array_keychains(on_device):
    arr1 = ivy.array([1], device=on_device)
    arr2 = ivy.array([2], device=on_device)
    container0 = Container({"a": arr1, "b": {"c": arr1, "d": arr2}})
    container1 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([1], device=on_device),
                "d": ivy.array([2], device=on_device),
            },
        }
    )
    res = ivy.Container.cont_duplicate_array_keychains(container0)
    assert res == (("a", "b/c"),)
    res = ivy.Container.cont_duplicate_array_keychains(container1)
    assert res == ()


def test_container_cont_inplace_update(on_device):
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([1], device=on_device),
                "d": ivy.array([2], device=on_device),
            },
        }
    )
    id0 = id(container0)
    container1 = Container(
        {
            "a": ivy.array([0], device=on_device),
            "b": {
                "c": ivy.array([0], device=on_device),
                "d": ivy.array([0], device=on_device),
            },
        }
    )
    id1 = id(container1)
    assert ivy.Container.cont_all_false(container0.all_equal(container1))
    container0.inplace_update(container1)
    assert id0 == id(container0)
    assert id1 == id(container1)
    assert ivy.Container.cont_all_true(container0.all_equal(container1))


def test_container_to_nested_list(on_device):
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([True], device=on_device),
                "d": {
                    "g": ivy.array([2.0], device=on_device),
                    "h": ivy.array([3], device=on_device),
                },
            },
        }
    )
    res = ivy.Container.cont_to_nested_list(container0)
    assert res == [1, [True, [2.0, 3]]]


def test_container_to_dict(on_device):
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([True], device=on_device),
                "d": {
                    "g": ivy.array([2.0], device=on_device),
                    "h": ivy.array([3], device=on_device),
                },
            },
        }
    )
    res = ivy.Container.cont_to_dict(container0)
    assert res == {"a": 1, "b": {"c": True, "d": {"g": 2.0, "h": 3}}}


def test_container_assert_contains(on_device):
    arr0 = ivy.array([0.0], device=on_device)
    arr1 = ivy.array([1.0], device=on_device)
    arr2 = ivy.array([2.0], device=on_device)
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
    container.cont_assert_contains_sub_container(container)
    container.cont_assert_contains_sub_container(sub_cont)
    assert sub_cont in container

    # partial sub-container
    partial_sub_cont = Container({"b": {"d": arr2}})
    container.cont_assert_contains_sub_container(container, partial=True)
    container.cont_assert_contains_sub_container(partial_sub_cont, partial=True)
    try:
        partial_sub_cont.cont_assert_contains_sub_container(container, partial=True)
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught
    # sub-structure
    sub_struc = Container(
        {
            "c": ivy.array([3.0], device=on_device),
            "d": ivy.array([4.0], device=on_device),
        }
    )
    try:
        not container.cont_assert_contains_sub_container(sub_struc)
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught
    assert sub_struc not in container
    container.cont_assert_contains_sub_structure(sub_struc)
    container.cont_assert_contains_sub_structure(container)

    # partial sub-structure
    partial_sub_struc = Container({"b": {"d": ivy.array([4.0], device=on_device)}})
    container.cont_assert_contains_sub_structure(container, partial=True)
    container.cont_assert_contains_sub_structure(partial_sub_struc, partial=True)
    try:
        partial_sub_struc.cont_assert_contains_sub_structure(container, partial=True)
        error_caught = False
    except IvyException:
        error_caught = True
    assert error_caught


def test_container_copy(on_device):
    dict_in = {
        "a": ivy.array([0.0], device=on_device),
        "b": {
            "c": ivy.array([1.0], device=on_device),
            "d": ivy.array([2.0], device=on_device),
        },
    }
    cont = Container(dict_in)
    cont_deepcopy = cont.cont_copy()
    assert np.allclose(ivy.to_numpy(cont.a), ivy.to_numpy(cont_deepcopy.a))
    assert np.allclose(ivy.to_numpy(cont.b.c), ivy.to_numpy(cont_deepcopy.b.c))
    assert np.allclose(ivy.to_numpy(cont.b.d), ivy.to_numpy(cont_deepcopy.b.d))
    assert id(cont) != id(cont_deepcopy)
    assert id(cont.a) == id(cont_deepcopy.a)
    assert id(cont.b.c) == id(cont_deepcopy.b.c)
    assert id(cont.b.d) == id(cont_deepcopy.b.d)


def test_container_try_kc(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    assert cont.cont_try_kc("a") == cont.a
    assert cont.cont_try_kc("b/c") == cont.b.c
    assert cont.cont_try_kc("b/d") == cont.b.d
    assert cont.cont_try_kc("b/e") is cont


def test_container_with_print_limit(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    default_print_limit = cont._print_limit
    id_cont = id(cont)
    cont1 = cont.cont_with_print_limit(default_print_limit + 5)
    assert cont1._print_limit == default_print_limit + 5
    assert id(cont1) != id(cont)
    assert cont._print_limit == default_print_limit
    assert cont._print_limit != cont1._print_limit
    cont.cont_with_print_limit(default_print_limit + 5, inplace=True)
    assert cont._print_limit == default_print_limit + 5
    assert cont.b._print_limit == default_print_limit + 5
    assert id(cont) == id_cont


def test_container_remove_print_limit(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    default_print_limit = cont._print_limit
    id_cont = id(cont)
    cont1 = cont.cont_remove_print_limit()
    assert cont1._print_limit is None
    assert id(cont1) != id(cont)
    assert cont._print_limit == default_print_limit
    assert cont._print_limit != cont1._print_limit
    assert cont.b._print_limit == default_print_limit
    cont.cont_remove_print_limit(inplace=True)
    assert cont._print_limit is None
    assert cont.b._print_limit is None
    assert id(cont) == id_cont


def test_container_with_key_length_limit(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    default_key_length_limit = cont._key_length_limit
    id_cont = id(cont)
    cont1 = cont.cont_with_key_length_limit(5)
    assert cont1._key_length_limit == 5
    assert id(cont1) != id(cont)
    assert cont._key_length_limit == default_key_length_limit
    assert cont.b._key_length_limit == default_key_length_limit
    assert cont._key_length_limit != cont1._key_length_limit
    cont.cont_with_key_length_limit(5, inplace=True)
    assert cont._key_length_limit == 5
    assert cont.b._key_length_limit == 5
    assert id(cont) == id_cont


def test_container_remove_key_length_limit(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    cont.cont_with_key_length_limit(5, inplace=True)
    default_key_length_limit = cont._key_length_limit
    id_cont = id(cont)
    cont1 = cont.cont_remove_key_length_limit()
    assert cont1._key_length_limit is None
    assert id(cont1) != id(cont)
    assert cont._key_length_limit == default_key_length_limit
    assert cont.b._key_length_limit == default_key_length_limit
    assert cont._key_length_limit != cont1._key_length_limit
    cont.cont_remove_key_length_limit(inplace=True)
    assert cont._key_length_limit is None
    assert cont.b._key_length_limit is None
    assert id(cont) == id_cont


def test_container_with_print_indent(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    default_print_indent = cont._print_indent
    id_cont = id(cont)
    cont1 = cont.cont_with_print_indent(default_print_indent + 5)
    assert cont1._print_indent == default_print_indent + 5
    assert id(cont1) != id(cont)
    assert cont._print_indent == default_print_indent
    assert cont.b._print_indent == default_print_indent
    assert cont._print_indent != cont1._print_indent
    cont.cont_with_print_indent(default_print_indent + 5, inplace=True)
    assert cont._print_indent == default_print_indent + 5
    assert cont.b._print_indent == default_print_indent + 5
    assert id(cont) == id_cont


def test_container_with_print_line_spacing(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    default_print_line_spacing = cont._print_line_spacing
    id_cont = id(cont)
    cont1 = cont.cont_with_print_line_spacing(default_print_line_spacing + 5)
    assert cont1._print_line_spacing == default_print_line_spacing + 5
    assert id(cont1) != id(cont)
    assert cont._print_line_spacing == default_print_line_spacing
    assert cont.b._print_line_spacing == default_print_line_spacing
    assert cont._print_line_spacing != cont1._print_line_spacing
    cont.cont_with_print_line_spacing(default_print_line_spacing + 5, inplace=True)
    assert cont._print_line_spacing == default_print_line_spacing + 5
    assert cont.b._print_line_spacing == default_print_line_spacing + 5
    assert id(cont) == id_cont


def test_container_with_default_key_color(on_device):
    cont = Container(
        {
            "a": ivy.array([0.0], device=on_device),
            "b": {
                "c": ivy.array([1.0], device=on_device),
                "d": ivy.array([2.0], device=on_device),
            },
        }
    )
    default_default_key_color = cont._default_key_color
    id_cont = id(cont)
    cont1 = cont.cont_with_default_key_color("red")
    assert cont1._default_key_color == "red"
    assert id(cont1) != id(cont)
    assert cont._default_key_color == default_default_key_color
    assert cont.b._default_key_color == default_default_key_color
    assert cont._default_key_color != cont1._default_key_color
    cont.cont_with_default_key_color("red", inplace=True)
    assert cont._default_key_color == "red"
    assert cont.b._default_key_color == "red"
    assert id(cont) == id_cont


def test_container_with_ivy_backend(on_device):
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([1], device=on_device),
                "d": ivy.array([2], device=on_device),
            },
        }
    )
    id_container0 = id(container0)
    container0 = ivy.Container.cont_with_ivy_backend(container0, "numpy")
    assert container0.cont_config["ivyh"] == "numpy"
    assert id_container0 != id(container0)
    id_container0 = id(container0)
    ivy.Container.cont_with_ivy_backend(container0, "torch", inplace=True)
    assert container0.cont_config["ivyh"] == "torch"
    assert id(container0) == id_container0


def test_container_trim_key(on_device):
    key = "abcdefg"
    max_length = 3
    trimmed_key = ivy.Container.cont_trim_key(key, max_length)
    assert trimmed_key == "adg"


def test_container_inplace(on_device):
    container0 = Container(
        {
            "a": ivy.array([1], device=on_device),
            "b": {
                "c": ivy.array([1], device=on_device),
                "d": ivy.array([2], device=on_device),
            },
        }
    )
    const = 3
    arr = ivy.array([1], device=on_device)
    container1 = Container(
        {
            "a": ivy.array([3], device=on_device),
            "b": {
                "c": ivy.array([4], device=on_device),
                "d": ivy.array([5], device=on_device),
            },
        }
    )

    special_funcs = [
        "__add__",
        "__and__",
        "__floordiv__",
        "__lshift__",
        "__matmul__",
        "__mod__",
        "__mul__",
        "__pow__",
        "__rshift__",
        "__sub__",
        "__truediv__",
        "__xor__",
    ]

    for func_str in special_funcs:
        func = getattr(Container, func_str)
        ifunc = getattr(Container, func_str[:2] + "i" + func_str[2:])

        for value in [
            const,
            arr,
            container1,
        ]:
            if value == const and func_str == "__matmul__":
                continue
            container0_copy = container0.cont_deep_copy()
            id_before_op = id(container0_copy)
            og_ids = container0_copy.cont_map(lambda x, _: id(x))
            ifunc(container0_copy, value)
            op_ids = container0_copy.cont_map(lambda x, _: id(x))

            assert func(container0, value) == container0_copy  # values
            assert id(container0_copy) == id_before_op  # container ids
            assert og_ids == op_ids  # value ids


# TODO: Test non-inplace operator functions like __add__ and __matmul__
