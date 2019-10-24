import pytest
from kdtree_labeling import KDTreeLabeling
import numpy as np


def all_the_same(ar):
    return ar[ar == ar[0]].shape[0] == len(ar)


def test_fail_on_negative_r():
    space = np.array([[0] * 5, [1] * 5])
    labeling = KDTreeLabeling(r=-3, bound=2, stream_count=2, params=3, len_=5)
    with pytest.raises(ValueError):
        labeling.fit(space)


def test_fail_on_empty_array():
    labeling = KDTreeLabeling(r=1, bound=2, stream_count=5, params=5, len_=1)
    space = np.array([])
    with pytest.raises(ValueError):
        labeling.fit(space)


def test_on_five_equal_points():
    labeling = KDTreeLabeling(r=2, bound=6, stream_count=1, params=5, len_=5)
    space = np.array([[1] * 5] * 5)
    assert all_the_same(labeling.fit(space))


def test_on_two_same_points():
    labeling = KDTreeLabeling(r=1, bound=0, stream_count=1, params=3, len_=2)
    space = np.array([[0, 0, 0], [1, 1, 1]])
    assert all_the_same(labeling.fit(space))


def test_on_drop_same_points_from_different_classes():
    labeling = KDTreeLabeling(r=1, bound=0, stream_count=2, params=3, len_=2)
    space = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [0, 0, 0]])
    assert np.array_equal(labeling.fit(space), np.array([-1, 0, 1, -1]))

# def test_on_two_different_points():
