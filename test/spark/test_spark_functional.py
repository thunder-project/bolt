import pytest
from numpy import arange, repeat, nan, float32, nanmean, nanmax, nanmin, nanvar, nanstd, nansum
from bolt import array
from bolt.utils import allclose
import generic

def test_map(sc):
    import random
    random.seed(42)

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=0)

    # Test all map functionality when the base array is split after the first axis
    generic.map_suite(x, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(x, sc, axis=(0, 1))
    generic.map_suite(x, b)

    # Split the BoltArraySpark after the third axis (scalar values) and rerun the tests
    b = array(x, sc, axis=(0, 1, 2))
    generic.map_suite(x, b)

def test_map_with_keys(sc):
    x = arange(2*3).reshape(2, 3)
    b = array(x, sc, axis=0)
    c = b.map(lambda kv: kv[0] + kv[1], with_keys=True)
    assert allclose(b.toarray() + [[0, 0, 0], [1, 1, 1]], c.toarray())

def test_reduce(sc):
    from numpy import asarray

    dims = (10, 10, 10)
    area = dims[0] * dims[1]
    arr = asarray([repeat(x, area).reshape(dims[0], dims[1]) for x in range(dims[2])])
    b = array(arr, sc, axis=0)

    # Test all reduce functionality when the base array is split after the first axis
    generic.reduce_suite(arr, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(arr, sc, axis=(0, 1))
    generic.reduce_suite(arr, b)

    # Split the BoltArraySpark after the third axis (scalar values) and rerun the tests
    b = array(arr, sc, axis=(0, 1, 2))
    generic.reduce_suite(arr, b)

def test_filter(sc):

    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=0)

    # Test all filter functionality when the base array is split after the first axis
    generic.filter_suite(x, b)

    # Split the BoltArraySpark after the second axis and rerun the tests
    b = array(x, sc, axis=(0, 1))
    generic.filter_suite(x, b)

    # Split the BoltArraySpark after the third axis (scalar values) and rerun the tests
    b = array(x, sc, axis=(0, 1, 2))
    generic.filter_suite(x, b)

    # Test that the split is correct after the filter (should always be 1
    # due to reindexing)
    b = array(x, sc, axis=(0, 1))
    r = b.filter(lambda x: True, axis=(1,))
    assert r.shape == (b.shape[1], b.shape[0], b.shape[2])
    assert r._split == 1

def test_mean(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.mean(), x.mean())
    assert allclose(b.mean(axis=0), x.mean(axis=0))
    assert allclose(b.mean(axis=(0, 1)), x.mean(axis=(0, 1)))
    assert b.mean(axis=(0, 1, 2)) == x.mean(axis=(0, 1, 2))

def test_std(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.std(), x.std())
    assert allclose(b.std(axis=0), x.std(axis=0))
    assert allclose(b.std(axis=(0, 1)), x.std(axis=(0, 1)))
    assert b.std(axis=(0, 1, 2)) == x.std(axis=(0, 1, 2))

def test_var(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.var(), x.var())
    assert allclose(b.var(axis=0), x.var(axis=0))
    assert allclose(b.var(axis=(0, 1)), x.var(axis=(0, 1)))
    assert b.var(axis=(0, 1, 2)) == x.var(axis=(0, 1, 2))

def test_sum(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.sum(), x.sum())
    assert allclose(b.sum(axis=0), x.sum(axis=0))
    assert allclose(b.sum(axis=(0, 1)), x.sum(axis=(0, 1)))
    assert b.sum(axis=(0, 1, 2)) == x.sum(axis=(0, 1, 2))

def test_min(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.min(), x.min())
    assert allclose(b.min(axis=0), x.min(axis=0))
    assert allclose(b.min(axis=(0, 1)), x.min(axis=(0, 1)))
    assert b.min(axis=(0, 1, 2)) == x.min(axis=(0, 1, 2))

def test_max(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    b = array(x, sc, axis=(0,))

    assert allclose(b.max(), x.max())
    assert allclose(b.max(axis=0), x.max(axis=0))
    assert allclose(b.max(axis=(0, 1)), x.max(axis=(0, 1)))
    assert b.max(axis=(0, 1, 2)) == x.max(axis=(0, 1, 2))

def test_nanmean(sc):
    x = arange(2 * 3 * 4).reshape(2, 3, 4).astype(float32)
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanmean(), nanmean(x))
    assert allclose(b.nanmean(axis=0), nanmean(x, axis=0))
    assert allclose(b.nanmean(axis=(0, 1)), nanmean(x, axis=(0, 1)))
    assert allclose(b.nanmean(axis=(0, 1, 2)), nanmean(x, axis=(0, 1, 2)))

    x[1, 2, 3] = nan
    x[0, 0, 2] = nan
    x[1, 1, 0] = nan
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanmean(), nanmean(x))
    assert allclose(b.nanmean(axis=0), nanmean(x, axis=0))
    assert allclose(b.nanmean(axis=(0, 1)), nanmean(x, axis=(0, 1)))
    assert allclose(b.nanmean(axis=(0, 1, 2)), nanmean(x, axis=(0, 1, 2)))

def test_nanstd(sc):
    x = arange(2 * 3 * 4).reshape(2, 3, 4).astype(float32)
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanstd(), nanstd(x))
    assert allclose(b.nanstd(axis=0), nanstd(x, axis=0))
    assert allclose(b.nanstd(axis=(0, 1)), nanstd(x, axis=(0, 1)))
    assert allclose(b.nanstd(axis=(0, 1, 2)), nanstd(x, axis=(0, 1, 2)))

    x[1, 2, 3] = nan
    x[0, 0, 2] = nan
    x[1, 1, 0] = nan
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanstd(), nanstd(x))
    assert allclose(b.nanstd(axis=0), nanstd(x, axis=0))
    assert allclose(b.nanstd(axis=(0, 1)), nanstd(x, axis=(0, 1)))
    assert allclose(b.nanstd(axis=(0, 1, 2)), nanstd(x, axis=(0, 1, 2)))

def test_nanvar(sc):
    x = arange(2 * 3 * 4).reshape(2, 3, 4).astype(float32)
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanvar(), nanvar(x))
    assert allclose(b.nanvar(axis=0), nanvar(x, axis=0))
    assert allclose(b.nanvar(axis=(0, 1)), nanvar(x, axis=(0, 1)))
    assert allclose(b.nanvar(axis=(0, 1, 2)), nanvar(x, axis=(0, 1, 2)))

    x[1, 2, 3] = nan
    x[0, 0, 2] = nan
    x[1, 1, 0] = nan
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanvar(), nanvar(x))
    assert allclose(b.nanvar(axis=0), nanvar(x, axis=0))
    assert allclose(b.nanvar(axis=(0, 1)), nanvar(x, axis=(0, 1)))
    assert allclose(b.nanvar(axis=(0, 1, 2)), nanvar(x, axis=(0, 1, 2)))

def test_nansum(sc):
    x = arange(2 * 3 * 4).reshape(2, 3, 4).astype(float32)
    b = array(x, sc, axis=(0,))

    assert allclose(b.nansum(), nansum(x))
    assert allclose(b.nansum(axis=0), nansum(x, axis=0))
    assert allclose(b.nansum(axis=(0, 1)), nansum(x, axis=(0, 1)))
    assert allclose(b.nansum(axis=(0, 1, 2)), nansum(x, axis=(0, 1, 2)))

    x[1, 2, 3] = nan
    x[0, 0, 2] = nan
    x[1, 1, 0] = nan
    b = array(x, sc, axis=(0,))

    assert allclose(b.nansum(), nansum(x))
    assert allclose(b.nansum(axis=0), nansum(x, axis=0))
    assert allclose(b.nansum(axis=(0, 1)), nansum(x, axis=(0, 1)))
    assert allclose(b.nansum(axis=(0, 1, 2)), nansum(x, axis=(0, 1, 2)))

def test_nanmin(sc):
    x = arange(2 * 3 * 4).reshape(2, 3, 4).astype(float32)
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanmin(), nanmin(x))
    assert allclose(b.nanmin(axis=0), nanmin(x, axis=0))
    assert allclose(b.nanmin(axis=(0, 1)), nanmin(x, axis=(0, 1)))
    assert allclose(b.nanmin(axis=(0, 1, 2)), nanmin(x, axis=(0, 1, 2)))

    x[1, 2, 3] = nan
    x[0, 0, 2] = nan
    x[1, 1, 0] = nan
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanmin(), nanmin(x))
    assert allclose(b.nanmin(axis=0), nanmin(x, axis=0))
    assert allclose(b.nanmin(axis=(0, 1)), nanmin(x, axis=(0, 1)))
    assert allclose(b.nanmin(axis=(0, 1, 2)), nanmin(x, axis=(0, 1, 2)))

def test_nanmax(sc):
    x = arange(2 * 3 * 4).reshape(2, 3, 4).astype(float32)
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanmax(), nanmax(x))
    assert allclose(b.nanmax(axis=0), nanmax(x, axis=0))
    assert allclose(b.nanmax(axis=(0, 1)), nanmax(x, axis=(0, 1)))
    assert allclose(b.nanmax(axis=(0, 1, 2)), nanmax(x, axis=(0, 1, 2)))

    x[1, 2, 3] = nan
    x[0, 0, 2] = nan
    x[1, 1, 0] = nan
    b = array(x, sc, axis=(0,))

    assert allclose(b.nanmax(), nanmax(x))
    assert allclose(b.nanmax(axis=0), nanmax(x, axis=0))
    assert allclose(b.nanmax(axis=(0, 1)), nanmax(x, axis=(0, 1)))
    assert allclose(b.nanmax(axis=(0, 1, 2)), nanmax(x, axis=(0, 1, 2)))