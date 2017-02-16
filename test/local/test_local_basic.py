from numpy import arange
from bolt import array
from bolt.spark.array import BoltArraySpark
from bolt.utils import allclose


def test_construct():
    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x)
    assert b.shape == (2, 3, 4)


def test_toarray():

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x)
    assert allclose(b.toarray(), x)

def test_tospark(sc):

    x = arange(2*3*4).reshape((2, 3, 4))
    b = array(x)
    s = b.tospark(sc, axis=0)
    assert isinstance(s, BoltArraySpark)
    assert s.shape == (2, 3, 4)
    assert allclose(s.toarray(), x)

def test_tordd(sc):

    from pyspark import RDD
    x = arange(2*3*4).reshape((2, 3, 4))

    b = array(x)
    r = b.tordd(sc, axis=0)
    assert isinstance(r, RDD)
    assert r.count() == 2

    r = b.tordd(sc, axis=(0, 1))
    assert isinstance(r, RDD)
    assert r.count() == 2*3

    r = b.tordd(sc, axis=(0, 1, 2))
    assert isinstance(r, RDD)
    assert r.count() == 2*3*4
