import pytest
from numpy import arange, true_divide
from bolt import array
from bolt.utils import allclose

from pyspark import SparkConf, SparkContext

@pytest.fixture(scope="session",
                params=[pytest.mark.spark_yarn('yarn'),
                        pytest.mark.spark_local('local')])
def sc(request):
    if request.param == 'local':
        conf = (SparkConf()
                .setMaster("local[2]")
                .setAppName("pytest-pyspark-local-testing")
                )
    elif request.param == 'yarn':
        conf = (SparkConf()
                .setMaster("yarn-client")
                .setAppName("pytest-pyspark-yarn-testing")
                .set("spark.executor.memory", "1g")
                .set("spark.executor.instances", 2)
                )
    request.addfinalizer(lambda: spark_context.stop())

    spark_context = SparkContext(conf=conf)
    return spark_context

def test_elementwise_spark(sc):
    x = arange(1, 2*3*4+1).reshape(2, 3, 4)
    y = 5*x
    bx = array(x, sc, axis=(0,))
    by = array(y, sc, axis=(0,))
    
    bxyadd = bx+by
    bxyaddarr = bxyadd.toarray()
    
    bxysub = bx-by
    bxysubarr = bxysub.toarray()
    
    bxymul = bx*by
    bxymularr = bxymul.toarray()
    
    bxydiv = bx/by
    bxydivarr = bxydiv.toarray()
    
    assert allclose(bxyaddarr, x+y)
    assert allclose(bxysubarr, x-y)
    assert allclose(bxymularr, x*y)
    assert allclose(bxydivarr, true_divide(x,y))

def test_elementwise_mix(sc):
    x = arange(1, 2*3*4+1).reshape(2, 3, 4)
    y = x*3
    bx = array(x, sc, axis=(0,))
    by = array(y, sc, axis=(0,))
    
    bxyadd = bx+by
    bxyaddarr = bxyadd.toarray()
    bxyaddloc = bx+y
    bxyaddlocarr = bxyaddloc.toarray()
    
    bxysub = bx-by
    bxysubarr = bxysub.toarray()
    bxysubloc = bx-y
    bxysublocarr = bxysubloc.toarray()
    
    bxymul = bx*by
    bxymularr = bxymul.toarray()
    bxymulloc = bx*y
    bxymullocarr = bxymulloc.toarray()
    
    bxydiv = bx/by
    bxydivarr = bxydiv.toarray()
    bxydivloc = bx/y
    bxydivlocarr = bxydivloc.toarray()
    
    assert allclose(bxyadd, bxyaddlocarr)
    assert allclose(bxysub, bxysublocarr)
    assert allclose(bxymul, bxymullocarr)
    assert allclose(bxydiv, bxydivlocarr)
    
def test_elementwise_scalar(sc):
    x = arange(2*3*4).reshape(2, 3, 4)
    bx = array(x, sc, axis=(0,))
    
    bxfive = bx + 5
    bxfivearr = bxfive.toarray()
    fivebx = 5 + bx
    fivebxarr = fivebx.toarray()
    
    bxten = bx * 10
    bxtenarr = bxten.toarray()
    tenbx = 10 * bx
    tenbxarr = tenbx.toarray()
    
    assert allclose(bxfivearr, fivebxarr)
    assert allclose(bxtenarr, tenbxarr)
