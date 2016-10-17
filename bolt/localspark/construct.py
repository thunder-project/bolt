from bolt.construct import ConstructBase
from bolt.spark.construct import ConstructSpark as cs
import numpy as np
from bolt.localspark import LocalSparkContext


class ConstructLocalSpark(ConstructBase):

    @staticmethod
    def array(a, context=None, axis=(0,), dtype=None, npartitions=None):
		return cs.array(a, context = LocalSparkContext(), axis = axis, 
                        dtype=dtype, npartitions=npartitions)

    @staticmethod
    def ones(shape, context=None, axis=(0,), dtype=np.float64, npartitions=None):
        return cs.ones(a, context = LocalSparkContext(), axis = axis, 
                       dtype=dtype, npartitions=npartitions)

    @staticmethod
    def zeros(shape, context=None, axis=(0,), dtype=np.float64, npartitions=None):
        return cs.zeros(a, context = LocalSparkContext(), axis = axis, 
                        dtype=dtype, npartitions=npartitions)

    @staticmethod
    def concatenate(arrays, axis=0):
        return cs.concatenate(array, axis = axis)

    @staticmethod
    def _argcheck(*args, **kwargs):
        """
        Check that arguments are consistent with localspark array construction.

        Condition is
        (1) keyword arg 'context' is the string 'fake'
        """
        
        return kwargs.get('context', '').find('fake')>=0

    @staticmethod
    def _format_axes(axes, shape):
        return cs._format_axes(axes, shape)

    @staticmethod
    def _wrap(func, shape, context=None, axis=(0,), dtype=None, npartitions=None):
        return cs._wrap(func, shape = shape, context = LocalSparkContext(), axis = axis, 
                        dtype = dtype, npartitions = npartitions)
