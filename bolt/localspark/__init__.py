from collections import defaultdict
class LocalRDD(object):
    """
    A simple, inefficient, approximation for a locally stored RDD object (a lazy list)
    """
    def __init__(self, kv_list, fcn_to_apply = None):
        """
        Parameters
        ----------
        kv_list : list
            the list of key value pairs
        fcn_to_apply : function
            the function that should be applied, ndarray -> ndarray
        """
        self.kv_list = kv_list
        self.lazy_fcn_list = [] if fcn_to_apply is None else fcn_to_apply
        self.cached_list = []
        
    def map(self, c_func):
        return LocalRDD(self.kv_list, fcn_to_apply = self.lazy_fcn_list + [c_func])
    
    def mapValues(self, c_func):
        return self.map(lambda (a,b): (a,c_func(b)))
    
    def flatMap(self, c_func):
        return self.map(c_func).map('flatten')
    
    def mapPartitions(self, c_func):
        """
        Make artificial partitions and executes the function on each one
        """
        return self.map(lambda (k,v): c_func([(k,v)]))
    
    def partitionBy(self, numPartitions, partitionFunc):
        """
        The current operations need to be executed before they can be handed off to the partition function
        """
        return LocalPartitionedRDD(self.collect(), partitionFunc)
    
    def values(self):
        return self.map(lambda (_, v): v)
    
    def collect(self):
        return LocalRDD.expand(self)
    
    @staticmethod
    def expand(curRDD):
        last_list = curRDD.kv_list
        for c_func in curRDD.lazy_fcn_list:
            if c_func == 'flatten':
                out_list = []
                for i in last_list: 
                    out_list += i
                last_list = out_list
            else:
                last_list = map(c_func, last_list)
        return last_list
            
        

class LocalPartitionedRDD(object):
    
    def __init__(self, kv_list, partitionFunc, part_rdd = None):
        """
        Creates a partitioned RDD which supports mapPartitions and values operations
        
        Parameters
        ----------
        kv_list : list[(k,v)]
            the list of key values
        partitionFunc : function
            apply to the keys to put them in distinct partitions
        part_rdd : dict
            to supply the already partitioned dataset with keys as the partition ids
            and values as the partition contents
        """
        if part_rdd is None:
            self.part_rdd = defaultdict(list)
            for (k,v) in kv_list:
                self.part_rdd[partitionFunc(k)]+=[(k,v)]
        else:
            self.part_rdd = part_rdd
        self.partitionFunc = partitionFunc
        self.kv_list = kv_list
    
    def mapPartitions(self, c_func):
        new_part_values = {}
        new_kv_list = []
        for partName, partValues in self.part_rdd.iteritems():
            new_values = c_func(partValues)
            new_part_values[partName] = new_values
            new_kv_list += new_values
        return LocalPartitionedRDD(new_kv_list, self.partitionFunc, part_rdd = new_part_values)
    
    def values(self):
        return LocalRDD(self.kv_list, [lambda (_, v): v])

class LocalSparkContext(object):
    def __init__(self):
        pass
    def parallelize(self, in_list, npartitions = 0):
        return LocalRDD(in_list)
    