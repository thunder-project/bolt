#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file is ported from spark/util/StatCounter.scala
#
# This code is based on pyspark's statcounter.py and used under the ASF 2.0 license.

import copy
import math
from itertools import chain

from numpy import zeros, maximum, minimum, sqrt, isnan, fmin, fmax, shape, reshape, invert, amax, amin, nansum, dstack


class StatCounter(object):

    REQUIRED_FOR = {
        'count': ('n',),
        'mean': ('mu',),
        'sum': ('mu','n'),
        'min': ('minValue',),
        'max': ('maxValue',),
        'variance': ('mu', 'n', 'm2'),
        'sampleVariance': ('mu', 'n', 'm2'),
        'stdev': ('mu', 'n', 'm2'),
        'sampleStdev': ('mu', 'n', 'm2'),
        'nancount': ('n_n',),
        'nanmean': ('mu_n',),
        'nansum': ('mu_n', 'n_n'),
        'nanmin': ('minValue_n',),
        'nanmax': ('maxValue_n',),
        'nanvariance': ('mu_n', 'n_n', 'm2_n'),
        'nansampleVariance': ('mu_n', 'n_n', 'm2_n'),
        'nanstdev': ('mu_n', 'n_n', 'm2_n'),
        'nansampleStdev': ('mu_n', 'n_n', 'm2_n'),
        'all': ('n', 'mu', 'm2', 'minValue', 'maxValue', 'n_n', 'mu_n', 'm2_n', 'minValue_n', 'maxValue_n')
    }

    def __init__(self, values=(), stats='all'):
        self.n = 0L    # Running count of our values
        self.mu = 0.0  # Running mean of our values
        self.m2 = 0.0  # Running variance numerator (sum of (x - mean)^2)
        self.maxValue = None
        self.minValue = None
        self.n_n = None    # Running count of our values
        self.mu_n = None  # Running mean of our values
        self.m2_n = None  # Running variance numerator (sum of (x - mean)^2)
        self.maxValue_n = None
        self.minValue_n = None

        if isinstance(stats, basestring):
            stats = [stats]

        self.requiredAttrs = frozenset(chain().from_iterable([StatCounter.REQUIRED_FOR[stat] for stat in stats]))

        for v in values:
            self.merge(v)

    # add a value into this StatCounter, updating the statistics
    def merge(self, value):
        self.n += 1
        if self.__requires('mu'):
            delta = value - self.mu
            self.mu += delta / self.n
            if self.__requires('m2'):
                self.m2 += delta * (value - self.mu)
        if self.__requires('maxValue'):
            self.maxValue = maximum(self.maxValue, value) if not self.maxValue is None else value
        if self.__requires('minValue'):
            self.minValue = minimum(self.minValue, value) if not self.minValue is None else value

        if self.n_n is None:
            #Create the initial counter and set it to zeros
            self.n_n = zeros(value.shape)
            self.mu_n = zeros(value.shape)
            self.m2_n = zeros(value.shape)

        self.n_n += ~isnan(value)
        if self.__requires('mu_n'):
            delta = value - self.mu_n
            delta[isnan(value)] = 0
            self.mu_n = nansum(dstack((self.mu_n, (delta / self.n_n))),axis=2)
            if self.__requires('m2_n'):
                #Since value can have nans - replace with zeros
                tmpVal = value;
                tmpVal[isnan(tmpVal)] = 0
                self.m2_n += delta * (tmpVal - self.mu_n)
        if self.__requires('maxValue_n'):
            self.maxValue_n = fmax(self.maxValue_n, value) if not self.maxValue_n is None else value
        if self.__requires('minValue_n'):
            self.minValue_n = fmin(self.minValue_n, value) if not self.minValue_n is None else value

        return self

    # checks whether the passed attribute name is required to be updated in order to support the
    # statistics requested in self.requested
    def __requires(self, attrname):
        return attrname in self.requiredAttrs

    # merge another StatCounter into this one, adding up the statistics
    def combine(self, other):
        if not isinstance(other, StatCounter):
            raise Exception("Can only merge Statcounters!")

        if other is self:  # reference equality holds
            self.merge(copy.deepcopy(other))  # Avoid overwriting fields in a weird order
        else:
            # accumulator should only be updated if it's valid in both statcounters:
            self.requiredAttrs = set(self.requiredAttrs).intersection(set(other.requiredAttrs))

            if self.n == 0:
                self.n = other.n
                for attrname in ('mu', 'm2', 'maxValue', 'minValue', 'n_n', 'mu_n', 'm2_n', 'maxValue_n', 'minValue_n'):
                    if self.__requires(attrname):
                        setattr(self, attrname, getattr(other, attrname))

            elif other.n != 0:
                if self.__requires('mu'):
                    delta = other.mu - self.mu
                    if other.n * 10 < self.n:
                        self.mu = self.mu + (delta * other.n) / (self.n + other.n)
                    elif self.n * 10 < other.n:
                        self.mu = other.mu - (delta * self.n) / (self.n + other.n)
                    else:
                        self.mu = (self.mu * self.n + other.mu * other.n) / (self.n + other.n)

                    if self.__requires('m2'):
                        self.m2 += other.m2 + (delta * delta * self.n * other.n) / (self.n + other.n)

                if self.__requires('maxValue'):
                    self.maxValue = maximum(self.maxValue, other.maxValue)
                if self.__requires('minValue'):
                    self.minValue = minimum(self.minValue, other.minValue)

                self.n += other.n

                if self.__requires('mu_n'):
                    delta = other.mu_n - self.mu_n
                    self.mu_n = (self.mu_n * self.n_n + other.mu_n * other.n_n) / (self.n_n + other.n_n)

                    #Set areas with no data to zero
                    self.mu_n[isnan(self.mu_n)] = 0


                    if self.__requires('m2_n'):
                        tmpAdd = (delta * delta * self.n_n * other.n_n) / (self.n_n + other.n_n)
                        tmpAdd[isnan(tmpAdd)] = 0
                        self.m2_n += other.m2_n + tmpAdd

                if self.__requires('maxValue_n'):
                    self.maxValue_n = fmax(self.maxValue_n, other.maxValue_n)
                if self.__requires('minValue_n'):
                    self.minValue_n = fmin(self.minValue_n, other.minValue_n)

                self.n_n += other.n_n



        return self

    # Clone this StatCounter
    def copy(self):
        return copy.deepcopy(self)


    def __isavail(self, attrname):
        if not all(attr in self.requiredAttrs for attr in StatCounter.REQUIRED_FOR[attrname]):
            raise ValueError("'%s' stat not available, must be requested at "
                             "StatCounter instantiation" % attrname)
    @property
    def count(self):
        self.__isavail('count')
        return self.n

    @property
    def mean(self):
        self.__isavail('mean')
        return self.mu

    @property
    def sum(self):
        self.__isavail('sum')
        return self.n * self.mu

    @property
    def min(self):
        self.__isavail('min')
        return self.minValue

    @property
    def max(self):
        self.__isavail('max')
        return self.maxValue

    # Return the variance of the values.
    @property
    def variance(self):
        self.__isavail('variance')
        if self.n == 0:
            return float('nan')
        else:
            return self.m2 / self.n

    @property
    def stdev(self):
        self.__isavail('stdev')
        return sqrt(self.variance)

    #
    # Return the sample standard deviation of the values, which corrects for bias in estimating the
    # variance by dividing by N-1 instead of N.
    #
    @property
    def sampleStdev(self):
        self.__isavail('sampleStdev')
        return sqrt(self.sampleVariance)

    @property
    def nancount(self):
        self.__isavail('nancount')
        return self.n_n

    @property
    def nanmean(self):
        self.__isavail('nanmean')
        return self.mu_n

    @property
    def nansum(self):
        self.__isavail('nansum')
        return self.n_n * self.mu_n

    @property
    def nanmin(self):
        self.__isavail('nanmin')
        return self.minValue_n

    @property
    def nanmax(self):
        self.__isavail('nanmax')
        return self.maxValue_n

    # Return the variance of the values.
    @property
    def nanvariance(self):
        self.__isavail('nanvariance')
        tmpVar = self.m2_n / self.n_n
        #set areas with no data to zero
        tmpVar[isnan(tmpVar)] = 0
        return tmpVar

    #
    # Return the sample variance, which corrects for bias in estimating the variance by dividing
    # by N-1 instead of N.
    #
    @property
    def nansampleVariance(self):
        self.__isavail('nansampleVariance')
        tmpVar = self.m2_n / (self.n_n - 1)
        #set areas with no data to zero
        tmpVar[isnan(tmpVar)] = 0
        return tmpVar

    # Return the standard deviation of the values.
    @property
    def nanstdev(self):
        self.__isavail('nanstdev')
        return sqrt(self.nanvariance)

    #
    # Return the sample standard deviation of the values, which corrects for bias in estimating the
    # variance by dividing by N-1 instead of N.
    #
    @property
    def nansampleStdev(self):
        self.__isavail('nansampleStdev')
        return sqrt(self.nansampleVariance)

    def __repr__(self):
        return ("(count: %s, mean: %s, stdev: %s, max: %s, min: %s, required: %s, nancount: %s, nanmean: %s, nanstdev: %s, nanmax: %s, nanmin: %s)" %
                (self.count(), self.mean(), self.stdev(), self.max(), self.min(), str(tuple(self.requiredAttrs)), self.nancount(), self.nanmean(), self.nanstdev(), self.nanmax(), self.nanmin()))
