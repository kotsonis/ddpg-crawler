import timeit
import time
import numpy as np
import random
p_total = 20000.0
x = [i for i in range (20000)]
bins = 512
def find_prefixsum_idx(val):
       return x[int(val)]

def test_array():
    bin_interval = p_total / bins                      # split the total sum of priorities into batch_size segments
    points_in_bin = np.random.random(size=bins)*bin_interval
    offset_per_bin = np.arange(0,bins)*bin_interval
    mass = offset_per_bin + points_in_bin
    new_m = np.expand_dims(mass, -1)
    idxes = [find_prefixsum_idx(x) for x in mass]
    return idxes

def current():
    results = []
    p_total = self._st_sum.sum(0, len(self._buffer) - 1)       # get total sum of priorites in the whole replay buffer
    bin_interval = p_total / bins                      # split the total sum of priorities into batch_size segments

    points_in_bin = np.random.random(size=bins)*bin_interval
    offset_per_bin = np.arange(0,bins)*avg_bin_length
    mass = offset_per_bin + points_in_bin
        
    for i in range(batch_size):
        # generate a random cummulative sum of priorites within this segment
        mass = random.random() * bin_length + i * bin_length 
        #Find index in the array of sampling probabilities such that sum of previous values is mass
        idx = self._st_sum.find_prefixsum_idx(mass)             
        results.append(idx)
    return results

def test_curr():
    results = []
    every_range_len = p_total / bins                      # split the total sum of priorities into batch_size segments
    for i in range(bins):
        # generate a random cummulative sum of priorites within this segment
        mass = random.random() * every_range_len + i * every_range_len 
        #Find index in the array of sampling probabilities such that sum of previous values is mass
        idx = x[int(mass)]          
        results.append(idx)
    return results


def test_vectorize():
    vf = np.vectorize(find_prefixsum_idx)
    bin_interval = p_total / bins                      # split the total sum of priorities into batch_size segments
    points_in_bin = np.random.random(size=bins)*bin_interval
    offset_per_bin = np.arange(0,bins)*bin_interval
    mass = offset_per_bin + points_in_bin
    idxes = vf(mass).tolist()
    return idxes
from operator import itemgetter
#print("array", timeit.timeit('test_array()', globals=globals(), number=10000))
#print("test_vectorize", timeit.timeit('test_vectorize()', globals=globals(), number=10000))
#print("test_curr", timeit.timeit('test_curr()', globals=globals(), number=10000))
#print("test random", timeit.timeit('x = [ random.random() for i in range(512)]',globals=globals(), number=10000))
print("test np.random", timeit.timeit('np.random.randint(512, size=512)',globals=globals(), number=1000))
print("test np arrange", timeit.timeit('idxes = [random.randint(0, 512) for _ in range(512)]',globals=globals(), number=1000))
print("access array list comprehension", timeit.timeit('res = [x[idx] for idx in range(512)]',globals=globals(), number=100000))
print("itemgetter access array list", timeit.timeit('idxes = range(512);res = list(itemgetter(*idxes)(x))',globals=globals(), number=100000))