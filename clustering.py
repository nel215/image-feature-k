# utf-8

import time
import pickle
import numpy
import sklearn.cluster

f = open('./features-2.db', 'r')
vectors = pickle.load(f)
f.close()

print 'start clustering.'
start_time = time.clock()
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=100)
kmeans.fit(numpy.array(vectors))
end_time = time.clock()
print "elasped time:", end_time - start_time

f = open('./cluster-2.db', 'w')
pickle.dump(kmeans, f)
f.close()
