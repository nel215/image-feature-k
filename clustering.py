# utf-8

import pickle
import numpy
import sklearn.cluster

f = open('./features.db', 'r')
vectors = pickle.load(f)
f.close()

print 'start clustering.'
kmeans = sklearn.cluster.MiniBatchKMeans()
kmeans.fit(numpy.array(vectors))

f = open('./cluster.db', 'w')
pickle.dump(kmeans, f)
f.close()
