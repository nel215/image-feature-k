# coding: utf-8
import random
import pickle
import sklearn.svm
import numpy

f = open('./train_data.db', 'r')
data = pickle.load(f)
f.close()

def trial(data):

    random.shuffle(data)
    train_data = data[:len(data)/2]
    test_data  = data[len(data)/2:]

    features = [feature for feature, category in train_data]
    categories = [category for feature, category in train_data]

    #svc = sklearn.svm.SVC()
    svc = sklearn.svm.LinearSVC()
    svc.fit(features, categories)

    correct = 0

    for feature, category in test_data:
        p = svc.predict(feature)[0]
        #print "ans:",category,
        #print "predict:", p
        if p == category: correct += 1

    return 1.0 * correct / len(test_data)

results = numpy.array([trial(data) for _ in xrange(100)])

print "accuracy:", results.mean()*100, "%"

