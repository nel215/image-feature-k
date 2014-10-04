# coding: utf-8
import random
import pickle
import sklearn.svm
import numpy
import sklearn.cross_validation

f = open('./train_data.db', 'r')
data = pickle.load(f)
f.close()

features = numpy.array([feature for feature, category in data])
categories = numpy.array([category for feature, category in data])

svc = sklearn.svm.LinearSVC()
scores = sklearn.cross_validation.cross_val_score(svc, features, categories, cv = 20)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
