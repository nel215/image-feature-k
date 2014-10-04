#coding: utf-8
import cv2
import os
import pickle
import sklearn.cluster
import numpy

import vectorizer

def extract_features_from_img(img, kmeans, patch_size=5):
    k = kmeans.get_params()['n_clusters']
    h,w,c = img.shape
    offset = patch_size/2
    feature = [0 for _ in xrange(4*k)]
    for y in xrange(offset, h-offset):
        for x in xrange(offset, w-offset):
            patch = cv2.getRectSubPix(img, (patch_size, patch_size), (x,y))
            vector = vectorizer.extract_vector_from_patch(patch)
            dist = kmeans.transform(vector)
            bin = dist.argmin()
            min_dist = dist.min()
            mean = dist.mean()
            strength = max(0, mean - min_dist)
            idx = (2*(2*y/h)+(2*x/w))*k + bin
            feature[idx] += strength
    feature = numpy.array(feature)
    feature /= feature.mean()
    feature /= feature.std()
    return feature

f = open('./cluster-2.db', 'r')
kmeans = pickle.load(f)
f.close()

base_dir = './101_ObjectCategories/'

train_data = []

categories = os.listdir(base_dir)
for category in categories[:2]:
    img_dir = base_dir + category + '/'
    image_names = os.listdir(img_dir)
    for image_name in image_names[10:20]:
        print img_dir, image_name
        img = cv2.imread(img_dir + image_name)
        h,w,c = img.shape
        small_img = cv2.resize(img, (w/2, h/2))
        feature = extract_features_from_img(small_img, kmeans)
        train_data.append((feature, category))

f = open('./train_data.db', 'w')
pickle.dump(train_data, f)
f.close()

print train_data




