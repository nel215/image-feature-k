# utf-8
import pickle
import os
import cv2
import time
import sklearn.cluster

import vectorizer

base_dir = './101_ObjectCategories/'

vectors = []

categories = os.listdir(base_dir)
for category in categories[:10]:
    img_dir = base_dir + category + '/'
    image_names = os.listdir(img_dir)
    for image_name in image_names[:10]:
        print "vectorize", img_dir, image_name
        img = cv2.imread(img_dir + image_name)
        h,w,c = img.shape
        resize_ratio = 64.1/max(h,w)
        resize_size = (int(w*resize_ratio), int(h*resize_ratio))
        print resize_size
        small_img = cv2.resize(img, resize_size)
        vectors += vectorizer.extract_vectors_from_img(small_img)

print 'start clustering.'
k = 1000
start_time = time.clock()
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=k, init_size=3*k)
kmeans.fit(vectors)
end_time = time.clock()
print "elasped time:", end_time - start_time

f = open('./cluster.db', 'w')
pickle.dump(kmeans, f)
f.close()
