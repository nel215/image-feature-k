# utf-8
import pickle
import os
import cv2

import vectorizer

base_dir = './101_ObjectCategories/'

file_name = './features.db'

vectors = []

categories = os.listdir(base_dir)
for category in categories[:2]:
    img_dir = base_dir + category + '/'
    image_names = os.listdir(img_dir)
    for image_name in image_names[0:1]:
        img = cv2.imread(img_dir + image_name)
        vectors += vectorizer.extract_vectors_from_img(img)

f = open(file_name, 'w')
pickle.dump(vectors, f)
f.close()




