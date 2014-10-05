#coding: utf-8

import cv2
import numpy

def extract_vector_from_channel_patch(channel_patch):
    flat_channel_patch = channel_patch.reshape(-1)
    mean = flat_channel_patch.mean()
    std  = flat_channel_patch.std()
    normalized_channel_patch = flat_channel_patch - mean
    if std > 1e-9: normalized_channel_patch /= std
    return normalized_channel_patch

def extract_vector_from_patch(patch):
    return numpy.concatenate([extract_vector_from_channel_patch(channel_patch)
        for channel_patch in cv2.split(patch)])

def extract_vectors_from_img(img,patch_size=7):
    img_size = img.shape
    offset = patch_size/2
    features = []
    for y in xrange(offset, img_size[0]-offset):
        for x in xrange(offset, img_size[1]-offset):
            patch = cv2.getRectSubPix(img, (patch_size, patch_size), (x,y))
            features.append(extract_vector_from_patch(patch))
    return features


if __name__=='__main__':
    img = cv2.imread('./Parrots.bmp')
    vectors = extract_vectors_from_img(img)
    print vectors[0]
