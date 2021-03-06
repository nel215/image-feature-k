#!/bin/sh

echo 'download dataset.'

if [ -e ./101_ObjectCategories.tar.gz ]; then
    echo 'already exists.'
else
    wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
fi

echo 'decompress dataset.'

if [ -e ./101_ObjectCategories ]; then
    echo 'already exists.'
else
    tar xzf 101_ObjectCategories.tar.gz
fi

echo 'vectorize images'

python make_cluster.py
python extractor.py
python train_test.py



