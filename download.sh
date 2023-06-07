#!/bin/bash

#ml-100k
mkdir -p data/ml-100k
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P data/ml-100k
unzip data/ml-100k/ml-100k.zip
rm data/ml-100k/ml-100k.zip
cd data/ml-100k
python3 preprocess.py 
cd ../../

#gisette
mkdir -p data/gisette
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2 -P data/gisette/
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2 -P data/gisette/
bzip2 -d data/gisette/gisette_scale.bz2
bzip2 -d data/gisette/gisette_scale.t.bz2

#rcv1
mkdir -p data/rcv1
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2 -P data/rcv1
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2 -P data/rcv1
bzip2 -d data/rcv1/rcv1_train.binary.bz2
bzip2 -d data/rcv1/rcv1_test.binary.bz2

#news20
mkdir -p data/news20
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2 -P data/news20
bzip2 -d data/news20/news20.binary.bz2