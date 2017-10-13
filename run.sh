#!/bin/bash

for run in {1..51}
do
    python3 clustering_suite.py -d ../data/luteo-1796-1798.txt -s 1100000 -P 3,6000 -N feature_scale,4,5 -f 4,12 -p 4,5,6,7 -a hdbscan -r true -m 300,1000,100 -t 8
done