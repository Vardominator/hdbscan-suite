# Data
import numpy as np

# Intelligence
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
import hdbscan

# Visualization
# import matplotlib.pyplot as plt

# System
import datetime
import os
import time
import sys
import importlib as imp
import random as rand
import shutil

now = datetime.datetime.now()
dateTime = str(now.strftime("%Y-%m-%d  %H:%M:%S"))
datetime_dir = str(now.strftime("%Y-%m-%d__%H-%M-%S"))

# Directory will be determined by the user
current_directory = ""

# CLUSTERING INTERFACE
class ClusteringSession(object):
    def run(self, option, params, threads):
        raise NotImplementedError

# HDBSCAN
class HDBSCANSession(ClusteringSession):
    def __init__(self):
        self.labels = []
        self.option = "min_samples"
        self.param = 0
        self.data = None
        self.n_clusters = 0
    
    def run(self, data, option, m, threads):
        self.param = int(m)
        self.data = data
        self.option = option
        
        if self.option == "min_samples":
            hdb = hdbscan.HDBSCAN(min_samples=self.param, core_dist_n_jobs=threads)
        elif self.option == "min_cluster_size":
            hdb = hdbscan.HDBSCAN(min_cluster_size=self.param, core_dist_n_jobs=threads)

        self.labels = hdb.fit_predict(data)
        self.n_clusters = len(set(self.labels)) - 1
        
        sample_size = 0
        if len(data) >= 20000:
            sample_size = 20000
        else:
            sample_size = len(data)

        del hdb
        
        return {'n_clusters': self.n_clusters, 'labels': self.labels.tolist()}