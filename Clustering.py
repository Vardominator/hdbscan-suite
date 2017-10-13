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
    def run(self, params, threads):
        raise NotImplementedError
    def save_results(self, location):
        raise NotImplementedError
    def save_plots(self, location):
        raise NotImplementedError

# HDBSCAN
class HDBSCANSession(ClusteringSession):
    def __init__(self):
        self.labels = []
        self.min_samples = 0
        self.data = None
        self.n_clusters = 0
    
    def run(self, data, m, threads):
        self.min_samples = int(m)
        self.data = data

        hdb = hdbscan.HDBSCAN(min_samples=self.min_samples, core_dist_n_jobs=threads)
 
        self.labels = hdb.fit_predict(data)
        self.n_clusters = len(set(self.labels)) - 1
        
        sample_size = 0
        if len(data) >= 20000:
            sample_size = 20000
        else:
            sample_size = len(data)

        del hdb
        
        return {'n_clusters': self.n_clusters, 'labels': self.labels.tolist()}

    def save_results(self, location):
        return super().save_results(location)

    def save_plots(self, location=""):
        current_directory = location + "RESULTS/HDBSCAN/" + datetime_dir
        print(current_directory)
        os.makedirs(current_directory)
