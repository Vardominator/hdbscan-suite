# HELPER CLASSES
import Normalizer as norm
import Clustering
from Partitioner import Partitioner

# DATA PROCESSING & MANIPULATION
import numpy as np
import pandas as pd
import itertools
import statistics as stat

# VISUALIZATION
import seaborn as sns

# MISC
import random as rand
import argparse
import datetime
import os
import time
import json
import importlib as imp
import sys
import logging

parser = argparse.ArgumentParser(
    description='Welcome to the super duper awesome clustering suite!'
)

# LOAD DATASET WITH DESIRED SAMPLE SIZE AND FEATURES TO BE CLUSTERED
parser.add_argument('-d', '--data', type=str, help="data to be clustered")
parser.add_argument('-s', '--sample', nargs='?', type=int, help="sample size of dataset")
parser.add_argument('-f', '--frange', nargs='?', type=str, help="features to be cluster")
parser.add_argument('-c', '--cnames', nargs='?', type=str, help="feature column names")
parser.add_argument('-P', '--part', nargs='?', type=str, help="partition by feature column and respective range")
parser.add_argument('-b', '--best', nargs='?', default=False, type=bool, help="report only best results")
parser.add_argument('-S', '--stats', default=1, type=int, help="number of runs for statistical assessment")
parser.add_argument('-N', '--norm', nargs='?', type=str, help="normalization method and columns to be normalized")
parser.add_argument('-r', '--range', nargs='?', default=False, type=bool, help="used to run with a range of parameters with a step size")
parser.add_argument('-o', '--option', type=str, default="min_samples", help="option to use min samples or min cluster size")
parser.add_argument('-m', '--min', type=str, help="min parameter range")
parser.add_argument('-t', '--threads', nargs='?', default=4, type=int, help="number of threads to use")
parser.add_argument('-R', '--runcount', nargs='?', default=1, type=int, help="run number")
parser.add_argument('--datetime', type=str, help="date and time")

args = parser.parse_args()
args_dict = vars(args)

# READ DATASET WITH ARBITRARY NUMBER OF ARGUMENTS
dataframe = pd.read_csv(args.data, sep='\s+', header=None)

# SAMPLE DATASET
if args.sample > 0:
    dataframe = Partitioner().sample(dataframe, args.sample)

# NORMALIZE DATASET
if args.norm:
    norm_cols_str = args.norm.split(',')
    norm_cols = list(map(int, norm_cols_str[1:]))
    dataframe = norm.Normalize(dataframe, norm_cols_str[0], norm_cols)

# PARTITION BY FEATURE COLUMN AND RESPECTIVE RANGE
if args.part:
    partition_arg = args.part.split(',')
    column = partition_arg[0]
    rows = list(map(int, partition_arg[1:]))
    dataframe = Partitioner().select_by_time(dataframe, rows[0], int(column))

# SELECT COLUMNS TO BE CLUSTERED
if args.frange:
    bounds = list(map(int, args.frange.split(',')))
    dataframe = Partitioner().select_by_column(dataframe, bounds)

# SET COLUMN NAMES
if args.cnames:
    dataframe.columns = args.colnames.split(',')

now = datetime.datetime.now()
final_results = {'datetime': str(now), 'paramruns': []}
best_results = []
current_dir = ''

# CREATE MAIN RESULTS DIRECTORY IF ONE DOES EXIST
if not os.path.exists('RESULTS'):
    os.makedirs('RESULTS')

# CREATE DATE TIME DIRECTORY
stats_dir = 'RESULTS/{}/{}'.format(args.datetime, 'statistics')
current_dir = 'RESULTS/{}'.format(args.datetime)
run_dir = 'RESULTS/{}/run_{}'.format(args.datetime, args.runcount)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# PREPARE LOGGING
logs_dir = "LOGS/" + args.datetime
logging.basicConfig(filename=logs_dir, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    
# STARTING TIME
start_time = time.time()

# RETRIEVE AND EXTRACT PARAMETER ARGUMENTS
session = Clustering.HDBSCANSession()

# CREATE PARAMETERS LISTS FROM ARGUMENTS
vals = list(map(float, args.min.split(',')))
if args.range:
    vals = np.arange(vals[0], vals[1] + vals[2], vals[2])

logging.info(len(dataframe))
# RUN HDBSCAN FOR EVERY PARAMETER
for min_val in list(vals):
    min_dir = '{}/min_{}'.format(run_dir, int(min_val))
    os.mkdir(min_dir)

    logging.info("HDBSCAN session started with {}={}".format(args.option, min_val))
    results = session.run(dataframe, args.option, min_val, args.threads)

    current_run = {'min':min_val, 'results': results}
    with open('{}/results.json'.format(min_dir), 'w') as f:
        f.write(json.dumps(current_run, indent=4))

    final_results['paramruns'].append(current_run)
    bounds = list(range(bounds[0], bounds[1] + 1))

    logging.info("Session successfully completed with {} clusters found".format(current_run['results']['n_clusters']))

    del results

# ENDING TIME
end_time = time.time()

# RECORD ELAPSED TIME
final_results['elapsed'] = '{}s'.format(int(end_time - start_time))
logging.info("Current run completed in {}".format(final_results['elapsed']))

# LOAD RESULTS JSON
results_json = {}
with open(current_dir + '/results.json', 'r') as j:
    results_json = json.load(j)

# UPDATE RESULTS WITH CURRENT CURRENT AND WRITE
results_json['runs'].append(final_results)
with open(current_dir + '/results.json', 'w') as j:
    j.write(json.dumps(results_json, indent=4))

hdbscan_runs = final_results['paramruns']
stats = {}

n_clus = [run['results']['n_clusters'] for run in hdbscan_runs]
logging.info("Clusters found for each {} in current run: {}".format(args.option, n_clus))

# APPEND MULTIRUN CSV
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
with open(stats_dir + '/hdbscan_multirun_results.csv', 'a') as j:
    j.write(','.join([str(n) for n in n_clus]) + '\n')