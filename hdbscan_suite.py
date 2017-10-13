"""
    Clustering suite launching point

    Used with Anaconda distro of Python 3.6
    
    1. Activate Anaconda environment: source activate SorinLab_env
    2. Install following packages using pip while environment is active
        a. numpy
        b. cython
        c. scikit-learn
        d. matplotlib
        e. seaborn
        f. hdbscan

"""

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
import matplotlib.pyplot as plt
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

def labels_to_colors(labels):
    """HELPER: Map labels assignments to colors"""
    color_palette = sns.color_palette('hls', 100)
    return [color_palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

def create_plots(dataframe, bounds, param_dir, cluster_colors):
    """HELPER: Creates and saves plots"""
    for pair in list(itertools.combinations(bounds, r = 2)):
        x = pair[0] - bounds[0]
        y = pair[1] - bounds[0]
        fig, ax = plt.subplots(1)
        ax.set_title('{} vs {}'.format(x + bounds[0], y + bounds[0]))
        ax.scatter(dataframe.iloc[:, x], dataframe.iloc[:, y], s=50, linewidth=0,c=cluster_colors, alpha=0.80)
        plot_filename = '{}/{}_vs_{}.png'.format(param_dir, x + bounds[0], y + bounds[0])
        fig.savefig(plot_filename)
        fig.clf()

parser = argparse.ArgumentParser(
    description='Welcome to the super duper awesome clustering suite!'
)

# LOAD DATASET WITH DESIRED SAMPLE SIZE AND FEATURES TO BE CLUSTERED + PLOTTED
parser.add_argument('-d', '--data', type=str, help="data to be clustered")
parser.add_argument('-s', '--sample', nargs='?', type=int, help="sample size of dataset")
parser.add_argument('-f', '--frange', nargs='?', type=str, help="features to be cluster")
parser.add_argument('-p', '--fplots', nargs='?', type=str, help="clustered features to be plotted")
parser.add_argument('-c', '--cnames', nargs='?', type=str, help="feature column names")
parser.add_argument('-P', '--part', nargs='?', type=str, help="partition by feature column and respective range")
parser.add_argument('-b', '--best', nargs='?', default=False, type=bool, help="report only best results")
parser.add_argument('-S', '--stats', default=1, type=int, help="number of runs for statistical assessment")

# NORMALIZATION
parser.add_argument('-N', '--norm', nargs='?', type=str, help="normalization method and columns to be normalized")

# FOR USING A RANGE OF PARAMETERS
parser.add_argument('-r', '--range', nargs='?', default=False, type=bool, help="used to run with a range of parameters with a step size")

# KMEANS PARAMETERS
parser.add_argument('-n', '--nclusters', nargs='?', type=str, help="number of clusters for kmeans")
parser.add_argument('-i', '--init', nargs='?', default='k-means++', type=str, help="initialization method of kmeans(random or k-means++)")

# DBSCAN PARAMETERS
parser.add_argument('-e', '--eps', nargs='?', type=str, help="eps radius for core points")

# DBSCAN & HDBSCAN PARAMETERS
parser.add_argument('-m', '--min', nargs='?', type=str, help="min number of samples for core points")

# THREADS
parser.add_argument('-t', '--threads', nargs='?', default=4, type=int, help="number of threads to use")

# RUN NUMBER
parser.add_argument('-R', '--runcount', nargs='?', default=1, type=int, help="run number")

# DATE TIME
parser.add_argument('--datetime', type=str, help="date and time")

# SET UP ARGS
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

# CREATE DATA TIME DIRECTORY
stats_dir = 'RESULTS/{}/{}'.format(args.datetime, 'statistics')
current_dir = 'RESULTS/{}'.format(args.datetime)
run_dir = 'RESULTS/{}/run_{}'.format(args.datetime, args.runcount)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    
# STARTING TIME
start_time = time.time()

# RETRIEVE AND EXTRACT PARAMETER ARGUMENTS
session = Clustering.HDBSCANSession()

# CREATE PARAMETERS LISTS FROM ARGUMENTS
vals = list(map(float, args.min.split(',')))
if args.range:
    vals = np.arange(vals[0], vals[1] + vals[2], vals[2])

for min_val in list(vals):
    min_dir = '{}/min_{}'.format(run_dir, int(min_val))
    os.mkdir(min_dir)

    results = session.run(dataframe, min_val, args.threads)

    current_run = {'min':min_val, 'results': results}
    with open('{}/results.json'.format(min_dir), 'w') as f:
        f.write(json.dumps(current_run, indent=4))

    final_results['paramruns'].append(current_run)

    if args.fplots:
        bounds = list(map(int, args.fplots.split(',')))
    else:
        bounds = list(range(bounds[0], bounds[1] + 1))

    cluster_colors = labels_to_colors(results['labels'])    
    create_plots(dataframe, bounds, min_dir, cluster_colors)

    del results

# ENDING TIME
end_time = time.time()

# RECORD ELAPSED TIME
final_results['elapsed'] = '{}s'.format(int(end_time - start_time))

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
print(n_clus)

# APPEND MULTIRUN CSV
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
with open(stats_dir + '/hdbscan_multirun_results.csv', 'a') as j:
    j.write(','.join([str(n) for n in n_clus]) + '\n')


# n_cluster_stds = tuple(dict((x[0], x) for x in n_cluster_stds).values())
# print(n_cluster_stds)



# print('Integrating results...')

# # CREATE MAIN RESULTS DIRECTORY IF ONE DOES EXIST
# if not os.path.exists('RESULTS'):
#     os.makedirs('RESULTS')

# # CREATE DIRECTORY FOR NEW RUN
# now = datetime.datetime.now()
# datetime_dir = str(now.strftime("%Y-%m-%d__%H-%M-%S"))
# current_directory = "RESULTS/" + datetime_dir
# os.makedirs(current_directory)


# with open(current_directory + '/summary.txt', 'w') as summary:

#     # PRINT FINAL RESULTS
#     summary.write('CLUSTERING SUMMARY:\n\n')

#     now_formatted = str(now.strftime("%Y-%m-%d  %H:%M:%S"))
#     summary.write('DATE & TIME COMPLETED: {}\n'.format(now_formatted))
#     summary.write('TIME ELAPSED: {}s\n\n'.format(int(end_time - start_time)))

#     filename = args.data.split('/')[-1]
#     summary.write('PREVIEW OF {}: \n\n'.format(filename))
#     summary.write(str(Partitioner().sample(dataframe, 10)))
#     summary.write('\n\n\nSAMPLE SIZE: {}'.format(args.sample))
#     summary.write('\n\n\n')

#     summary.write('METHODS USED: {}\n\n\n'.format(', '.join(algs)))

#     for alg in final_results.keys():

#         # CREATE ALGORITHM DIRECTORY
#         alg_dir = current_directory + '/{}'.format(alg)
#         os.makedirs(alg_dir)

#         summary.write('PARAMETERS CHOSEN FOR {}:\n\n'.format(alg))
#         if alg in ['hdbscan', 'dbscan']:
#             summary.write('min samples(m): \n{}'.format('\n'.join([str(m) for m in list(min_vals)])))
#             if alg is 'dbscan':
#                 summary.write('eps(e): \n{}'.format('\n'.join([str(eps) for eps in list(eps_vals)])))

#         if alg is 'kmeans':
#             summary.write('n clusters(n): \n{}'.format('\n'.join([str(n) for n in list(n_clusters_vals)])))
        
#         if args.best is True:
#             best_params = max(final_results[alg], key=lambda x:x['sil_score'])
#             best_params['algorithm'] = alg
#             best_results.append(best_params)

#             if args.range:
#                 summary.write('\n\nPARAMETERS OF {} WITH BEST SILHOUETTE SCORE: \n\n'.format(alg))
#             else:
#                 summary.write('\n\nSILHOUETTE SCORE: \n\n')

#             for item in sorted(best_params):
#                 if item not in ['labels', 'algorithm']:
#                     summary.write('{}: {}\n'.format(item, best_params[item]))

#         else:
#             for result in final_results[alg]:
#                 run_dir = current_directory + '/' + result_to_dirname(alg, result)
#                 os.makedirs(run_dir)

#         summary.write('\n\n\n')



# # CREATE RESULTS JSON
# with open(current_directory + '/results.json', 'w') as j:
#     j.write(json.dumps(final_results, sort_keys=True, indent=4))


# # CREATE PLOTS FOR BEST RESULTS
# print('Creating plots...')

# if args.fplots:
#     bounds = list(map(int, args.fplots.split(',')))
# else:
#     bounds = list(range(bounds[0], bounds[1] + 1))

# color_palette = sns.color_palette('hls', 100)


# if args.best is True:
#     for best_result in best_results:
#         cluster_colors = labels_to_colors(best_result['labels']) 
#         run_dir = current_directory + '/' + best_result['algorithm'] 
#         create_plots(dataframe, bounds, run_dir, cluster_colors)

# else:
#     for alg in final_results.keys():
#         for result in final_results[alg]:
#             cluster_colors = labels_to_colors(result['labels'])
#             run_dir = current_directory + '/' + result_to_dirname(alg, result)
#             create_plots(dataframe, bounds, run_dir, cluster_colors)

# print('Completed!')
# print('Results stored in {}/{}/'.format(os.getcwd(), current_directory))