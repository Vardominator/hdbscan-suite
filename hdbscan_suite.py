import json
import subprocess
import os
import datetime
import sys

# DATA PROCESSING AND ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CREATE MAIN RESULTS DIRECTORY IF ONE DOES EXIST
if not os.path.exists('RESULTS'):
    os.makedirs('RESULTS')

# CREATE DIRECTORY FOR CURRENT SET OF RUNS
now = datetime.datetime.now()
datetime_dir = str(now.strftime("%H-%M-%S__%Y-%m-%d"))
current_directory = "RESULTS/" + datetime_dir
os.makedirs(current_directory)

# READ RUN CONFIG
with open('config.json', 'r') as f:
    config = json.load(f)

# CHECK IF DATASET EXISTS
if not os.path.exists(config['data']):
    print("Dataset does not exist. Check the path and try again.")
    sys.exit()

stats_dir = '{}/statistics'.format(current_directory)
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

# CREATE REULTS JSON
with open(current_directory + '/results.json', 'w') as f:
    f.write(json.dumps({'runs': []}))

# INITIALIZE HDBSCAN_RESULTS CSV
params = []
vals = []
if config['parameters']['range'] is True:
    params = config['parameters']['min']
    vals = list(np.arange(params[0], params[1] + params[2], params[2]))
with open(stats_dir + '/hdbscan_multirun_results.csv', 'w') as f:
    f.write(','.join([str(m) for m in vals]) + '\n')

# RUN SUBPROCESSES AND WRITE TO FILE
for run in range(config['runs']):
    print('run: {}'.format(run))
    proc = subprocess.Popen([
        'python3',
        'hdbscan_runner.py',
        '-d',
        str(config['data']),
        '-P',
        '{},{}'.format(config['partition']['column'], config['partition']['range']),
        '-s',
        str(config['sample']),
        '-N',
        '{},{}'.format(config['norm']['method'], ','.join([str(c) for c in config['norm']['columns']])),
        '-f',
        ','.join([str(f) for f in config['range']]),
        '-p',
        ','.join([str(p) for p in config['plot_cols']]),
        '-r',
        str(config['parameters']['range']),
        '-o',
        str(config['parameters']['option']),
        '-m',
        ','.join([str(x) for x in config['parameters']['min']]),
        '--datetime',
        datetime_dir,
        '-R',
        str(run + 1),
        '-t',
        str(config['threads'])
    ], stdout=subprocess.PIPE)

    proc.wait()
    n_clusters = str(proc.stdout.readline().rstrip(), 'utf-8')
    print(n_clusters)


# STATISTICAL ANALYSIS AND PLOT CREATION OF RUNS
# if config['runs'] > 1:
#     df = pd.read_csv(stats_dir + '/hdbscan_multirun_results.csv')
#     print(df)
#     min_samples = list(map(int, list(df.columns)))
#     df_mean = df.mean()
#     df_std = df.std()

#     fig, ax = plt.subplots()
#     ax.errorbar(min_samples, df_mean, yerr=df_std, fmt='-o')
#     fig.savefig(stats_dir + '/hdbscan_multirun_stats.png')
#     fig.clf()
