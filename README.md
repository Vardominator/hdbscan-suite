# HDBSCAN Suite
Clustering suite designed to run HDBSCAN on a dataset multiple times with multiple parameters and perform statistical analysis on the results.

## Description
For each run, HDBSCAN will be performed on the dataset with each parameter. The results for each run will be accumulated and stored in its respective directory.

## Prerequisities
Install git, python3.5+, python3-pip and python3-virtualenv if necessary.
```
sudo apt-get install git
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install python3-virtualenv
```

## Installation
Go to desired directory and create a virtual environment.
IMPORTANT: Make sure to make set the default interpreter as python3.
```
virtualenv -p python3 --no-site-packages [desired environment name]
```
Go into the environment directory and activate the environment.
```
cd [environment]
source bin/activate
```
Clone this repository into the environment.
```
git clone https://github.com/Vardominator/hdbscan-suite.git
```
Install necessary packages. Order is important.
```
pip install numpy
pip install -r requirements.txt
```

## Usage
Save config.json.template as config.json.
```
cp config.json.template config.json
```
Using the table below, set the parameters in config.json as desired.

| Parameter | Description | Example |
| --------- | ----------- | --------
| runs | Number of times to run HDBSCAN with set of parameters | 50 |
| data | Path to dataset to be clustered | "data/luteo-1796-1798.txt" |
| partition:column | Column used to partition data | 3 |
| partition:start | Starting value for partition (everything less is left out) | 6000 |
| sample | Sample size of the dataset. Set 0 to use entire dataset post partition | 12000 |
| norm:method | Desired method for normalization. Available methods: [standard_score](https://en.wikipedia.org/wiki/Standard_score), [feature_scale](https://en.wikipedia.org/wiki/Feature_scaling) | "feature_scale" |
| norm:columns | Columns to be normalized (ex. [4,5,10]) | [4,5] |
| range | Columns to be included in the clustering, starting at 0 | [4,12] (columns 4 through 12 will be used in clustering ) | parameters:range | Whether or the set of desired parameters are a range. Set to false if running with individual values | true |
| parameters:option | Cluster criterion, minimum cluster size or minimum sample size. Look below for more information. | "min_cluster_size" |
| parameters:min | Range or set of values to be used for each run. [2,10,1] will use parameters from 2 to 10 in steps of 1 if  range is set to true. [2,5,10,30] will use parameters 2,5,10, and 30. | [2,10,1] |
| threads | Number of threads to use within HDBSCAN algorithm | 4 |




## Resources
Open-source HDBSCAN extenstion to Python's scikit-learn machine learning library

## TODOs
* Implement logging (DONE)
* Run on Xavier's machine with entire stable luteo dataset (DONE)
* Run with min samples = 100 to 1500 in steps of 100
* Run with setting min samples = 1 and varying cluster size from 100 to 1500 in steps of 100
* Write install and usage instructions in README
* Write a table of parameters in config

## RANDOM NOTES
* min_cluster_size must be greater than 1
