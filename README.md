# HDBSCAN Suite

## TODOs
* Implement logging (DONE)
* Run on Xavier's machine with entire stable luteo dataset (DONE)
* Run with min samples = 100 to 1500 in steps of 100
* Run with setting min samples = 1 and varying cluster size from 100 to 1500 in steps of 100
* Write install and usage instructions in README
* Write a table of parameters in config

## RANDOM NOTES
* min_cluster_size must be greater than 1

## Prerequisities
Git, Python 3.5+, pip3, virtualenv
```
sudo apt-get install git
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install python3-virtualenv
```

## Installing
Go to desired directory and create a virtual environment.
IMPORTANT: Make sure to make set the default interpreter as python3
```
virtualenv -p python3 --no-site-packages [desired environment name]
```
Go into the environment directory and activate the environment
```
cd [environment]
source bin/activate
```
Clone this repository into the environment
```
git clone https://github.com/Vardominator/hdbscan-suite.git
```
Install necessary packages. Order is important.
```
pip install numpy
pip install -r requirements.txt
```