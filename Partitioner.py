"""
    Protein data partitioner
"""

import random

class Partitioner:

    # select by project
    def select_by_project(dataframe, projectNumber):
        dataframe = dataframe.loc[dataframe['Proj'] == projectNumber]
        return dataframe

    # select by run
    def select_by_run(dataframe, runNumber):
        dataframe = dataframe.loc[dataframe['Run'] == runNumber]
        return dataframe

    # select by clone
    def select_by_clone(dataframe, cloneNumber):
        dataframe = dataframe.loc[dataframe['Clone'] == cloneNumber]
        return dataframe

    # remove all bookkeeping data (project, run, clone, time, and date?) this is necessary for clustering
    def remove_all_bookkeeping(dataframe, remove_native_contacts=False):
        dataframe = dataframe.iloc[:, 4:]
        if remove_native_contacts:
            dataframe = dataframe.drop('NC', 1)
        return dataframe

    @staticmethod
    def sample(dataframe, sample_size):
        return dataframe.loc[random.sample(list(dataframe.index), sample_size)]

    @staticmethod
    def select_by_column(dataframe, bounds):
        partitioned_data = dataframe.iloc[:, bounds[0]:(bounds[1] + 1)]
        return partitioned_data

    @staticmethod
    def select_by_time(dataframe, startime, timecolumn):
        ix_true = dataframe.index[dataframe.iloc[:, timecolumn] > startime]
        temp_df = dataframe
        temp_df.columns = list(map(str, range(len(dataframe.iloc[0,:]))))
        partitioned_data = dataframe.loc[dataframe[str(timecolumn)] >= startime]
        return partitioned_data
