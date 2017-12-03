"""
    Protein data normalizer

    source: https://en.wikipedia.org/wiki/Normalization_(statistics)
"""

def Normalize(dataframe, method, cols):
    return function_map[method](dataframe, cols)       

def standard_score(dataframe, cols):
    dataframe = (dataframe - dataframe.mean()) / dataframe.std()
    return dataframe

def feature_scale(dataframe, cols):
    dataframe.iloc[:, cols] = (dataframe.iloc[:, cols] - dataframe.iloc[:, cols].min()) / (dataframe.iloc[:, cols].max() - dataframe.iloc[:, cols].min())
    return dataframe

function_map = {'standard_score': standard_score,
                        'feature_scale': feature_scale}
