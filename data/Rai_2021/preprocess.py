
import numpy as np
import pandas as pd

def preprocess(data):
    """Finds the probability of punishment in different IVs"""
    formatted_df = data
    if 'Large-gain' in list(data['IV']):
        formatted_df.IV = pd.Categorical(formatted_df.IV,
                                         categories=['No-gain', 'Small-gain', 'Large-gain'],
                                         ordered=True)
    else:
        formatted_df.IV = pd.Categorical(formatted_df.IV,
                                                categories=['No-gain', 'Small-gain'],
                                                ordered=True)
    return formatted_df
