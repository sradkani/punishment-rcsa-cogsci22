
import numpy as np
import pandas as pd

def preprocess(data):
    """Finds the probability of punishment in different IVs"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': []}

    # Base punisher
    for IV, policy in data['average_base_punisher_policy'].items():
        formatted_data['condition'].append('Base punisher')
        formatted_data['IV'].append(IV)
        formatted_data['punish_prob'].append(6*policy['Punish']+1)

    # Pragmatic punisher
    for IV, policy in data['average_pragmatic_punisher_policy_high'].items():
        formatted_data['condition'].append('High audience value')
        formatted_data['IV'].append(IV)
        formatted_data['punish_prob'].append(6*policy['Punish']+1)

    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.IV = pd.Categorical(formatted_df.IV,
                                     categories=['No-gain', 'Small-gain', 'Large-gain'],
                                     ordered=True)
    formatted_df.condition = pd.Categorical(formatted_df.condition,
                                     categories=['Base punisher', 'High audience value'],
                                     ordered=True)
    return formatted_df
