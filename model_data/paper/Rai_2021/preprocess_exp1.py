
import numpy as np
import pandas as pd

def preprocess(data):
    """Finds the probability of punishment in different IVs"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': []}

    # Base punisher
    for IV, policy in data['average_base_punisher_policy'].items():
        formatted_data['condition'].append('Base punisher')
        formatted_data['IV'].append(IV)
        formatted_data['punish_prob'].append(policy['Punish'])

    # Pragmatic punisher
    for IV, policy in data['average_pragmatic_punisher_policy_low'].items():
        formatted_data['condition'].append('Low audience value')
        formatted_data['IV'].append(IV)
        formatted_data['punish_prob'].append(policy['Punish'])

    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.IV = pd.Categorical(formatted_df.IV,
                                     categories=['No-gain', 'Small-gain'],
                                     ordered=True)
    formatted_df.condition = pd.Categorical(formatted_df.condition,
                                     categories=['Base punisher', 'Low audience value'],
                                     ordered=True)
    return formatted_df
