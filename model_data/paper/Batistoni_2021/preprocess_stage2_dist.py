
import numpy as np
import pandas as pd

def preprocess(data):
    """Finds the probability of punishment in different IVs"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': []}

    # # Base punisher
    # for IV, policy in data['average_base_punisher_policy'].items():
    #     for severity, prob in policy.items():
    #         formatted_data['condition'].append('Base punisher')
    #         formatted_data['IV'].append(severity)
    #         formatted_data['punish_prob'].append(prob)

    # Pragmatic punisher - low
    for IV, policy in data['average_pragmatic_punisher_policy_low'].items():
        for severity, prob in policy.items():
            formatted_data['condition'].append('Low audience value')
            formatted_data['IV'].append(severity)
            formatted_data['punish_prob'].append(prob)

    # Pragmatic punisher - high
    for IV, policy in data['average_pragmatic_punisher_policy_high'].items():
        for severity, prob in policy.items():
            formatted_data['condition'].append('High audience value')
            formatted_data['IV'].append(severity)
            formatted_data['punish_prob'].append(prob)

    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.condition = pd.Categorical(formatted_df.condition,
                                    #  categories=['Base punisher', 'Pragmatic punisher - Low', 'Pragmatic punisher - High'],
                                    categories=['Low audience value', 'High audience value'],
                                     ordered=True)
    return formatted_df
