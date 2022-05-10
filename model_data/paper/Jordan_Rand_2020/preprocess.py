
import numpy as np
import pandas as pd

def preprocess(data):
    """Finds the probability of punishment in different IVs"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': []}

    ### Punishment-only
    # Base punisher
    # formatted_data['IV'].append('P-only')
    # formatted_data['condition'].append('Base punisher')
    # formatted_data['punish_prob'].append(data['average_base_punisher_policy']['Punishment-only']['Punish'])
    # Low audience value
    formatted_data['IV'].append('P-only')
    formatted_data['condition'].append('Low audience value')
    formatted_data['punish_prob'].append(data['average_pragmatic_punisher_policy_low']['Punishment-only']['Punish'])
    # High audience value
    formatted_data['IV'].append('P-only')
    formatted_data['condition'].append('High audience value')
    formatted_data['punish_prob'].append(data['average_pragmatic_punisher_policy_high']['Punishment-only']['Punish'])

    # # Both - H
    # formatted_data['IV'].append('Both - H')
    # formatted_data['punish_prob'].append(data['average_pragmatic_punisher_policy']['Both - H']['Punish'])
    #
    # # Both - NH
    # formatted_data['IV'].append('Both - NH')
    # formatted_data['punish_prob'].append(data['average_pragmatic_punisher_policy']['Both - NH']['Punish'])

    ### Both - H+P
    # Base punisher
    # formatted_data['IV'].append('H + P')
    # formatted_data['condition'].append('Base punisher')
    # formatted_data['punish_prob'].append(0.8 * data['average_base_punisher_policy']['Both - H']['Punish'] + \
    #                                      0.2 * data['average_base_punisher_policy']['Both - NH']['Punish'])
    # Low audience value
    formatted_data['IV'].append('H + P')
    formatted_data['condition'].append('Low audience value')
    formatted_data['punish_prob'].append(0.65 * data['average_pragmatic_punisher_policy_low']['Both - H']['Punish'] + \
                                         0.35 * data['average_pragmatic_punisher_policy_low']['Both - NH']['Punish'])
    # High audience value
    formatted_data['IV'].append('H + P')
    formatted_data['condition'].append('High audience value')
    formatted_data['punish_prob'].append(0.83 * data['average_pragmatic_punisher_policy_high']['Both - H']['Punish'] + \
                                         0.17 * data['average_pragmatic_punisher_policy_high']['Both - NH']['Punish'])

    formatted_df = pd.DataFrame(formatted_data)
    # formatted_df.IV = pd.Categorical(formatted_df.IV,
    #                                  categories=['P-only', 'H + P'],
    #                                  ordered=True)
    formatted_df.condition = pd.Categorical(formatted_df.condition,
                                     categories=['Low audience value', 'High audience value'],
                                     ordered=True)
    return formatted_df
