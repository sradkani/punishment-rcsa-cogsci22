
import numpy as np
import pandas as pd

def preprocess(data):
    # take only the third-party punisher data
    punish_data = data[data['role']=='TP']

    """Finds the probability of punishing with certain severities as different IVs in different conditions"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': [], 'punish_prob_se': []}


    def format_data(data, IV, condition, treatment, low_bound, up_bound):
        formatted_data['IV'].append(IV)
        formatted_data['condition'].append(condition)
        total_indices = (punish_data['treat']==treatment)
        indices = (punish_data['treat']==treatment) & (low_bound<=punish_data['tp_invest_perc']) & (punish_data['tp_invest_perc']<up_bound)
        punish_prob = np.sum(indices) / np.sum(total_indices)
        formatted_data['punish_prob'].append(punish_prob)
        formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / np.sum(total_indices)))

    #### Random/Anonymous condition
    boundaries = range(0, 101, 10)
    for i in range(len(boundaries)-1):
        format_data(punish_data, f"{boundaries[i]}-{boundaries[i+1]}", 'Private/Anonymous', 'Random / Anonymous', boundaries[i], boundaries[i+1])

    #### Random/Knowledge condition
    boundaries = range(0, 101, 10)
    for i in range(len(boundaries)-1):
        format_data(punish_data, f"{boundaries[i]}-{boundaries[i+1]}", 'Public/Observed', 'Random / Knowledge', boundaries[i], boundaries[i+1])

    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.condition = pd.Categorical(formatted_df.condition,
                                            categories=['Private/Anonymous', 'Public/Observed'],
                                            ordered=True)
    IVs_ordered = [f"{boundaries[i]}-{boundaries[i+1]}" for i in range(len(boundaries)-1)]
    formatted_df.IV = pd.Categorical(formatted_df.IV,
                                            categories=IVs_ordered,
                                            ordered=True)
    return formatted_df
