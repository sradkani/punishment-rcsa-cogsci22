
import numpy as np
import pandas as pd

def preprocess(data):
    punish_data = data[data['role']=='TP'].copy(deep=True)
    punish_data['punished'] = punish_data['tp_invest']>0

    """Finds the probability of punishment in different IVs and conditions"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': [], 'punish_prob_se': [],
                      'punish_severity': [], 'punish_severity_se': []}

    # Random/Anonymous condition
    formatted_data['IV'].append('Punishment')
    formatted_data['condition'].append('Private/Anonymous')
    indices = (punish_data['treat']=='Random / Anonymous')
    punish_prob = np.mean(punish_data[indices]['punished'])
    formatted_data['punish_prob'].append(punish_prob)
    formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    indices = (punish_data['treat']=='Random / Anonymous') & (punish_data['punished'])
    formatted_data['punish_severity'].append(np.mean(punish_data[indices]['tp_invest_perc']))
    formatted_data['punish_severity_se'].append(np.std(punish_data[indices]['tp_invest_perc']) / np.sqrt(len(indices)))

    # Random/Knowledge condition
    formatted_data['IV'].append('Punishment')
    formatted_data['condition'].append('Public/Observed')
    indices = (punish_data['treat']=='Random / Knowledge')
    punish_prob = np.mean(punish_data[indices]['punished'])
    formatted_data['punish_prob'].append(punish_prob)
    formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    indices = (punish_data['treat']=='Random / Knowledge') & (punish_data['punished'])
    formatted_data['punish_severity'].append(np.mean(punish_data[indices]['tp_invest_perc']))
    formatted_data['punish_severity_se'].append(np.std(punish_data[indices]['tp_invest_perc']) / np.sqrt(len(indices)))

    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.condition = pd.Categorical(formatted_df.condition,
                                            categories=['Private/Anonymous', 'Public/Observed'],
                                            ordered=True)
    return formatted_df
