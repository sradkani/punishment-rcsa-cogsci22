
import numpy as np
import pandas as pd

def preprocess(all_data):
    """Finds the probability of punishment in different IVs"""
    formatted_data = {'IV': [], 'condition': [], 'punish_prob': [], 'punish_prob_se': []}

    # take only the data from experiment 9 and 10
    indices = (all_data['study']==9) | (all_data['study']==10)
    data = all_data[indices]

    ###### punishment-only IV ########
    # Unspecified (i.e., no TG)
    formatted_data['IV'].append('Punishment-only')
    formatted_data['condition'].append('Private/Unspecified')
    indices = (data['punonly']==1) & (data['TG']==0)
    punish_prob = np.mean(data[indices]['punish'])
    formatted_data['punish_prob'].append(punish_prob)
    formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    # formatted_data['punish_prob_se'].append(np.std(data[indices]['punish']) / np.sqrt(len(indices)))

    # Observed (i.e., TG)
    formatted_data['IV'].append('Punishment-only')
    formatted_data['condition'].append('Observed')
    indices = (data['punonly']==1) & (data['TG']==1)
    punish_prob = np.mean(data[indices]['punish'])
    formatted_data['punish_prob'].append(punish_prob)
    formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    # formatted_data['punish_prob_se'].append(np.std(data[indices]['punish']) / np.sqrt(len(indices)))

    ####### helping+punishment IV ########
    # Unspecified (i.e., no TG)
    formatted_data['IV'].append('Helping+Punishment')
    formatted_data['condition'].append('Private/Unspecified')
    indices = (data['condition']=='Punishment + Helping') & (data['helpfirst']==1) & (data['TG']==0)
    punish_prob = np.mean(data[indices]['punish'])
    formatted_data['punish_prob'].append(punish_prob)
    formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    # formatted_data['punish_prob_se'].append(np.std(data[indices]['punish']) / np.sqrt(len(indices)))

    # Observed (i.e., TG)
    formatted_data['IV'].append('Helping+Punishment')
    formatted_data['condition'].append('Observed')
    indices = (data['condition']=='Punishment + Helping') & (data['helpfirst']==1) & (data['TG']==1)
    punish_prob = np.mean(data[indices]['punish'])
    formatted_data['punish_prob'].append(punish_prob)
    formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    # formatted_data['punish_prob_se'].append(np.std(data[indices]['punish']) / np.sqrt(len(indices)))


    # # punishment+helping IV
    # formatted_data['IV'].append('Punishment+Helping')
    # indices = (data['condition']=='both') & (data['helpfirst']==0)
    # punish_prob = np.mean(data[indices]['punish'])
    # formatted_data['punish_prob'].append(punish_prob)
    # formatted_data['punish_prob_se'].append(np.sqrt(punish_prob * (1-punish_prob) / len(indices)))
    # # formatted_data['punish_prob_se'].append(np.std(data[indices]['punish']) / np.sqrt(len(indices)))

    formatted_df = pd.DataFrame(formatted_data)
    formatted_df.IV = pd.Categorical(formatted_df.IV,
                                            categories=['Punishment-only', 'Helping+Punishment'],#, 'Punishment+Helping'],
                                            ordered=True)

    return formatted_df
