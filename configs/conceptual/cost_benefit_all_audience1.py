
import numpy as np
from scipy import stats
import itertools

from models import base_punisher_models
from models import typed_audience_models as audience_models
from models import pragmatic_punisher_models
from utils import joint_density

def get_available_actions(key):
    available_actions = ['Not-punish', 'Punish']
    return available_actions

def get_selfish_utilities(key, available_actions):
    selfish_utilities = {'Not-punish': 0,
                         'Punish': key}
    return selfish_utilities

def get_social_utilities(key, available_actions):
    social_utilities = {'Not-punish': 0,
                        'Punish': 10}
    return social_utilities

def get_target_utilities(key, available_actions):
    target_utilities = {'Not-punish': 0,
                        'Punish': -5}
    return target_utilities

def get_alpha_set():
    # In defining alpha_selfish_set and alpha_social_set, the types of agents are also determined as the keys of these two dictionaries
    alpha_selfish_set = {'unselfish': 2, 'selfish': 8}
    alpha_social_set = {'unsocial': 2, 'social': 8}
    alpha_target = 1
    return alpha_selfish_set, alpha_social_set, alpha_target

def get_config(alpha_reputation):
    alpha_selfish_set, alpha_social_set, alpha_target = get_alpha_set()
    permut = itertools.permutations(list(alpha_selfish_set.keys()), len(alpha_social_set))
    all_types_combinations = []
    # zip() is called to pair each permutation
    # and shorter list element into combination
    for comb in permut:
        zipped = zip(comb, (alpha_social_set.keys()))
        all_types_combinations.append(list(zipped))
    all_types_combinations = list(itertools.chain.from_iterable(all_types_combinations))

    # the axis from high cost to high benefit
    punishment_selfish_utilities = np.arange(-50, 50)

    config = {}
    for U_selfish in punishment_selfish_utilities:
        available_actions = get_available_actions(U_selfish)
        base_punisher_config = {
            'constructor': base_punisher_models.BasePunisher,
            'kwargs':{
                'alpha_selfish': 5.0,
                'alpha_social': 5.0,
                'alpha_target': alpha_target,
                'softmax_beta': 0.02,
                'selfish_utilities': get_selfish_utilities(U_selfish, available_actions),
                'social_utilities': get_social_utilities(U_selfish, available_actions),
                'target_utilities': get_target_utilities(U_selfish, available_actions)
            }
        }

        audience_alpha_prior = joint_density.JointDiscrete(pmf_values= {type: 1/(len(alpha_selfish_set) * len(alpha_social_set))    # uniform prior
                                                                        for type in all_types_combinations},
                                                            domain=[list(alpha_selfish_set.keys()), list(alpha_social_set.keys())])

        base_audience_config = {
            'constructor': audience_models.Audience1,
            'kwargs':{
                'alpha_prior': audience_alpha_prior,
                'alpha_selfish_set': alpha_selfish_set,
                'alpha_social_set': alpha_social_set,
                'alpha_target': alpha_target,
                'punisher_model': base_punisher_config['constructor'],
                'punisher_model_kwargs': base_punisher_config['kwargs']
            }
        }

        pragmatic_punisher_config = {
            'constructor': pragmatic_punisher_models.PragmaticPunisher,
            'kwargs':{
                'alpha_reputation': alpha_reputation,
                'base_punisher': base_punisher_config['constructor'],
                'base_punisher_kwargs': base_punisher_config['kwargs'],
                'audience': base_audience_config['constructor'],
                'audience_kwargs': base_audience_config['kwargs']
            }
        }

        pragmatic_audience_config = {
            'constructor': audience_models.Audience1,
            'kwargs':{
                'alpha_prior': audience_alpha_prior,
                'alpha_selfish_set': alpha_selfish_set,
                'alpha_social_set': alpha_social_set,
                'alpha_target': alpha_target,
                'punisher_model': pragmatic_punisher_config['constructor'],
                'punisher_model_kwargs': pragmatic_punisher_config['kwargs']
            }
        }

        config[U_selfish] = {
            'available_actions': available_actions,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'base_punisher': base_punisher_config,
            'base_audience': base_audience_config,
            'pragmatic_punisher': pragmatic_punisher_config,
            'pragmatic_audience': pragmatic_audience_config,
        }

    return config
