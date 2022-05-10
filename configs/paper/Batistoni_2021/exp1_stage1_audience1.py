
import numpy as np
from scipy import stats
import itertools

from models import base_punisher_models
from models import typed_audience_models as audience_models
from models import pragmatic_punisher_models
from utils import joint_density


import numpy as np
from scipy import stats
import itertools

from models import base_punisher_models
from models import typed_audience_models as audience_models
from models import pragmatic_punisher_models
from utils import joint_density

def get_available_actions(key):
    if key=='Punisher':
        available_actions = ['Not-punish', 'Punish']
    return available_actions

def get_selfish_utilities(key, available_actions):
    if key=='Punisher':
        selfish_utilities = {'Not-punish': 0,
                             'Punish': -25}
    return selfish_utilities

def get_social_utilities(key, available_actions):
    if key=='Punisher':
        social_utilities = {'Not-punish': 0,
                            'Punish': 5.5}
    return social_utilities

def get_target_utilities(key, available_actions):
    if key=='Punisher':
        target_utilities = {'Not-punish': 0,
                            'Punish': -55}
    return target_utilities


def get_alpha_set():
    # In defining alpha_selfish_set and alpha_social_set, the types of agents are also determined as the keys of these two dictionaries
    alpha_selfish_set = {'unselfish': 2, 'selfish': 8}
    alpha_social_set = {'unsocial': 2, 'social': 8}
    alpha_target = 2
    return alpha_selfish_set, alpha_social_set, alpha_target

def get_config():
    alpha_selfish_set, alpha_social_set, alpha_target = get_alpha_set()
    permut = itertools.permutations(list(alpha_selfish_set.keys()), len(alpha_social_set))
    all_types_combinations = []
    # zip() is called to pair each permutation
    # and shorter list element into combination
    for comb in permut:
        zipped = zip(comb, (alpha_social_set.keys()))
        all_types_combinations.append(list(zipped))
    all_types_combinations = list(itertools.chain.from_iterable(all_types_combinations))

    # config 1
    available_actions1 = get_available_actions('Punisher')
    base_punisher_config1 = {
        'constructor': base_punisher_models.BasePunisher,
        'kwargs':{
            'alpha_selfish': 5.0,
            'alpha_social': 5.0,
            'alpha_target': alpha_target,
            'softmax_beta': 0.02,
            'selfish_utilities': get_selfish_utilities('Punisher', available_actions1),
            'social_utilities': get_social_utilities('Punisher', available_actions1),
            'target_utilities': get_target_utilities('Punisher', available_actions1)
        }
    }

    audience_alpha_prior1 = joint_density.JointDiscrete(pmf_values= {type: 1/(len(alpha_selfish_set) * len(alpha_social_set))    # uniform prior
                                                                     for type in all_types_combinations},
                                                        domain=[list(alpha_selfish_set.keys()), list(alpha_social_set.keys())])

    base_audience_config1 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior1,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': base_punisher_config1['constructor'],
            'punisher_model_kwargs': base_punisher_config1['kwargs']
        }
    }

    pragmatic_punisher_config1 = {
        'constructor': pragmatic_punisher_models.PragmaticPunisher,
        'kwargs':{
            'alpha_reputation': 350,
            'base_punisher': base_punisher_config1['constructor'],
            'base_punisher_kwargs': base_punisher_config1['kwargs'],
            'audience': base_audience_config1['constructor'],
            'audience_kwargs': base_audience_config1['kwargs']
        }
    }

    pragmatic_audience_config1 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior1,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': pragmatic_punisher_config1['constructor'],
            'punisher_model_kwargs': pragmatic_punisher_config1['kwargs']
        }
    }

    config1 = {
        'available_actions': available_actions1,
        'alpha_selfish_set': alpha_selfish_set,
        'alpha_social_set': alpha_social_set,
        'alpha_target': alpha_target,
        'base_punisher': base_punisher_config1,
        'base_audience': base_audience_config1,
        'pragmatic_punisher': pragmatic_punisher_config1,
        'pragmatic_audience': pragmatic_audience_config1,
    }

    config = {'Punisher': config1}

    return config
