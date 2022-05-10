
import numpy as np
from scipy import stats
import itertools

from models import base_punisher_models
from models import typed_audience_models as audience_models
from models import pragmatic_punisher_models
from utils import joint_density

def get_available_actions(key):
    if key=='Both-punishment':
        available_actions = ['Not-punish', 'Punish']
    elif key=='Both-helping':
        available_actions = ['Not-help', 'Help']
    return available_actions

def get_selfish_utilities(key, available_actions):
    if key=='Both-punishment':
        selfish_utilities = {'Not-punish': 0,
                             'Punish': -5}
    elif key=='Both-helping':
        selfish_utilities = {'Not-help': 0,
                             'Help': -20}
    return selfish_utilities

def get_social_utilities(key, available_actions):
    if key=='Both-punishment':
        social_utilities = {'Not-punish': 0,
                            'Punish': 10}
    elif key=='Both-helping':
        social_utilities = {'Not-help': 0,
                            'Help': 10}
    return social_utilities

def get_target_utilities(key, available_actions):
    if key=='Both-punishment':
        target_utilities = {'Not-punish': 0,
                            'Punish': -50}
    elif key=='Both-helping':
        target_utilities = {'Not-help': 0,
                            'Help': 15}
    return target_utilities


def get_alpha_set():
    # In defining alpha_selfish_set and alpha_social_set, the types of agents are also determined as the keys of these two dictionaries
    alpha_selfish_set = {'unselfish': 2, 'selfish': 8}
    alpha_social_set = {'unsocial': 2, 'social': 8}
    alpha_target = 2
    return alpha_selfish_set, alpha_social_set, alpha_target

def get_config(alpha_priors):
    alpha_selfish_set, alpha_social_set, alpha_target = get_alpha_set()

    # config 1 - Agent who helped before and now decides whether to punish
    available_actions1 = get_available_actions('Both-punishment')
    base_punisher_config1 = {
        'constructor': base_punisher_models.BasePunisher,
        'kwargs':{
            'alpha_selfish': 5.0,
            'alpha_social': 5.0,
            'alpha_target': alpha_target,
            'softmax_beta': 0.02,
            'selfish_utilities': get_selfish_utilities('Both-punishment', available_actions1),
            'social_utilities': get_social_utilities('Both-punishment', available_actions1),
            'target_utilities': get_target_utilities('Both-punishment', available_actions1)
        }
    }

    a = 'H'
    audience_alpha_prior1 = alpha_priors['Both - '+ a]

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

    # config 2 - Agent who did not-help before and now decides whether to punish
    available_actions2 = get_available_actions('Both-punishment')
    base_punisher_config2 = {
        'constructor': base_punisher_models.BasePunisher,
        'kwargs':{
            'alpha_selfish': 5.0,
            'alpha_social': 5.0,
            'alpha_target': alpha_target,
            'softmax_beta': 0.02,
            'selfish_utilities': get_selfish_utilities('Both-punishment', available_actions2),
            'social_utilities': get_social_utilities('Both-punishment', available_actions2),
            'target_utilities': get_target_utilities('Both-punishment', available_actions2)
        }
    }

    a = 'NH'
    audience_alpha_prior2 = alpha_priors['Both - '+ a]

    base_audience_config2 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior2,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': base_punisher_config2['constructor'],
            'punisher_model_kwargs': base_punisher_config2['kwargs']
        }
    }

    pragmatic_punisher_config2 = {
        'constructor': pragmatic_punisher_models.PragmaticPunisher,
        'kwargs':{
            'alpha_reputation': 350,
            'base_punisher': base_punisher_config2['constructor'],
            'base_punisher_kwargs': base_punisher_config2['kwargs'],
            'audience': base_audience_config2['constructor'],
            'audience_kwargs': base_audience_config2['kwargs']
        }
    }

    pragmatic_audience_config2 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior2,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': pragmatic_punisher_config2['constructor'],
            'punisher_model_kwargs': pragmatic_punisher_config2['kwargs']
        }
    }

    config2 = {
        'available_actions': available_actions2,
        'alpha_selfish_set': alpha_selfish_set,
        'alpha_social_set': alpha_social_set,
        'alpha_target': alpha_target,
        'base_punisher': base_punisher_config2,
        'base_audience': base_audience_config2,
        'pragmatic_punisher': pragmatic_punisher_config2,
        'pragmatic_audience': pragmatic_audience_config2,
    }

    # config 3 - Agent who punished before and now decides whether to help
    available_actions3 = get_available_actions('Both-helping')
    base_punisher_config3 = {
        'constructor': base_punisher_models.BasePunisher,
        'kwargs':{
            'alpha_selfish': 5.0,
            'alpha_social': 5.0,
            'alpha_target': alpha_target,
            'softmax_beta': 0.02,
            'selfish_utilities': get_selfish_utilities('Both-helping', available_actions3),
            'social_utilities': get_social_utilities('Both-helping', available_actions3),
            'target_utilities': get_target_utilities('Both-helping', available_actions3)
        }
    }

    a = 'P'
    audience_alpha_prior3 = alpha_priors['Both - '+ a]

    base_audience_config3 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior3,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': base_punisher_config3['constructor'],
            'punisher_model_kwargs': base_punisher_config3['kwargs']
        }
    }

    pragmatic_punisher_config3 = {
        'constructor': pragmatic_punisher_models.PragmaticPunisher,
        'kwargs':{
            'alpha_reputation': 350,
            'base_punisher': base_punisher_config3['constructor'],
            'base_punisher_kwargs': base_punisher_config3['kwargs'],
            'audience': base_audience_config3['constructor'],
            'audience_kwargs': base_audience_config3['kwargs']
        }
    }

    pragmatic_audience_config3 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior3,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': pragmatic_punisher_config3['constructor'],
            'punisher_model_kwargs': pragmatic_punisher_config3['kwargs']
        }
    }

    config3 = {
        'available_actions': available_actions3,
        'alpha_selfish_set': alpha_selfish_set,
        'alpha_social_set': alpha_social_set,
        'alpha_target': alpha_target,
        'base_punisher': base_punisher_config3,
        'base_audience': base_audience_config3,
        'pragmatic_punisher': pragmatic_punisher_config3,
        'pragmatic_audience': pragmatic_audience_config3,
    }

    # config 4 - Agent who did not-punish before and now decides whether to help
    available_actions4 = get_available_actions('Both-helping')
    base_punisher_config4 = {
        'constructor': base_punisher_models.BasePunisher,
        'kwargs':{
            'alpha_selfish': 5.0,
            'alpha_social': 5.0,
            'alpha_target': alpha_target,
            'softmax_beta': 0.02,
            'selfish_utilities': get_selfish_utilities('Both-helping', available_actions4),
            'social_utilities': get_social_utilities('Both-helping', available_actions4),
            'target_utilities': get_target_utilities('Both-helping', available_actions4)
        }
    }

    a = 'NP'
    audience_alpha_prior4 = alpha_priors['Both - '+ a]

    base_audience_config4 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior4,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': base_punisher_config4['constructor'],
            'punisher_model_kwargs': base_punisher_config4['kwargs']
        }
    }

    pragmatic_punisher_config4 = {
        'constructor': pragmatic_punisher_models.PragmaticPunisher,
        'kwargs':{
            'alpha_reputation': 350,
            'base_punisher': base_punisher_config4['constructor'],
            'base_punisher_kwargs': base_punisher_config4['kwargs'],
            'audience': base_audience_config4['constructor'],
            'audience_kwargs': base_audience_config4['kwargs']
        }
    }

    pragmatic_audience_config4 = {
        'constructor': audience_models.Audience1,
        'kwargs':{
            'alpha_prior': audience_alpha_prior4,
            'alpha_selfish_set': alpha_selfish_set,
            'alpha_social_set': alpha_social_set,
            'alpha_target': alpha_target,
            'punisher_model': pragmatic_punisher_config4['constructor'],
            'punisher_model_kwargs': pragmatic_punisher_config4['kwargs']
        }
    }

    config4 = {
        'available_actions': available_actions4,
        'alpha_selfish_set': alpha_selfish_set,
        'alpha_social_set': alpha_social_set,
        'alpha_target': alpha_target,
        'base_punisher': base_punisher_config4,
        'base_audience': base_audience_config4,
        'pragmatic_punisher': pragmatic_punisher_config4,
        'pragmatic_audience': pragmatic_audience_config4,
    }

    config = {'Both - H': config1,
              'Both - NH': config2,
              'Both - P': config3,
              'Both - NP': config4}

    return config
