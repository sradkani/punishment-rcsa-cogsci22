
import json
import numpy as np
import matplotlib.pyplot as plt

def _simulate_base_punisher(base_punisher, keys, params):
    base_punisher_policy = {}
    for key in keys:
        base_punisher_policy[key] = {a: {} for a in params['available_actions'][key]}
        for alpha_selfish in params['alpha_selfish_set'][key]:
            for alpha_social in params['alpha_social_set'][key]:
                base_punisher[key].set_alpha_selfish(alpha_selfish)
                base_punisher[key].set_alpha_social(alpha_social)
                policy = base_punisher[key].policy(params['available_actions'][key])
                for action in params['available_actions'][key]:
                    base_punisher_policy[key][action][(alpha_selfish, alpha_social)] = policy[action]

    return base_punisher_policy


def _simulate_pragmatic_punisher(pragmatic_punisher, keys, params, alpha_reputation):
    pragmatic_punisher_policy = {}
    for key in keys:
        pragmatic_punisher_policy[key] = {a: {} for a in params['available_actions'][key]}
        for alpha_selfish in params['alpha_selfish_set'][key]:
            for alpha_social in params['alpha_social_set'][key]:
                pragmatic_punisher[key].set_alpha_selfish(alpha_selfish)
                pragmatic_punisher[key].set_alpha_social(alpha_social)
                pragmatic_punisher[key].set_alpha_reputation(alpha_reputation[key])
                policy = pragmatic_punisher[key].policy(params['available_actions'][key])
                for action in params['available_actions'][key]:
                    pragmatic_punisher_policy[key][action][(alpha_selfish, alpha_social)] = policy[action]

    return pragmatic_punisher_policy

def _simulate_internal_model_of_audience(pragmatic_punisher, keys, params):
    average_audience_judgement = {}
    for key in keys:
        tmp = {a: 0 for a in params['available_actions'][key]}
        for alpha_selfish in params['alpha_selfish_set'][key]:
            for alpha_social in params['alpha_social_set'][key]:
                pragmatic_punisher[key].set_alpha_selfish(alpha_selfish)
                pragmatic_punisher[key].set_alpha_social(alpha_social)
                prevalence = params['alpha_prior'][(alpha_selfish, alpha_social)]
                audience_judgement = pragmatic_punisher[key].simulate_audience_judgement(params['available_actions'][key])
                for a in params['available_actions'][key]:
                    tmp[a] += prevalence * audience_judgement[a]
        average_audience_judgement[key] = tmp
    return average_audience_judgement

def _find_population_average(policy, keys, params):
    average_policy = {}
    for key in keys:
        average_policy[key] = {a: 0 for a in params['available_actions'][key]}
        for action in params['available_actions'][key]:
            for alpha_selfish in params['alpha_selfish_set'][key]:
                for alpha_social in params['alpha_social_set'][key]:
                    prevalence = params['alpha_prior'][(alpha_selfish, alpha_social)]   # the prevalence of a base punisher with (alpha_selfish, alpha_social) in the population
                    average_policy[key][action] += prevalence * policy[key][action][(alpha_selfish, alpha_social)]
    return average_policy

def simulate(config_name, config, params, save=True):
    # instantiate the base punisher
    base_punisher = {}
    for key in config.keys():
        base_punisher[key] = config[key]['base_punisher']['constructor'](**config[key]['base_punisher']['kwargs'])

    # simulate the base punishers to obtain their policy as a function of alpha_selfish and alpha_social
    base_punisher_policy = _simulate_base_punisher(base_punisher, config.keys(), params)

    # instantiate the pragmatic punisher
    pragmatic_punisher = {}
    alpha_reputations_high = {}
    alpha_reputations_low = {}
    for key in config.keys():
        pragmatic_punisher[key] = config[key]['pragmatic_punisher']['constructor'](**config[key]['pragmatic_punisher']['kwargs'])
        # the pragmatic punisher with high audience value uses the alpha_reputation set in the config file
        alpha_reputations_high[key] = pragmatic_punisher[key].alpha_reputation
        # the pragmatic punisher with low audience value uses an alpha_reputation equal to ~0.7 of the alpha_reputation set in the config file
        # note that the extra 0.0143 is for rounding the numbers for the paper, and it does not make any significant difference in the results.
        alpha_reputations_low[key] = alpha_reputations_high[key] * 0.7143
        print(f"{key}: alpha_reputation_high = {alpha_reputations_high[key]}")
        print(f"{key}: alpha_reputation_low = {alpha_reputations_low[key]}")

    # simulate the pragmatic punishers to obtain their policy as a function of alpha_selfish and alpha_social
    pragmatic_punisher_policy_high = _simulate_pragmatic_punisher(pragmatic_punisher, config.keys(), params, alpha_reputations_high)
    pragmatic_punisher_policy_low = _simulate_pragmatic_punisher(pragmatic_punisher, config.keys(), params, alpha_reputations_low)

    # find the population average behavior
    average_base_punisher_policy = _find_population_average(base_punisher_policy, config.keys(), params)
    average_pragmatic_punisher_policy_high = _find_population_average(pragmatic_punisher_policy_high, config.keys(), params)
    average_pragmatic_punisher_policy_low = _find_population_average(pragmatic_punisher_policy_low, config.keys(), params)

    # simulate the audience judgement in the mind of pragmatic punishers
    audience_judgement = _simulate_internal_model_of_audience(pragmatic_punisher, config.keys(), params)

    model_data = {'average_base_punisher_policy': average_base_punisher_policy,
                  'average_pragmatic_punisher_policy_high': average_pragmatic_punisher_policy_high,
                  'average_pragmatic_punisher_policy_low': average_pragmatic_punisher_policy_low,
                  'audience_judgement': audience_judgement}

    if save:
        # save the simulated data in the 'model_data' folder
        tmp = config_name.split(".")
        file_dir = ""
        for i in tmp[:-1]:
        	file_dir = file_dir + i + "/"
        file_dir = file_dir + tmp[-1]
        print(file_dir)
        with open(f"model_data/{file_dir}.json", "w") as outfile:
            json.dump(model_data, outfile)

    return model_data
