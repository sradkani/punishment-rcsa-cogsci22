
import json
import numpy as np
import importlib
import itertools
from scipy import stats
from tabulate import tabulate
from utils import joint_density

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

def simulate(config_name, config_both_name, save=True):
    config_module = importlib.import_module(f"configs.{config_name}")
    config = config_module.get_config()

    alpha_range = np.arange(0,10,0.1)
    available_actions = {key: config[key]['available_actions'] for key in config.keys()}
    alpha_selfish_set = {key: alpha_range for key in config.keys()}
    alpha_social_set = {key: alpha_range for key in config.keys()}
    alpha_target = {key: config[key]['alpha_target'] for key in config.keys()}
    selfish_utilities = {key: config_module.get_selfish_utilities(key, available_actions[key]) for key in config.keys()}
    social_utilities = {key: config_module.get_social_utilities(key, available_actions[key]) for key in config.keys()}
    target_utilities = {key: config_module.get_target_utilities(key, available_actions[key]) for key in config.keys()}

    population_alpha_prior = joint_density.JointIndependent([stats.expon(scale=1/(1/3)),   # prior over alpha_selfish
                                                  stats.uniform(0, 10)])  # prior over alpha_social

    alpha_prior_discretized = {}
    for alpha_selfish in alpha_range:
        for alpha_social in alpha_range:
            alpha_prior_discretized[(alpha_selfish, alpha_social)] = population_alpha_prior.pdf((alpha_selfish, alpha_social))
    normalizing_factor = np.sum(list(alpha_prior_discretized.values()))
    for alpha_selfish in alpha_range:
        for alpha_social in alpha_range:
            alpha_prior_discretized[(alpha_selfish, alpha_social)] = alpha_prior_discretized[(alpha_selfish, alpha_social)] / normalizing_factor

    params = {'available_actions': available_actions,
              'alpha_selfish_set': alpha_selfish_set,
              'alpha_social_set': alpha_social_set,
              'alpha_target': alpha_target,
              'selfish_utilities': selfish_utilities,
              'social_utilities': social_utilities,
              'target_utilities': target_utilities,
              'alpha_prior': alpha_prior_discretized}

    for j, key in enumerate(config.keys()):
        utilities = [[i+1, action, selfish_utilities[key][action], social_utilities[key][action], target_utilities[key][action]]
                     for i, action in enumerate(available_actions[key])]
        print(f"{key} utilities")
        print (tabulate(utilities, headers=["Action", "U_selfish", "U_social", "U_target"]))
        print("\n")

    # instantiate the base punisher
    base_punisher = {}
    for key in config.keys():
        base_punisher[key] = config[key]['base_punisher']['constructor'](**config[key]['base_punisher']['kwargs'])

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

    # simulate the audience
    base_audience = {}
    for key in config.keys():
        base_audience[key] = config[key]['base_audience']['constructor'](**config[key]['base_audience']['kwargs'])
    alpha_posterior = {}
    for key in config.keys():
        alpha_posterior[key] = base_audience[key].infer_posterior(params['available_actions'][key])

    # change the prior of the audience for the "both" agent who previously has decided whether to help or not
    # separately for helped and not-helped decisions
    key = list(config.keys())[0]
    audience_alpha_prior = {}
    for action in params['available_actions']['Help-only']:
        a = 'H' if action=='Help' else 'NH'
        audience_alpha_prior['Both - '+ a] = joint_density.JointDiscrete(pmf_values=alpha_posterior['Help-only'][action], domain=None)

    # change the prior of the audience for the "both" agent who previously has decided whether to punish or not
    # separately for punished and not-punished decisions
    for action in params['available_actions']['Punishment-only']:
        a = 'P' if action=='Punish' else 'NP'
        audience_alpha_prior['Both - '+ a] = joint_density.JointDiscrete(pmf_values=alpha_posterior['Punishment-only'][action], domain=None)

    config_module = importlib.import_module(f"configs.{config_both_name}")
    config_both = {}
    config_both = config_module.get_config(audience_alpha_prior)

    for key in config_both.keys():
        available_actions[key] = config_both[key]['available_actions']
        alpha_selfish_set[key] = alpha_range
        alpha_social_set[key] = alpha_range
        alpha_target[key] = config_both[key]['alpha_target']
    params = {'available_actions': available_actions,
              'alpha_selfish_set': alpha_selfish_set,
              'alpha_social_set': alpha_social_set,
              'alpha_target': alpha_target,
              'selfish_utilities': selfish_utilities,
              'social_utilities': social_utilities,
              'target_utilities': target_utilities,
              'alpha_prior': alpha_prior_discretized}

    # instantiate the base punishers
    for key in config_both.keys():
        base_punisher[key] = config_both[key]['base_punisher']['constructor'](**config_both[key]['base_punisher']['kwargs'])

    # simulate the base punishers to obtain their policy as a function of alpha_selfish and alpha_social
    base_punisher_policy = _simulate_base_punisher(base_punisher, base_punisher.keys(), params)

    # instantiate the pragmatic punishers
    for key in config_both.keys():
        pragmatic_punisher[key] = config_both[key]['pragmatic_punisher']['constructor'](**config_both[key]['pragmatic_punisher']['kwargs'])
        alpha_reputations_high[key] = pragmatic_punisher[key].alpha_reputation
        alpha_reputations_low[key] = alpha_reputations_high[key] * 0.7143
        print(f"{key}: alpha_reputation_high = {alpha_reputations_high[key]}")
        print(f"{key}: alpha_reputation_low = {alpha_reputations_low[key]}")

    # simulate the pragmatic punishers to obtain their policy as a function of alpha_selfish and alpha_social
    pragmatic_punisher_policy_high = _simulate_pragmatic_punisher(pragmatic_punisher, pragmatic_punisher.keys(), params, alpha_reputations_high)
    pragmatic_punisher_policy_low = _simulate_pragmatic_punisher(pragmatic_punisher, pragmatic_punisher.keys(), params, alpha_reputations_low)

    # find the population average behavior
    average_base_punisher_policy = _find_population_average(base_punisher_policy, base_punisher_policy.keys(), params)
    average_pragmatic_punisher_policy_high = _find_population_average(pragmatic_punisher_policy_high, pragmatic_punisher_policy_high.keys(), params)
    average_pragmatic_punisher_policy_low = _find_population_average(pragmatic_punisher_policy_low, pragmatic_punisher_policy_low.keys(), params)

    # simulate the audience judgement in the mind of pragmatic punishers
    audience_judgement = _simulate_internal_model_of_audience(pragmatic_punisher, pragmatic_punisher.keys(), params)

    model_data = {'average_base_punisher_policy': average_base_punisher_policy,
                  'average_pragmatic_punisher_policy_high': average_pragmatic_punisher_policy_high,
                  'average_pragmatic_punisher_policy_low': average_pragmatic_punisher_policy_low,
                  'audience_judgement': audience_judgement}

    if save:
        tmp = config_name.split(".")
        file_dir = ""
        for i in tmp[:-1]:
        	file_dir = file_dir + i + "/"
        file_dir = file_dir + tmp[-1]
        print(file_dir)
        with open(f"model_data/{file_dir}.json", "w") as outfile:
            json.dump(model_data, outfile)

    return model_data
