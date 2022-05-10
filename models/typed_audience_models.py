
import numpy as np

class Audience():
    def __init__(self,
                 alpha_prior,
                 alpha_selfish_set,
                 alpha_social_set,
                 alpha_target,
                 punisher_model,
                 punisher_model_kwargs={}):
        self.alpha_prior = alpha_prior
        self.punisher_model = punisher_model
        self.punisher_model_kwargs = punisher_model_kwargs
        self.alpha_selfish_set = alpha_selfish_set
        self.alpha_social_set = alpha_social_set
        self.alpha_target = alpha_target    # TODO: makes this general, so that we can have different type_targets with different alpha_targets as well

    def infer_posterior(self, available_actions):
        punisher_model = self.punisher_model(**self.punisher_model_kwargs)
        punisher_policies = {(type_selfish, type_social): None
                             for type_selfish, type_social in zip(self.alpha_selfish_set.keys(), self.alpha_social_set.keys())}
        for type_selfish, alpha_selfish in self.alpha_selfish_set.items():
            for type_social, alpha_social in self.alpha_social_set.items():
                punisher_model.set_alpha_selfish(alpha_selfish)
                punisher_model.set_alpha_social(alpha_social)
                punisher_model.set_alpha_target(self.alpha_target)
                policy = punisher_model.policy(available_actions)
                punisher_policies[(type_selfish, type_social)] = policy

        alpha_posterior = {action: None for action in available_actions}
        for action in available_actions:
            tmp = {}
            for type_selfish, alpha_selfish in self.alpha_selfish_set.items():
                for type_social, alpha_social in self.alpha_social_set.items():
                    if type(self.alpha_prior) is dict:
                        prior = self.alpha_prior[action].pmf((type_selfish, type_social))
                    else:
                        prior = self.alpha_prior.pmf((type_selfish, type_social))
                    likelihood = punisher_policies[(type_selfish, type_social)][action]
                    # for each action, alpha_posterior contains a dictionary of poterior values,
                    # for all the possible combinations of agent types
                    tmp[(type_selfish, type_social)] = likelihood * prior
            # normalize the posterior
            alpha_posterior[action] = {key: posterior/np.sum(list(tmp.values())) for key, posterior in tmp.items()}

        return alpha_posterior

    def judge_legitimacy(self, available_actions, all_alpha0=False):
        pass


class Audience1(Audience):
    """
    This audience model considers P(unselfish) as the definition/degree of legitimacy
    """
    def __init__(self,
                 alpha_prior,
                 alpha_selfish_set,
                 alpha_social_set,
                 alpha_target,
                 punisher_model,
                 punisher_model_kwargs):
        super().__init__(alpha_prior,
                         alpha_selfish_set,
                         alpha_social_set,
                         alpha_target,
                         punisher_model,
                         punisher_model_kwargs)

    def judge_selfishness(self, available_actions):
        alpha_posterior = self.infer_posterior(available_actions)
        P_unselfish = {action: None for action in available_actions}
        for action in available_actions:
            tmp = 0
            for type_social in self.alpha_social_set.keys():
                tmp = tmp + alpha_posterior[action][('unselfish', type_social)]
            P_unselfish[action] = tmp
        return P_unselfish

    def judge_legitimacy(self, available_actions, visualization=False):
        if visualization:
            P_unselfish = self.judge_selfishness(available_actions)
            P_legitimate = P_unselfish
            legitimacy_definition = "P(unselfish)"
            return P_legitimate, legitimacy_definition
        else:
            P_unselfish = self.judge_selfishness(available_actions)
            P_legitimate = P_unselfish
            return P_legitimate


'''
The following audience models are not used in the paper. However, they demonstrate
how the RCSA model introduced in the paper can be extended to include other
desired impressions as well (as discussed in the 'Discussion' section of the paper). 
'''

class Audience2(Audience):
    """
    This audience model considers P(social) as the definition/degree of legitimacy
    """
    def __init__(self,
                 alpha_prior,
                 alpha_selfish_set,
                 alpha_social_set,
                 alpha_target,
                 punisher_model,
                 punisher_model_kwargs):
        super().__init__(alpha_prior,
                         alpha_selfish_set,
                         alpha_social_set,
                         alpha_target,
                         punisher_model,
                         punisher_model_kwargs)

    def judge_socialness(self, available_actions):
        alpha_posterior = self.infer_posterior(available_actions)
        P_social = {action: None for action in available_actions}
        for action in available_actions:
            tmp = 0
            for type_selfish in self.alpha_selfish_set.keys():
                tmp = tmp + alpha_posterior[action][(type_selfish, 'social')]
            P_social[action] = tmp
        return P_social

    def judge_legitimacy(self, available_actions, visualization=False):
        if visualization:
            P_social = self.judge_socialness(available_actions)
            P_legitimate = P_social
            legitimacy_definition = "P(social)"
            return P_legitimate, legitimacy_definition
        else:
            P_social = self.judge_socialness(available_actions)
            P_legitimate = P_social
            return P_legitimate

class Audience3(Audience):
    """
    This audience model considers P(unselfish, social) as the definition/degree of illegitimacy
    """
    def __init__(self,
                 alpha_prior,
                 alpha_selfish_set,
                 alpha_social_set,
                 alpha_target,
                 punisher_model,
                 punisher_model_kwargs):
        super().__init__(alpha_prior,
                         alpha_selfish_set,
                         alpha_social_set,
                         alpha_target,
                         punisher_model,
                         punisher_model_kwargs)

    def judge_selfishness_socialness(self, available_actions):
        alpha_posterior = self.infer_posterior(available_actions)
        P_unselfish_social = {action: None for action in available_actions}
        for action in available_actions:
            P_unselfish_social[action] = alpha_posterior[action][('unselfish', 'social')]
        return P_unselfish_social

    def judge_legitimacy(self, available_actions, visualization=False):
        if visualization:
            P_unselfish_social = self.judge_selfishness_socialness(available_actions)
            P_legitimate = P_unselfish_social
            legitimacy_definition = "P(unselfish, social)"
            return P_legitimate, legitimacy_definition
        else:
            P_unselfish_social = self.judge_selfishness_socialness(available_actions)
            P_legitimate = P_unselfish_social
            return P_legitimate

class Audience4(Audience):
    """
    This audience model considers [P(unselfish) + P(social)]/2 as the definition/degree of legitimacy
    """
    def __init__(self,
                 alpha_prior,
                 alpha_selfish_set,
                 alpha_social_set,
                 alpha_target,
                 punisher_model,
                 punisher_model_kwargs):
        super().__init__(alpha_prior,
                         alpha_selfish_set,
                         alpha_social_set,
                         alpha_target,
                         punisher_model,
                         punisher_model_kwargs)

    def judge_selfishness_socialness(self, available_actions):
        alpha_posterior = self.infer_posterior(available_actions)
        P_unselfish_social = {action: None for action in available_actions}
        for action in available_actions:
            # unselfish
            tmp = 0
            for type_social in self.alpha_social_set.keys():
                tmp = tmp + alpha_posterior[action][('unselfish', type_social)]
            P_unselfish = tmp
            # social
            tmp = 0
            for type_selfish in self.alpha_selfish_set.keys():
                tmp = tmp + alpha_posterior[action][(type_selfish, 'social')]
            P_social = tmp
            P_unselfish_social[action] = (P_unselfish + P_social) / 2
        return P_unselfish_social

    def judge_legitimacy(self, available_actions, visualization=False):
        if visualization:
            P_unselfish_social = self.judge_selfishness_socialness(available_actions)
            P_legitimate = P_unselfish_social
            legitimacy_definition = "[P(unselfish) + P(social)]/2"
            return P_legitimate, legitimacy_definition
        else:
            P_unselfish_social = self.judge_selfishness_socialness(available_actions)
            P_legitimate = P_unselfish_social
            return P_legitimate
