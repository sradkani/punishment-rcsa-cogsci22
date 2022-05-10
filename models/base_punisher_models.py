
import numpy as np

class BasePunisher():
    def __init__(self,
                 alpha_selfish,
                 alpha_social,
                 alpha_target,
                 softmax_beta,
                 selfish_utilities,
                 social_utilities,
                 target_utilities):
        self.softmax_beta = softmax_beta
        self.alpha_selfish = alpha_selfish
        self.alpha_social = alpha_social
        self.alpha_target = alpha_target
        self.selfish_utilities = selfish_utilities
        self.social_utilities = social_utilities
        self.target_utilities = target_utilities

    def get_utility(self, action):
        U_selfish = self.selfish_utilities[action]
        U_social = self.social_utilities[action]
        U_target = self.target_utilities[action]
        U_total = self.alpha_selfish * U_selfish + self.alpha_social * U_social + self.alpha_target * U_target
        return U_total

    def policy(self, available_actions):
        policy = {a: None for a in available_actions}
        for action in available_actions:
            U_total = self.get_utility(action)
            policy[action] = np.exp(self.softmax_beta * U_total)

        normalized_policy = {a: policy[a]/np.sum(list(policy.values())) for a in available_actions}
        return normalized_policy

    def decide(self, available_actions):
        policy = self.policy(available_actions)
        current_action_idx = np.argwhere(np.random.multinomial(1, policy.value()))[0][0]
        current_action = available_actions[current_action_idx]
        return current_action

    def set_alpha_selfish(self, alpha_selfish):
        self.alpha_selfish = alpha_selfish

    def set_alpha_social(self, alpha_social):
        self.alpha_social = alpha_social
    
    def set_alpha_target(self, alpha_target):
        self.alpha_target = alpha_target

    def get_softmax_beta(self):
        return self.softmax_beta

    def get_alpha_selfish(self):
        return self.alpha_selfish

    def get_alpha_social(self):
        return self.alpha_social

    def get_alpha_target(self):
        return self.alpha_target
