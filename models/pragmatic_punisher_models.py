
import numpy as np
import collections

class PragmaticPunisher():
    def __init__(self,
                 alpha_reputation,
                 base_punisher,
                 base_punisher_kwargs,
                 audience,
                 audience_kwargs):
        self.base_punisher = base_punisher(**base_punisher_kwargs)
        self.audience = audience(**audience_kwargs)
        self.alpha_reputation = alpha_reputation
        self.P_legitimate = None       # judgements of audience about illegitimacy of punisher's actions

    def simulate_audience_judgement(self, available_actions=None):
        if available_actions is None:
            self.P_legitimate = self.audience.judge_legitimacy(self.available_actions)
        else:
            self.P_legitimate = self.audience.judge_legitimacy(available_actions)
        return self.P_legitimate

    def get_utility(self, action):
        # TODO: think more about this to see whether it's ok to find P_legitimate only once
        # and use it for whatever function calls to get_utility of pragmatic punisher with
        # whatever pragmatism depth
        if self.P_legitimate is None:
            # if the pragmatic punisher has never calculated this before
            self.simulate_audience_judgement()
        else:
            # or the pragmatic punisher now wants to calculate this for another set of available actions
            if collections.Counter(list(self.P_legitimate.keys())) != collections.Counter(self.available_actions):
                self.simulate_audience_judgement()
        U_base_punisher = self.base_punisher.get_utility(action)
        U_reputation = self.P_legitimate[action]
        U_total = U_base_punisher + self.alpha_reputation * U_reputation
        return U_total

    def policy(self, available_actions):
        self.available_actions = available_actions
        policy = {a: None for a in available_actions}
        for action in available_actions:
            U_total = self.get_utility(action)
            policy[action] = np.exp(self.get_softmax_beta() * U_total)

        normalized_policy = {a: policy[a]/np.sum(list(policy.values())) for a in available_actions}
        return normalized_policy

    def decide(self, available_actions):
        policy = self.policy(available_actions)
        current_action_idx = np.argwhere(np.random.multinomial(1, policy.value()))[0][0]
        current_action = available_actions[current_action_idx]
        return current_action

    def set_alpha_selfish(self, alpha_selfish):
        # this is a recursive function that calls the set_alpha_selfish on the tree
        # of base_punishers until it hits the punisher agent that has the alpha_selfish in it
        self.base_punisher.set_alpha_selfish(alpha_selfish)

    def set_alpha_social(self, alpha_social):
        # this is a recursive function that calls the set_alpha_social on the tree
        # of base_punishers until it hits the punisher agent that has the alpha_social in it
        self.base_punisher.set_alpha_social(alpha_social)

    def set_alpha_target(self, alpha_target):
        # this is a recursive function that calls the set_alpha_target on the tree
        # of base_punishers until it hits the punisher agent that has the alpha_target in it
        self.base_punisher.set_alpha_target(alpha_target)

    def set_alpha_reputation(self, alpha_reputation):
        self.alpha_reputation = alpha_reputation

    def get_softmax_beta(self):
        return self.base_punisher.get_softmax_beta()

    def get_alpha_selfish(self):
        return self.base_punisher.get_alpha_selfish()

    def get_alpha_social(self):
        return self.base_punisher.get_alpha_social()

    def get_alpha_target(self):
        return self.base_punisher.get_alpha_target()
