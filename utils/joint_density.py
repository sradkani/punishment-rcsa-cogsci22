
import numpy as np

class ContinuousDistribution():
    def __init__(self):
        pass

    def pdf(self, x):
        # return the joint pdf at point x
        pass

    def marginal_pdf(self, x_i, i):
        # return the marginal pdf of ith variable at point x_i
        pass


class DiscreteDistribution():
    def __init__(self):
        pass

    def pmf(self, x):
        # return the joint pmf at point x
        pass

    def marginal_pmf(self, x_i, i):
        # return the marginal pmf of ith variable at point x_i
        pass

class JointIndependent(ContinuousDistribution):
    def __init__(self,
                 marginals):
        super().__init__()
        self.marginals = marginals      # the list of scipy distribution objects for each dimension of the multidimensional RV

    def pdf(self, x):
        pdf = 1
        for x_i, marginal_i in zip(x, self.marginals):
            pdf = pdf * marginal_i.pdf(x_i)

        return pdf

    def marginal_pdf(self, x_i, i):
        return self.marginals[i].pdf(x_i)


class DistFromEstimate(ContinuousDistribution):
    def __init__(self,
                 pdf_values,
                 domain):
        super().__init__()
        self.pdf_values = pdf_values
        self.domain = domain

    def pdf(self, x):
        x_indices = []
        for i, x_i in enumerate(x):
            x_indices.append(np.where(np.array(self.domain[i])==x_i)[0][0])

        pdf = np.array(self.pdf_values)[tuple(x_indices)]
        return pdf


class JointDiscrete(DiscreteDistribution):
    def __init__(self,
                 pmf_values,
                 domain):
        super().__init__()
        self.pmf_values = pmf_values   # dictionary of pmf values for each combination of possible values of random variable
        self.domain = domain

    def pmf(self, x):
        pmf = self.pmf_values[x]
        return pmf
