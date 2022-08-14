class FitResults():
    def __init__(self):

        self.distributions = {}


    def add_distribution(self, distribution, parameters, aic, bic):
        self.distributions[distribution] = {'parameters': parameters,
                                            'aic': aic,
                                            'bic': bic}


    def distributions_fitted(self):
        return list(self.distributions.keys())


    def distribution_parameters(self, distribution):
        return self.distributions[distribution]['parameters']


    def fit_stats(self, distribution, statistic='all'):
        if isinstance(statistic, str):
            statistic = [statistic]

        return [self.distributions[distribution][_stat] for _stat in statistic] 