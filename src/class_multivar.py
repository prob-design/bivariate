import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad


class NormalCopula():
    def __init__(self, rho):
        self.rho = rho
    
    def pdf(self, u, v):
        x = self.X.ppf(u)
        y = self.Y.ppf(v)
        rho = self.rho
        pdf = 1/(2*np.pi*np.sqrt(1-rho**2)) np.exp(-(x**2 - 2*rho*x*y + y**2)/(2*(1-rho**2)))
        return pdf
    
    def cdf(self, u, v):
        x = self.X.ppf(u)
        y = self.Y.ppf(v)
        cdf = dblquad(self.pdf, a=-np.inf, b=y, gfun=-np.inf, hfun=x)[0]
        return cdf
    
class ClaytonCopula():
    def __init__(self, theta):
        self.theta = theta
    
    def pdf(self, u, v):
        theta = self.theta
        pdf = (1+theta)*(u*v)^(-1-teta) * (u^(-theta) + v^(-theta) - 1)^(-1/theta -2) 
        return pdf
    
    def cdf(self, u, v):
        cdf = (u^(-theta) + v^(-theta) - 1)^(-1/theta)
        return cdf


class Multivar():

    def __init__(self, X:list, copulas:dict):
        ''' X: vector of random variables. 
        copula: vector of copulas '''
        self.X = X
        self.copula = bivar_copula(copula)

    def getMarginal(self, i):
        ''' Method to extract marginal distribution of index i from vector X of random variables. '''
        return self.X[i]
    
    def drawMarginalPdf(self, i):
        f, ax = plt.subplot(1)
        X = self.X[i]
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = self.X[i].pdf(x)
        ax.plot(x, y, label="pdf")
        ax.set_title(r"Probability density function of X_{" + str(i) + "}")
        return f, ax

    def drawMarginalCdf(self, i):
        f, ax = plt.subplot(1)
        X = self.X[i]
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = self.X[i].cdf(x)
        ax.plot(x, y, label="cdf")
        ax.set_title(r"Cumulative density function of X_{" + str(i) + "}")
        return f, ax

    @staticmethod
    def bivar_copula(dic):
        ''' dic: dictionary of copulas {"name": parameter}. Example: {"normal":0.5, "clayton":0.3}. '''
        cop = []
        names = dic.keys()
        param = dic.values()
        for i in range(len(names)):
            if names[i] == "normal":
                o = NormalCopula(param[i])   
            elif name[i] == "clayton":
                o = ClaytonCopula(param[i])
            else:
                RaiseExceptions   # Raise Value exception "not acceptable copula name"
            cop.append(o)
        return o
    
    def bivar_pdf(self, marg_1:int, x_1, marg_2:int, x_2):
        if marg_1==0 and marg_2==1:
            copula = self.copula[0]
            return copula.pdf(self.X[0].cdf(x_1), self.X[1].cdf(x_2))
        elif marg_1==1 and marg_2==2:
            copula = self.copula[1]
            return copula.pdf(self.X[1].cdf(x_1), self.X[2].cdf(x_2))
        else:
            pass        # Add calculation for X0 and X2 since we have no copula

    def bivar_cdf(self, marg_1:int, x_1, marg_2:int, x_2):
        if marg_1==0 and marg_2==1:
            copula = self.copula[0]
            return copula.cdf(self.X[0].cdf(x_1), self.X[1].cdf(x_2))
        elif marg_1==1 and marg_2==2:
            copula = self.copula[1]
            return copula.cdf(self.X[1].cdf(x_1), self.X[2].cdf(x_2))
        else:
            pass        # Add calculation for X0 and X2 since we have no copula

    def bivariate_plot(self, marg_1, marg_2, func):
        X = self.X[marg_1]
        Y = self.X[marg_2]

        f, ax = plt.subplot(1)
        # add LSF plot
        # add contour plot
        ax.set_xlim(X.ppf(0.01), X.ppf(0.99))
        ax.set_ylim(Y.ppf(0.01), Y.ppf(0.99))
        ax.set_aspect("scaled")
        return f, ax