import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

# from scipy.integrate import dblquad
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal as mvn

import pyvinecopulib as pyc      # Used for sampling: maybe used manually defined classes in the future

from functools import  singledispatch, update_wrapper

class NormalCopula():
    def __init__(self, rho):
        self.rho = rho
    
    def pdf(self, u, v):
        x = st.norm.ppf(u)
        y = st.norm.ppf(v)
        rho = self.rho
        pdf = 1/(2*np.pi*np.sqrt(1-rho**2)) * np.exp(-(x**2 - 2*rho*x*y + y**2)/(2*(1-rho**2)))
        return pdf
    
    def cdf(self, u, v):
        rho = self.rho
        cdf = mvn.cdf(x=[st.norm.ppf(u), st.norm.ppf(v)], cov=np.array([[1, rho], [rho, 1]]))
        # Can use double integration, although very slow
        # cdf = dblquad(self.pdf, a=0, b=v, gfun=0, hfun=u)[0]
        return cdf
    
class ClaytonCopula():
    def __init__(self, theta):
        self.theta = theta
    
    def pdf(self, u, v):
        theta = self.theta
        pdf = (1+theta)*(u*v)^(-1-theta) * (u^(-theta) + v^(-theta) - 1)^(-1/theta -2) 
        return pdf
    
    def cdf(self, u, v):
        theta = self.theta
        cdf = (u^(-theta) + v^(-theta) - 1)^(-1/theta)
        return cdf
    
class IndependentCopula():
    def __init__(self):
        pass
    
    #def pdf(self, u, v):
    #    pdf = u*v
    #    return pdf
    
    def cdf(self, u, v):
        cdf = u*v
        return cdf

def methdispatch(func):
    ''' 
    Source: https://stackoverflow.com/questions/24601722/how-can-i-use-functools-singledispatch-with-instance-methods 
    For use in pointLSF method (avoids multiple if statements elegantly).
    '''
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[3].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

class Bivariate():
    def __init__(self, X, Y, family, parameter=0):
        ''' copula: string. Either "Normal", "Clayton" or "Independent". '''
        self.X = X
        self.Y = Y
        self.family = family
        if family == "Normal":
            #self.copula = NormalCopula(parameter)
            self.copula = pyc.Bicop(family=pyc.BicopFamily.gaussian, parameters=[parameter])
        elif family == "Clayton":
            #self.copula = ClaytonCopula(parameter)
            self.copula = pyc.Bicop(family=pyc.BicopFamily.clayton, parameters=[parameter])
        elif family == "Independent":
            #self.copula = IndependentCopula()
            self.copula = pyc.Bicop(family=pyc.BicopFamily.indep)
        else:
            raise ValueError("Invalid copula. Please choose between Normal, Clayton or Independent copulae. ")            
    
    def getMarginal(self, i):
            ''' Method to extract marginal distribution of index i from vector X of random variables. '''
            if i==0:
                return self.X
            elif i==1:
                return self.Y
            else:
                raise ValueError("Index out of range. Please select i=0 or i=1.")
        
    def drawMarginalPdf(self, i):
        f, ax = plt.subplots(figsize=(12,8))
        X = self.getMarginal(i)
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = X.pdf(x)
        ax.plot(x, y, label="pdf")
        ax.set_title(r"Probability density function of $X_{" + str(i) + "}$", fontsize=18)
        ax.set_xlabel(r"$X_" + str(i) + "}$")
        ax.set_ylabel(r"f($X_" + str(i) + "})$")
        return f, ax

    def drawMarginalCdf(self, i):
        f, ax = plt.subplots(figsize=(12,8))
        X = self.getMarginal(i)
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = X.cdf(x)
        ax.plot(x, y, label="cdf")
        ax.set_title(r"Cumulative density function of $X_{" + str(i) + "}$", fontsize=18)
        ax.set_xlabel(r"$X_" + str(i) + "}$")
        ax.set_xlabel(r"$F(X_" + str(i) + "})$")
        return f, ax    

    def bivariatePdf(self, x, y):
        ''' 
        Bivariate probability density function.

        Arguments:
        x, y: scalar. Coordinates at which the bivariate PDF is evaluated.
        '''
        X = self.X
        Y = self.Y
        copula = self.copula
        #pdf = copula.pdf(X.cdf(x), Y.cdf(y))
        pdf = float(copula.pdf([[X.cdf(x), Y.cdf(y)]]))
        return pdf
    
    def bivariateCdf(self, x, y):
        '''
        Bivariate cumulative density function.

        Arguments:
        x, y: scalar. Coordinates at which the bivariate CDF is evaluated.
        '''
        X = self.X
        Y = self.Y
        copula = self.copula
        #cdf = copula.cdf(X.cdf(x), Y.cdf(y))
        cdf = float(copula.cdf([[X.cdf(x), Y.cdf(y)]]))
        return cdf
    
    @methdispatch
    def pointLSF(self, myLSF, i, point, start=0):   # For now: supposes failure condition is myLSF() < 0
        ''' points: int, float, list or 1D ndarray. '''
        if i ==0:
            func = lambda y: myLSF([point, y])
            return fsolve(func, start)[0]
        elif i==1:
            func = lambda x: myLSF([x, point])
            return fsolve(func, start)[0]
        else:
            raise ValueError("Index out of range. Please select i=0 or i=1.")
            
    @pointLSF.register(list)
    @pointLSF.register(np.ndarray)
    def _(self, myLSF, i, point, start=0):
        return [self.pointLSF(myLSF, i, x, start) for x in point]   

    def plotLSF(self, myLSF, ax=None, xlim=None, ylim=None, reverse=False):
        
        if reverse:
            X = self.Y
            Y = self.X
        else:
            X = self.X
            Y = self.Y

        if xlim is None:
            if ax is None:
                xlim = (X.ppf(0.01), X.ppf(0.99))
            else: 
                xlim = ax.get_xlim()
        if ylim is None:
            if ax is None:
                ylim = (Y.ppf(0.01), Y.ppf(0.99))
            else:
                ylim = ax.get_ylim()

        if ax is None:
            f, ax = plt.subplots(figsize=(12,8))
        else:
            f = plt.gcf()
            
        x = np.linspace(xlim[0], xlim[1], 1000)
        y = self.pointLSF(myLSF, 0, x, start=(ylim[0]+ylim[1])/2)
        if reverse:
            ax.plot(y, x, label="LSF", color="r")
        else:
            ax.plot(x, y, label="LSF", color="r")
        return f, ax
        
    def plot_contour(self, ax=None, xlim=None, ylim=None, reverse=False, nb_points=200):
        
        if reverse:    
            X = self.Y
            Y = self.X
        else:
            X = self.X
            Y = self.Y  

        if xlim is None:
            if ax is None:
                xlim = (X.ppf(0.01), X.ppf(0.99))
            else: 
                xlim = ax.get_xlim()
        if ylim is None:
            if ax is None:
                ylim = (Y.ppf(0.01), Y.ppf(0.99))
            else:
                ylim = ax.get_ylim()

        if ax is None:
            f, ax = plt.subplots(figsize=(12,8))
        else:
            f = plt.gcf()
            
        x = np.linspace(xlim[0], xlim[1], nb_points).reshape(-1, 1)
        y = np.linspace(ylim[0], ylim[1], nb_points).reshape(-1, 1)
        X,Y = np.meshgrid(x,y)
        pdf = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i,j] = self.bivariatePdf(X[i,j], Y[i,j])

        ax.contour(X, Y, pdf, levels=8, cmap=cm.Blues) 
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")
        return f, ax

    
def sampling_cop(cop1, cop2, cond_cop=pyc.Bicop(), n=10000):
    ''' 
    cop1: copula between variables X0 and X1.
    cop2: copula between variables X1 and X2.
    Assumption: conditional copula (default: independent copula).
    
    ---------------
    See sampling procedure in http://dx.doi.org/10.1016/j.insmatheco.2007.02.001
    '''
    u = np.random.rand(n, 3)
    x0 = u[:,0]
    x1 = cop1.hinv1(np.concatenate((x0.reshape(-1,1), u[:,1].reshape(-1,1)), axis=1))
    a = cop1.hfunc1(np.concatenate((x0.reshape(-1,1), x1.reshape(-1,1)), axis=1))
    b = cond_cop.hinv1(np.concatenate((a.reshape(-1,1), u[:,2].reshape(-1,1)), axis=1))
    x2 = cop2.hinv1(np.concatenate((x1.reshape(-1,1), b.reshape(-1,1)), axis=1))
    #x = np.concatenate((X0.ppf(x0).reshape(-1,1), X1.ppf(x1).reshape(-1,1), X2.ppf(x2).reshape(-1,1)), axis=1)
    x = np.concatenate((x0.reshape(-1,1), x1.reshape(-1,1), x2.reshape(-1,1)), axis=1)
    return x

def fit_copula(x, y, family=pyc.BicopFamily.gaussian):
    fitted_cop = pyc.Bicop(family=family)
    fitted_cop.fit(np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1))
    return fitted_cop


class Multivariate():
    def __init__(self, X:list, copulas:dict):
        ''' X: vector of random variables. 
        copula: vector of copulas including the conditional one. '''
        self.X = X
        self.copulas = copulas        
        self.B1 = Bivariate(X[0], X[1], copulas[0][0], copulas[0][1])
        self.B2 = Bivariate(X[1], X[2], copulas[1][0], copulas[1][1])
        if self.B1.family == "Normal" and self.B2.family == "Normal" and copulas[2][0] == "Normal":
            family=pyc.BicopFamily.gaussian
            x = sampling_cop(self.B1.copula, self.B2.copula, 
                             cond_cop=pyc.Bicop(family=family, parameters=[copulas[2,1]]))
            fitted_copula = fit_copula(x[:,0], x[:,2], family=family)
            self.B3 = Bivariate(X[0], X[2], "Normal", float(fitted_copula.parameters))
        else:
            pass       # Determine which are the particular cases where the copula of X0, X2 is defined

    def getMarginal(self, i):
        ''' Method to extract marginal distribution of index i from vector X of random variables. '''
        return self.X[i]
    
    def drawMarginalPdf(self, i):
        f, ax = plt.subplot(1)
        X = self.X[i]
        x = np.linspace(X.ppf( .01), X.ppf(0.99), 1000)
        y = self.X[i].pdf(x)
        ax.plot(x, y, label="pdf")
        ax.set_ylabel(fr"$X_{{{i}}}$")
        ax.set_title(fr"Probability density function of $X_{{{i}}}$")
        return f, ax

    def drawMarginalCdf(self, i):
        f, ax = plt.subplot(1)
        X = self.X[i]
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = X.cdf(x)
        ax.plot(x, y, label="cdf")
        ax.set_title(fr"Cumulative density function of $X_{{{i}}}$")
        return f, ax

    def bivariate_plot(self, x_index, y_index, myLSF=None):
        X = self.X[x_index]
        Y = self.X[y_index]
        reverse = x_index > y_index

        f, ax = plt.subplot(1)
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)

        if (x_index+y_index)==1:
            c = self.B1
        elif (x_index+y_index)==3:
            c = self.B2
        else:
            c = self.B3
            
        c.plotLSF(myLSF, ax, xlim=(x[0], x[-1]), reverse=reverse)
        c.plot_contour(ax, xlim=(x[0], x[-1]), reverse=reverse)
        
        ax.set_title(fr"Bivariate contours and limit-state function in the plane $(X_{{{x_index}}}, X_{{{y_index}}})$")
        ax.set_xlabel(fr"$X_{{{x_index}}}$")
        ax.set_ylabel(fr"$X_{{{y_index}}}$")
        ax.set_xlim(X.ppf(0.01), X.ppf(0.99))
        ax.set_ylim(Y.ppf(0.01), Y.ppf(0.99))
        ax.set_aspect("scaled")
        return f, ax