import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

from scipy.integrate import dblquad
from scipy.optimize import fsolve

from functools import  singledispatch, update_wrapper
class NormalCopula():
    def __init__(self, rho):
        self.rho = rho
    
    def pdf(self, u, v):
        x = st.ppf(u)
        y = st.ppf(v)
        rho = self.rho
        pdf = 1/(2*np.pi*np.sqrt(1-rho**2)) * np.exp(-(x**2 - 2*rho*x*y + y**2)/(2*(1-rho**2)))
        return pdf
    
    def cdf(self, u, v):
        x = st.ppf(u)
        y = st.ppf(v)
        cdf = dblquad(self.pdf, a=-np.inf, b=y, gfun=-np.inf, hfun=x)[0]
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
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

class Bivariate():
    def __init__(self, X, Y, copula, parameter=0):
        ''' copula: string. Either "Normal", "Clayton" or "Independent". '''
        self.X = X
        self.Y = Y
        if copula == "Normal":
            self.copula = NormalCopula(parameter)
        elif copula == "Clayton":
            self.copula = ClaytonCopula(parameter)
        elif copula == "Independent":
            self.copula = IndependentCopula()
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
        f, ax = plt.subplot(1)
        X = self.getMarginal(i)
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = X.pdf(x)
        ax.plot(x, y, label="pdf")
        ax.set_title(r"Probability density function of X_{" + str(i) + "}")
        return f, ax

    def drawMarginalCdf(self, i):
        f, ax = plt.subplot(1)
        X = self.getMarginal(i)
        x = np.linspace(X.ppf(0.01), X.ppf(0.99), 1000)
        y = X.cdf(x)
        ax.plot(x, y, label="cdf")
        ax.set_title(r"Cumulative density function of X_{" + str(i) + "}")
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
        return copula.pdf(X.cdf(x), Y.cdf(y))
    
    def bivariateCdf(self, x, y):
        '''
        Bivariate cumulative density function.

        Arguments:
        x, y: scalar. Coordinates at which the bivariate CDF is evaluated.
        '''
        X = self.X
        Y = self.Y
        copula = self.copula
        return copula.cdf(X.cdf(x), Y.cdf(y))
    
    @methdispatch
    def pointLSF(self, myLSF, i, points, start=0):   # For now: supposes failure condition is myLSF() < 0
        ''' points: int, float, list or 1D ndarray. '''
        if i ==0:
            func = lambda y: myLSF(points, y)
            return fsolve(func, start)[0]
        elif i==1:
            func = lambda x: myLSF(x, points)
            return fsolve(func, start)[0]
        else:
            raise ValueError("Index out of range. Please select i=0 or i=1.")
            
    @pointLSF.register(list)
    @pointLSF.register(np.ndarray)
    def _(self, myLSF, i, points, start=0):
        return [self.pointLSF(myLSF, i, x, start) for x in points]   

    def plotLSF(self, myLSF, ax=None, xlim=None, ylim=None, reverse=False):
        if ax is None:
            f, ax = plt.subplot(1)
        else:
            f = plt.gcf()

        if xlim is None:
            xlim = (X.ppf(0.01), X.ppf(0.99))
        if ylim is None:
            ylim = (Y.ppf(0.01), Y.ppf(0.99))

        x = np.linspace(xlim[0], xlim[1], 1000)
        y = self.pointLSF(myLSF, 0, x, start=(ylim[0]+ylim[1])/2)
        if reverse:
            ax.plot(y, x, label="LSF", color="r")
        else:
            ax.plot(x, y, label="LSF", color="r")
        return f, ax
        
    def plot_contour(self, ax=None, xlim=None, ylim=None, reverse=False, nb_points=200):
        if ax is None:
            f, ax = plt.subplot(1)
        else:
            f = plt.gcf()

        if reverse:    # Reverse allows to plot Y (2nd RV) on the x-axis
            X = self.Y
            Y = self.X
        else:
            X = self.X
            Y = self.Y  

        if xlim is None:
            xlim = (X.ppf(0.01), X.ppf(0.99))
        if ylim is None:
            ylim = (Y.ppf(0.01), Y.ppf(0.99))

        x = np.linspace(xlim[0], xlim[1], nb_points).reshape(-1, 1)
        y = np.linspace(ylim[0], ylim[1], nb_points).reshape(-1, 1)
        X,Y = np.meshgrid(x,y)
        pdf = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i,j] = self.bivariatePdf([X[i,j], Y[i,j]])
        if reverse:
            ax.contour(Y, X, pdf, levels=8, cmap=cm.Blues) 
        else:
            ax.contour(X, Y, pdf, levels=8, cmap=cm.Blues)
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y")
        return f, ax

    
class Multivar():
    def __init__(self, X:list, copulas:dict):
        ''' X: vector of random variables. 
        copula: vector of copulas '''
        self.X = X
        self.copulas = copulas
        keys = copulas.keys()
        params = copulas.values()
        self.C1 = Bivariate(X[0], X[1], keys[0], params[0])
        self.C2 = Bivariate(X[1], X[2], keys[1], params[1])

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
            c = self.C1
            c.plotLSF(myLSF, ax, xlim=(x[0], x[-1]), reverse=reverse)
            c.plot_contour(ax, xlim=(x[0], x[-1]), reverse=reverse)
        elif (x_index+y_index)==3:
            c = self.C2
            c.plotLSF(myLSF, ax, xlim=(x[0], x[-1]), reverse=reverse)
            c.plot_contour(ax, xlim=(x[0], x[-1]), reverse=reverse)



        ax.set_title(fr"Bivariate contours and limit-state function in the plane $(X_{{{x_index}}}, X_{{{y_index}}})$")
        ax.set_xlabel(fr"$X_{{{x_index}}}$")
        ax.set_ylabel(fr"$X_{{{y_index}}}$")
        ax.set_xlim(X.ppf(0.01), X.ppf(0.99))
        ax.set_ylim(Y.ppf(0.01), Y.ppf(0.99))
        ax.set_aspect("scaled")
        return f, ax
