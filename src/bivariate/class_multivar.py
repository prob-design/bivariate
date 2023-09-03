import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats.distributions

from scipy.optimize import fsolve

import pyvinecopulib as pyc


class LimitStateFunction():
    def __init__(self):
        pass


class Bivariate():
    def __init__(self,
                 X: list,
                 family,
                 parameter=0):
        '''
        Define a Bivariate object with three attributes:

        X: list[scipy.stats.distributions.rv_frozen]. Contains the two random variables of the object.
        family: string. Defines the type of bivariate copula of the elements of X.
        parameter: float. Parameter of the bivariate copula family. Optional for the independent copula.
        '''
        self.X = X
        self.family = family
        if family == "Normal":
            self.copula = pyc.Bicop(family=pyc.BicopFamily.gaussian,
                                    parameters=[parameter])
        elif family == "Clayton":
            self.copula = pyc.Bicop(family=pyc.BicopFamily.clayton,
                                    parameters=[parameter])
        elif family == "Independent":
            self.copula = pyc.Bicop(family=pyc.BicopFamily.indep)
        else:
            raise ValueError(
                "Invalid copula. Please choose between Normal, Clayton or Independent copulae. "
            )

    def getMarginal(self, i: int) -> scipy.stats.distributions.rv_frozen:
        '''
        Returns the marginal distribution of variable X_i.

        i: int. Index of the marginal distribution returned (0 or 1).
        '''
        try:
            return self.X[i]
        except IndexError:
            raise ValueError(
                'Index i out of range. Please select i=0 or i=1.'
            )

    def drawMarginalPdf(self, i: int):
        '''
        Plot of the marginal probability density function (pdf) of variable X_i.

        i: int. Index of the marginal distribution plotted (0 or 1).
        '''

        f, ax = plt.subplots(figsize=(10, 6))

        var = self.getMarginal(i)
        x = np.linspace(var.ppf(0.01), var.ppf(0.99), 1000)
        y = var.pdf(x)

        ax.plot(x, y, label="pdf")
        ax.set_title(r"Probability density function of $X_{" + str(i) + "}$",
                     fontsize=18)
        ax.set_xlabel(r"$x_" + str(i) + "$")
        ax.set_ylabel(r"f($x_" + str(i) + ")$")
        return f, ax

    def drawMarginalCdf(self, i: int):
        '''
        Plot of the marginal cumulative density function (cdf) of variable X_i.

        i: int. Index of the marginal distribution plotted (0 or 1).
        '''

        f, ax = plt.subplots(figsize=(10, 6))

        var = self.getMarginal(i)
        x = np.linspace(var.ppf(0.01), var.ppf(0.99), 1000)
        y = var.cdf(x)

        ax.plot(x, y, label="cdf")
        ax.set_title(r"Cumulative density function of $X_{" + str(i) + "}$",
                     fontsize=18)
        ax.set_xlabel(r"$x_" + str(i) + "$", fontsize=15)
        ax.set_ylabel(r"$F(x_" + str(i) + ")$", fontsize=15)
        return f, ax

    def bivariatePdf(self, x: list[float]) -> float:
        '''
        Computes the bivariate probability density function evaluated in x.

        x: list. Coordinates at which the bivariate PDF is evaluated.
        '''
        # Compute the ranks of the coordinates
        u0 = self.X[0].cdf(x[0])
        u1 = self.X[1].cdf(x[1])

        pdf = float(self.copula.pdf([[u0, u1]]))
        return pdf

    def bivariateCdf(self, x: list[float]):
        '''
        Computes the bivariate cumulative density function evaluated in x.

        x: list. Coordinates at which the bivariate CDF is evaluated.
        '''
        # Compute the ranks of the coordinates
        u0 = self.X[0].cdf(x[0])
        u1 = self.X[1].cdf(x[1])

        cdf = float(self.copula.cdf([[u0, u1]]))
        return cdf

# See how the methods below are affected by the creation of the LimitStateFunction class
# Robustness of fsolve: some start values lead to converge at wrong values (e.g. start=0 and quadratic functions)
    @staticmethod
    def calculate_root_LSF(myLSF, i: int, values: list[float],
                           start: float = 0, max_attempts: int = 5) -> float:
        '''
        Calculates the value of X_i that cancels myLSF given the values in values.

        myLSF: function. (Limit state) function whose root is to be determined.
        i: int. Index of the missing value, see example.
        values: (list) of values of all variables of myLSF but X_i.
        start: float. Default start value for the execution of fsolve.
        max_attempts: int. Maximum number of attempts for the convergence of fsolve.

        Example:
        func = x**2 - y
        calculate_root_LSF(func, i=1, values=[3]) = 9
        '''
        # Define a temporary function with the missing variable as argument
        def temp_func(x):
            return myLSF(values[:i] + [x] + values[i:])

        # For certain (non-monotonic) functions, inappropriate start_values of fsolve lead
        # to RuntimeWarning and no convergence.
        for _ in range(max_attempts):
            start_value = start
            try:
                y = fsolve(temp_func, start_value)[0]
            except RuntimeWarning:
                start_value += 1
            else:
                return y
        raise Exception(
            'An exception has occurred. Please reiterate with a different value of start.'
        )

    def plot_contour(self, ax=None, xlim=None, ylim=None,
                     x_index: int = 0, nb_points=200):
        '''
        Plots the contour of the probability density function in the plane (X_{x_index}, X_{1-x_index}).

        myLSF: function. Limit state function.
        ax: matplotlib.axes.Axes. Passed if the limit state function is plotted on an existing Axes object.
        xlim, ylim: tuples. Limits on the x- and y-axes of the plot.
        x_index: int. Defines which variable is set on the x-axis.
        '''
        rv_x = self.X[x_index]
        rv_y = self.X[1-x_index]

        if xlim is None:
            if ax is None:
                xlim = (rv_x.ppf(0.01), rv_x.ppf(0.99))
            else:
                xlim = ax.get_xlim()

        if ylim is None:
            if ax is None:
                ylim = (rv_y.ppf(0.01), rv_y.ppf(0.99))
            else:
                ylim = ax.get_ylim()

        if ax is None:
            f, ax = plt.subplots(figsize=(12, 8))
        else:
            f = plt.gcf()

        x = np.linspace(xlim[0], xlim[1], nb_points).reshape(-1, 1)
        y = np.linspace(ylim[0], ylim[1], nb_points).reshape(-1, 1)
        X, Y = np.meshgrid(x, y)

        pdf = np.zeros(X.shape)
        if x_index == 0:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pdf[i, j] = self.bivariatePdf([X[i, j], Y[i, j]])
        else:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pdf[i, j] = self.bivariatePdf([Y[i, j], X[i, j]])

        ax.contour(X, Y, pdf, levels=8, cmap=cm.Blues)
        ax.set_aspect("equal")    # ensures scales of both axis are the same (interpretation of the contours shapes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")
        return f, ax


    def plotLSF(self, myLSF, ax=None, xlim=None, ylim=None, x_index=0):
        '''
        Plots the limit state function in the 2D plane (X_{x_index}, X_{1-x_index}).

        myLSF: function. Limit state function.
        ax: matplotlib.axes.Axes. Passed if the limit state function is plotted on an existing Axes object.
        xlim, ylim: tuple[float]. Limits on the x- and y-axes of the plot.
        x_index: int. Defines which variable is set on the x-axis.
        '''
        rv_x = self.X[x_index]

        if xlim is None:
            if ax is None:
                xlim = (rv_x.ppf(0.01), rv_x.ppf(0.99))
            else:
                xlim = ax.get_xlim()
        if ylim is None and ax is not None:
            ylim = ax.get_ylim()

        if ax is None:
            f, ax = plt.subplots(figsize=(10, 8))
        else:
            f = plt.gcf()

        x = np.linspace(xlim[0], xlim[1], 1000)
        # Calculate the root of the limit state function for all points of the x-axis
        y = [self.calculate_root_LSF(myLSF, i=1-x_index, values=[k]) for k in x]

        ax.plot(x, y, label="LSF", color="r")
        ax.set_xlim(xlim)
        ax.set_title('Limit state function in 2D plane.', fontsize=18)
        ax.set_xlabel('$x_' + str(x_index) + '$', fontsize=15)
        ax.set_ylabel('$x_' + str(1 - x_index) + '$', fontsize=15)

        # Fix y-limits if they exist
        try:
            ax.set_ylim(ylim)
        except:
            pass

        # Add shading of the failure region (myLSF<0 or else)

        return f, ax


# Functions to sample copula of X0 and X2 in Multivariate class
def sampling_cop(cop1, cop2, cond_cop=pyc.Bicop(), n=10000):
    '''
    Generates a random sample of the bivariate copula of X0 and X2.

    cop1: pyvinecopulib.Bicop. Copula between variables X0 and X1.
    cop2: pyvinecopulib.Bicop. Copula between variables X1 and X2.
    cond_cop: pyvinecopulib.Bicop. Conditional copula between X0 and X2 given X1.
    (default: independent copula).
    n: int. Number of samples.

    ---------------
    See sampling procedure in http://dx.doi.org/10.1016/j.insmatheco.2007.02.001
    '''
    u = np.random.rand(n, 3)
    x0 = u[:, 0]
    x1 = cop1.hinv1(np.concatenate((x0.reshape(-1, 1),
                                    u[:, 1].reshape(-1, 1)), axis=1))
    a = cop1.hfunc1(np.concatenate((x0.reshape(-1, 1),
                                    x1.reshape(-1, 1)), axis=1))
    b = cond_cop.hinv1(np.concatenate((a.reshape(-1, 1),
                                       u[:, 2].reshape(-1, 1)), axis=1))
    x2 = cop2.hinv1(np.concatenate((x1.reshape(-1, 1),
                                    b.reshape(-1, 1)), axis=1))
    x = np.concatenate((x0.reshape(-1, 1),
                        x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
    return x


def fit_copula(x, y, family=pyc.BicopFamily.gaussian):
    '''
    Fits a bivariate copula to a sample of random variables X and Y.

    x: ndarray. Sample of variable X.
    y: ndarray. Sample of variable Y.
    family: pyvinecopulib.BicopFamily. Family of the copula to fit.
    '''

    fitted_cop = pyc.Bicop(family=family)
    fitted_cop.fit(np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1))
    return fitted_cop


class Multivariate():
    def __init__(self, rv: list, copulas: list):
        '''
        Multivariate object for multidimensional probabilistic analyses.

        rv: list. Vector of random variables.
        copulas: list. Vector of copulas [cop1, cop2, cop_cond] such that
        cop1 is the copula of rv[0] and rv[1], cop2 of rv[1] and rv[2] and
        cond_cop the conditional copula of rv[0], rv[2] given rv[1].
        '''
        self.rv = rv
        self.copulas = copulas        
        self.B1 = Bivariate(rv[0], rv[1], copulas[0][0], copulas[0][1])
        self.B2 = Bivariate(rv[1], rv[2], copulas[1][0], copulas[1][1])
        if self.B1.family == "Normal" and self.B2.family == "Normal" and copulas[2][0] == "Normal":
            family=pyc.BicopFamily.gaussian
            x = sampling_cop(self.B1.copula, self.B2.copula, 
                             cond_cop=pyc.Bicop(family=family, parameters=[copulas[2][1]]))
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

        f, ax = plt.subplots(figsize=(10,6))
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
        return f, ax
