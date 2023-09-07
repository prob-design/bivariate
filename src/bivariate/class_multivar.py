import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats.distributions

from scipy.optimize import fsolve

import pyvinecopulib as pyc


class LimitStateFunction():
    def __init__(self):
        pass


# See how the methods below are affected by the creation of the LimitStateFunction class
# Robustness of fsolve: some start values lead to converge at wrong values (e.g. start=0 and quadratic functions)
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


class Bivariate():
    def __init__(self,
                 rv: list,
                 family,
                 parameter=0):
        '''
        Define a Bivariate object with three attributes:

        rv: list[scipy.stats.distributions.rv_frozen]. Contains the two random variables of the object.
        family: string. Defines the type of bivariate copula of the elements of rv.
        parameter: float. Parameter of the bivariate copula family. Optional for the independent copula.
        '''
        self.rv = rv
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
                "Invalid copula. Please choose between Normal, Clayton or Independent copula. "
            )

    def getMarginal(self, i: int) -> scipy.stats.distributions.rv_frozen:
        '''
        Returns the marginal distribution of variable X_i.

        i: int. Index of the marginal distribution returned (0 or 1).
        '''
        try:
            return self.rv[i]
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
        u0 = self.rv[0].cdf(x[0])
        u1 = self.rv[1].cdf(x[1])

        pdf = float(self.copula.pdf([[u0, u1]])) * self.rv[0].pdf(x[0]) * self.rv[1].pdf(x[1])
        return pdf

    def bivariateCdf(self, x: list[float]):
        '''
        Computes the bivariate cumulative density function evaluated in x.

        x: list. Coordinates at which the bivariate CDF is evaluated.
        '''
        # Compute the ranks of the coordinates
        u0 = self.rv[0].cdf(x[0])
        u1 = self.rv[1].cdf(x[1])

        cdf = float(self.copula.cdf([[u0, u1]]))
        return cdf

    def plot_contour(self, ax=None, xlim=None, ylim=None,
                     x_index=0, nb_points=200):
        '''
        Plots the contour of the probability density function in the plane (X_{x_index}, X_{1-x_index}).

        myLSF: function. Limit state function.
        ax: matplotlib.axes.Axes. Passed if the limit state function is plotted on an existing Axes object.
        xlim, ylim: tuples. Limits on the x- and y-axes of the plot.
        x_index: int. Defines which variable is set on the x-axis.
        '''
        rv_x = self.rv[x_index]
        rv_y = self.rv[1-x_index]

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
        ax.set_aspect('equal')    # ensures scales of both axis are the same (interpretation of the contours shapes)
        #ax.set_xlim(xlim)
        #ax.set_ylim(ylim)
        ax.set_xlabel('$x_' + str(x_index) + '$', fontsize=15)
        ax.set_ylabel('$x_' + str(1 - x_index) + '$', fontsize=15)
        return f, ax


    def plotLSF(self, myLSF, ax=None, xlim=None, ylim=None, x_index=0):
        '''
        Plots the limit state function in the 2D plane (X_{x_index}, X_{1-x_index}).

        myLSF: function. Limit state function.
        ax: matplotlib.axes.Axes. Passed if the limit state function is plotted on an existing Axes object.
        xlim, ylim: tuple[float]. Limits on the x- and y-axes of the plot.
        x_index: int. Defines which variable is set on the x-axis.
        '''
        rv_x = self.rv[x_index]

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
        y = [calculate_root_LSF(myLSF, i=1-x_index, values=[k]) for k in x]

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

    def p_and(self, x: float, y: float):
        """ Computes the probability P(X>x,Y>y). """
        u = self.rv[0].cdf(x)
        v = self.rv[1].cdf(y)
        c = self.bivariateCdf([x, y])
        return 1 - u - v + c

    def plot_and(self, x, y, x_index=0, ax=None, contour=False, xlim=None, ylim=None, compare=False):
        """ Computes the probability P(X>x,Y>y) and draws the related figure. """
        rv_x = self.rv[x_index]
        rv_y = self.rv[1-x_index]

        if ax is None:
            f, ax = plt.subplots(1)
        else:
            f = plt.gcf()

        if xlim is None:
            xlim = (rv_x.ppf(0.01), rv_x.ppf(0.99))
        if ylim is None:
            ylim = (rv_y.ppf(0.01), rv_y.ppf(0.99))

        if contour:
            self.plot_contour(ax, xlim=xlim, ylim=ylim)

        if not compare:
            color = "lightgrey"
            zorder = 3
        else:
            color = "grey"
            zorder = 5

        ax.vlines(x, ymin=ylim[0], ymax=ylim[1], colors='k', linestyles="dashed", zorder=zorder+1)
        ax.hlines(y, xmin=xlim[0], xmax=xlim[1], colors='k', linestyles="dashed", zorder=zorder+1)
        ax.vlines(x, ymin=ylim[0], ymax=ylim[1], colors='k', linestyles="dashed", zorder=zorder+1)
        ax.hlines(y, xmin=xlim[0], xmax=xlim[1], colors='k', linestyles="dashed", zorder=zorder+1)
        ax.fill_between([x, xlim[1]], y, ylim[1], color='lightgrey', linewidth=2, edgecolor="k", alpha=1, zorder=zorder)
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        p = self.p_and(x, y)
        if not compare:
            if p < 0.01:
                ax.text(.01, .01, r"$(P=$" + "{:.2e})".format(p), ha="left", va="bottom", transform=ax.transAxes)
            else:
                ax.text(.01, .01, r"$(P=$" + "{:.4f})".format(p), ha="left", va="bottom", transform=ax.transAxes)
        else:
            if p < 0.01:
                ax.text(.01, .07, r"$(P_{2}=$" + "{:.2e})".format(p), ha="left", transform=ax.transAxes)
            else:
                ax.text(.01, .07, r"$(P_{2}=$" + "{:.4f})".format(p), ha="left", transform=ax.transAxes)

        ax.set_title("$p_{AND}$ (=" + str(p) + ")", fontsize=18)
        ax.set_xlabel('$x_' + str(x_index) + '$', fontsize=15)
        ax.set_ylabel('$x_' + str(1 - x_index) + '$', fontsize=15)
        return f, ax

    def p_or(self, x, y):
        """ Computes the probability P(X>x OR Y>y). """
        c = self.bivariateCdf([x, y])
        return 1 - c

    def plot_or(self, x, y, x_index=0, ax=None, contour=False, xlim=None, ylim=None, compare=False):
        """ Computes the probability P(X>x OR Y>y) and draws the related figure. """
        rv_x = self.rv[x_index]
        rv_y = self.rv[1 - x_index]

        if ax is None:
            f, ax = plt.subplots(1)
        else:
            f = plt.gcf()

        if xlim is None:
            xlim = (rv_x.ppf(0.01), rv_x.ppf(0.99))
        if ylim is None:
            ylim = (rv_y.ppf(0.01), rv_y.ppf(0.99))

        if contour:
            self.plot_contour(ax, xlim=xlim, ylim=ylim)

        if not compare:
            color = "lightgrey"
            zorder = 1
        else:
            color = "grey"
            zorder = 4

        ax.vlines(x, ymin=y, ymax=ylim[1], colors='k', linestyles="dashed", zorder=zorder+2)
        ax.hlines(y, xmin=x, xmax=xlim[1], colors='k', linestyles="dashed", zorder=zorder+2)
        ax.fill_between(np.array([xlim[0], x, x, xlim[1]]), np.array([y, y, ylim[0], ylim[0]]),
                        ylim[1], color='lightgrey', linewidth=2, edgecolor="k", zorder=zorder)

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        p = self.p_or(x, y)
        # if not compare:
        if p < 0.01:
            ax.text(.01, .01, r"$(P=$" + "{:.2e})".format(p), ha="left", va="bottom", transform=ax.transAxes)
        else:
            ax.text(.01, .01, r"$(P=$" + "{:.4f})".format(p), ha="left", va="bottom", transform=ax.transAxes)
        # else:
        #     if p < 0.01:
        #         ax.text(.01, .07, r"$(P_{2}=$" + "{:.2e})".format(p), ha="left", transform=ax.transAxes)
        #     else:
        #         ax.text(.01, .07, r"$(P_{2}=$" + "{:.4f})".format(p), ha="left", transform=ax.transAxes)
        ax.set_title("$p_{OR}$ (=" + str(p) + ")", fontsize=18)
        ax.set_xlabel('$x_' + str(x_index) + '$', fontsize=15)
        ax.set_ylabel('$x_' + str(1 - x_index) + '$', fontsize=15)
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
    def __init__(self, rv: list, copulas: list[tuple]):
        '''
        Multivariate object for multidimensional probabilistic analyses.

        rv: list. Vector of random variables.
        copulas: list. Vector of copulas [cop1, cop2, cop_cond] such that
        cop1 is the copula of rv[0] and rv[1], cop2 of rv[1] and rv[2] and
        cond_cop the conditional copula of rv[0], rv[2] given rv[1].
        '''

        self.rv = rv
        self.copulas = copulas        
        self.B1 = Bivariate([rv[0], rv[1]], copulas[0][0], copulas[0][1])
        self.B2 = Bivariate([rv[1], rv[2]], copulas[1][0], copulas[1][1])

        if all(cop[0] == 'Normal' for cop in copulas):
            family = pyc.BicopFamily.gaussian
            parameter = [copulas[2][1]]
            self.__cond_copula = pyc.Bicop(family=pyc.BicopFamily.gaussian, parameters=[parameter])
            x = sampling_cop(self.B1.copula, self.B2.copula, 
                             cond_cop=self.__cond_copula)
            fitted_copula = fit_copula(x[:, 0], x[:, 2], family=family)
            self.B3 = Bivariate([rv[0], rv[2]], 'Normal', parameter=float(fitted_copula.parameters))
        else:
            pass # Determine which are the particular cases where the copula of X0, X2 is analytically defined

    def getMarginal(self, i):
        '''
        Returns the marginal distribution of variable X_i.

        i: int. Index of the marginal distribution returned (0, 1 or 2).
        '''
        assert (i >= 0 and i <= 2), \
            'Index i out of range. Please select a value between 0 and 2.'
        return self.rv[i]

    def drawMarginalPdf(self, i):
        '''
        Plot of the marginal probability density function (pdf) of variable X_i.

        i: int. Index of the marginal distribution plotted (0, 1 or 2).
        '''
        assert (i >= 0 and i <= 2), \
            'Index i out of range. Please select a value between 0 and 2.'

        f, ax = plt.subplots(figsize=(10, 6))
        rv_x = self.rv[i]
        x = np.linspace(rv_x.ppf(0.01), rv_x.ppf(0.99), 1000)
        y = self.rv[i].pdf(x)
        ax.plot(x, y, label="pdf")
        ax.set_xlabel('$x_' + str(i) + '$', fontsize=15)
        ax.set_ylabel('$f(x_' + str(i) + ')$', fontsize=15)
        ax.set_title(fr'Probability density function of $X_{{{i}}}$', fontsize=18)
        return f, ax

    def drawMarginalCdf(self, i):
        '''
        Plot of the marginal cumulative density function (cdf) of variable X_i.

        i: int. Index of the marginal distribution plotted (0 or 1).
        '''
        assert (i >= 0 and i <= 2), \
            'Index i out of range. Please select a value between 0 and 2.'

        f, ax = plt.subplots(figsize=(10, 6))
        rv_x = self.rv[i]
        x = np.linspace(rv_x.ppf(0.01), rv_x.ppf(0.99), 1000)
        y = rv_x.cdf(x)
        ax.plot(x, y, label="cdf")
        ax.set_xlabel('$x_' + str(i) + '$', fontsize=15)
        ax.set_ylabel('$F(x_' + str(i) + ')$', fontsize=15)
        ax.set_title(fr'Cumulative density function of $X_{{{i}}}$', fontsize=18)
        return f, ax


    def bivariate_plot(self, x_index, y_index, myLSF, z:float, xlim=None, ylim=None):
        '''
        Plots the limit state function and the contours of the probability density function in the 2D plane.

        x_index: int. Index of the variable on the x-axis (0 <= x_index <= 2).
        y_index: int. Index of the variable on the y-axis (0 <= y_index <= 2).
        myLSF: function. Limit state function.
        z: float. Value attributed to the third (non represented) variable.
        '''

        assert x_index != y_index, \
            'Values of x_index and y_index are identical. Please select different values of x_index or y_index.'
        assert (x_index >= 0 and x_index <= 2), \
            'Index x_index out of range. Please select a value between 0 and 2.'
        assert (y_index >= 0 and y_index <= 2), \
            'Index y_index out of range. Please select a value between 0 and 2.'

        rv_x = self.rv[x_index]
        rv_y = self.rv[y_index]

        f, ax = plt.subplots(figsize=(10,6))
        if xlim is None:
            xlim = (rv_x.ppf(0.01), rv_x.ppf(0.99))
        x = np.linspace(xlim[0], xlim[1], 1000)
        cond = np.full(x.shape, z)

        if (x_index+y_index) == 1:
            bivar = self.B1
            values = np.concatenate((x.reshape(-1, 1),
                                     cond.reshape(-1, 1)), axis=1)
        elif (x_index+y_index) == 3:
            bivar = self.B2
            values = np.concatenate((cond.reshape(-1, 1),
                                     x.reshape(-1, 1)),  axis=1)
        else:
            bivar = self.B3
            if x_index > y_index:
                values = np.concatenate((cond.reshape(-1, 1),
                                         x.reshape(-1, 1)), axis=1)
            else:
                values = np.concatenate((x.reshape(-1, 1),
                                         cond.reshape(-1, 1)), axis=1)

        if ylim is None:
            ylim = (rv_y.ppf(0.01), rv_y.ppf(0.99))
        relat_index = 1 * (x_index < y_index)
        bivar.plot_contour(ax, xlim=(x[0], x[-1]), ylim=ylim, x_index=relat_index)

        y = [calculate_root_LSF(myLSF, i=y_index, values=list(k)) for k in values]
        ax.plot(x, y, color='k', label='LSF')

        # Add shading of the failure domain (and symbol \Omega in the domain)

        ax.set_title(fr"Bivariate contours and limit-state function in the plane $(X_{{{x_index}}}, X_{{{y_index}}})$")
        ax.set_xlabel(fr"$x_{{{x_index}}}$")
        ax.set_ylabel(fr"$x_{{{y_index}}}$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        return f, ax

    def plot_or(self, x, y, x_index, y_index, contour=False, compare=False):
        assert x_index != y_index, \
            'Values of x_index and y_index are identical. Please select different values of x_index or y_index.'
        assert (x_index >= 0 and x_index <= 2), \
            'Index x_index out of range. Please select a value between 0 and 2.'
        assert (y_index >= 0 and y_index <= 2), \
            'Index y_index out of range. Please select a value between 0 and 2.'

        relat_index = 1 * (x_index > y_index)
        if x_index + y_index == 1:
            bivar = self.B1
        elif x_index + y_index ==3:
            bivar = self.B2
        else:
            bivar = self.B3
        f, ax = bivar.plot_or(x, y, x_index=relat_index, contour=contour, compare=compare)
        ax.set_xlabel(fr"$x_{{{x_index}}}$")
        ax.set_ylabel(fr"$x_{{{y_index}}}$")
        return f, ax

    def plot_and(self, x, y, x_index, y_index, contour=False, compare=False):
        assert x_index != y_index, \
            'Values of x_index and y_index are identical. Please select different values of x_index or y_index.'
        assert (x_index >= 0 and x_index <= 2), \
            'Index x_index out of range. Please select a value between 0 and 2.'
        assert (y_index >= 0 and y_index <= 2), \
            'Index y_index out of range. Please select a value between 0 and 2.'

        relat_index = 1 * (x_index > y_index)
        if x_index + y_index == 1:
            bivar = self.B1
        elif x_index + y_index ==3:
            bivar = self.B2
        else:
            bivar = self.B3
        f, ax = bivar.plot_and(x, y, x_index=relat_index, contour=contour, compare=compare)
        ax.set_xlabel(fr"$x_{{{x_index}}}$")
        ax.set_ylabel(fr"$x_{{{y_index}}}$")
        return f, ax

