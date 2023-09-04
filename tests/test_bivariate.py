import scipy.stats as st
import pytest
import itertools

from bivariate.class_multivar import Bivariate, Multivariate

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
)

log = logging.getLogger()

X1 = st.norm(0,1)
X2 = st.norm(1,0.3)
X3 = st.norm(3,1.7)


# TESTS BIVARIATE CLASS

@pytest.fixture
def bivar():
    return [Bivariate([X1, X2], 'Normal', 0.5), Bivariate([X2, X3], 'Clayton', 1.5),
            Bivariate([X1, X3], 'Independent')]


@pytest.fixture
def limitstatefunc2D():

    def myLSF_2D(x):
        return x[0] - x[1]**2

    return myLSF_2D


def test_bivariate_marginal_plots(bivar, limitstatefunc2D):
    for b in bivar:
        b.drawMarginalCdf(0)
        b.drawMarginalPdf(1)
        b.plotLSF(myLSF=limitstatefunc2D)
        b.plot_contour(nb_points=100)


# TESTS MULTIVARIATE CLASS

@pytest.fixture
def multivar():
    return Multivariate([X1, X2, X3], [('Normal', 0.5), ('Normal', 0.2), ('Normal', 0.4)])


@pytest.fixture
def limitstatefunc3D():

    def myLSF_3D(x):
        return x[0] + 3*x[1]**3 - 8*x[2]

    return myLSF_3D


def test_marginal_plots(multivar):
    multivar.drawMarginalPdf(0)
    multivar.drawMarginalCdf(2)
    with pytest.raises(AssertionError):
        multivar.drawMarginalPdf(4)
        multivar.drawMarginalCdf('a')

def test_bivariate_plot(multivar, limitstatefunc3D):
    ''' Test the plotting of the bivariate plot for all combinations of (x_index, y_index). '''
    multivar.bivariate_plot(x_index=0,
                            y_index=1,
                            myLSF=limitstatefunc3D,
                            z=1)

    multivar.bivariate_plot(x_index=2,
                            y_index=0,
                            myLSF=limitstatefunc3D,
                            z=1)

    with pytest.raises(AssertionError):
        multivar.bivariate_plot(x_index=1, y_index=1,
                                myLSF=limitstatefunc3D, z=1)
        multivar.bivariate_plot(x_index=2, y_index=3,
                                myLSF=limitstatefunc3D, z=1)

def test_and_or_plots(multivar):
    multivar.plot_or(1.2, 0.8, x_index=0, y_index=2)
    multivar.plot_or(-1, 0.8, x_index=2, y_index=1)