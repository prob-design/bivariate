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
        b.plotLSF(limitstatefunc2D())
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
    for i in range(2):
        multivar.drawMarginalPdf(i)
        multivar.drawMarginalCdf(i)


def test_multivariate_plot(multivar, limitstatefunc3D):
    ''' Test the plotting of the bivariate plot for all combinations of (x_index, y_index). '''
    indices = [0, 1, 2]
    for combi in list(itertools.permutations(indices, 2)):
        multivar.bivariate_plot(x_index=combi[0],
                                y_index=combi[1],
                                myLSF=limitstatefunc3D,
                                z=1)
