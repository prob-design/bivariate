import scipy.stats as st
import pytest

from bivariate.class_multivar import Bivariate, Multivariate

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
)

log = logging.getLogger()

X1 = st.norm(0,1)
X2 = st.norm(1,0.3)
X3 = st.norm(3,1.7)


@pytest.fixture
def bivar():
    return [Bivariate(X1, X2, 'Normal', 0.5), Bivariate(X2, X3, 'Clayton', 1.5),
            Bivariate(X1, X3, 'Independent')]

@pytest.fixture
def limitstatefunc():
    return lambda x: x[0] - x[1]**2

def test_bivariate_marginal_plots(bivar, limitstatefunc):
    for b in bivar:
        b.drawMarginalCdf(0)
        b.drawMarginalPdf(1)
        b.plotLSF(limitstatefunc)
        b.plot_contour(nb_points=100)

