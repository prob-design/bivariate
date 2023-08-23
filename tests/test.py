import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import sys
sys.path.append(r'H:/bivariate/src')
from src.class_multivar import Bivariate, Multivariate

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
)

log = logging.getLogger()

X1 = st.norm(0,1)
X2 = st.norm(1,0.3)
X3 = st.norm(3,1.7)

assert Bivariate(X1, X2, 'bdhdb', 0.5)

