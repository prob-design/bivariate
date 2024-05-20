import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import openturns as ot


import scipy.linalg as scp
# import matplotlib.pylab as plt
import openturns.viewer as viewer
ot.Log.Show(ot.Log.NONE)

import time
import scipy


from scipy.optimize import fsolve # for plotting the LSF
from matplotlib.pylab import rcParams

# ------------------------------------------------- Run FORM and MCS ------------------------------------------------- #



def input_OpenTurns(X1, X2, X3, myLSF):
    # Definition of the dependence structure: here, multivariate normal with correlation coefficient rho between two RV's.
    global R, multinorm_copula, inputDistribution, inputRandomVector, myfunction, outputvector, failureevent, optimAlgo 
    global start_pt, start, algo
    R = ot.CorrelationMatrix(3)   
    multinorm_copula = ot.NormalCopula(R)

    # inputDistribution = ot.ComposedDistribution((wave_height_windsea, wave_height_swell, u_current), multinorm_copula)
    inputDistribution = ot.ComposedDistribution((X1, X2, X3), multinorm_copula)
    inputDistribution.setDescription(["windsea","swell", "u_current"])
    inputRandomVector = ot.RandomVector(inputDistribution)

    myfunction = ot.PythonFunction(3, 1, myLSF)

    # Vector obtained by applying limit state function to X1 and X2
    outputvector = ot.CompositeRandomVector(myfunction, inputRandomVector)

    # Define failure event: here when the limit state function takes negative values
    failureevent = ot.ThresholdEvent(outputvector, ot.Less(), 0)
    failureevent.setName('LSF inferior to 0')

    optimAlgo = ot.Cobyla()
    optimAlgo.setMaximumEvaluationNumber(100000)
    optimAlgo.setMaximumAbsoluteError(1.0e-10)
    optimAlgo.setMaximumRelativeError(1.0e-10)
    optimAlgo.setMaximumResidualError(1.0e-10)
    optimAlgo.setMaximumConstraintError(1.0e-10)


def run_FORM_analysis():
    global result, x_star, u_star, pf, beta
    # Starting point for FORM
    start_pt = []

    # Start timer
    start = time.time()
    algo = ot.FORM(optimAlgo, failureevent, inputDistribution.getMean())  # Maybe change 3rd argument and put start_pt defined above
    algo.run()
    result = algo.getResult()
    x_star = result.getPhysicalSpaceDesignPoint()        # Design point: original space
    u_star = result.getStandardSpaceDesignPoint()        # Design point: standard normal space
    pf = result.getEventProbability()                    # Failure probability
    beta = result.getHasoferReliabilityIndex()           # Reliability index
    
    # End timer
    end = time.time()
    print(f'The FORM analysis took {end-start:.3f} seconds')
    
    print('FORM result, pf = {:.4f}'.format(pf))
    print('FORM result, beta = {:.3f}\n'.format(beta))
    print('The design point in the u space: ', u_star)
    print('The design point in the x space: ', x_star)
    
    return result, x_star, u_star, pf, beta    


def run_MonteCarloSimulation(mc_size):
    # Monte Carlo simulation
    # Start timer
    start = time.time()
    montecarlosize = mc_size
    outputSample = outputvector.getSample(montecarlosize)

    number_failures = sum(i < 0 for i in np.array(outputSample))[0]      # Count the failures, i.e the samples for which g(x)<0
    pf_mc = number_failures/montecarlosize                               # Calculate the failure probability                       

    # End timer and print the time
    end = time.time()
    print(f'The Monte Carlo simulation took {end-start:.3f} seconds')
    
    print('pf for MCS: ', pf_mc)
    
    return pf_mc



def importance_factors():
    print(f'---------------Running part 2 of FORM and MCS, importantce factors-------------------')
    import matplotlib.pyplot as plt
    plt.ion()
    print()
    alpha_ot = result.getImportanceFactors()
    print(f'The importance factors {alpha_ot}')
    result.drawImportanceFactors()
    
    plt.show()
    
    alpha = u_star/beta
    print("The importance factors as defined in the textbook are: ", alpha)
    sens = result.getHasoferReliabilityIndexSensitivity()
    print("The sensitivity factors of the reliability index "
        "with regards to the distribution parameters are:\n")
    for i in range(sens.getSize()):
        print(sens.at(i))
        
    result.drawImportanceFactors()
    
    return alpha_ot, alpha, sens