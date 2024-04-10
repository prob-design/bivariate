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

# ----------------------------------------------- FEM model properties ----------------------------------------------- #

def Beam2DMatrices(m, EA, EI, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl], [xr, yr])      
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2)    # length
    alpha = math.atan2(yr - yl, xr - xl)        # angle

    # 2 - calculate transformation matrix T
    C = np.cos(alpha)
    S = np.sin(alpha)
    T = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))
    

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, -EA/L, 0, 0], 
                    [0, 12*EI/L3, 6*EI/L2, 0, -12*EI/L3, 6*EI/L2], 
                    [0, 6*EI/L2, 4*EI/L, 0, -6*EI/L2, 2*EI/L], 
                    [-EA/L, 0, 0, EA/L, 0, 0], 
                    [0, -12*EI/L3, -6*EI/L2, 0, 12*EI/L3, -6*EI/L2], 
                    [0, 6*EI/L2, 2*EI/L, 0, -6*EI/L2, 4*EI/L]])
    M = m*L/420*np.array([[140, 0, 0, 70, 0, 0], 
                            [0, 156, 22*L, 0, 54, -13*L], 
                            [0, 22*L, 4*L2, 0, 13*L, -3*L2], 
                            [70, 0, 0, 140, 0, 0], 
                            [0, 54, 13*L, 0, 156, -22*L], 
                            [0, -13*L, -3*L2, 0, -22*L, 4*L2]])
    

    # 4 - rotate the matrices
    # K = K
    # M = M
    # 4 - rotate the matrices
    K = np.matmul(np.transpose(T), np.matmul(K, T))
    M = np.matmul(np.transpose(T), np.matmul(M, T))
    return M, K, alpha, T


def Beam3DMatrices(m, EA, EI, GJ, Im, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl, zl], [xr, yr, zr])
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    zl = NodeCoord[0][2]    # z-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    zr = NodeCoord[1][2]    # z-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2 + (zr - zl)**2)    # length
    
    # 2 - calculate transformation matrix T
    C = (xr-xl)/L
    S = (yr-yl)/L
    # T in this is different from T in the above Beam 2D function
    T = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])
    
    # Only support rotation in XY plane
    if(abs(zr - zl) > 1e-6):
        print("Error, Only supports rotation in XY plane")
        T = np.array([[1, 0, 0], [0, 1, 0], [0,0,1]])        
    
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))    
    T = np.asarray(np.bmat([[T, np.zeros((6,6))], [np.zeros((6, 6)), T]]))    
    # print(T)

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, 0, 0, 0, -EA/L, 0, 0, 0, 0, 0], 
                  [0, 12*EI/L3, 0, 0, 0, 6*EI/L2, 0, -12*EI/L3, 0, 0, 0, 6*EI/L2], 
                  [0, 0, 12*EI/L3, 0, -6*EI/L2, 0, 0, 0, -12*EI/L3, 0, -6*EI/L2, 0], 
                  [0, 0, 0, GJ/L, 0, 0, 0, 0, 0, -GJ/L, 0, 0],
                  [0, 0, -6*EI/L2, 0, 4*EI/L, 0, 0, 0, 6*EI/L2, 0, 2*EI/L, 0], 
                  [0, 6*EI/L2, 0, 0, 0, 4*EI/L, 0, -6*EI/L2, 0, 0, 0, 2*EI/L], 
                  [-EA/L, 0, 0, 0, 0, 0, EA/L, 0, 0, 0, 0, 0], 
                  [0, -12*EI/L3, 0, 0, 0, -6*EI/L2, 0, 12*EI/L3, 0, 0, 0, -6*EI/L2], 
                  [0, 0, -12*EI/L3, 0, 6*EI/L2, 0, 0, 0, 12*EI/L3, 0, 6*EI/L2, 0], 
                  [0, 0, 0, -GJ/L, 0, 0, 0, 0, 0, GJ/L, 0, 0],                  
                  [0, 0, -6*EI/L2, 0, 2*EI/L, 0, 0, 0, 6*EI/L2, 0, 4*EI/L, 0],
                  [0, 6*EI/L2, 0, 0, 0, 2*EI/L, 0, -6*EI/L2, 0, 0, 0, 4*EI/L]])    
    
    M = np.array([[140, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0], 
                  [0, 156, 0, 0, 0, 22*L, 0, 54, 0, 0, 0, -13*L], 
                  [0, 0, 156, 0, 22*L, 0, 0, 0, 54, 0, 13*L, 0], 
                  [0, 0, 0, 140*Im, 0, 0, 0, 0, 0, 70*Im, 0, 0],
                  [0, 0, 22*L, 0, 4*L2, 0, 0, 0, -13*L, 0, -3*L2, 0], 
                  [0, 22*L, 0, 0, 0, 4*L2, 0, 13*L, 0, 0, 0, -3*L2], 
                  [70, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 0], 
                  [0, 54, 0, 0, 0, 13*L, 0, 156, 0, 0, 0, -22*L], 
                  [0, 0, 54, 0, -13*L, 0, 0, 0, 156, 0, 22*L, 0], 
                  [0, 0, 0, 70*Im, 0, 0, 0, 0, 0, 140*Im, 0, 0],
                  [0, 0, 13*L, 0, -3*L2, 0, 0, 0, 22*L, 0, 4*L2, 0],
                  [0, -13*L, 0, 0, 0, -3*L2, 0, -22*L, 0, 0, 0, 4*L2]])
    M = m*L/420 * M

    # 4 - rotate the matrices
    K = np.matmul(np.transpose(T), np.matmul(K, T))
    M = np.matmul(np.transpose(T), np.matmul(M, T))
    return M, K    


# ----------------------------------------- Creating matrix of the FEM model ----------------------------------------- #
# Mesh
DistLandings = 17000 # Distance between landings
Deviation = 2000 # The maximum deviation

r = sp.symbols('r')
equation = sp.Eq((DistLandings/2)**2 + (r - Deviation)**2, r**2)
solutions = sp.solve(equation, r)
TunRad = float(solutions[0]) #m

TunAng = (180-np.arccos((DistLandings/2)/TunRad)*180/np.pi, np.arccos((DistLandings/2)/TunRad)*180/np.pi) #deg

dth = 0.0509 #deg
TunCX = TunRad*np.cos( np.deg2rad( np.arange(TunAng[0], TunAng[1]-dth, -dth) ) ) + DistLandings/2
TunCY = TunRad*np.sin( np.deg2rad( np.arange(TunAng[0], TunAng[1]-dth, -dth) ) )
TunCY = TunCY - min(TunCY)

# Properties of the beam
a = 23.8                            #m
b = 9.2                             #m
A = 38                              #m2
E_c = 50e9                          #E-modulus of high strength concrete
J = (np.pi*a**3*b**3)/(a**2 + b**2) #Torsion constant of the beam
G = E_c / (2*(1+0.3))
rho_c = 2500                        #kg/m^3 of concrete

Beam_m = 150000                    # [kg/m]
Beam_EI = E_c*(np.pi*a**3*b)/4      # [N.m2]
Beam_EIz = E_c*(np.pi*a*b**3)/4
Beam_EA = E_c*A                    # [N]
Beam_GJ = G*J                      # [N.m2]
Beam_Im = rho_c*J                  #[m2]

# ------------------------------------------------ Creating the nodes ------------------------------------------------ #
NodeC = [ [x,y] for x,y in zip(TunCX, TunCY) ]
nNode = len(NodeC)


# Define elements (and their properties
#             NodeLeft    NodeRight          m         EA        EI
Ele = [ [n1, n2, Beam_m, Beam_EA, Beam_EI] 
           for n1,n2 in zip(range(0,nNode-1), range(1,nNode)) ]
nEle = len(Ele)



# ------------------------------------------------ Creating the matrix ----------------------------------------------- #
LDOF = 3
nDof = LDOF*nNode  #3 dof per node
K = np.zeros(nDof*nDof)
M = np.zeros(nDof*nDof)
Q = np.zeros(nDof)
    

exeTime = [0.0, 0.0]
exeTime[0] = time.time()

for iEle in range(0, nEle):
    n1, n2, m, EA, EI = Ele[iEle]
    
    x1, y1 = NodeC[n1]
    x2, y2 = NodeC[n2]
    
    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    n1dof = LDOF*n1 + np.arange(0,LDOF)
    n2dof = LDOF*n2 + np.arange(0,LDOF)    
    
    # Calculate elemental matrices
    Me, Ke, alpha, T_transformation = Beam2DMatrices(m, EA, EI, (NodeC[n1], NodeC[n2]) )
    
    
    indexes = np.append(n1dof, n2dof)
    for i in range(0, 2*LDOF):
        for j in range(0, 2*LDOF):
            ij = indexes[i]*nDof + indexes[j]
            M[ij] = M[ij] + Me[i,j]
            K[ij] = K[ij] + Ke[i,j]
    

# Reshape the global matrix from a 1-dimensional array to a 2-dimensional array
M = M.reshape((nDof, nDof))
K = K.reshape((nDof, nDof))

# ------------------------------------------------ Applying the boundary conditions ----------------------------------- #
NodesClamp = (0, nNode-1)

# Prescribed dofs
DofsP = np.empty([0], dtype=int)
for n0 in NodesClamp:
    DofsP = np.append(DofsP, n0*LDOF + np.arange(0,LDOF))

# Free dofs
DofsF = np.arange(0, nDof)       # free DOFs
DofsF = np.delete(DofsF, DofsP)  # remove the fixed DOFs from the free DOFs array


M_FF = [ M[iRow,DofsF].tolist() for iRow in DofsF ]
K_FF = [ K[iRow,DofsF].tolist() for iRow in DofsF ]


# -------------------------------------------- Redoing some notation stuff ------------------------------------------- #
# free & fixed array indices
fx = DofsF[:, np.newaxis]
fy = DofsF[np.newaxis, :]
bx = DofsP[:, np.newaxis]
by = DofsP[np.newaxis, :]

# Mass
Mii = M[fx, fy]
Mib = M[fx, by]
Mbi = M[bx, fy]
Mbb = M[bx, by]

# Stiffness
Kii = K[fx, fy]
Kib = K[fx, by]
Kbi = K[bx, fy]
Kbb = K[bx, by]

# External force
Qi = Q[fx]
Qb = Q[bx]


# Rewriting some notation 
M_FP = Mib
K_FP = Kib

# Return stiffness vector
def return_stiffness_vector():
    return K_FF


# Calculate force matrix
def Force_vector(total_force = 100):
    Q = np.zeros(nDof)
    for i in range(0, nNode):
        for k in range(0, LDOF):
            if i == 0 or i == nNode-1:
                Q[LDOF*i + k] = 0.5*total_force*17000/nEle
            elif k == 1:
                Q[LDOF*i + k] = total_force*17000/nEle
                
    # Construct a matrix to obtain the correct entities of the Q vector
    R1 = np.identity(nDof)
    R1 = R1[fx, 0:nDof]
    F_external = np.dot(R1, Q).reshape(1,-1)
    # Get rid of matrix form and transform this obtained matrix back into a vector
    F_external = F_external[0,:]
    return F_external

# Calculate stress FEM 
def calculate_stress_FEM(Total_force):
    u_displ = scipy.linalg.solve(K_FF, Force_vector(Total_force))
    
    # Length of the element
    L_elem = 46.157	

    # Observation point
    x_obs = L_elem/2

    ux1 = np.array(u_displ[0::6])
    uy1 = np.array(u_displ[1::6])
    ry1 = np.array(u_displ[2::6])



    ux2 = np.array(u_displ[3::6])
    # ux2 = np.append(ux2, 0)

    uy2 = np.array(u_displ[4::6])
    # uy2 = np.append(uy2, 0)

    ry2 = np.array(u_displ[5::6])
    # ry2 = np.append(ry2, 0)

    Myk = Beam_EI/(L_elem**2) * ((-6+(12*x_obs/L_elem))*uy1 + (4*L_elem + 6*x_obs)*ry1 + -(-6+(12*x_obs/L_elem))*uy2 + (-3*L_elem + 6*x_obs)*ry2)

    Iy = (Beam_EI/E_c)

    sigma_M = ((Myk*(a/2))/Iy)*1e-6 #MPa
    return sigma_M
    
    
    
# ------------------------------------------------- Run FORM and MCS ------------------------------------------------- #
def run_FORM_and_MCS(X1, X2, X3, myLSF, mc_size = 10):
    # Definition of the dependence structure: here, multivariate normal with correlation coefficient rho between two RV's.
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

    # Starting point for FORM
    start_pt = []

    algo = ot.FORM(optimAlgo, failureevent, inputDistribution.getMean())  # Maybe change 3rd argument and put start_pt defined above
    algo.run()
    global result
    result = algo.getResult()
    x_star = result.getPhysicalSpaceDesignPoint()        # Design point: original space
    global u_star 
    u_star = result.getStandardSpaceDesignPoint()        # Design point: standard normal space
    pf = result.getEventProbability()                    # Failure probability
    global beta 
    beta = result.getHasoferReliabilityIndex()           # Reliability index
    print('FORM result, pf = {:.4f}'.format(pf))
    print('FORM result, beta = {:.3f}\n'.format(beta))
    print('The design point in the u space: ', u_star)
    print('The design point in the x space: ', x_star)

    montecarlosize = mc_size
    outputSample = outputvector.getSample(montecarlosize)

    number_failures = sum(i < 0 for i in np.array(outputSample))[0]      # Count the failures, i.e the samples for which g(x)<0
    pf_mc = number_failures/montecarlosize                               # Calculate the failure probability                       


    import matplotlib.pyplot as plt
    plt.ion()
    print('pf for MCS: ', pf_mc)
    print()
    print()
    alpha_ot = result.getImportanceFactors()
    print(f'The importance factors {alpha_ot}')
    result.drawImportanceFactors()
    
    plt.show()

    
def run_FORM_and_MCS_part2():
    print(f'---------------Running part 2 of FORM and MCS-------------------')
    alpha = u_star/beta
    print("The importance factors as defined in the textbook are: ", alpha)
    sens = result.getHasoferReliabilityIndexSensitivity()
    print("The sensitivity factors of the reliability index "
        "with regards to the distribution parameters are:\n")
    for i in range(sens.getSize()):
        print(sens.at(i))
        
    result.drawImportanceFactors()