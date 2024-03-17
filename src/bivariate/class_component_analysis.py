# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import fsolve # for plotting the LSF

import openturns as ot
import openturns.viewer as viewer
ot.Log.Show(ot.Log.NONE)

import numpy as np
import scipy.linalg as scp
import matplotlib.pylab as plt
import time

import sympy as sp

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


class FEM_floating_submerged_curved_tunnel():
    
    def __init__(self):
        
        
        
    def create_beam_matrices(length = 17000, Deviation = 2000, a = 23.8, b = 9.2, A = 38, E_c = 50e9, rho_c = 2500, Beam_m = 150000):
            
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

        # print(DofsP)
        # print(DofsF)

        M_FF = [ M[iRow,DofsF].tolist() for iRow in DofsF ]
        K_FF = [ K[iRow,DofsF].tolist() for iRow in DofsF ]
        np.shape(M_FF)
        
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
 
 


       
import numpy as np
import math
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