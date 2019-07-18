from gurobipy import *
#import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op


##### Helper functions

def get_A(m, ncoef, deg):
    A = [[None for t in range(ncoef[k])] for k in range(deg + 1)]
    for k in range(deg + 1):
        for t in range(ncoef[k]):
            A[k][t] = m.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(t), lb = -GRB.INFINITY)
    return m, A

def get_interior(xDisc):
    xVals = list(np.linspace(0, 1, math.floor(1.0/xDisc + 1))[1:-1])
    yVals = list(np.linspace(0, 1, math.floor(1.0/xDisc + 1))[1:-1])
    xy = []
    for cx in xVals:
        for cy in yVals:
            xy.append([cx,cy]) 
    #print(intXY)
    return xy

def get_boundary(xDisc):
    ####xy1 are the points along y = 0
    xy1 = [[newX, 0.0] for newX in np.linspace(0,1, math.floor(1.0/xDisc + 1))]
    ####xy2 are the points along y = 1
    xy2 = [[newX, 1.0] for newX in np.linspace(0,1, math.floor(1.0/xDisc + 1))]
    ####xy3 are the points along x = 0
    xy3 = [[0.0, newY] for newY in np.linspace(0,1, math.floor(1.0/xDisc + 1))]
    ####xy4 are the points along x = 1
    xy4 = [[1.0, newY] for newY in np.linspace(0,1, math.floor(1.0/xDisc + 1))]    
    #print(xy1)
    #print(xy2)
    #print(xy3)
    #print(xy4)
    return xy1, xy2, xy3, xy4

def get_z(m, xy1, xy2, xy3, xy4):
    z1 = [None for l in range(len(xy1))]
    z2 = [None for l in range(len(xy2))]
    z3 = [None for l in range(len(xy3))]
    z4 = [None for l in range(len(xy4))]
    z11 = [None for l in range(len(xy1))]
    z22 = [None for l in range(len(xy2))]
    z33 = [None for l in range(len(xy3))]
    z44 = [None for l in range(len(xy4))]

    for l in range(len(xy1)):
        # add upper bound, UZ1 = upper bound for z1, UZ2 etc.
        z1[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(l))
        z2[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(l))
        z3[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(l))
        z4[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(l))
        z11[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z11_" + str(l))
        z22[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z22_" + str(l))
        z33[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z33_" + str(l))
        z44[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z44_" + str(l))

    return m, z1, z2, z3, z4, z11, z22, z33, z44

def get_fg(xy, xy1, xy2, xy3, xy4):
    g1Vals = [g1(ele[0],ele[1]) for ele in xy1]
    g2Vals = [g2(ele[0],ele[1]) for ele in xy2]
    g3Vals = [g3(ele[0],ele[1]) for ele in xy3]
    g4Vals = [g4(ele[0],ele[1]) for ele in xy4]
    fVals = [f(ele[0],ele[1]) for ele in xy]

    g11Vals = [g11(ele[0],ele[1]) for ele in xy1]
    g22Vals = [g22(ele[0],ele[1]) for ele in xy2]
    g33Vals = [g33(ele[0],ele[1]) for ele in xy3]
    g44Vals = [g44(ele[0],ele[1]) for ele in xy4]
    return fVals, g1Vals, g2Vals, g3Vals, g4Vals, g11Vals, g22Vals, g33Vals, g44Vals

def get_M_int(ncoef, xy, deg, terms):
    M = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy))]
    for l in range(len(xy)):
        for k in range(deg + 1):
            for t in range(ncoef[k]):
                for dTerm in terms: ###each run in this loop is for a separate term in the LHS of the PDE. Need to sum up 
                    temp = dTerm[2]
                    for r in range(0,dTerm[0]):
                        temp = temp*(k - t - r)
                    for s in range(0,dTerm[1]):
                        temp = temp*(t - s)
                    exp1 = max(0,k - t - dTerm[0]) ##avoid negative exponents for 0. If negative exponent, the if statement below will set this to zero anyway
                    exp2 = max(0, t - dTerm[1])
                    temp = temp*(xy[l][0]**(exp1))*xy[l][1]**(exp2)
                    if dTerm[0] > k-t or dTerm[1] > t:
                        temp = 0.0
                    M[l][k][t] = M[l][k][t] + temp
    return M
'''
    M = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy))]

     for l in range(len(xy)):
          for k in range(deg + 1):
               for t in range(ncoef[k]):
                    
                    for dTerm in terms: ###each run in this loop is for a separate term in the LHS of the PDE. Need to sum up 
                         temp = dTerm[2]
                         for r in range(0,dTerm[0]):
                              temp = temp*(k - t - r)
                         for s in range(0,dTerm[1]):
                              temp = temp*(t - s)
                         exp1 = max(0,k - t - dTerm[0]) ##avoid negative exponents for 0. If negative exponent, the if statement below will set this to zero anyway
                         exp2 = max(0, t - dTerm[1])
                         temp = temp*(xy[l][0]**(exp1))*xy[l][1]**(exp2)
                         if dTerm[0] > k-t or dTerm[1] > t:
                              temp = 0.0
                         M[l][k][t] = M[l][k][t] + temp
'''



def get_N_bc(ncoef, deg, xy1, xy2, xy3, xy4):
    # also needs M1,M2,M3,M4 for Dirichlet case
    M11 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy1))]
    M22 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy2))]
    M33 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy3))]
    M44 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy4))]

    for l in range(len(xy1)):
        for k in range(deg + 1):
            for t in range(ncoef[k]):
                temp3 = -max(k-t,0.0)*(xy3[l][0]**max(k-t-1,0))*xy3[l][1]**t
                M33[l][k][t] = temp3
                temp4 = max(k-t,0.0)*(xy4[l][0]**max(k-t-1,0))*xy4[l][1]**t
                M44[l][k][t] = temp4
                temp1 = -max(t,0.0)*(xy1[l][0]**(k-t))*xy1[l][1]**max(t-1,0)
                temp2 = max(t,0.0)*(xy2[l][0]**(k-t))*xy2[l][1]**max(t-1,0)
                M11[l][k][t] = temp1
                M22[l][k][t] = temp2
    return M11, M22, M33, M44


def PDE_const(m, xy, M, A, ncoef, deg, fVals):
    for l in range(len(xy)):
        m.addConstr(sum(sum( M[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1) ) == fVals[l], "fCon_" + str(l) ) 
    return m  

def D_const(m, xy1, xy2, xy3, xy4, A, g1Vals, g2Vals, g3Vals, g4Vals, ncoef, deg, z1, z2, z3, z4):
    for l in range(len(xy1)):
        m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy1[l][1]**t - g1Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z1[l] )
        m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy1[l][1]**t + g1Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z1[l] )
        m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy2[l][1]**t - g2Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z2[l] )
        m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy2[l][1]**t + g2Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z2[l] )
        m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy3[l][1]**t - g3Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z3[l] )
        m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy3[l][1]**t + g3Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z3[l] )
        m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy4[l][1]**t - g4Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z4[l] )
        m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy4[l][1]**t + g4Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z4[l] )
    return m

def new_D_const(m, xyi, A, giVals, ncoef, deg, zi):
    for l in range(len(xyi)):
        m.addConstr(sum(sum( A[k][t]*(xyi[l][0]**(k-t))*xyi[l][1]**t - giVals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= zi[l] )
        m.addConstr(-sum(sum( A[k][t]*(xyi[l][0]**(k-t))*xyi[l][1]**t + giVals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= zi[l] )
    return m

def N_const(m, xy1, M11, M22, M33, M44, A, ncoef, deg, g11Vals, g22Vals, g33Vals, g44Vals, z11, z22, z33, z44):
    for l in range(len(xy1)):
        m.addConstr(sum(sum( M11[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g11Vals[l] <= z11[l])
        m.addConstr(-sum(sum( M11[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g11Vals[l] <= z11[l])
        m.addConstr(sum(sum( M22[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g22Vals[l] <= z22[l])
        m.addConstr(-sum(sum( M22[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g22Vals[l] <= z22[l])
        m.addConstr(sum(sum( M33[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g33Vals[l] <= z33[l])
        m.addConstr(-sum(sum( M33[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g33Vals[l] <= z33[l])
        m.addConstr(sum(sum( M44[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g44Vals[l] <= z44[l])
        m.addConstr(-sum(sum( M44[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g44Vals[l] <= z44[l])
    return m
   
def new_N_const(m, xyi, Mii, A, ncoef, deg, giiVals, zii):
    for l in range(len(xyi)):
        m.addConstr(sum(sum( Mii[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - giiVals[l] <= zii[l])
        m.addConstr(-sum(sum( Mii[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + giiVals[l] <= zii[l])
    return m


def get_norms(m, xy1, dual1, dual2, dual3, dual4, dual11, dual22, dual33, dual44):
    norm1 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm1_", obj = dual1/(4*len(xy1)))
    norm2 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm2_", obj = dual2/(4*len(xy1)))
    norm3 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm3_", obj = dual3/(4*len(xy1)))
    norm4 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm4_", obj = dual4/(4*len(xy1)))
    norm11 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm11_", obj = dual11/(4*len(xy1)))
    norm22 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm22_", obj = dual22/(4*len(xy1)))
    norm33 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm33_", obj = dual33/(4*len(xy1)))
    norm44 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm44_", obj = dual44/(4*len(xy1)))
    return m, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44

def l1_Norm_const(m, z1, z2, z3, z4, z11, z22, z33, z44, xy1, xy2, xy3, xy4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim):
    # need to convert to abs(sum - norm) <= norm_lim 
    
    m.addConstr(sum(z1[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm1)
    m.addConstr(sum(z2[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm2)
    m.addConstr(sum(z3[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm3)
    m.addConstr(sum(z4[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm4)

    m.addConstr(sum(z11[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm11)
    m.addConstr(sum(z22[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm22)
    m.addConstr(sum(z33[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm33)
    m.addConstr(sum(z44[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm44)
    
    #norm_i <= norm_lim
    # norm_ii <= norm_lim
    m.addConstr(norm1 <= norm_lim)
    m.addConstr(norm2 <= norm_lim)
    m.addConstr(norm3 <= norm_lim)
    m.addConstr(norm4 <= norm_lim)
    
    m.addConstr(norm11 <= norm_lim)
    m.addConstr(norm22 <= norm_lim)
    m.addConstr(norm33 <= norm_lim)
    m.addConstr(norm44 <= norm_lim)
    
    return m

def l_max_Norm_const(m, z1, z2, z3, z4, z11, z22, z33, z44, xy1, xy2, xy3, xy4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim):
    # take max of all abs(z_i) or abs(z_ii), set <= norm_lim? 
    
    # norm_i >= z_i[j] for all j
    # norm_i <= lim
    for p in range(len(xy1)):
        m.addConstr(z1[p] <= norm1)
        m.addConstr(z11[p] <= norm11)
    for p in range(len(xy2)):
        m.addConstr(z2[p] <= norm2)
        m.addConstr(z22[p] <= norm22)
    for p in range(len(xy3)):
        m.addConstr(z3[p] <= norm3)
        m.addConstr(z33[p] <= norm33)
    for p in range(len(xy4)):
        m.addConstr(z4[p] <= norm4)
        m.addConstr(z44[p] <= norm44)
    
    m.addConstr(norm1 <= norm_lim)
    m.addConstr(norm2 <= norm_lim)
    m.addConstr(norm3 <= norm_lim)
    m.addConstr(norm4 <= norm_lim)
    
    m.addConstr(norm11 <= norm_lim)
    m.addConstr(norm22 <= norm_lim)
    m.addConstr(norm33 <= norm_lim)
    
    return m




####The goal of this code is to solve PDEs with polynomials as best we can even when the boundary conditions cannot be met exactly. This should be for general linear PDEs.

#xDisc is the discretization level of the grid 
#f is the source function
#g_{i} is the boundary condition on side i of the unit square
#g_{ii} is the neumann boundary condition on side i of the unit square
#deg is the maximum degree of the solution
#terms has the number of x and y derivatives for each term.
##terms[v] = [xd, yd, signMagnitude]

def LPGenELPDE(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, dual1, dual2, dual3, dual4, \
dual11, dual22, dual33, dual44):

    # 'dual' variables are intended to be Boolean switches dual1-4 for dirichlet
    # dual 11-44 for neumann
    # true if constraint for problem
    # false if the boundary condition has to be met with equality
    # add some new variable 'slack' to act as maximum error
    
    # 1-4 are dirichlet boundary conditions, see line 74-77, similar for 11-44 with neumann
    
    
     m = Model()
     m.setAttr("ModelSense", 1) ###minimize
     ###First, need to determine how many coefficients there are
     ncoef = [k + 1 for k in range(deg + 1)]

     m, A = get_A(m, ncoef, deg)

     

     ####make a list of interior sample points
     xy = get_interior(xDisc)


     ####points along y = 0, y = 1, x = 0, x = 1 respectively (for boundary conditions)
     xy1, xy2, xy3, xy4 = get_boundary(xDisc)
     
     ####Make auxiliary vars for boundary objectives
     m, z1, z2, z3, z4, z11, z22, z33, z44 = get_z(m, xy1, xy2, xy3, xy4)

     ####Now we need to evaluate the functions at our grid points
     fVals, g1Vals, g2Vals, g3Vals, g4Vals, g11Vals, g22Vals, g33Vals, g44Vals = get_fg(xy, xy1, xy2, xy3, xy4)

     ####Now we make the constraint matrix for the interior.
     M = get_M_int(ncoef, xy, deg, terms)

    # print(M)
    # print(" ")
    # print(xy)
#     ####Making constraint matrices for the boundary conditions
#     M1 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy1))]
#     M2 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy2))]
#     M3 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy3))]
#     M4 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy4))]

     M11 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy1))]
     M22 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy2))]
     M33 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy3))]
     M44 = [[[0.0 for t in range(ncoef[k])] for k in range(deg + 1)] for l in range(len(xy4))]

     for l in range(len(xy1)):
          for k in range(deg + 1):
               for t in range(ncoef[k]):
                    temp3 = -max(k-t,0.0)*(xy3[l][0]**max(k-t-1,0))*xy3[l][1]**t
                    M33[l][k][t] = temp3
                    temp4 = max(k-t,0.0)*(xy4[l][0]**max(k-t-1,0))*xy4[l][1]**t
                    M44[l][k][t] = temp4
                    temp1 = -max(t,0.0)*(xy1[l][0]**(k-t))*xy1[l][1]**max(t-1,0)
                    temp2 = max(t,0.0)*(xy2[l][0]**(k-t))*xy2[l][1]**max(t-1,0)
                    M11[l][k][t] = temp1
                    M22[l][k][t] = temp2


     ####First constraint: PDE(u(x_l)) = f(x_l)
     for l in range(len(xy)):
          m.addConstr(sum(sum( M[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1) ) == fVals[l], "fCon_" + str(l)  )

     ####Second constraint: |u(x) - g_i(x)| = z_i, x in gamma_i
     for l in range(len(xy1)):
          m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy1[l][1]**t - g1Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z1[l] )
          m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy1[l][1]**t + g1Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z1[l] )
          m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy2[l][1]**t - g2Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z2[l] )
          m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy2[l][1]**t + g2Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z2[l] )
          m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy3[l][1]**t - g3Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z3[l] )
          m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy3[l][1]**t + g3Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z3[l] )
          m.addConstr(sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy4[l][1]**t - g4Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z4[l] )
          m.addConstr(-sum(sum( A[k][t]*(xy1[l][0]**(k-t))*xy4[l][1]**t + g4Vals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= z4[l] )



     ####Third constraint: |du/dx_{i} - g_{ii}| = z_{ii} x in gamma_{i}
     for l in range(len(xy1)):
          m.addConstr(sum(sum( M11[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g11Vals[l] <= z11[l])
          m.addConstr(-sum(sum( M11[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g11Vals[l] <= z11[l])
          m.addConstr(sum(sum( M22[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g22Vals[l] <= z22[l])
          m.addConstr(-sum(sum( M22[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g22Vals[l] <= z22[l])
          m.addConstr(sum(sum( M33[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g33Vals[l] <= z33[l])
          m.addConstr(-sum(sum( M33[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g33Vals[l] <= z33[l])
          m.addConstr(sum(sum( M44[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g44Vals[l] <= z44[l])
          m.addConstr(-sum(sum( M44[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + g44Vals[l] <= z44[l])
    # could use total error?
    # l_1 norm: sum(z) < N
    # l_norm: z_i < N

     
     ####One more set of vars so we can see norms of different constraints
     norm1 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm1_", 
           obj = dual1/(4*len(xy1)))
     norm2 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm2_", 
           obj = dual2/(4*len(xy1)))
     norm3 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm3_",
           obj = dual3/(4*len(xy1)))
     norm4 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm4_",
           obj = dual4/(4*len(xy1)))
     norm11 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm11_",
           obj = dual11/(4*len(xy1)))
     norm22 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm22_",
           obj = dual22/(4*len(xy1)))
     norm33 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm33_",
           obj = dual33/(4*len(xy1)))
     norm44 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm44_",
           obj = dual44/(4*len(xy1)))

     m.addConstr(sum(z1[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm1)
     m.addConstr(sum(z2[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm2)
     m.addConstr(sum(z3[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm3)
     m.addConstr(sum(z4[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm4)

     m.addConstr(sum(z11[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm11)
     m.addConstr(sum(z22[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm22)
     m.addConstr(sum(z33[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm33)
     m.addConstr(sum(z44[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm44)
  
     ####update model
     m.update()
     m.optimize()
     return m




def useSol(m, xDisc, deg, shouldPlot, saveName, plotTitle):
     ncoef = [k + 1 for k in range(deg + 1)]
     ####discretize space
     intX = list(np.linspace(0, 1, 1.0/xDisc))
     intY = list(np.linspace(0, 1, 1.0/xDisc))
     intXY = []
     for cx in intX:
          for cy in intY:
               intXY.append([cx,cy])



     xArray = np.linspace(0,1, 1.0/xDisc)
     yArray = np.linspace(0,1, 1.0/xDisc)


     ####get the coefficients
     A = []
     for k in range(deg + 1):
          A.append([])
          for t in range(ncoef[k]):
               A[k].append(m.getVarByName("A_" + str(k) + "_" + str(t)).x); print(A[k][t])

     ####get function values in a plot friendly format

     print("u(.5,0): "); print(evalWithA(.5, 0.0, A, deg))

     xArray, yArray = np.meshgrid(xArray, yArray)
     uArray = evalWithA(xArray, yArray, A, deg); #print("uArray:"); print(uArray)

     ####Plot if requested

     if shouldPlot == True:
          fig = plt.figure()
          ax = plt.axes(projection = '3d')
          theX = [intXY[p][0] for p in range(len(intXY))]
          theY = [intXY[p][1] for p in range(len(intXY))];# print("theX: " + str(theX)); print("theY: " + str(theY))
          ax.plot_surface(xArray, yArray, uArray,cmap=plt.cm.jet)
          ax.set_xlabel("x"); ax.set_ylabel("y")
          plt.title(plotTitle)
          plt.savefig(saveName)
     ####Return the coefficients
     return A


def evalWithA(x,y, A, deg):
     ncoef = [k + 1 for k in range(deg + 1)]
     retVal = 0.0
     for k in range(deg + 1):
          for t in range(ncoef[k]):
               retVal = retVal + A[k][t]*(x**t)*(y**(k - t))
     return retVal

xDisc = .25;  
def g1(x,y):
     return x**2
def g2(x,y):
     return x**2 + 1
def g3(x,y):
     return y**3
def g4(x,y):
     return 1 + y**3
def f(x,y):
     return  2 + 6.0*y
def g11(x,y):
     return -3.0*y**2
def g22(x,y):
     return 3.0*y**2
def g33(x,y):
     return -2.0*x
def g44(x,y):
     return 2.0*x
d1 = []; d2 = []; d3 = []; d4 = []; deg = 3
terms = [[0,2,1.0], [2,0,1.0]]
newm = LPGenELPDE(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, 1.0, 1.0, 1.0, 1.0, \
1.0, 1.0, 1.0, 1.0)


#newm.write("PDE_LP1A.sol")

useSol(newm, .05, deg, True, "prob1ALP_plot.png", r'Computed Solution to $\Delta u = 1 - 2y$ (Prob4)')