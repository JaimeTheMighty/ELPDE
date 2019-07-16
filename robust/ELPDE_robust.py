# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:35:47 2019

@author: James Liu
"""

#Based on ELPDESplit by owenmichals
#This version attempts to formulate the problem under a stochastic programming/robust optimization setting, each different
#scenario corresponds to a different set of values for the rhs of the boundary conditions. The formulation focuses on the 
#feasibility of the problem.

from gurobipy import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op

'''
Variables that appear throughout:
Disc - (float) discretization of the grid on which the solution is computes. This number gets rounded if it does not go into 1 evenly
f - function that computes the Laplacian of the solution
g1 - value of the function along y = 0 (for Dirichlet BC)
g11 - value of a derivative along y = 0 (for Neumann BC) 
g2 - value of the function along y = 1 (for Dirichlet BC)
g22 - value of a derivative along y = 1 (for Neumann BC)  
g3 - value of the function along x = 0 (for Dirichlet BC)
g33 - value of a derivative along x = 0 (for Neumann BC) 
g4 - value of the function along x = 1 (for Dirichlet BC)
g44 - value of a derivative along x = 0 (for Neumann BC) 
deg - (int) degree of the solution 
terms - (list) list of terms in the PDE, each term is one element in the list
    in the form [xd, yd, coef] where xd is the degree of the derivative with respect to x
    for that term, yd is the degree of the derivative with respect to y for that term, and
    coef is the coefficient of that term
DBC1 - (0 or 1) 1 if g1 is being used as a Dirichlet BC, 0 otherwise
DBC2 - (0 or 1) 1 if g2 is being used as a Dirichlet BC, 0 otherwise
DBC3 - (0 or 1) 1 if g3 is being used as a Dirichlet BC, 0 otherwise
DBC4 - (0 or 1) 1 if g4 is being used as a Dirichlet BC, 0 otherwise
NBC1 - (0 or 1) 1 if g11 is being used as a Neumann BC, 0 otherwise 
NBC2 - (0 or 1) 1 if g22 is being used as a Neumann BC, 0 otherwise 
NBC3 - (0 or 1) 1 if g33 is being used as a Neumann BC, 0 otherwise 
NBC4 - (0 or 1) 1 if g44 is being used as a Neumann BC, 0 otherwise 
l1_norm - (0 or 1) 1 if using the L1 norm as a method of evaluating potential solutions, 0 if using max norm (infinity norm)
norm_lim - (float) upper limit on the norms in potential solutions
'''


##### Helper functions

# Creates the A matrix, UNCHANGED
def get_A(m, ncoef, deg):    
    A = [[None for t in range(ncoef[k])] for k in range(deg + 1)]
    for k in range(deg + 1):
        for t in range(ncoef[k]):
            A[k][t] = m.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(t), lb = -GRB.INFINITY)
    return m, A

# Finds all points in the interior of the grid, UNCHANGED
def get_interior(Disc):
    xVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    yVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    xy = []
    for cx in xVals:
        for cy in yVals:
            xy.append([cx,cy]) 
    return xy

# Finds all points on the boundary of the grid, UNCHANGED
def get_boundary(Disc):
    ####xy1 are the points along y = 0
    xy1 = [[newX, 0.0] for newX in np.linspace(0,1, math.floor(1.0/Disc + 1))]
    ####xy2 are the points along y = 1
    xy2 = [[newX, 1.0] for newX in np.linspace(0,1, math.floor(1.0/Disc + 1))]
    ####xy3 are the points along x = 0
    xy3 = [[0.0, newY] for newY in np.linspace(0,1, math.floor(1.0/Disc + 1))]
    ####xy4 are the points along x = 1
    xy4 = [[1.0, newY] for newY in np.linspace(0,1, math.floor(1.0/Disc + 1))]    
#    print(xy1)
#    print(xy2)
#    print(xy3)
#    print(xy4)
    return xy1, xy2, xy3, xy4

# Creates the z variables， CHANGED, it is used iteratively with each scenario
    #added new input argument s, for the scenario
    #added zint, for interior
def get_z(m, xy, xy1, xy2, xy3, xy4, s):
    z1 = [None for l in range(len(xy1))]
    z2 = [None for l in range(len(xy2))]
    z3 = [None for l in range(len(xy3))]
    z4 = [None for l in range(len(xy4))]
    z11 = [None for l in range(len(xy1))]
    z22 = [None for l in range(len(xy2))]
    z33 = [None for l in range(len(xy3))]
    z44 = [None for l in range(len(xy4))]
    
    zint = [None for l in range(len(xy))]

    for l in range(len(xy1)):
        # add upper bound, UZ1 = upper bound for z1, UZ2 etc.
        z1[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(s) + "_" + str(l))
        z2[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(s) + "_" + str(l))
        z3[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(s) + "_" + str(l))
        z4[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(s) + "_" + str(l))
        z11[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z11_" + str(s) + "_" + str(l))
        z22[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z22_" + str(s) + "_" + str(l))
        z33[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z33_" + str(s) + "_" + str(l))
        z44[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z44_" + str(s) + "_" + str(l))
    
    for l in range(len(xy)):
        zint[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "zint_" + str(s) + "_" + str(l))

    return m, zint, z1, z2, z3, z4, z11, z22, z33, z44

# Computes the Laplacian on the interior, and the value and derivatives along the boundaries, CHANGED
    #added argument s, for each scenario
    #g_i and g_ii's are assumed to take in the indicator of the scenario to return different values
    #fVals is deterministic
def get_fg(xy, xy1, xy2, xy3, xy4, s):
    g1Vals = [g1(ele[0],ele[1], s) for ele in xy1]
    g2Vals = [g2(ele[0],ele[1], s) for ele in xy2]
    g3Vals = [g3(ele[0],ele[1], s) for ele in xy3]
    g4Vals = [g4(ele[0],ele[1], s) for ele in xy4]
    fVals = [f(ele[0],ele[1]) for ele in xy]

    g11Vals = [g11(ele[0],ele[1], s) for ele in xy1]
    g22Vals = [g22(ele[0],ele[1], s) for ele in xy2]
    g33Vals = [g33(ele[0],ele[1], s) for ele in xy3]
    g44Vals = [g44(ele[0],ele[1], s) for ele in xy4]
    return fVals, g1Vals, g2Vals, g3Vals, g4Vals, g11Vals, g22Vals, g33Vals, g44Vals

# Creates the M matrix, no reason to change, the lhs coefficients are the same
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

# Creates the Mii matrices, no reason to change
def get_N_bc(ncoef, deg, xy1, xy2, xy3, xy4):
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


# Applies the Laplacian constraint, CHANGED
    #adds a new argument s, for scenarios
    #adds new auxiliary variables zint_s_l, so the constraints can be rewritten in a norm-like fashion
        #i.e. |u(x)-f(x)|=zint, zint<=ub
def PDE_const(m, xy, M, A, ncoef, deg, fVals, zint):
    for l in range(len(xy)):
        m.addConstr(sum(sum( M[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1) ) - fVals[l] == zint[l], "fCon_" + str(l) ) 
    return m


# Applies Dirichlet constraint, uniform expression across all scenarios
def new_D_const(m, xyi, A, giVals, ncoef, deg, zi):
    for l in range(len(xyi)):
        m.addConstr(sum(sum( A[k][t]*(xyi[l][0]**(k-t))*xyi[l][1]**t - giVals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= zi[l] )
        m.addConstr(-sum(sum( A[k][t]*(xyi[l][0]**(k-t))*xyi[l][1]**t + giVals[l] for t in range(ncoef[k]) ) for k in range(deg + 1)) <= zi[l] )
    return m


# Applies Neumann constraint, uniform expression across all scenarios
def new_N_const(m, xyi, Mii, A, ncoef, deg, giiVals, zii):
    for l in range(len(xyi)):
        m.addConstr(sum(sum( Mii[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - giiVals[l] <= zii[l])
        m.addConstr(-sum(sum( Mii[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + giiVals[l] <= zii[l])
    return m


#Creates the variable for interior norm, for scenario s
def get_norms_int(m, s):
    normint = m.addVar(vtype = GRB.CONTINUOUS, name="normint_" + str(s))
    return m, normint


# Creates the variables for each normi and normii, for scenario s
def get_norms(m, xy1, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4, s):
    norm1 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm1_" + str(s))
    norm2 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm2_" + str(s))
    norm3 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm3_" + str(s))
    norm4 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm4_" + str(s))
    norm11 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm11_" + str(s))
    norm22 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm22_" + str(s))
    norm33 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm33_" + str(s))
    norm44 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm44_" + str(s))
    return m, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44

# Computes the L1 norms and enforces that they are less than norm_lim, both on the boundaries and interior
def l1_Norm_const(m, zint, z1, z2, z3, z4, z11, z22, z33, z44, xy, xy1, xy2, xy3, xy4, \
                  normint, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim_b, norm_lim_int):
    
    m.addConstr(sum(z1[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm1)
    m.addConstr(sum(z2[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm2)
    m.addConstr(sum(z3[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm3)
    m.addConstr(sum(z4[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm4)

    m.addConstr(sum(z11[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm11)
    m.addConstr(sum(z22[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm22)
    m.addConstr(sum(z33[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm33)
    m.addConstr(sum(z44[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm44)
    
    m.addConstr(sum(zint[p] for p in range(len(xy)))*(1.0/len(xy)) == normint)
    
    m.addConstr(norm1 <= norm_lim_b)
    m.addConstr(norm2 <= norm_lim_b)
    m.addConstr(norm3 <= norm_lim_b)
    m.addConstr(norm4 <= norm_lim_b)
    
    m.addConstr(norm11 <= norm_lim_b)
    m.addConstr(norm22 <= norm_lim_b)
    m.addConstr(norm33 <= norm_lim_b)
    m.addConstr(norm44 <= norm_lim_b)
    
    m.addConstr(normint <= norm_lim_int)
    return m

# Computes the L-Inf norms (max norms) and enforces that they are less than norm_lim, both on the boundaries and the interior
def l_max_Norm_const(m, zint, z1, z2, z3, z4, z11, z22, z33, z44, xy, xy1, xy2, xy3, xy4, \
                     normint, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim_b, norm_lim_int):
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
    
    for p in range(len(xy)):
        m.addConstr(zint[p] <= normint)
    
    m.addConstr(norm1 <= norm_lim_b)
    m.addConstr(norm2 <= norm_lim_b)
    m.addConstr(norm3 <= norm_lim_b)
    m.addConstr(norm4 <= norm_lim_b)
    
    m.addConstr(norm11 <= norm_lim_b)
    m.addConstr(norm22 <= norm_lim_b)
    m.addConstr(norm33 <= norm_lim_b)
    m.addConstr(norm44 <= norm_lim_b)
    
    m.addConstr(normint <= norm_lim_int)
    
    return m

# Creates the objective function, unsure of the objective function yet, needs refinement
def createObjective(m, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44):
    objective = LinExpr()
    if DBC1:
        objective.addTerms(1, norm1)
    if DBC2:
        objective.addTerms(1, norm2)
    if DBC3:
        objective.addTerms(1, norm3)
    if DBC4:
        objective.addTerms(1, norm4)
        
    if NBC1:
        objective.addTerms(1, norm11)
    if NBC2:
        objective.addTerms(1, norm22)
    if NBC3:
        objective.addTerms(1, norm33)
    if NBC4:
        objective.addTerms(1, norm44)
    
    m.setObjective(objective)
    return m


#New version for setting the objective, the input includes the existing objective, an update
    #Normint is neglected for now
def UpdateObjective_robust(m, objective,\
                           DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44):
    if DBC1:
        objective.addTerms(1, norm1)
    if DBC2:
        objective.addTerms(1, norm2)
    if DBC3:
        objective.addTerms(1, norm3)
    if DBC4:
        objective.addTerms(1, norm4)
        
    if NBC1:
        objective.addTerms(1, norm11)
    if NBC2:
        objective.addTerms(1, norm22)
    if NBC3:
        objective.addTerms(1, norm33)
    if NBC4:
        objective.addTerms(1, norm44)
    
    m.setObjective(objective)
    return m
              
#####




'''
This function attempts to find approximate polynomial solutions for linear PDEs.

Disc - (float) discretization of the grid on which the solution is computes. This number gets rounded if it does not go into 1 evenly
f - function that computes the Laplacian of the solution
g1 - value of the function along y = 0 (for Dirichlet BC)
g11 - value of a derivative along y = 0 (for Neumann BC) 
g2 - value of the function along y = 1 (for Dirichlet BC)
g22 - value of a derivative along y = 1 (for Neumann BC)  
g3 - value of the function along x = 0 (for Dirichlet BC)
g33 - value of a derivative along x = 0 (for Neumann BC) 
g4 - value of the function along x = 1 (for Dirichlet BC)
g44 - value of a derivative along x = 0 (for Neumann BC) 
deg - (int) degree of the solution 
terms - (list) list of terms in the PDE, each term is one element in the list
    in the form [xd, yd, coef] where xd is the degree of the derivative with respect to x
    for that term, yd is the degree of the derivative with respect to y for that term, and
    coef is the coefficient of that term
DBC1 - (0 or 1) 1 if g1 is being used as a Dirichlet BC, 0 otherwise
DBC2 - (0 or 1) 1 if g2 is being used as a Dirichlet BC, 0 otherwise
DBC3 - (0 or 1) 1 if g3 is being used as a Dirichlet BC, 0 otherwise
DBC4 - (0 or 1) 1 if g4 is being used as a Dirichlet BC, 0 otherwise
NBC1 - (0 or 1) 1 if g11 is being used as a Neumann BC, 0 otherwise 
NBC2 - (0 or 1) 1 if g22 is being used as a Neumann BC, 0 otherwise 
NBC3 - (0 or 1) 1 if g33 is being used as a Neumann BC, 0 otherwise 
NBC4 - (0 or 1) 1 if g44 is being used as a Neumann BC, 0 otherwise 
l1_norm - (0 or 1) 1 if using the L1 norm as a method of evaluating potential solutions, 0 if using max norm (infinity norm)
norm_lim - (float) upper limit on the norms in potential solutions
'''


#Updated vesion, the model is input, and added input s
     #could be further refined so the generating of xy and xyi are done outside the loop
     #Also the generating of M and N can be done outside the loop
def LPGen_Scenario(Disc, m, s, A, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, DBC1, DBC2, DBC3, DBC4, \
NBC1, NBC2, NBC3, NBC4, l1_norm, norm_lim_b, norm_lim_int, objective):        

     ###First, need to determine how many coefficients there are
     ncoef = [k + 1 for k in range(deg + 1)]

     # UPDATED: A is now added outside the loop !!!!!!!
     # A is the variables representing the coefficients of polynomial terms
     # these variables get added to m
     # m, A = get_A(m, ncoef, deg)
     
     # xy is a list of interior sample points
     xy = get_interior(Disc)

     # points along y = 0, y = 1, x = 0, x = 1 respectively (for boundary conditions)
     xy1, xy2, xy3, xy4 = get_boundary(Disc)
     
     # auxiliary variables for points and derivatives along y = 0, y = 1, x = 0, x = 1 
     # and for the values on the interior
     m, zint, z1, z2, z3, z4, z11, z22, z33, z44 = get_z(m, xy, xy1, xy2, xy3, xy4, s)


     
     # fvals - value of the Laplacian at every point in the grid
     # g1vals - value along y = 0
     # g2vals - value along y = 1
     # g3vals - value along x = 0
     # g4vals - value along x = 1
     # g11vals - value of the derivative along y = 0
     # g22vals - value of the derivative along y = 1
     # g33vals - value of the derivative along x = 0
     # g44vals - value of the derivative along x = 1
     fVals, g1Vals, g2Vals, g3Vals, g4Vals, g11Vals, g22Vals, g33Vals, g44Vals = get_fg(xy, xy1, xy2, xy3, xy4, s)

     # M - Constraint matrix that can be combined with A to enforce the Laplacian constraint
     M = get_M_int(ncoef, xy, deg, terms)
     
     # M11 - Constraint matrix be combined with A to enforce a Neumann BC on y = 0
     # M22 - Constraint matrix be combined with A to enforce a Neumann BC on y = 1
     # M33 - Constraint matrix be combined with A to enforce a Neumann BC on x = 0
     # M44 - Constraint matrix be combined with A to enforce a Neumann BC on x = 1
     M11, M22, M33, M44 = get_N_bc(ncoef, deg, xy1, xy2, xy3, xy4)

     # This is not needed for Dirichlet BC, as A can be combined with xy1, xy2, xy3, xy4 directly

     # Force the Laplacian as a constraint using M, A, zint
     # this will be done outside the loop !!!!!!!
     # m = PDE_const(m, xy, M, A, ncoef, deg, fVals, zint)
     
     
     # Force any desired Dirichlet BC using the corresponding xyi, A, giVals, zi
     # Constraints take form: |u(x) - g_i(x)| = z_i
     if DBC1:
         m = new_D_const(m, xy1, A, g1Vals, ncoef, deg, z1)
     if DBC2:
         m = new_D_const(m, xy2, A, g2Vals, ncoef, deg, z2)
     if DBC3:
         m = new_D_const(m, xy3, A, g3Vals, ncoef, deg, z3)
     if DBC4:
         m = new_D_const(m, xy4, A, g4Vals, ncoef, deg, z4)

     # Force any desired Neumann BC using the corresponding xyi, Mii, A, giiVals, zii
     # Constraints take form: |du/dx_{i} - g_{ii}| = z_{ii}
     if NBC1:
         m = new_N_const(m, xy1, M11, A, ncoef, deg, g11Vals, z11)
     if NBC2:
         m = new_N_const(m, xy2, M22, A, ncoef, deg, g22Vals, z22)
     if NBC3:
         m = new_N_const(m, xy3, M33, A, ncoef, deg, g33Vals, z33)
     if NBC4:
         m = new_N_const(m, xy4, M44, A, ncoef, deg, g44Vals, z44)
     
     # Create variables for the norms 
     m, normint = get_norms_int(m, s)
     m, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44 = get_norms(m, xy1, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4, s)
     
     # Calculate the values that the norms take, where normi is based on zi and normii is based on zii
     if l1_norm:
         # use the L1 calculation of norm
         m = l1_Norm_const(m, zint, z1, z2, z3, z4, z11, z22, z33, z44, xy, xy1, xy2, xy3, xy4, \
                  normint, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim_b, norm_lim_int)
     else:
         # use the L-Inf calculation of norm (max norm)
         m = l_max_Norm_const(m, zint, z1, z2, z3, z4, z11, z22, z33, z44, xy, xy1, xy2, xy3, xy4, \
                  normint, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim_b, norm_lim_int)
    
     # create an objective function by summing over the norms that correspond the boundary conditions
     m = UpdateObjective_robust(m, objective, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44)
     
     ####update model
     m.update()
     
     #No need for optimization now
     #m.optimize()
     
     return m

'''
This will be the main function to wrap up the scenarios.
The inputs are added with variable sce, giving us the scenarios
'''
def LPGenDEP (Disc, f, g1, g11, g2, g22, g3, g33, g4, g44, sce, deg, terms, DBC1, DBC2, DBC3, DBC4, \
              NBC1, NBC2, NBC3, NBC4, l1_norm, norm_lim_b, norm_lim_int):
    
    ##First we build up the model by specifying m and A
    m=Model()
    m.setAttr("ModelSense", 1) 
    ncoef = [k + 1 for k in range(deg + 1)]
    m, A=get_A(m, ncoef, deg)
    
    #Second build a universal objective that will be updated in each scenario problem
    objective = LinExpr()
    
    for s in sce:
        m = LPGen_Scenario(Disc, m, s, A, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, DBC1, DBC2, DBC3, DBC4, \
                       NBC1, NBC2, NBC3, NBC4, l1_norm, norm_lim_b, norm_lim_int, objective)
    
    
    m.update()
    m.optimize()
    
    return m



###### Function to display results

# Plots the results of a model such as the one outputed by LPGenELPDE
def useSol(m, Disc, deg, shouldPlot, saveName, plotTitle):
     ncoef = [k + 1 for k in range(deg + 1)]
     ####discretize space
     intX = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
     intY = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
     intXY = []
     for cx in intX:
          for cy in intY:
               intXY.append([cx,cy])


     xArray = np.linspace(0,1, math.floor(1.0/Disc + 1))[1:-1]
     yArray = np.linspace(0,1, math.floor(1.0/Disc + 1))[1:-1]


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


def evalWithA(x, y, A, deg):
     ncoef = [k + 1 for k in range(deg + 1)]
     retVal = 0.0
     for k in range(deg + 1):
          for t in range(ncoef[k]):
               retVal = retVal + A[k][t]*(x**t)*(y**(k - t))
     return retVal


############# Test case
     ######## For each scenario input into the gi and gii's, the output is different

Disc = .05;  


def g1(x, y, s):
    if s==1:
        return x**3
    elif s==2:
        return x**2
def g2(x, y, s):
    if s==1:
        return x**3
    elif s==2:
        return x**2
def g3(x, y, s):
    if s==1:
        return y**3
    elif s==2:
        return y**2
def g4(x, y, s):
    if s==1:
        return y**3
    elif s==2:
        return y**2
 
def g11(x, y, s):
     return 0
def g22(x, y, s):
     return 0
def g33(x, y, s):
     return 0
def g44(x, y, s):
     return 0    


def f(x, y):
     return x + y

deg = 3
terms = [[0,2,1.0], [2,0,1.0]]
sce=[1,2]


#Norm_lim_int set to 0
newm = LPGenDEP (Disc, f, g1, g11, g2, g22, g3, g33, g4, g44, sce, deg, terms, 1.0, 1.0, 1.0, 1.0, \
              0, 0, 0, 0, 0, 1, 0)

newm.write("PDE_LPori.sol")
useSol(newm, Disc, deg, True, "probori_plot.png", r'Computed Solution to $\Delta u = x + y $')









