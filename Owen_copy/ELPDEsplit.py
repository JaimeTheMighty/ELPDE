#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Mar 2019

@author: owenmichals
"""
from gurobipy import *
#import matplotlib as mpl
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

# Creates the A matrix
def get_A(m, ncoef, deg):    
    A = [[None for t in range(ncoef[k])] for k in range(deg + 1)]
    for k in range(deg + 1):
        for t in range(ncoef[k]):
            A[k][t] = m.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(t), lb = -GRB.INFINITY)
    return m, A

# Finds all points in the interior of the grid
def get_interior(Disc):
    xVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    yVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    xy = []
    for cx in xVals:
        for cy in yVals:
            xy.append([cx,cy]) 
    return xy

# Finds all points on the boundary of the grid
def get_boundary(Disc):
    ####xy1 are the points along y = 0
    xy1 = [[newX, 0.0] for newX in np.linspace(0,1, math.floor(1.0/Disc + 1))]
    ####xy2 are the points along y = 1
    xy2 = [[newX, 1.0] for newX in np.linspace(0,1, math.floor(1.0/Disc + 1))]
    ####xy3 are the points along x = 0
    xy3 = [[0.0, newY] for newY in np.linspace(0,1, math.floor(1.0/Disc + 1))]
    ####xy4 are the points along x = 1
    xy4 = [[1.0, newY] for newY in np.linspace(0,1, math.floor(1.0/Disc + 1))]    
    #print(xy1)
    #print(xy2)
    #print(xy3)
    #print(xy4)
    return xy1, xy2, xy3, xy4

# Creates the z variables
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

# Computes the Laplacian on the interior, and the value and derivatives along the boundaries
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

# Creates the M matrix
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

# Creates the Mii matrices
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


# Applies the Laplacian constraint
def PDE_const(m, xy, M, A, ncoef, deg, fVals):
    for l in range(len(xy)):
        m.addConstr(sum(sum( M[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1) ) == fVals[l], "fCon_" + str(l) ) 
    return m  

# Applies Dirichlet constraints - OUTDATED, use new_D_const for more control
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

# Applies Dirichlet constraint
def new_D_const(m, xyi, A, giVals, ncoef, deg, zi):
    for l in range(len(xyi)):
        m.addConstr(sum(sum( A[k][t]*(xyi[l][0]**(k-t))*xyi[l][1]**t  for t in range(ncoef[k]) ) for k in range(deg + 1))  - giVals[l]  <= zi[l] )
        m.addConstr(-sum(sum( A[k][t]*(xyi[l][0]**(k-t))*xyi[l][1]**t  for t in range(ncoef[k]) ) for k in range(deg + 1)) + giVals[l] <= zi[l] )
    return m

# Applies Neumann constraints - OUTDATED, use new_N_const for more control
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

# Applies Neumann constraint  
def new_N_const(m, xyi, Mii, A, ncoef, deg, giiVals, zii):
    for l in range(len(xyi)):
        m.addConstr(sum(sum( Mii[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - giiVals[l] <= zii[l])
        m.addConstr(-sum(sum( Mii[l][k][t]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) + giiVals[l] <= zii[l])
    return m


# Creates the variables for each normi and normii
def get_norms(m, xy1, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4):
    norm1 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm1_")
    norm2 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm2_")
    norm3 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm3_")
    norm4 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm4_")
    norm11 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm11_")
    norm22 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm22_")
    norm33 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm33_")
    norm44 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm44_")
    return m, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44

# Computes the L1 norms and enforces that they are less than norm_lim
def l1_Norm_const(m, z1, z2, z3, z4, z11, z22, z33, z44, xy1, xy2, xy3, xy4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim):
    
    m.addConstr(sum(z1[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm1)
    m.addConstr(sum(z2[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm2)
    m.addConstr(sum(z3[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm3)
    m.addConstr(sum(z4[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm4)

    m.addConstr(sum(z11[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm11)
    m.addConstr(sum(z22[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm22)
    m.addConstr(sum(z33[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm33)
    m.addConstr(sum(z44[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm44)
    
    m.addConstr(norm1 <= norm_lim)
    m.addConstr(norm2 <= norm_lim)
    m.addConstr(norm3 <= norm_lim)
    m.addConstr(norm4 <= norm_lim)
    
    m.addConstr(norm11 <= norm_lim)
    m.addConstr(norm22 <= norm_lim)
    m.addConstr(norm33 <= norm_lim)
    m.addConstr(norm44 <= norm_lim)
    
    return m

# Computes the L-Inf norms (max norms) and enforces that they are less than norm_lim
def l_max_Norm_const(m, z1, z2, z3, z4, z11, z22, z33, z44, xy1, xy2, xy3, xy4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim):
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

# Creates the objective function
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
def LPGenELPDE(Disc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, DBC1, DBC2, DBC3, DBC4, \
NBC1, NBC2, NBC3, NBC4, l1_norm, norm_lim):        
     m = Model()
     m.setAttr("ModelSense", 1) ###minimize
     ###First, need to determine how many coefficients there are
     ncoef = [k + 1 for k in range(deg + 1)]

     # A is the variables representing the coefficients of polynomial terms
     # these variables get added to m
     m, A = get_A(m, ncoef, deg)
     
     # xy is a list of interior sample points
     xy = get_interior(Disc)

     # points along y = 0, y = 1, x = 0, x = 1 respectively (for boundary conditions)
     xy1, xy2, xy3, xy4 = get_boundary(Disc)
     
     # auxiliary variables for points and derivatives along y = 0, y = 1, x = 0, x = 1 respectively (for boundary objectives)
     m, z1, z2, z3, z4, z11, z22, z33, z44 = get_z(m, xy1, xy2, xy3, xy4)

     # Not really sure what to do with this. This was present in the original ELPDE code, but was already
     # commented out, and seemed unnecessary
     ####will fill in with x^t y^k-t for each of the grid points, do for all exponent combinations
     # stuff had been here, but was already commented out
     
     # fvals - value of the Laplacian at every point in the grid
     # g1vals - value along y = 0
     # g2vals - value along y = 1
     # g3vals - value along x = 0
     # g4vals - value along x = 1
     # g11vals - value of the derivative along y = 0
     # g22vals - value of the derivative along y = 1
     # g33vals - value of the derivative along x = 0
     # g44vals - value of the derivative along x = 1
     fVals, g1Vals, g2Vals, g3Vals, g4Vals, g11Vals, g22Vals, g33Vals, g44Vals = get_fg(xy, xy1, xy2, xy3, xy4)

     # M - Constraint matrix that caa be combined with A to enforce the Laplacian constraint
     M = get_M_int(ncoef, xy, deg, terms)
     
     # M11 - Constraint matrix be combined with A to enforce a Neumann BC on y = 0
     # M22 - Constraint matrix be combined with A to enforce a Neumann BC on y = 1
     # M33 - Constraint matrix be combined with A to enforce a Neumann BC on x = 0
     # M44 - Constraint matrix be combined with A to enforce a Neumann BC on x = 1
     M11, M22, M33, M44 = get_N_bc(ncoef, deg, xy1, xy2, xy3, xy4)

     # This is not needed for Dirichlet BC, as A can be combined with xy1, xy2, xy3, xy4 directly

     # Force the Laplacian as a constraint using M and A
     m = PDE_const(m, xy, M, A, ncoef, deg, fVals)
     
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
     m, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44 = get_norms(m, xy1, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4)
     
     # Calculate the values that the norms take, where normi is based on zi and normii is based on zii
     if l1_norm:
         # use the L1 calculation of norm
         m = l1_Norm_const(m, z1, z2, z3, z4, z11, z22, z33, z44, xy1, xy2, xy3, xy4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim)
     else:
         # use the L-Inf calculation of norm (max norm)
         m = l_max_Norm_const(m, z1, z2, z3, z4, z11, z22, z33, z44, xy1, xy2, xy3, xy4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44, norm_lim)
    
     # create an objective function by summing over the norms that correspond the boundary conditions
     m = createObjective(m, DBC1, DBC2, DBC3, DBC4, NBC1, NBC2, NBC3, NBC4, norm1, norm2, norm3, norm4, norm11, norm22, norm33, norm44)
     
     ####update model
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


def evalWithA(x,y, A, deg):
     ncoef = [k + 1 for k in range(deg + 1)]
     retVal = 0.0
     for k in range(deg + 1):
          for t in range(ncoef[k]):
               retVal = retVal + A[k][t]*(x**t)*(y**(k - t))
     return retVal


############# Test cases etc.

'''
Disc = .25;  
def g1(x,y):
     return 0.0
def g2(x,y):
     return 0
def g3(x,y):
     return 0
def g4(x,y):
     return 0
def f(x,y):
     return  -2.0*y*(1.0 - y) - 2.0*x*(1.0-x)
def g11(x,y):
     return -(1 - 2.0*y)*x*(1-x)
def g22(x,y):
     return (1 - 2.0*y)*x*(1-x)
def g33(x,y):
     return -(1 - 2.0*x)*y*(1-y)
def g44(x,y):
     return (1 - 2.0*x)*y*(1-y)
deg = 4
terms = [[2, 0, 1.0], [0, 3, 1.0] ]
newm = LPGenELPDE(Disc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 1, 1)
'''
'''
####BC1: u(x,y) on y = 0
####BC2: u(x,y) on y = 1
####BC3: u(x,y) on x = 0
####BC4: u(x,y) on x = 1
####Problem 10: Lap(u) = -2y(1 - y) - 2x(1-x)
Disc = .01;  
def g1(x,y):
     return 0
def g2(x,y):
     return 0
def g3(x,y):
     return 0
def g4(x,y):
     return 0
def f(x,y):
     return -2.0*y*(1.0 - y) - 2.0*x*(1.0-x)
def g11(x,y):
     return -(1 - 2.0*y)*x*(1-x) + .1
def g22(x,y):
     return (1 - 2.0*y)*x*(1-x)
def g33(x,y):
     return -(1 - 2.0*x)*y*(1-y) - .1
def g44(x,y):
     return (1 - 2.0*x)*y*(1-y)
#terms = [[4, 0, 1.0], [3, 0, -1.0], [0, 4, 1.0], [0, 3, -1.0] ]
terms = [ [2, 0, 1.0], [0, 2, 1.0] ]
deg = 4
newm = LPGenELPDE(Disc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1, 100)
newm.write("PDE_LP1.sol")
useSol(newm, Disc, deg, True, "prob1LP_plot.png", r'Computed Solution to $\Delta u = 4$ (Prob1)')
'''
####BC1: u(x,y) on y = 0
####BC2: u(x,y) on y = 1
####BC3: u(x,y) on x = 0
####BC4: u(x,y) on x = 1
Disc = .05;  
def g1(x,y):
     return x**3
def g2(x,y):
     return x**3 
def g3(x,y):
     return y**3
def g4(x,y):
     return y**3
def f(x,y):
     return x + y
def g11(x,y):
     return 0
def g22(x,y):
     return 0
def g33(x,y):
     return 0
def g44(x,y):
     return 0
deg = 3
terms = [[0,2,1.0], [2,0,1.0]]
newm = LPGenELPDE(Disc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, 1.0, 1.0, 1.0, 1.0, \
0, 0, 0, 0, 0, 1)
newm.write("PDE_LP1A.sol")
useSol(newm, Disc, deg, True, "prob1ALP_plot.png", r'Computed Solution to $\Delta u = x+y$ (Prob4)')







