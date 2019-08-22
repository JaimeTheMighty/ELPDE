# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:25:11 2019

@author: James Lau
"""

#This version allows us to enforce bound constraints 
# Based on ELPDE_3D, has x, y and t dimensions

##The approximate function takes the following form:
## U(x,y,t)=\sum_{k=0}^{K}\sum_{m=0}^{k}\sum_{n=0}^m A_{k,m,n} x^n y^{m-n} t^{k-m}

from gurobipy import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op


"""
Notations used throughout the code:
    xyt   : set of interior points
    b1~b6 : set of boundary planes points
"""

"""
Helper functions to create needed data
"""

##Creates the decision variables A
##Note that A can be used in A[k][m][n] fashion, with deg>=k>=m>=n>=0 
def get_A(model, ncoef, deg):
    A = [[[None for n in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg+1)]
    for k in range(deg + 1):
        for m in range(k + 1):
            for n in range(ncoef[k][m]):
                A[k][m][n] = model.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(m)+"_"+ str(n), lb = -GRB.INFINITY)
    return model, A


##Creates the interior points
##Note that now the interior points are 3-D
def get_interior(Disc):
    xVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    yVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    tVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1))[1:-1])
    xyt = []
    for cx in xVals:
        for cy in yVals:
            for ct in tVals:
                xyt.append([cx,cy, ct]) 
    return xyt

def get_interior_vals(xyt):
    fVals = [f(ele[0],ele[1], ele[2]) for ele in xyt]
    return fVals


##Creates the boundary planes
    ##Named as: b1,b2: x=0,a; b3,b4: y=0,b; b5,b6: t=0,c
def get_boundary(Disc):
    xVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1)))
    yVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1)))
    tVals = list(np.linspace(0, 1, math.floor(1.0/Disc + 1)))
    
    b1=[]
    b2=[]
    b3=[]
    b4=[]
    b5=[]
    b6=[]
    
    for cy in yVals:
        for ct in tVals:
            b1.append([0.0,cy,ct])
    for cy in yVals:
        for ct in tVals:
            b2.append([1.0,cy,ct])
    for cx in xVals:
        for ct in tVals:
            b3.append([cx,0.0,ct])
    for cx in xVals:
        for ct in tVals:
            b4.append([cx,1.0,ct])
    for cx in xVals:
        for cy in yVals:
            b5.append([cx,cy,0.0])
    for cx in xVals:
        for cy in yVals:
            b6.append([cx,cy,1.0])
    
    return b1,b2,b3,b4,b5,b6

def get_boundary_vals(b1,b2,b3,b4,b5,b6):
    b1Vals = [g1(ele[0],ele[1], ele[2]) for ele in b1]
    b2Vals = [g2(ele[0],ele[1], ele[2]) for ele in b2]
    b3Vals = [g3(ele[0],ele[1], ele[2]) for ele in b3]
    b4Vals = [g4(ele[0],ele[1], ele[2]) for ele in b4]
    b5Vals = [g5(ele[0],ele[1], ele[2]) for ele in b5]
    b6Vals = [g6(ele[0],ele[1], ele[2]) for ele in b6]
    
    b11Vals = [g11(ele[0],ele[1], ele[2]) for ele in b1]
    b22Vals = [g22(ele[0],ele[1], ele[2]) for ele in b2]
    b33Vals = [g33(ele[0],ele[1], ele[2]) for ele in b3]
    b44Vals = [g44(ele[0],ele[1], ele[2]) for ele in b4]
    b55Vals = [g55(ele[0],ele[1], ele[2]) for ele in b5]
    b66Vals = [g66(ele[0],ele[1], ele[2]) for ele in b6]
    
    return b1Vals, b2Vals, b3Vals, b4Vals, b5Vals, b6Vals, b11Vals, b22Vals, b33Vals, b44Vals, b55Vals, b66Vals,

"""
Constraint matrices
"""

##Creates the coefficient matrix for the interior
def get_M_int(ncoef, xyt, deg, terms):
    M = [[[[0.0 for n in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(xyt))]
    for l in range(len(xyt)):
        for k in range(deg + 1):
            for m in range(k+1):
                for n in range(ncoef[k][m]):
                    for dTerm in terms: ###each run in this loop is for a separate term in the LHS of the PDE. 
                        temp = dTerm[3] ###terms has four components: (alpha, beta, gamma, c), c for the constant
                        for alpha in range(0,dTerm[0]):
                            temp = temp*(n - alpha)
                        for beta in range(0,dTerm[1]):
                            temp = temp*(m - n - beta)
                        for gamma in range(0, dTerm[2]):
                            temp = temp*(k - m - gamma)
                        exp1 = max(0, n - dTerm[0]) 
                        ##avoid negative exponents for 0. 
                        #If negative exponent, the if statement below will set this to zero anyway
                        exp2 = max(0, m - n - dTerm[1])
                        exp3 = max(0, k - m - dTerm[2])
                        temp = temp*(xyt[l][0]**(exp1))*(xyt[l][1]**(exp2))*(xyt[l][2]**(exp3))
                        if dTerm[0] > n or dTerm[1] > m-n or dTerm[2]>k-m:
                            temp = 0.0
                        M[l][k][m][n] = M[l][k][m][n] + temp
    return M


##Creates constraint matrix for boundary conditions with first order derivative
def get_M_boundary(ncoef, deg, b1, b2, b3, b4, b5, b6):
    M11 = [[[[0.0 for t in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(b1))]
    M22 = [[[[0.0 for t in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(b2))]
    M33 = [[[[0.0 for t in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(b3))]
    M44 = [[[[0.0 for t in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(b4))]
    M55 = [[[[0.0 for t in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(b5))]
    M66 = [[[[0.0 for t in range(ncoef[k][m])] for m in range(k+1)] for k in range(deg + 1)] for l in range(len(b6))]


    for l in range(len(b1)):
        for k in range(deg + 1):
            for m in range(k+1):
                for n in range(ncoef[k][m]):
                    temp1 = max(n,0)*b1[l][0]**max(n-1, 0)*b1[l][1]**(m-n)*b1[l][2]**(k-m)
                    temp2 = max(n,0)*b2[l][0]**max(n-1, 0)*b2[l][1]**(m-n)*b2[l][2]**(k-m)
                    temp3 = max(m-n,0)*b3[l][0]**(n)*b3[l][1]**max(m-n-1,0)*b3[l][2]**(k-m)
                    temp4 = max(m-n,0)*b4[l][0]**(n)*b4[l][1]**max(m-n-1,0)*b4[l][2]**(k-m)
                    temp5 = max(k-m,0)*b5[l][0]**(n)*b5[l][1]**(m-n)*b5[l][2]**max(k-m-1, 0)
                    temp6 = max(k-m,0)*b6[l][0]**(n)*b6[l][1]**(m-n)*b6[l][2]**max(k-m-1, 0)
                    M11[l][k][m][n] = temp1
                    M22[l][k][m][n] = temp2
                    M33[l][k][m][n] = temp3
                    M44[l][k][m][n] = temp4
                    M55[l][k][m][n] = temp5
                    M66[l][k][m][n] = temp6
    return M11,M22,M33,M44,M55,M66

"""
Auxiliary variables and norms
"""

##Creates the auxiliary variables for the interior
def get_z_interior(model, xyt):
    zint = [None for l in range(len(xyt))]
    for l in range(len(xyt)):
        zint[l] = model.addVar(vtype = GRB.CONTINUOUS, name = "zint_" + "_" + str(l))
    return model, zint

##Creates the auxiliary variables for the boundaries
def get_z_boundary(m, b1, b2, b3, b4, b5, b6):
    z1 = [None for l in range(len(b1))]
    z2 = [None for l in range(len(b2))]
    z3 = [None for l in range(len(b3))]
    z4 = [None for l in range(len(b4))]
    z5 = [None for l in range(len(b5))]
    z6 = [None for l in range(len(b6))]
    z11 = [None for l in range(len(b1))]
    z22 = [None for l in range(len(b2))]
    z33 = [None for l in range(len(b3))]
    z44 = [None for l in range(len(b4))]
    z55 = [None for l in range(len(b5))]
    z66 = [None for l in range(len(b6))]

    for l in range(len(b1)):
        # add upper bound, UZ1 = upper bound for z1, UZ2 etc.
        z1[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(l))
        z2[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(l))
        z3[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(l))
        z4[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(l))
        z5[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z5_" + str(l))
        z6[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z6_" + str(l))
        z11[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z11_" + str(l))
        z22[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z22_" + str(l))
        z33[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z33_" + str(l))
        z44[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z44_" + str(l))
        z55[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z55_" + str(l))
        z66[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z66_" + str(l))
    
    return m, z1, z2, z3, z4, z5, z6, z11, z22, z33, z44, z55, z66


##Creates the auxiliary variables for norms
def get_norms(model):
    norm1 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm1_")
    norm2 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm2_")
    norm3 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm3_")
    norm4 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm4_")
    norm5 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm5_")
    norm6 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm6_")
    norm11 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm11_")
    norm22 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm22_")
    norm33 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm33_")
    norm44 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm44_")
    norm55 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm55_")
    norm66 = model.addVar(vtype = GRB.CONTINUOUS, name = "norm66_")
    return model, norm1, norm2, norm3, norm4, norm5, norm6, norm11, norm22, norm33, norm44, norm55, norm66

def get_norms_int(model):
    normint = model.addVar(vtype = GRB.CONTINUOUS, name="normint_")
    return model, normint

"""
Helper functions to add constraints
"""

def interior_const(model, xyt, M, A, ncoef, deg, intVals, zint):
    for l in range(len(xyt)):
        model.addConstr(sum(sum(sum(M[l][k][m][n]*A[k][m][n] for n in range(ncoef[k][m])) for m in range(k+1)) for k in range(deg + 1))-intVals[l]==zint[l], "fCon_" + str(l)) 
    return model

##For Dirichlet boundary conditions
def boundary_const(model, bi, A, biVals, ncoef, deg, zi):
    for l in range(len(bi)):
        model.addConstr(sum(sum(sum( A[k][m][n]*bi[l][0]**n*bi[l][1]**(m-n)*bi[l][2]**(k-m) for n in range(ncoef[k][m]) ) for m in range(k+1) ) for k in range(deg + 1))  - biVals[l] <= zi[l] )
        model.addConstr(-sum(sum(sum( A[k][m][n]*bi[l][0]**n*bi[l][1]**(m-n)*bi[l][2]**(k-m) for n in range(ncoef[k][m]) ) for m in range(k+1) ) for k in range(deg + 1))  + biVals[l] <= zi[l] )
    return model

##For boundary conditions with first order derivative
def boundary_der_const(model, bi, Mii, A, ncoef, deg, biiVals, zii):
    for l in range(len(bi)):
        model.addConstr(sum(sum(sum( Mii[l][k][m][n]*A[k][m][n] for n in range(ncoef[k][m])) for m in range(k+1)) for k in range(deg + 1)) - biiVals[l] <= zii[l])
        model.addConstr(-sum(sum(sum( Mii[l][k][m][n]*A[k][m][n] for n in range(ncoef[k][m])) for m in range(k+1)) for k in range(deg + 1)) + biiVals[l] <= zii[l])
    return model

##For bound constraints
    ##The input: approximate solution (coefficients matrix 'A'), points to enforce this bound constraint 'xyt'
    ## upper bound 'ub', lower bound 'lb', the matrix to be combined with A (can obtain from get_M_int with a different L operator)
        ##The constraint looks like: lb<=Lu(x)<=ub, for all x in 'xyt'
def bound_const(model, A, Mbound, xyt, lb, ub, ncoef, deg):
    ##enforces lower bound constraint if lb is not NaN
    if 1-np.isnan(lb):
        for l in range(len(xyt)):
            model.addConstr(sum(sum(sum( A[k][m][n]*xyt[l][0]**n*xyt[l][1]**(m-n)*xyt[l][2]**(k-m) for n in range(ncoef[k][m]) ) for m in range(k+1) ) for k in range(deg + 1))>=lb)
    if 1-np.isnan(ub):
        for l in range(len(xyt)):
            model.addConstr(sum(sum(sum( A[k][m][n]*xyt[l][0]**n*xyt[l][1]**(m-n)*xyt[l][2]**(k-m) for n in range(ncoef[k][m]) ) for m in range(k+1) ) for k in range(deg + 1))<=ub)
    return model

            

    

"""
Helper functions for calculating the norms & addding norm constraints
"""

# Computes the L1 norms and enforces that they are less than norm_lim
def l1_Norm_const(model, z1, z2, z3, z4, z5, z6, z11, z22, z33, z44, z55, z66,\
                  b1, b2, b3, b4, b5, b6, norm1, norm2, norm3, norm4, norm5, \
                  norm6, norm11, norm22, norm33, norm44, norm55, norm66, norm_lim):
    
    model.addConstr(sum(z1[p] for p in range(len(b1)))*(1.0/len(b1)) == norm1)
    model.addConstr(sum(z2[p] for p in range(len(b2)))*(1.0/len(b2)) == norm2)
    model.addConstr(sum(z3[p] for p in range(len(b3)))*(1.0/len(b3)) == norm3)
    model.addConstr(sum(z4[p] for p in range(len(b4)))*(1.0/len(b4)) == norm4)
    model.addConstr(sum(z5[p] for p in range(len(b5)))*(1.0/len(b5)) == norm5)
    model.addConstr(sum(z6[p] for p in range(len(b6)))*(1.0/len(b6)) == norm6)

    model.addConstr(sum(z11[p] for p in range(len(b1)))*(1.0/len(b1)) == norm11)
    model.addConstr(sum(z22[p] for p in range(len(b2)))*(1.0/len(b2)) == norm22)
    model.addConstr(sum(z33[p] for p in range(len(b3)))*(1.0/len(b3)) == norm33)
    model.addConstr(sum(z44[p] for p in range(len(b4)))*(1.0/len(b4)) == norm44)
    model.addConstr(sum(z55[p] for p in range(len(b5)))*(1.0/len(b5)) == norm55)
    model.addConstr(sum(z66[p] for p in range(len(b6)))*(1.0/len(b6)) == norm66)
    
    model.addConstr(norm1 <= norm_lim)
    model.addConstr(norm2 <= norm_lim)
    model.addConstr(norm3 <= norm_lim)
    model.addConstr(norm4 <= norm_lim)
    model.addConstr(norm5 <= norm_lim)
    model.addConstr(norm6 <= norm_lim)
 
    model.addConstr(norm11 <= norm_lim)
    model.addConstr(norm22 <= norm_lim)
    model.addConstr(norm33 <= norm_lim)
    model.addConstr(norm44 <= norm_lim)
    model.addConstr(norm55 <= norm_lim)
    model.addConstr(norm66 <= norm_lim)
    
    return model

# Computes the L-Inf norms (max norms) and enforces that they are less than norm_lim
def l_max_Norm_const(model, z1, z2, z3, z4, z5, z6, z11, z22, z33, z44, z55, z66,\
                  b1, b2, b3, b4, b5, b6, norm1, norm2, norm3, norm4, norm5, \
                  norm6, norm11, norm22, norm33, norm44, norm55, norm66, norm_lim):
    for p in range(len(b1)):
        model.addConstr(z1[p] <= norm1)
        model.addConstr(z11[p] <= norm11)
    for p in range(len(b2)):
        model.addConstr(z2[p] <= norm1)
        model.addConstr(z22[p] <= norm11)
    for p in range(len(b3)):
        model.addConstr(z3[p] <= norm1)
        model.addConstr(z33[p] <= norm11)
    for p in range(len(b4)):
        model.addConstr(z4[p] <= norm1)
        model.addConstr(z44[p] <= norm11)
    for p in range(len(b5)):
        model.addConstr(z5[p] <= norm1)
        model.addConstr(z55[p] <= norm11)
    for p in range(len(b6)):
        model.addConstr(z6[p] <= norm1)
        model.addConstr(z66[p] <= norm11)

    
    model.addConstr(norm1 <= norm_lim)
    model.addConstr(norm2 <= norm_lim)
    model.addConstr(norm3 <= norm_lim)
    model.addConstr(norm4 <= norm_lim)
    model.addConstr(norm5 <= norm_lim)
    model.addConstr(norm6 <= norm_lim)
    
    model.addConstr(norm11 <= norm_lim)
    model.addConstr(norm22 <= norm_lim)
    model.addConstr(norm33 <= norm_lim)
    model.addConstr(norm44 <= norm_lim)
    model.addConstr(norm55 <= norm_lim)
    model.addConstr(norm66 <= norm_lim)
    
    return model


def l1_Norm_const_i(model, zint, xyt, normint, norm_lim_int):
    
    model.addConstr(sum(zint[p] for p in range(len(xyt)))*(1.0/len(xyt)) == normint)
    model.addConstr(normint <= norm_lim_int)
    
    return model

def l_max_Norm_const_i(model, zint, xyt, normint, norm_lim_int):
    
    for p in range(len(xyt)):
        model.addConstr(zint[p] <= normint)
    
    model.addConstr(normint <= norm_lim_int)
    
    return model


"""
Function to add objective
"""

def createObjective(model, BC1, BC2, BC3, BC4, BC5, BC6, \
                    NBC1, NBC2, NBC3, NBC4, NBC5, NBC6, \
                    norm1, norm2, norm3, norm4, norm5, norm6,\
                    norm11, norm22, norm33, norm44, norm55, norm66):
    objective = LinExpr()
    if BC1:
        objective.addTerms(1, norm1)
    if BC2:
        objective.addTerms(1, norm2)
    if BC3:
        objective.addTerms(1, norm3)
    if BC4:
        objective.addTerms(1, norm4)
    if BC5:
        objective.addTerms(1, norm5)
    if BC6:
        objective.addTerms(1, norm6)
        
    if NBC1:
        objective.addTerms(1, norm11)
    if NBC2:
        objective.addTerms(1, norm22)
    if NBC3:
        objective.addTerms(1, norm33)
    if NBC4:
        objective.addTerms(1, norm44)
    if NBC5:
        objective.addTerms(1, norm55)
    if NBC6:
        objective.addTerms(1, norm66)
    
    model.setObjective(objective)
    return model



"""
The main function
"""


def LPGenELPDE(Disc, deg, terms, BC1, BC2, BC3, BC4, BC5, BC6,\
               NBC1, NBC2, NBC3, NBC4, NBC5, NBC6, l1_norm, norm_lim, norm_lim_int, BoundC, lb, ub, bTerms):   
    
    ##Set up the model
    model = Model()
    model.setAttr("ModelSense", 1) ###minimize
    
    ##Number of A_k,m,n for a given degree k is (k+1)(k+2)/2
    ncoef = [[ m+1 for m in range(k+1)] for k in range(deg + 1)]
    
    
    ##Setup the decision variables
    model, A = get_A(model, ncoef, deg)
    
    
    ##Setup the interior points
    xyt = get_interior(Disc)
    
    ##Get the interior values
    intVals = get_interior_vals(xyt)
    
    
    ##Setup the boundaries
    ##Note that the boundaries here are actually plains, 
    ##  six plains are used to bound the whole space
    b1, b2, b3, b4, b5, b6 = get_boundary(Disc)
    
    ##Get the boundary values
    b1Vals, b2Vals, b3Vals, b4Vals, b5Vals,\
        b6Vals, b11Vals, b22Vals, b33Vals, b44Vals, b55Vals, b66Vals = get_boundary_vals(b1,b2,b3,b4,b5,b6)
    
    
    ##Creates the cnstraint matrix for the interior
    M = get_M_int(ncoef, xyt, deg, terms)
    ##M in the form M[l][k][m][n]
    
    
    ##Creates the constraint matrix for constraints u_{x_i}=g
    ##In the case of the wave equation, only on b5 will we have this type of constraints
    M11, M22, M33, M44, M55, M66 = get_M_boundary(ncoef, deg, b1, b2, b3, b4, b5, b6)
    
    
    ##Creates the auxiliary variables z, both interior and on the boundaries
    model, zint= get_z_interior(model, xyt)
    model, z1, z2, z3, z4, z5, z6, z11, z22, z33, z44, z55, z66 = get_z_boundary(model, b1, b2, b3, b4, b5, b6)
    
    
        
    ##Create variables for the norms
    model, norm1, norm2, norm3, norm4, norm5, norm6, norm11,\
         norm22, norm33, norm44, norm55, norm66 = get_norms(model)
         
    model, normint = get_norms_int(model)
    
    
    ##After this point all variables are created and added
    model.update()
    
    
    
    ######Start adding constraints to the model######
    
    ##Force interior constraints: |L(u(x,y,t))-f(x,y,t)|=zint
    model = interior_const(model, xyt, M, A, ncoef, deg, intVals, zint)
    
    ##Force boundary constraints: |u(x,y,t)-g_i(x,y,t)|=zi
    if BC1:
        model = boundary_const(model, b1, A, b1Vals, ncoef, deg, z1)
    if BC2:
        model = boundary_const(model, b2, A, b2Vals, ncoef, deg, z2)
    if BC3:
        model = boundary_const(model, b3, A, b3Vals, ncoef, deg, z3)
    if BC4:
        model = boundary_const(model, b4, A, b4Vals, ncoef, deg, z4)  
    if BC5:
        model = boundary_const(model, b5, A, b5Vals, ncoef, deg, z5)
    if BC6:
        model = boundary_const(model, b6, A, b6Vals, ncoef, deg, z6)
    
    ##Force boundary constraints: |u_t(x,y,t) - g_ii(x,y,t)|=zii
    if NBC1:
        model = boundary_der_const(model, b1, M11, A, ncoef, deg, b11Vals, z11)
    if NBC2:
        model = boundary_der_const(model, b2, M22, A, ncoef, deg, b22Vals, z22)
    if NBC3:
        model = boundary_der_const(model, b3, M33, A, ncoef, deg, b33Vals, z33)
    if NBC4:
        model = boundary_der_const(model, b4, M44, A, ncoef, deg, b44Vals, z44)
    if NBC5:
        model = boundary_der_const(model, b5, M55, A, ncoef, deg, b55Vals, z55)
    if NBC6:
        model = boundary_der_const(model, b6, M66, A, ncoef, deg, b66Vals, z66)
    

         
    if l1_norm:
         # use the L1 calculation of norm
         model = l1_Norm_const(model, z1, z2, z3, z4, z5, z6, z11, z22, z33, z44, z55, z66,\
                               b1, b2, b3, b4, b5, b6, norm1, norm2, norm3, norm4, norm5, \
                               norm6, norm11, norm22, norm33, norm44, norm55, norm66, norm_lim)
         model = l1_Norm_const_i(model, zint, xyt, normint, norm_lim_int)
    else:
         # use the L-Inf calculation of norm 
         model = l_max_Norm_const(model, z1, z2, z3, z4, z5, z6, z11, z22, z33, z44, z55, z66,\
                                  b1, b2, b3, b4, b5, b6, norm1, norm2, norm3, norm4, norm5, \
                                  norm6, norm11, norm22, norm33, norm44, norm55, norm66, norm_lim)
         model = l_max_Norm_const_i(model, zint, xyt, normint, norm_lim_int)
    
    ##Add bound constraints
    ##For now we assume that this bound constraint is enforced everywhere in the interior, but it could be specified with a little change
    if BoundC:
        Mbound = get_M_int(ncoef, xyt, deg, bTerms)   
            ##the structure of the lhs of constraint is similar so we can use this function to get
        model = bound_const(model, A, Mbound, xyt, lb, ub, ncoef, deg)
    
    
    "def get_M_int(ncoef, xyt, deg, terms)"
    
    ##After this point all constraints are added to the model
    model.update()
    
    
    ######Start formulating the objective######
    model = createObjective(model, BC1, BC2, BC3, BC4, BC5, BC6, \
                            NBC1, NBC2, NBC3, NBC4, NBC5, NBC6, \
                            norm1, norm2, norm3, norm4, norm5, norm6,\
                            norm11, norm22, norm33, norm44, norm55, norm66)
    
    ##Update and solve
    model.update()
    model.optimize()   
    
    
    return model

"""
Evaluation functions
"""

##Returns the approximation value with given coordinates
def evalWithA(x, y, t, A, deg):
     ncoef = [[ m+1 for m in range(k+1)] for k in range(deg + 1)]
     retVal = 0.0
     for k in range(deg + 1):
          for m in range(k+1):
              for n in range(ncoef[k][m]):
                  retVal = retVal + A[k][m][n]*(x**n)*(y**(m - n))*(t**(k - m))
     return retVal


##For a set of t values defined by DiscT, plot a set of U(x,y,t_i) surfaces
def useSol(model, Disc, deg, DiscT, shouldPlot, saveName, plotTitle):
     ncoef = [[ m+1 for m in range(k+1)] for k in range(deg + 1)]
     ####Discretize space, including the boundary points
     X = list(np.linspace(0, 1, math.floor(1.0/Disc + 1)))
     Y = list(np.linspace(0, 1, math.floor(1.0/Disc + 1)))
     
     XY = []
     for cx in X:
          for cy in Y:
               XY.append([cx,cy])
     
     
     ####Discretize time for plots, using DiscT
     plotT = list(np.linspace(0, 1, math.floor(1.0/DiscT + 1)))
     

     xArray = np.linspace(0,1, math.floor(1.0/Disc + 1))
     yArray = np.linspace(0,1, math.floor(1.0/Disc + 1))
     ####Get the coefficients
     A = []
     for k in range(deg + 1):
         A.append([])
         for m in range(k+1):
             A[k].append([])
             for n in range(ncoef[k][m]):
                 A[k][m].append(model.getVarByName("A_" + str(k) + "_" + str(m) + "_" +str(n)).x); print(A[k][m][n])

     ####Get grid of x, y values
     xArray, yArray = np.meshgrid(xArray, yArray)
     
     
     ####Plot for each t
     
     if shouldPlot == True:
         for t in plotT:
             uArray = evalWithA(xArray, yArray, t, A, deg); #print("uArray:"); print(uArray)
             
             fig = plt.figure()
             ax = plt.axes(projection = '3d')
             ax.plot_surface(xArray, yArray, uArray, cmap=plt.cm.jet)
             ax.set_xlabel("x"); ax.set_ylabel("y")
             ax.set_zlim([-0.025,0.025]) 
             plt.title(" ")####plotTitle+"t="+str(t)[0:3]
             plt.savefig(saveName+"_t="+str(t)[0:3]+".png")
     
     ####Return the coefficients
     return A



"""
Test instance
"""


def g1(x,y,t):
     return 0
def g2(x,y,t):
     return 0 
def g3(x,y,t):
     return 0
def g4(x,y,t):
     return 0
def g5(x,y,t):
     return 0
def g6(x,y,t):
     return 0
 
def f(x,y,t):
     return 0
 
def g11(x,y,t):
     return 0
def g22(x,y,t):
     return 0
def g33(x,y,t):
     return 0
def g44(x,y,t):
     return 0
def g55(x,y,t):
     return x*(1-x)+y*(1-y)
def g66(x,y,t):
     return 0





deg = 6
terms = [(2,0,0,-9), (0,2,0,-9),(0,0,2,1)]
DiscD = 0.05
Disc = 0.05
DiscT = 0.05

bTerms = [(0, 0, 0, 1)]
lb = np.nan
ub = 0.015

#def LPGenELPDE(Disc, deg, terms, BC1, BC2, BC3, BC4, BC5, BC6,\
#               NBC1, NBC2, NBC3, NBC4, NBC5, NBC6, l1_norm, norm_lim, norm_lim_int, BoundC, lb, ub, bTerms):   

model = LPGenELPDE(Disc, deg, terms, 1,1,1,1,1,0,\
           0,0,0,0,1,0,\
           1,\
           0.05,0.05,\
           1, lb, ub, bTerms)

model.write("model.lp")
model.write("model.sol")


useSol(model, DiscD, deg, DiscT, True, "prob1_plot", r' ')