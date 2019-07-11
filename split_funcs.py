#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
March 2019
Owen Michals
"""
from gurobipy import *
#import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op

##### New Helper functions to make smaller #####

def get_interior(xDisc):
    intX = list(np.linspace(0, 1, math.floor(1.0/xDisc + 1))[1:-1])
    intY = list(np.linspace(0, 1, math.floor(1.0/xDisc + 1))[1:-1])
    intXY = []
    for cx in intX:
        for cy in intY:
            intXY.append([cx,cy])  
    print(intXY)
    return intXY

def get_boundary(xDisc):
    ####xy1 are the points along y = 0
    xy1 = [[newX, 0.0] for newX in list(np.linspace(0,1, math.floor(1.0/xDisc + 1)))]
    ####xy2 are the points along y = 1
    xy2 = [[newX, 1.0] for newX in list(np.linspace(0,1, math.floor(1.0/xDisc + 1)))]

    ####xy3 are the points along x = 0
    xy3 = [[0.0, newY] for newY in list(np.linspace(0,1, math.floor(1.0/xDisc + 1)))]
    ####xy4 are the points along x = 1
    xy4 = [[1.0, newY] for newY in list(np.linspace(0,1, math.floor(1.0/xDisc + 1)))] 
    
    #print(xy1)
    #print(xy2)
    #print(xy3)
    #print(xy4)
    return xy1, xy2, xy3, xy4

def get_powers(deg, ncoef, intXY, xy1, xy2, xy3, xy4):
    xyIntPower = []
    xy1Power = []
    xy2Power = []
    xy3Power = []
    xy4Power = []
    for k in range(deg + 1):
        xyIntPower.append([])
        xy1Power.append([])
        xy2Power.append([])
        xy3Power.append([])
        xy4Power.append([])
        for t in range(ncoef[k]):
            xyIntPower[k].append([(ele[0]**t)*(ele[1]**(k - t)) for ele in intXY])
            #print("xyIntPower, (k,t) = " + str(k) + "," + str(t)); print(xyIntPower[k][t])
            xy1Power[k].append([(ele[0]**t)*(ele[1]**(k - t)) for ele in xy1])               
            xy2Power[k].append([(ele[0]**t)*(ele[1]**(k - t)) for ele in xy2])               
            xy3Power[k].append([(ele[0]**t)*(ele[1]**(k - t)) for ele in xy3])               
            xy4Power[k].append([(ele[0]**t)*(ele[1]**(k - t)) for ele in xy4])    
    
    return xyIntPower, xy1Power, xy2Power, xy3Power, xy4Power

def get_Lcoef(deg, ncoef, intXY):
    LCoef = []
    for k in range(deg + 1):
        LCoef.append([])
        for t in range(ncoef[k]):
            LCoef[k].append([])
            for p in range(len(intXY)):
                LCoef[k][t].append(0.0)
                LCoef[k][t][p] = max(t - 1, 0)*t*(intXY[p][0]**(t - 2))*(intXY[p][1]**(k - t)) \
                + max(k - t - 1, 0)*(k - t)*(intXY[p][0]**t)*(intXY[p][1]**(k - t - 2))
                #print("lapl coef k: " + str(k) + " t " + str(t) + " p " + str(p)); print(laplaceCoef[k][t][p])
    
    return LCoef
    

##### End of additions #####



#xDisc is the discretization level of the grid 
#f is the source function
#g_{i} is the boundary condition on side i of the unit square
#d_{i} is a boolean on whether or not to dualize that boundary's constraint
#deg is the maximum degree of the solution
def makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg):

     m = Model()
     m.setAttr("ModelSense", 1) ###minimize
     ###First, need to determine how many coefficients there are
     ncoef = [k + 1 for k in range(deg + 1)]

     A = [[None for t in range(ncoef[k])] for k in range(deg + 1)]

     for k in range(deg + 1):
          for t in range(ncoef[k]):
               A[k][t] = m.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(t), lb = -GRB.INFINITY)


     ####Need to get values of x, x^2, x^3, so on
     # xDisc works a little bit wierdly
     intXY = get_interior(xDisc)
     
     # xy1 contains the points along y=0
     # xy2 contains the points along y=1
     # xy3 contains the points along x=0
     # xy4 contains the points along x=1
     xy1, xy2, xy3, xy4 = get_boundary(xDisc)
     

     # get the powers
     xyIntPower, xy1Power, xy2Power, xy3Power, xy4Power = get_powers(deg, ncoef, intXY, xy1, xy2, xy3, xy4)
              

     ####Now need to evaluate the boundary conditions and f at the points
     g1Vals = [g1(ele[0], ele[1]) for ele in xy1]
     g2Vals = [g2(ele[0], ele[1]) for ele in xy2]
     g3Vals = [g3(ele[0], ele[1]) for ele in xy3]
     g4Vals = [g4(ele[0], ele[1]) for ele in xy4]    
     fVals = [f(ele[0], ele[1]) for ele in intXY]

     ####We are going to minimize the 1 norm of boundary condition errors. Thus, we need auxiliary variables
     z1 = [None for p in g1Vals]
     z2 = [None for p in g2Vals]    
     z3 = [None for p in g3Vals]           
     z4 = [None for p in g4Vals]

     for p in range(len(g1Vals)):
          z1[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g2Vals)):
          z2[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g3Vals)):
          z3[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g4Vals)):
          z4[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(p), obj = 1.0/(4.0*len(xy1)))


     ####That long line should be the coefficient of u_{xx} + u_{yy} for each coefficient term in the polynomial 
     LCoef = get_Lcoef(deg, ncoef, intXY)


     ####Now we need to add the constraint Laplacian u = f
     for p in range(len(fVals)):
          m.addConstr(sum(sum(LCoef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) == fVals[p], "Cf_" + str(p))

     ####Now we add constraint to compute |u - g_{i}|
     for p in range(len(g1Vals)):
          m.addConstr(sum(sum(xy1Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g1Vals[p] <= z1[p], "Cg1L_" + str(p))

     for p in range(len(g2Vals)):
          m.addConstr(sum(sum(xy2Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g2Vals[p] <= z2[p], "Cg2L_" + str(p))

     for p in range(len(g3Vals)):
          m.addConstr(sum(sum(xy3Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g3Vals[p] <= z3[p], "Cg3L_" + str(p))

     for p in range(len(g4Vals)):
          m.addConstr(sum(sum(xy4Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g4Vals[p] <= z4[p], "Cg4L_" + str(p))


     for p in range(len(g1Vals)):
          m.addConstr(sum(sum(xy1Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g1Vals[p] >= -z1[p], "Cg1U_" + str(p))

     for p in range(len(g2Vals)):
          m.addConstr(sum(sum(xy2Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g2Vals[p] >= -z2[p], "Cg2U_" + str(p))

     for p in range(len(g3Vals)):
          m.addConstr(sum(sum(xy3Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g3Vals[p] >= -z3[p], "Cg3U_" + str(p))

     for p in range(len(g4Vals)):
          m.addConstr(sum(sum(xy4Power[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g4Vals[p] >= -z4[p], "Cg4U_" + str(p))

     m.update()
     m.optimize()
     return m

##### Miscellaneous testing individual parts

get_interior(.15)
#get_boundary(.1)
'''
xDisc = .2;  
def g1(x,y):
     return x**2
def g2(x,y):
     return x**2
def g3(x,y):
     return y**2
def g4(x,y):
     return y**2
def f(x,y):
     return 4.0
d1 = []; d2 = []; d3 = []; d4 = []; deg = 2
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)
'''




