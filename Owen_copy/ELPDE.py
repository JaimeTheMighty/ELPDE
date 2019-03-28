from gurobipy import *
#import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op

####The goal of this code is to solve PDEs with polynomials as best we can even when the boundary conditions cannot be met exactly. This should be for general linear PDEs.

#xDisc is the discretization level of the grid 
#f is the source function
#g_{i} is the boundary condition on side i of the unit square
#g_{ii} is the neumann boundary condition on side i of the unit square
#deg is the maximum degree of the solution
#terms has the number of x and y derivatives for each term.
##terms[v] = [xd, yd, signMagnitude]
from gurobipy import *
#import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op

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

     A = [[None for t in range(ncoef[k])] for k in range(deg + 1)]

     for k in range(deg + 1):
          for t in range(ncoef[k]):
               A[k][t] = m.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(t), lb = -GRB.INFINITY)

     

     ####make a list of interior sample points
     xVals = list(np.linspace(0,1,1.0/xDisc)[1:-1])
     yVals = list(np.linspace(0,1,1.0/xDisc)[1:-1])

     xy = []
     for cx in xVals:
          for cy in yVals:
               xy.append([cx,cy])


     ####points along y = 0, y = 1, x = 0, x = 1 respectively (for boundary conditions)
     xy1 = [[newX, 0.0] for newX in np.linspace(0,1,1.0/xDisc)]
     xy2 = [[newX, 1.0] for newX in np.linspace(0,1,1.0/xDisc)]
     xy3 = [[0.0, newY] for newY in np.linspace(0,1,1.0/xDisc)]
     xy4 = [[1.0, newY] for newY in np.linspace(0,1,1.0/xDisc)]
     
     ####Make auxiliary vars for boundary objectives
     z1 = [None for l in range(len(xy1))]
     z2 = [None for l in range(len(xy2))]
     z3 = [None for l in range(len(xy3))]
     z4 = [None for l in range(len(xy4))]
     z11 = [None for l in range(len(xy1))]
     z22 = [None for l in range(len(xy2))]
     z33 = [None for l in range(len(xy3))]
     z44 = [None for l in range(len(xy4))]

     for l in range(len(xy1)):
          z1[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(l))
          z2[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(l))
          z3[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(l))
          z4[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(l))
          z11[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z11_" + str(l))
          z22[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z22_" + str(l))
          z33[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z33_" + str(l))
          z44[l] = m.addVar(vtype = GRB.CONTINUOUS, name = "z44_" + str(l))

     ####will fill in with x^t y^k-t for each of the grid points, do for all exponent combinations
#     xyPower = []

#     xy1Power = []
#     xy2Power = []
#     xy3Power = []
#     xy4Power = []

#     for k in range(deg + 1):
#          xyPower.append([])
#          xy1Power.append([])
#          xy2Power.append([])
#          xy3Power.append([])
#          xy4Power.append([])
#          
#          for t in range(ncoef[k]):
#               print(k)
#               xyPower[k].append([(ele[0]**t)*(ele[1]**(k-t)) for ele in xyPower]) 
#               xy1Power[k].append([(ele[0]**t)*(ele[1]**(k-t)) for ele in xy1Power]) 
#               xy2Power[k].append([(ele[0]**t)*(ele[1]**(k-t)) for ele in xy2Power]) 
#               xy3Power[k].append([(ele[0]**t)*(ele[1]**(k-t)) for ele in xy3Power]) 
#               xy4Power[k].append([(ele[0]**t)*(ele[1]**(k-t)) for ele in xy4Power]) 
#
     ####Now we need to evaluate the functions at our grid points
     g1Vals = [g1(ele[0],ele[1]) for ele in xy1]
     g2Vals = [g2(ele[0],ele[1]) for ele in xy2]
     g3Vals = [g3(ele[0],ele[1]) for ele in xy3]
     g4Vals = [g4(ele[0],ele[1]) for ele in xy4]
     fVals = [f(ele[0],ele[1]) for ele in xy]

     g11Vals = [g11(ele[0],ele[1]) for ele in xy1]
     g22Vals = [g22(ele[0],ele[1]) for ele in xy2]
     g33Vals = [g33(ele[0],ele[1]) for ele in xy3]
     g44Vals = [g44(ele[0],ele[1]) for ele in xy4]


     ####Now we make the constraint matrix for the interior.
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

