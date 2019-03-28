from gurobipy import *
#import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator as op

####The goal of this code is to solve PDEs with polynomials as best we can even when the boundary conditions cannot be met exactly.

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
     intX = list(np.linspace(0, 1, 1.0/xDisc)[1:-1])
     intY = list(np.linspace(0, 1, 1.0/xDisc)[1:-1])
     intXY = []
     for cx in intX:
          for cy in intY:
               intXY.append([cx,cy])

    # print("intXY: "); print(intXY)
     ####xy1 are the points along y = 0
     xy1 = [[newX, 0.0] for newX in list(np.linspace(0,1, 1.0/xDisc))]
     ####xy2 are the points along y = 1
     xy2 = [[newX, 1.0] for newX in list(np.linspace(0,1, 1.0/xDisc))]

     ####xy3 are the points along x = 0
     xy3 = [[0.0, newY] for newY in list(np.linspace(0,1, 1.0/xDisc))]
     ####xy4 are the points along x = 1
     xy4 = [[1.0, newY] for newY in list(np.linspace(0,1, 1.0/xDisc))]
     

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




def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom


####function to compute estimated solution at grid points using coefficients from solved lp
####m is a gurobi model
####xDisc is the spacing of the grid points
####deg is the degree of the solution
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

          
  


####The goal of this code is to solve PDEs with polynomials as best we can even when the boundary conditions cannot be met exactly.

#xDisc is the discretization level of the grid 
#f is the source function
#g_{i} is the boundary condition on side i of the unit square
#g_{ii} is the neumann boundary condition on side i of the unit square
#deg is the maximum degree of the solution
def makeLPCauchy(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg):

     m = Model()
     m.setAttr("ModelSense", 1) ###minimize
     ###First, need to determine how many coefficients there are
     ncoef = [k + 1 for k in range(deg + 1)]

     A = [[None for t in range(ncoef[k])] for k in range(deg + 1)]

     for k in range(deg + 1):
          for t in range(ncoef[k]):
               A[k][t] = m.addVar(vtype = GRB.CONTINUOUS, name = "A_" + str(k) + "_" + str(t), lb = -GRB.INFINITY)


     ####Need to get values of x, x^2, x^3, so on
     intX = list(np.linspace(0, 1, 1.0/xDisc)[1:-1])
     intY = list(np.linspace(0, 1, 1.0/xDisc)[1:-1])
     intXY = []
     for cx in intX:
          for cy in intY:
               intXY.append([cx,cy])

    # print("intXY: "); print(intXY)
     ####xy1 are the points along y = 0
     xy1 = [[newX, 0.0] for newX in list(np.linspace(0,1, 1.0/xDisc))]
     ####xy2 are the points along y = 1
     xy2 = [[newX, 1.0] for newX in list(np.linspace(0,1, 1.0/xDisc))]

     ####xy3 are the points along x = 0
     xy3 = [[0.0, newY] for newY in list(np.linspace(0,1, 1.0/xDisc))]
     ####xy4 are the points along x = 1
     xy4 = [[1.0, newY] for newY in list(np.linspace(0,1, 1.0/xDisc))]
     

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

     ####Now need to evaluate the boundary conditions and f at the points
     g1Vals = [g1(ele[0], ele[1]) for ele in xy1]
     g2Vals = [g2(ele[0], ele[1]) for ele in xy2]
     g3Vals = [g3(ele[0], ele[1]) for ele in xy3]
     g4Vals = [g4(ele[0], ele[1]) for ele in xy4]    
     fVals = [f(ele[0], ele[1]) for ele in intXY]

     g11Vals = [g11(ele[0], ele[1]) for ele in xy1]
     g22Vals = [g22(ele[0], ele[1]) for ele in xy2]
     g33Vals = [g33(ele[0], ele[1]) for ele in xy3]
     g44Vals = [g44(ele[0], ele[1]) for ele in xy4]      

     ####We are going to minimize the 1 norm of boundary condition errors. Thus, we need auxiliary variables
     z1 = [None for p in g1Vals]
     z2 = [None for p in g2Vals]    
     z3 = [None for p in g3Vals]           
     z4 = [None for p in g4Vals]

     z11 = [None for p in g11Vals]
     z22 = [None for p in g22Vals]    
     z33 = [None for p in g33Vals]           
     z44 = [None for p in g44Vals]

     for p in range(len(g1Vals)):
          z1[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g2Vals)):
          z2[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g3Vals)):
          z3[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g4Vals)):
          z4[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(p), obj = 1.0/(4.0*len(xy1)))

     for p in range(len(g11Vals)):
          z11[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z11_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g22Vals)):
          z22[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z22_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g33Vals)):
          z33[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z33_" + str(p), obj = 1.0/(4.0*len(xy1)))
     for p in range(len(g44Vals)):
          z44[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z44_" + str(p), obj = 1.0/(4.0*len(xy1)))


     ####That long line should be the coefficient of u_{xx} + u_{yy} for each coefficient term in the polynomial 
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


     ####This long line should be coefficient of u_{-y}
     g11Coef = []
     for k in range(deg + 1):
          g11Coef.append([])
          for t in range(ncoef[k]):
               g11Coef[k].append([])
               for p in range(len(xy1)):
                    g11Coef[k][t].append(0.0)
                    if k - t >= 1:
                         g11Coef[k][t][p] = (k - t)*(xy1[p][0]**t)*(xy1[p][1]**(k - t - 1))
                    g11Coef[k][t][p] = -1.0*g11Coef[k][t][p]


     ####This long line should be coefficient of u_{y}
     g22Coef = []
     for k in range(deg + 1):
          g22Coef.append([])
          for t in range(ncoef[k]):
               g22Coef[k].append([])
               for p in range(len(xy2)):
                    g22Coef[k][t].append(0.0)
                    if k - t >= 1:
                         g22Coef[k][t][p] = (k - t)*(xy2[p][0]**t)*(xy2[p][1]**(k - t - 1))

     ####This long line should be coefficient of u_{-x}
     g33Coef = []
     for k in range(deg + 1):
          g33Coef.append([])
          for t in range(ncoef[k]):
               g33Coef[k].append([])
               for p in range(len(xy3)):
                    g33Coef[k][t].append(0.0)
                    if t >= 1:
                         g33Coef[k][t][p] = (t)*(xy3[p][0]**(t-1))*(xy3[p][1]**(k - t))
                    g33Coef[k][t][p] = -1.0*g33Coef[k][t][p]

     ####This long line should be coefficient of u_{x}
     g44Coef = []
     for k in range(deg + 1):
          g44Coef.append([])
          for t in range(ncoef[k]):
               g44Coef[k].append([])
               for p in range(len(xy4)):
                    g44Coef[k][t].append(0.0)
                    if t >= 1:
                         g44Coef[k][t][p] = (t)*(xy4[p][0]**(t-1))*(xy4[p][1]**(k - t))



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


     ####Now we need to add constraints z_{ii} = |(du/dni) - g_{ii}|
     for p in range(len(g11Vals)):
          m.addConstr(sum(sum(g11Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g11Vals[p] <= z11[p], "Cg11L_" + str(p))

     for p in range(len(g22Vals)):
          m.addConstr(sum(sum(g22Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g22Vals[p] <= z22[p], "Cg22L_" + str(p))

     for p in range(len(g33Vals)):
          m.addConstr(sum(sum(g33Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g33Vals[p] <= z33[p], "Cg33L_" + str(p))

     for p in range(len(g44Vals)):
          m.addConstr(sum(sum(g44Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g44Vals[p] <= z44[p], "Cg44L_" + str(p))


     for p in range(len(g11Vals)):
          m.addConstr(sum(sum(g11Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g11Vals[p] >= -z11[p], "Cg11U_" + str(p))

     for p in range(len(g22Vals)):
          m.addConstr(sum(sum(g22Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g22Vals[p] >= -z22[p], "Cg22U_" + str(p))

     for p in range(len(g33Vals)):
          m.addConstr(sum(sum(g33Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g33Vals[p] >= -z33[p], "Cg33U_" + str(p))

     for p in range(len(g44Vals)):
          m.addConstr(sum(sum(g44Coef[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) - g44Vals[p] >= -z44[p], "Cg44U_" + str(p))


     ####One more set of vars so we can see norms of different constraints
     norm1 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm1_" + str(p))
     norm2 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm2_" + str(p))
     norm3 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm3_" + str(p))
     norm4 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm4_" + str(p))
     norm11 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm11_" + str(p))
     norm22 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm22_" + str(p))
     norm33 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm33_" + str(p))
     norm44 = m.addVar(vtype = GRB.CONTINUOUS, name = "norm44_" + str(p))

     m.addConstr(sum(z1[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm1)
     m.addConstr(sum(z2[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm2)
     m.addConstr(sum(z3[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm3)
     m.addConstr(sum(z4[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm4)

     m.addConstr(sum(z11[p] for p in range(len(xy1)))*(1.0/len(xy1)) == norm11)
     m.addConstr(sum(z22[p] for p in range(len(xy2)))*(1.0/len(xy2)) == norm22)
     m.addConstr(sum(z33[p] for p in range(len(xy3)))*(1.0/len(xy3)) == norm33)
     m.addConstr(sum(z44[p] for p in range(len(xy4)))*(1.0/len(xy4)) == norm44)

     m.update()
     m.optimize()
     return m



