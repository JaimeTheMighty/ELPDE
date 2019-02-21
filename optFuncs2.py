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
def makeLPGenPDE(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, dual1, dual2, dual3, dual4, \
dual11, dual22, dual33, dual44):

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
          z1[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z1_" + str(p), obj = dual1*1.0/(4.0*len(xy1)))
     for p in range(len(g2Vals)):
          z2[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z2_" + str(p), obj = dual2*1.0/(4.0*len(xy1)))
     for p in range(len(g3Vals)):
          z3[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z3_" + str(p), obj = dual3*1.0/(4.0*len(xy1)))
     for p in range(len(g4Vals)):
          z4[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z4_" + str(p), obj = dual4*1.0/(4.0*len(xy1)))

     for p in range(len(g11Vals)):
          z11[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z11_" + str(p), obj = dual11*1.0/(4.0*len(xy1)))
     for p in range(len(g22Vals)):
          z22[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z22_" + str(p), obj = dual22*1.0/(4.0*len(xy1)))
     for p in range(len(g33Vals)):
          z33[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z33_" + str(p), obj = dual33*1.0/(4.0*len(xy1)))
     for p in range(len(g44Vals)):
          z44[p] = m.addVar(vtype = GRB.CONTINUOUS, name = "z44_" + str(p), obj = dual44*1.0/(4.0*len(xy1)))


###getting coefficients for a_{i} by looping through the terms
     tCoefs = []
     Coefs = [ [ [0.0  for p in range(len(intXY))]  for t in range(ncoef[k])] for k in range(deg + 1)]
     for v in range(len(terms)):
          tCoefs.append([])
          for k in range(deg + 1):
               tCoefs[v].append([])
               for t in range(ncoef[k]):
                    tCoefs[v][k].append([])
                    for p in range(len(intXY)):
                         tCoefs[v][k][t].append(0.0)
                         temp = 1.0
                         for s in range(terms[v][0]):
                              temp = temp*max(0,(t - s))
                         temp = temp*intXY[p][0]**(t - terms[v][0])
                         for s in range(terms[v][1]):
                              temp = temp*max(0, (k - t - s))
                         temp = temp*intXY[p][1]**(k - t - terms[v][1])
                         tCoefs[v][k][t][p] = temp
                         Coefs[k][t][p] = Coefs[k][t][p] + terms[v][2]*tCoefs[v][k][t][p]
     print(Coefs[1])
                         
     ####That long line should be the coefficient of u_{xx} + u_{yy} for each coefficient term in the polynomial 
#     LCoef = []
#     for k in range(deg + 1):
#          LCoef.append([])
#          for t in range(ncoef[k]):
#               LCoef[k].append([])
#               for p in range(len(intXY)):
#                    LCoef[k][t].append(0.0)
#                    LCoef[k][t][p] = max(t - 1, 0)*t*(intXY[p][0]**(t - 2))*(intXY[p][1]**(k - t)) \
#                    + max(k - t - 1, 0)*(k - t)*(intXY[p][0]**t)*(intXY[p][1]**(k - t - 2))
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
          m.addConstr(sum(sum(Coefs[k][t][p]*A[k][t] for t in range(ncoef[k])) for k in range(deg + 1)) == fVals[p], "Cf_" + str(p))
          for k in range(deg + 1):
               for t in range(ncoef[k]):
                    print("Coefs["+str(k)+"]["+str(t)+"]["+str(p)+"]: " + str(Coefs[k][t][p]))

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



