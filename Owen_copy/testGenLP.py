from gurobipy import *
from optProjectFuncs import *
from optFuncs2 import *
import math
from ELPDE import *

#####Problem 9: Lap(u) = -2y(1 - y) - 2x(1-x)
#####BC1: u(x,y) = 0 on y = 0
#####BC2: u(x,y) = 0 on y = 1
#####BC3: u(x,y) = 0 on x = 0
#####BC4: u(x,y) = 0 on x = 1
#####NC1: u_{-y}(x,y) = -(1 - 2y)x(1-x) on y = 0
#####NC2: u_{y}(x,y) = (1 - 2y)x(1-x) on y = 1
#####NC3: u_{-x}(x,y) = -(1 - 2x)y(1-y) on x = 0
#####NC4: u_{x}(x,y) = (1 - 2x)y(1-y) on x = 1
#
#xDisc = .1;  
#def g1(x,y):
#     return 0.0
#def g2(x,y):
#     return 0
#def g3(x,y):
#     return 0
#def g4(x,y):
#     return 0
#def f(x,y):
#     return  -2.0*y*(1.0 - y) - 2.0*x*(1.0-x)
#def g11(x,y):
#     return -(1 - 2.0*y)*x*(1-x)
#def g22(x,y):
#     return (1 - 2.0*y)*x*(1-x)
#def g33(x,y):
#     return -(1 - 2.0*x)*y*(1-y)
#def g44(x,y):
#     return (1 - 2.0*x)*y*(1-y)
#deg = 4
#newm = makeLPCauchy(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg)
#
#
#newm.write("PDE_LP9.sol")
#
#useSol(newm, .05, deg, True, "prob9LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob9)')
#




####Problem 11: Lap(u) = -2y(1 - y) - 2x(1-x)
####BC1: u(x,y) = 0 on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = 0 on x = 0
####BC4: u(x,y) = 0 on x = 1
####NC1: u_{-y}(x,y) = -(1 - 2y)x(1-x) on y = 0
####NC2: u_{y}(x,y) = (1 - 2y)x(1-x) on y = 1
####NC3: u_{-x}(x,y) = -(1 - 2x)y(1-y) on x = 0
####NC4: u_{x}(x,y) = (1 - 2x)y(1-y) on x = 1
###USING GENERAL PDE CODE HERE
'''
xDisc = .25;  
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
newm = LPGenELPDE(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
'''

#newm.write("PDE_LP11.sol")

#useSol(newm, .05, deg, True, "prob11LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob11)')




#####Problem 11: Lap(u) = -2y(1 - y) - 2x(1-x)
#####BC1: u(x,y) = 0 on y = 0
#####BC2: u(x,y) = 0 on y = 1
#####BC3: u(x,y) = 0 on x = 0
#####BC4: u(x,y) = 0 on x = 1
#####NC1: u_{-y}(x,y) = -(1 - 2y)x(1-x) on y = 0
#####NC2: u_{y}(x,y) = (1 - 2y)x(1-x) on y = 1
#####NC3: u_{-x}(x,y) = -(1 - 2x)y(1-y) on x = 0
#####NC4: u_{x}(x,y) = (1 - 2x)y(1-y) on x = 1
####USING GENERAL PDE CODE HERE
#
#xDisc = .2;  
#def g1(x,y):
#     return 3*x
#def g2(x,y):
#     return 3*x + 3
#def g3(x,y):
#     return 3*y
#def g4(x,y):
#     return 3 + 3*y
#def f(x,y):
#     return  6
#def g11(x,y):
#     return -2*(x**3)*y
#def g22(x,y):
#     return 2*(x**3)*y
#def g33(x,y):
#     return -(2*x + 3*(x**2)*(y**2))
#def g44(x,y):
#     return (2*x + 3*(x**2)*(y**2))
#deg = 1
#terms = [[1, 0, 1.0], [0, 1, 3.0]]
#newm = makeLPGenPDE(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg, terms, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
#
#
#newm.write("PDE_LP12.sol")
#
#useSol(newm, .05, deg, True, "prob12LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob12)')


####Problem 1A: Lap(u) = 1 - 2y
####BC1: u(x,y) = 0 on y = 0
####BC2: u(x,y) = x(1-x) on y = 1
####BC3: u(x,y) = 0 on x = 0
####BC4: u(x,y) = 0 + 1 on x = 1

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


newm.write("PDE_LP1A.sol")

useSol(newm, .05, deg, True, "prob1ALP_plot.png", r'Computed Solution to $\Delta u = 1 - 2y$ (Prob4)')


