from optProjectFuncs import *
from optFuncs2 import *
import math

####Problem 1: Lap(u) = 4
####BC1: u(x,y) = x^2 on y = 0
####BC2: u(x,y) = x^2 + 1 on y = 1
####BC3: u(x,y) = y^2 on x = 0
####BC4: u(x,y) = y^2 + 1 on x = 1
xDisc = .2;  
def g1(x,y):
     return x**2
def g2(x,y):
     return x**2 + 1
def g3(x,y):
     return y**2
def g4(x,y):
     return y**2 + 1
def f(x,y):
     return 4.0
d1 = []; d2 = []; d3 = []; d4 = []; deg = 2
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)
newm.write("PDE_LP1.sol")


useSol(newm, .05, deg, True, "prob1LP_plot.png", r'Computed Solution to $\Delta u = 4$ (Prob1)')



####Problem 2: Lap(u) = 6x + 6y
####BC1: u(x,y) = x^3 on y = 0
####BC2: u(x,y) = x^3 + 1 on y = 1
####BC3: u(x,y) = y^3 on x = 0
####BC4: u(x,y) = y^3 + 1 on x = 1
xDisc = .2;  
def g1(x,y):
     return x**3
def g2(x,y):
     return x**3 + 1
def g3(x,y):
     return y**3
def g4(x,y):
     return y**3 + 1
def f(x,y):
     return 6.0*x + 6.0*y
d1 = []; d2 = []; d3 = []; d4 = []; deg = 3
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)
newm.write("PDE_LP2.sol")

useSol(newm, .05, deg, True, "prob2LP_plot.png", r'Computed Solution to $\Delta u = 6x + 6y$ (Prob2)')

####Problem 3: Lap(u) = 3x + 3y
####BC1: u(x,y) = x^3 on y = 0
####BC2: u(x,y) = x^3 + 1 on y = 1
####BC3: u(x,y) = y^3 on x = 0
####BC4: u(x,y) = y^3 + 1 on x = 1
####Note that this problem does not have a solution. However, we will dualize the boundary conditions and do our best
xDisc = .2;  
def g1(x,y):
     return x**3
def g2(x,y):
     return x**3 + 1
def g3(x,y):
     return y**3
def g4(x,y):
     return y**3 + 1
def f(x,y):
     return 3.0*x + 3.0*y
d1 = []; d2 = []; d3 = []; d4 = []; deg = 3
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)


newm.write("PDE_LP3.sol")

useSol(newm, .05, deg, True, "prob3LP_plot.png", r'Computed Solution to $\Delta u = 3x + 3y$ (Prob3)')



####Problem 4: Lap(u) = 1 - 2y
####BC1: u(x,y) = 0 on y = 0
####BC2: u(x,y) = x(1-x) on y = 1
####BC3: u(x,y) = 0 on x = 0
####BC4: u(x,y) = 0 + 1 on x = 1

xDisc = .2;  
def g1(x,y):
     return 0.0
def g2(x,y):
     return x*(1-x)
def g3(x,y):
     return 0
def g4(x,y):
     return 0
def f(x,y):
     return  - 2.0*y
d1 = []; d2 = []; d3 = []; d4 = []; deg = 3
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)


newm.write("PDE_LP4.sol")

useSol(newm, .05, deg, True, "prob4LP_plot.png", r'Computed Solution to $\Delta u = 1 - 2y$ (Prob4)')



####Problem 5: Lap(u) = -2y(1 - y) - 2x(1-x)
####BC1: u(x,y) = 0 on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = 0 on x = 0
####BC4: u(x,y) = 0 + 1 on x = 1

xDisc = .1;  
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
d1 = []; d2 = []; d3 = []; d4 = []; deg = 4
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)


newm.write("PDE_LP5.sol")

useSol(newm, .05, deg, True, "prob5LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob5)')



####Problem 6: Lap(u) = -2y(1 - y) - 2x(1-x)
####BC1: u(x,y) = .1 on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = -.1 on x = 0
####BC4: u(x,y) = 0 + 1 on x = 1
####No actual solution
xDisc = .1;  
def g1(x,y):
     return 0.1
def g2(x,y):
     return 0
def g3(x,y):
     return -.1
def g4(x,y):
     return 0
def f(x,y):
     return  -2.0*y*(1.0 - y) - 2.0*x*(1.0-x)
d1 = []; d2 = []; d3 = []; d4 = []; deg = 4
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)


newm.write("PDE_LP6.sol")

useSol(newm, .05, deg, True, "prob6LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob6)')



####Problem 7: Lap(u) = -2y(1 - y) - 2x(1-x)
####BC1: u(x,y) = epsilon on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = -epsilon on x = 0
####BC4: u(x,y) = 0 + 1 on x = 1
####No actual solution
epsilon = .01
xDisc = .1;  
def g1(x,y):
     return epsilon
def g2(x,y):
     return 0
def g3(x,y):
     return -epsilon
def g4(x,y):
     return 0
def f(x,y):
     return  -2.0*y*(1.0 - y) - 2.0*x*(1.0-x)
d1 = []; d2 = []; d3 = []; d4 = []; deg = 4
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)


newm.write("PDE_LP7.sol")

useSol(newm, .05, deg, True, "prob7LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob7)')


####Problem 8: Lap(u) = 0
####BC1: u(x,y) = e(x) on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = e(y) on x = 0
####BC4: u(x,y) = 0 on x = 1
####No actual solution
epsilon = .01
xDisc = .1;  
def g1(x,y):
     return math.exp(x)
def g2(x,y):
     return 0
def g3(x,y):
     return math.exp(y)
def g4(x,y):
     return 0
def f(x,y):
     return  -2.0*y*(1.0 - y) - 2.0*x*(1.0-x)
d1 = []; d2 = []; d3 = []; d4 = []; deg = 8
newm = makeLP(xDisc, f, g1, d1, g2, d2, g3, d3, g4, d4, deg)


newm.write("PDE_LP8.sol")

useSol(newm, .05, deg, True, "prob8LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob8)')




####Problem 9: Lap(u) = -2y(1 - y) - 2x(1-x)
####BC1: u(x,y) = 0 on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = 0 on x = 0
####BC4: u(x,y) = 0 on x = 1
####NC1: u_{-y}(x,y) = -(1 - 2y)x(1-x) on y = 0
####NC2: u_{y}(x,y) = (1 - 2y)x(1-x) on y = 1
####NC3: u_{-x}(x,y) = -(1 - 2x)y(1-y) on x = 0
####NC4: u_{x}(x,y) = (1 - 2x)y(1-y) on x = 1

xDisc = .1;  
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
newm = makeLPCauchy(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg)


newm.write("PDE_LP9.sol")

useSol(newm, .05, deg, True, "prob9LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob9)')




####Problem 10: Lap(u) = -2y(1 - y) - 2x(1-x)
####BC1: u(x,y) = 0 on y = 0
####BC2: u(x,y) = 0 on y = 1
####BC3: u(x,y) = 0 on x = 0
####BC4: u(x,y) = 0 on x = 1
####NC1: u_{-y}(x,y) = -(1 - 2y)x(1-x) + eps on y = 0
####NC2: u_{y}(x,y) = (1 - 2y)x(1-x) on y = 1
####NC3: u_{-x}(x,y) = -(1 - 2x)y(1-y) - eps on x = 0
####NC4: u_{x}(x,y) = (1 - 2x)y(1-y) on x = 1
eps = .1
xDisc = .1;  
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
     return -(1 - 2.0*y)*x*(1-x) + eps
def g22(x,y):
     return (1 - 2.0*y)*x*(1-x)
def g33(x,y):
     return -(1 - 2.0*x)*y*(1-y) - eps
def g44(x,y):
     return (1 - 2.0*x)*y*(1-y)
deg = 4
newm = makeLPCauchy(xDisc, f, g1, g11, g2, g22, g3, g33, g4, g44, deg)


newm.write("PDE_LP10.sol")

useSol(newm, .05, deg, True, "prob10LP_plot.png", r'Computed Solution to $\Delta u = -2y(1 - y) - 2x(1-x)$ (Prob10)')
