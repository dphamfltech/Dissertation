
from dolfin import *
import numpy

tol = 1E-14 # tolerance for coordinate comparisons

# Create mesh and define fu_curction space
nx = 80
ny = 20

# define the rectangle
x_left  = 0.0   # right bdr
y_low  = 0.0
x_right = 4.0
y_upper = 1.0

# parameter
E  = 1.0
nu = 0.3
lamda  = ( E*nu )/( (1.0 + nu )*( 1.0- 2.0 *nu) )
mu     = E/( 2.0* (1.0 + nu) )

# print " Lamda = ", lamda
# print " mu = ", mu

mesh =  RectangleMesh(x_left,y_low,x_right,y_upper,nx,ny)
V    =  VectorFunctionSpace(mesh, 'Lagrange', 1)

u = TrialFunction( V  )

q = TestFunction( V )

h  = mesh.hmax()


# Define initial data
u_0 = Constant( (0.0, 0.0) )
v_0 = Constant( (0.0, 0.0) )


# Boby forcing term
F = Constant(  (0.0, 0.0) )


#Define left bdr domain
u_left = Expression ( ('t','0.0' ), t=0.0 )
# v_left = Constant   ( (1.0, 0.0) )


def left_bdr(x, on_bdr):
    return on_bdr and abs(x[0]) < tol

Uleft  = DirichletBC(V, u_left, left_bdr)
# Vleft  = DirichletBC(V, v_left, left_bdr)

u_right =  Constant( (0.0, 0.0) )
# v_right =  Constant( (0.0, 0.0) )


def right_bdr(x, on_bdr):
    return on_bdr and abs(x[0] - x_right) < tol

Uright = DirichletBC(V, u_right, right_bdr)
# Vright = DirichletBC(V, v_right, right_bdr)

#v_upper = Constant(0.0)
    #def upper_bdr(x, on_bdr):
#return on_bdr and abs(x[1] - y_upper) < tol
#Vupper = DirichletBC(V, v_upper, upper_bdr)


#v_lower = Constant(0.0)
    #def lower_bdr(x, on_bdr):
#return on_bdr and abs(x[1])  < tol
#Vlower = DirichletBC(V, v_lower, lower_bdr)

#bcu  =  [Uleft,Uright]
#bcv  =  [Vleft,Vright]
#bcut =  [Uleft,Vright,Utleft,Vtleft,Uright,Vright, Utright,Vtright]

bc =  [Uleft,Uright]

dt = 0.01*h      # time step
k = Constant( dt )

# Split mixed functions

u0  = project( u_0  , V )
u1  = project( u_0 + k * v_0  , V )

u_cur  = Function( V )  # u at step nth

a = inner( u, q ) * dx + k * k * (lamda + mu) * div(u) * div(q) * dx + k * k * mu * inner( grad(u), grad(q) ) * dx

L =  inner( 2.0 * u1 - u0 , q ) * dx + k * k * inner (F, q) * dx

A = assemble( a )


for bcu in bc:
    bcu.apply(A)

t = 0.0
T = 13.0

solver = LUSolver(A)


solver.parameters["reuse_factorization"] = True


# Create files for storing solution
ufile = File("results/Displacement.pvd")


while t < T:
    t += dt
    u_left.t= t
    #print 'time t=', t
    #print "u_left = ", u_left(0)
    b = assemble(L)
    for bcu in bc:
        bcu.apply(b)
    solver.solve(u_cur.vector() , b)

    # Save to file
#ufile << u_cur
    u0.assign(u1)
    u1.assign(u_cur)

u0x, v0y = u0.split(deepcopy=True)  # extract components
ux, vy = u1.split(deepcopy=True)  # extract components

plot(ux,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "u_component",  # default plotfile name
     rescale = False)

plot(vy,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "v_component",  # default plotfile name
     rescale = False)


plot((ux-u0x)*1/k,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "u_component",  # default plotfile name
     rescale = False)

plot((vy-v0y)*1/k,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "v_component",  # default plotfile name
     rescale = False)




