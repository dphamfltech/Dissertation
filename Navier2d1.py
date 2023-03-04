
from dolfin import *
import numpy

tol = 1E-14 # tolerance for coordinate comparisons

# Create mesh and define fu_curction space
nx = 100
ny = 100

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
V    =  VectorFunctionSpace(mesh, 'Lagrange', 2)


u  = TrialFunction( V )
v  = TrialFunction( V )

q  = TestFunction( V )
r  = TestFunction( V )

h  = mesh.hmax()


# Define initial data
u_0 = Constant( (0.0, 0.0) )
v_0 = Constant( (0.0, 0.0) )


# Boby forcing term
F = Constant(  (0.0, 0.0) )


#Define left bdr domain
u_left = Expression ( ('t','0.0' ), t=0.0 )
v_left = Constant   ( (1.0, 0.0) )


def left_bdr(x, on_bdr):
    return on_bdr and abs(x[0]) < tol

Uleft  = DirichletBC(V, u_left, left_bdr)
Vleft  = DirichletBC(V, v_left, left_bdr)

u_right =  Constant( (0.0, 0.0) )
v_right =  Constant( (0.0, 0.0) )


def right_bdr(x, on_bdr):
    return on_bdr and abs(x[0] - x_right) < tol

Uright = DirichletBC(V, u_right, right_bdr)
Vright = DirichletBC(V, v_right, right_bdr)

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

bc =  [Uleft,Uright,Vleft,Vright]

dt = 0.001*h      # time step
k = Constant( dt )

u0  = project( u_0  , V )
v0  = project( v_0  , V )


u_cur  = Function( V )  # u at step nth
v_cur  = Function( V )  # v at step nth


a1 = inner( u, q ) * dx

a2 = inner( v, r ) * dx


L1 =  inner( u0, q ) * dx + k * inner( v0, q ) * dx

L2 = inner( v0, r ) * dx  - k * (lamda + mu) * div(u_cur) * div(r) * dx - k * mu * inner( grad(u_cur), grad(r) ) * dx + k * inner (F, r) * dx

#L2 = inner( v0, r ) * dx - k * div(u_cur) * div(r) * dx - k * inner( grad(u_cur), grad(r) ) * dx + k * inner (F, r) * dx

A1 = assemble( a1 )

A2 = assemble( a2 )

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"



for bcu in bc:
    bcu.apply(A1)
    bcu.apply(A2)

t = 0.0
T = 2.0
    
solver1 = LUSolver(A1)
solver2 = LUSolver(A2)

solver1.parameters["reuse_factorization"] = True
solver2.parameters["reuse_factorization"] = True

# Create files for storing solution
#ufile = File("results/Displacement.pvd")
#vfile = File("results/Velocity.pvd")



while t < T:
    u_left.t= t
    #print 'time t=', t
    #print "u_left = ", u_left(0)
    b1 = assemble(L1)
    for bcu in bc:
        bcu.apply(b1)
    solver1.solve(u_cur.vector() , b1)

    b2 = assemble(L2)
    for bcu in bc:
        bcu.apply(b2)
    solver2.solve(v_cur.vector() , b2)

    # Save to file
    # ufile << u_cur
    # vfile << v_cur

    u0.assign(u_cur)
    v0.assign(v_cur)
    t += dt

u1, v1 = u_cur.split(deepcopy=True)  # extract components

plot(u1,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "u_component",  # default plotfile name
     rescale = False)

plot(v1,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "v_component",  # default plotfile name
     rescale = False)

u2, v2 = v_cur.split(deepcopy=True)  # extract components

plot(u2,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "u_component",  # default plotfile name
     rescale = False)

plot(v2,
     mode = "displacement",
     mesh = mesh,
     wireframe = False,
     interactive = True,         # hold plot on screen
     axes = True,                # include axes
     basename = "v_component",  # default plotfile name
     rescale = False)



#plot(u2, title='x-component of flux ')
#plot(v2, title='y-component of flux ')


#plot(u_cur, title="Displacement u", rescale= True )
#plot(u_cur[1], title="Displacement u")
#plot(v_cur, title="Velocity v", rescale= True)
#plot(v0[1], title="Velocity v")

