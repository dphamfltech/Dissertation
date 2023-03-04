
from dolfin import *
import numpy

# Create mesh and define function space
nx = 500
R_bdr=4.0   # right boundary
mesh = mesh = IntervalMesh(nx,0,R_bdr)
V = FunctionSpace(mesh, 'Lagrange', 4)
u = TrialFunction( V )
v = TrialFunction( V )
q = TestFunction( V )
r = TestFunction( V )

h=mesh.hmax()



# Define initial data
v_0 = Constant(0.0)
u_0 = Constant(0.0)


#Define left boundary domain
u_L = Expression("t", t=0.0)
v_L = Constant("1.0")

def left_boundary(x, on_boundary):
    tol = 1E-14 # tolerance for coordinate comparisons
    return on_boundary and abs(x[0]) < tol

UL = DirichletBC(V, u_L, left_boundary)
VL = DirichletBC(V, v_L, left_boundary)

u_R = Constant("0.0")
v_R = Constant("0.0")

def right_boundary(x, on_boundary):
    tol = 1E-14 # tolerance for coordinate comparisons
    return on_boundary and abs(x[0] - R_bdr) < tol

UR = DirichletBC(V, u_R, right_boundary)
VR = DirichletBC(V, v_R, right_boundary)

bc_u =  [UL, UR]
bc_v =  [VL, VR]


dt = 0.1*h      # time step
k = Constant( dt )
g =  Constant(0.0)


u0 = project( u_0 , V )
v0 = project( v_0, V )


un = Function( V )
vn = Function( V )
    
    
a1 = u * q * dx
a2 = v * r * dx + k * inner( grad(u) , grad( r ) ) * dx

L1 = (u0 * q + k*v0*q)* dx
L2 = v0 * r * dx 


M1 = assemble( a1 )
M2 = assemble( a2 )

for bcu in bc_u:
    bcu.apply(M1)

for bcv in bc_v:
    bcv.apply(M2)


t=0.0
T=1.0
    
solver = LUSolver(M1)
solver2 = LUSolver(M2)
    
solver.parameters["reuse_factorization"] = True
solver2.parameters["reuse_factorization"] = True
    
    
while t < T:
    u_L.t
    b1 = assemble( L1 )
    for bcu in bc_u:
        bcu.apply(b1)
    solver.solve( un.vector() , b1 )
    b2 = assemble( L2 )
    for bcv in bc_v:
        bcv.apply(b2)
    solver2.solve( vn.vector() , b2 )
    u0.vector()[:] = un.vector()
    v0.vector()[:] = vn.vector()
    t += dt
    

plot(un, wireframe=True ,  title="Solution u")
plot(vn, wireframe=True ,  title="Velocity v")
interactive()
