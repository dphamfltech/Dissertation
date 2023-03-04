
from dolfin import *
import numpy

# Create mesh and define function space
nx = 500
mesh = mesh = IntervalMesh(nx,0,1)
V = FunctionSpace(mesh, 'Lagrange', 1)
u = TrialFunction( V )
v = TrialFunction( V )
q = TestFunction( V )
r = TestFunction( V )

h=mesh.hmax()



# Define initial data
v_0 = Constant(0.0)
u_0 = Constant(0.0)

# Define boundary data
ub = Expression("(1-x[0])*t", t = 0.0)
vb = Expression("1.0-x[0]")


# Define the boudary data

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
bc_u = DirichletBC(V, ub, boundary)
bc_v = DirichletBC(V, vb, boundary)


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
    
bc_u.apply(M1)
bc_v.apply(M2)
    
t=0.0
T=1.0
    
solver = LUSolver(M1)
solver2 = LUSolver(M2)
    
solver.parameters["reuse_factorization"] = True
solver2.parameters["reuse_factorization"] = True
    
    
while t < T:
    b1 = assemble( L1 )
    bc_u.apply( b1 )
    solver.solve( un.vector() , b1 )
    b2 = assemble( L2 )
    bc_v.apply( b2 )
    solver2.solve( vn.vector() , b2 )
    u0.vector()[:] = un.vector()
    v0.vector()[:] = vn.vector()
    t = t + dt
    

plot(un, wireframe=True ,  title="Solution u")
plot(vn, wireframe=True ,  title="Velocity v")
interactive()
