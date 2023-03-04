
from dolfin import *
import numpy

tol = 1E-14 # tolerance for coordinate comparisons

# Create mesh and define fu_curction space
nx = 50
ny = 50

# define the rectangle
x_left  = 0.0   # right bdr
y_low  = 0.0
x_right = 4.0
y_upper = 1.0

# parameter
E  = 1.0
nu = 0.3
lamda  = ( E*nu )/( (1.0+nu)*(1-2*nu) )
mu      = E/( 2.0*(1+nu) )
# print " Lamda = ", lamda
# print " mu = ", mu

mesh =  RectangleMesh(x_left,y_low,x_right,y_upper,nx,ny)
V = FunctionSpace(mesh, 'Lagrange', 1)

u  = TrialFunction( V )
v  = TrialFunction( V )
ut = TrialFunction( V )
vt = TrialFunction( V )

q  = TestFunction( V )
r  = TestFunction( V )
s  = TestFunction( V )
w  = TestFunction( V )


h=mesh.hmax()


# Define initial data
v_0 = Constant(0.0)
u_0 = Constant(0.0)
ut_0 = Constant(0.0)
vt_0 = Constant(0.0)


# Boby forcing term
F_x = Constant(0.0)
F_y = Constant(0.0)


#Define left bdr domain
u_left = Expression('2*t', t=0.0)
v_left = Constant(0.0)
ut_left = Constant(1.0)
vt_left = Constant(0.0)

def left_bdr(x, on_bdr):
    return on_bdr and abs(x[0]) < tol

Uleft  = DirichletBC(V, u_left, left_bdr)
Vleft  = DirichletBC(V, v_left, left_bdr)
Utleft = DirichletBC(V, ut_left, left_bdr)
Vtleft = DirichletBC(V, vt_left, left_bdr)

u_right =  Constant(0.0)
v_right =  Constant(0.0)
ut_right = Constant(0.0)
vt_right = Constant(0.0)


def right_bdr(x, on_bdr):
    return on_bdr and abs(x[0] - x_right) < tol

Uright = DirichletBC(V, u_right, right_bdr)
Vright = DirichletBC(V, v_right, right_bdr)
Utright = DirichletBC(V, ut_right, right_bdr)
Vtright = DirichletBC(V, vt_right, right_bdr)

v_upper = Constant(0.0)
def upper_bdr(x, on_bdr):
    return on_bdr and abs(x[1] - y_upper) < tol
Vupper = DirichletBC(V, v_upper, upper_bdr)


v_lower = Constant(0.0)
def lower_bdr(x, on_bdr):
    return on_bdr and abs(x[1])  < tol
Vlower = DirichletBC(V, v_lower, lower_bdr)

#bcu  =  [Uleft,Uright]
#bcv  =  [Vleft,Vright]
#bcut =  [Uleft,Vright,Utleft,Vtleft,Uright,Vright, Utright,Vtright]
bc =  [Uleft,Uright,Utleft,Utright,Vleft,Vright,Vtleft,Vtright]



dt = 0.1*h      # time step
k = Constant( dt )


u0  = project( u_0  , V )
v0  = project( v_0  , V )
ut0 = project( ut_0 , V )
vt0 = project( vt_0 , V )


u_cur  = Function( V )  # u at step nth
ut_cur = Function( V )  # u_t at step nth
v_cur  = Function( V )  # v at step nth
vt_cur = Function( V )  # v_t at step nth


a1 = u * q * dx

a2 = v * r * dx

a3 = ut * s * dx

a4 = vt * w * dx


L1 = (u0 + k * ut0 )* q * dx

L2 = (v0 + k * vt0 )* r * dx

L3 =  ut0 * s * dx - k * inner( grad(u_cur), grad (s) )* dx - k * (lamda + mu ) * inner( grad(u_cur)[0] + grad(v_cur)[1] , grad(s)[0]) * dx +  k * F_x * s * dx

L4 = vt0 * w * dx - k * inner( grad(v_cur), grad (w) ) * dx - k * (lamda + mu ) * inner ( grad(u_cur)[0] + grad(v_cur)[1] , grad(w)[1]) * dx +  k * F_y * w * dx

A1 = assemble( a1 )

A2 = assemble( a2 )

A3 = assemble( a3 )

A4 = assemble( a4 )

for bcu in bc:
    bcu.apply(A1)
    bcu.apply(A2)
    bcu.apply(A3)
    bcu.apply(A4)


t=0.0
T=1.0
    
solver1 = LUSolver(A1)
solver2 = LUSolver(A2)
solver3 = LUSolver(A3)
solver4 = LUSolver(A4)

solver1.parameters["reuse_factorization"] = True
solver2.parameters["reuse_factorization"] = True
solver3.parameters["reuse_factorization"] = True
solver4.parameters["reuse_factorization"] = True
    
while t < T:
    t += dt
    u_left.t=t
    b1 = assemble(L1 )
    b2 = assemble(L2 )
    for bcu in bc:
        bcu.apply(b1)
        bcu.apply(b2)
    solver1.solve( u_cur.vector() , b1 )
    solver2.solve( v_cur.vector() , b2 )
    
    b3 = assemble(L3)
    b4 = assemble(L4 )
    for bcu in bc:
        bcu.apply(b3)
        bcu.apply(b4)
    solver3.solve( ut_cur.vector() , b3 )
    solver4.solve( vt_cur.vector() , b4 )
 
    u0.vector()[:] = u_cur.vector()
    v0.vector()[:] = v_cur.vector()
    ut0.vector()[:] = ut_cur.vector()
    vt0.vector()[:] = vt_cur.vector()
    
#u0.assign(u_cur)
#v0.assign(v_cur)
#ut0.assign(ut_cur)
#vt0.assign(vt_cur)

plot(u_cur, wireframe=True ,  title="Solution u")
plot(v_cur, wireframe=True ,  title="Solution v")
interactive()
