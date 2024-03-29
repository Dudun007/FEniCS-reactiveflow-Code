#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:45:39 2021

@author: C00257297
"""

from fenics import *
from mshr import *
import numpy as np
from datetime import date
import sys
import random
import math
from statistics import mean


#Simulation Parameters
T = 10000000
#num_steps = 100
#dt = T/num_steps

#Initial and Boundary Condition
u_inlet = 7.2E-6 # m/sec = 6000 cm/min)
C_inlet = 0.198              #mass fraction #mol/m3

dt = 0.01/u_inlet

num_steps = int(T/dt)

#Rock and Fluid Properties
D = Constant(3.6E-9)              #m2/sec
mu = Constant(0.01)            #pa-sec (1 pa-sec = 1000 cp)
poro_ref = 0.2                  #fraction
perm_ref =1E-15              #m2 (1 mD = 1E-15 m2)
rho_f = 1140                    #kg/m3
rho_m = 2710                    #kg/m3
beta100 = 1.37                 #fraction   #kg/mol
k_s = 0.000252                    #m/s
a_v0 = 14.286                      #1/m

print("Completed!")
today = date.today()
print("Today's date:", today)

# Create a computational domain
DimX = 1
DimY = 0.4 
domain = Rectangle(Point(0,0), Point(DimX,DimY))

# Create a subdomain for fractures (each frac is a rectangle)
def createFracSubDomain (fracdata):
  fracdomain = []

  for i in range (0, len(fracdata)):
    L = fracdata[i][0]      # the length of your rectangle
    b = fracdata[i][1]      # aperture
    a = fracdata[i][2]      # orientation angle
    Cx = fracdata[i][3]     # center point x
    Cy = fracdata[i][4]     # center point y

    # coordinate of each corner point for rectangle with the center at (0,0)
    LeftTop_x = -L/2; LeftTop_y = b/2; LeftBot_x = -L/2; LeftBot_y = -b/2
    RightBot_x = L/2; RightBot_y = -b/2; RightTop_x = L/2; RightTop_y = b/2

    #The rotated position of each corner    
    Rx1 = Cx + (LeftTop_x  * math.cos(a)) - (LeftTop_y * math.sin(a))
    Ry1 = Cy + (LeftTop_x  * math.sin(a)) + (LeftTop_y * math.cos(a))
    Rx2 = Cx + (LeftBot_x  * math.cos(a)) - (LeftBot_y * math.sin(a))
    Ry2 = Cy + (LeftBot_x  * math.sin(a)) + (LeftBot_y * math.cos(a))
    Rx3 = Cx + (RightBot_x  * math.cos(a)) - (RightBot_y * math.sin(a))
    Ry3 = Cy + (RightBot_x  * math.sin(a)) + (RightBot_y * math.cos(a))
    Rx4 = Cx + (RightTop_x  * math.cos(a)) - (RightTop_y * math.sin(a))
    Ry4 = Cy + (RightTop_x  * math.sin(a)) + (RightTop_y * math.cos(a))

    domain_vertices = [Point(Rx1,Ry1),Point(Rx2,Ry2),Point(Rx3,Ry3),Point(Rx4,Ry4),Point(Rx1,Ry1)]

    if i == 0:
      fracdomain = Polygon(domain_vertices)
    else:
      fracdomain = fracdomain + Polygon(domain_vertices)

  return fracdomain


# generate fracture subdmain based on fracture data 
# fracdata = [length, aperture, angle, center_x, center_y]

Length = 0.2 
fracwidth = 0.0001

fracdata = [
            [Length,fracwidth, 1.3, 0.1, 0.2],
            [Length,fracwidth,0.5, 0.25, 0.18],
            [Length,fracwidth, -1.3, 0.4, 0.2],
            [Length,fracwidth, -0.7, 0.42, 0.3],
            [Length,fracwidth, -0.5, 0.58, 0.23],
            [Length,fracwidth, 1.3,  0.65, 0.2],
            [Length,fracwidth, 0.2,  0.53, 0.3],
            [Length,fracwidth, 0.6,  0.63, 0.26],
            [Length,fracwidth, -1.3,  0.73, 0.2],
            [Length,fracwidth, -0.25,  0.36, 0.23],
            [Length,fracwidth, 0.5, 0.8, 0.25],            
            [Length,fracwidth, -1.3, 0.93, 0.2]
           ]
 
  

fracdomain = createFracSubDomain (fracdata)


# Add fractures into the domain and mark them as subdomain '1'
domain.set_subdomain(1, fracdomain)

# Generature mesh for the combined system
mesh = generate_mesh(domain,50)

##local refinement
#cellsRefined = MeshFunction("bool", mesh, 2)
#cellsRefined.set_all(False)
#
#class RefineArea(SubDomain):
#   def inside(self, x, on_boundary):
#       return (0.2 <= x[0] <= 0.8 and 0.1 <= x[1] <= 0.3)
#
#RefineDomain = RefineArea()
#RefineDomain.mark(cellsRefined, True)
#
#mesh = refine(mesh, cellsRefined)

print("number of element", mesh.num_cells())

# Create a cell list for fractures -> frac_cells
domains = mesh.domains()
subdomains = MeshFunction('size_t',mesh,2,domains)
frac_cells = SubsetIterator(subdomains, 1)

#plot(mesh)
#
#
#sys.exit(0)

# Define function spaces and mixed (product) space
scalar = FiniteElement("CG", mesh.ufl_cell(), 1)
vector = VectorElement("CG", mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, MixedElement(vector, scalar))

# Define trial and test functions
(u, P) = TrialFunctions(W)
(tf1, tf2) = TestFunctions(W)

# Define trial and test functions for C
CFS = FunctionSpace(mesh, "CG", 1)
Ctf = TrialFunction(CFS)
tf3 = TestFunction(CFS)
C = Function(CFS)
C_n = Function(CFS)

#C = project(Constant(0.0), CFS)
C_n = project(Constant(0.0), CFS)

#Initialize Properties and Generate Cell Functions for Properties
poro = MeshFunction("double", mesh, 2)
perm = MeshFunction("double", mesh, 2)
a_v = MeshFunction("double", mesh, 2)

random.seed(500)

for cell in cells(mesh):
    #poro[cell] = (random.uniform(0,1)*0.15)+0.05
    poro[cell] = random.gauss(0.2, 0.01)
    perm[cell] = perm_ref*(poro[cell]/poro_ref)*(pow(poro[cell]/poro_ref,2))*pow((1-poro_ref)/(1-poro[cell]),2)
    a_v[cell] = a_v0*pow((poro[cell]/poro_ref),1)*pow((perm_ref*poro[cell])/(perm[cell]*poro_ref),0.5)

#update properties in fracture cells
for cell in frac_cells:
    poro[cell] = 0.9999999
    perm[cell] = perm_ref*(poro[cell]/poro_ref)*(pow(poro[cell]/poro_ref,2))*pow((1-poro_ref)/(1-poro[cell]),2)
    a_v[cell] = a_v0*pow((poro[cell]/poro_ref),1)*pow((perm_ref*poro[cell])/(perm[cell]*poro_ref),0.5)

# Define Properties Functions and Initialize###############################
#plot(poro)
print(max(poro.array()[:]),min(poro.array()[:]))
print(max(perm.array()[:]),min(perm.array()[:]))
print(max(a_v.array()[:]),min(a_v.array()[:]))

def calculatePV(poro, mesh):
  PV = 0
  for cell in cells(mesh):
    PV = PV + poro[cell]*cell.volume()

  return PV

Pore_volume = calculatePV(poro, mesh)

Pore_volume = Pore_volume +(13*Length*fracwidth)

print("pore volume ", Pore_volume )
 

# Code for C++ evaluation of conductivity
property_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class Property : public dolfin::Expression
{
public:

  // Create expression with 3 components
  Property() : dolfin::Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    const uint cell_index = cell.index;
    values[0] = (*c0)[cell_index];
    values[1] = (*c1)[cell_index];
    values[2] = (*c2)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<dolfin::MeshFunction<double>> c0;
  std::shared_ptr<dolfin::MeshFunction<double>> c1;
  std::shared_ptr<dolfin::MeshFunction<double>> c2;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Property, std::shared_ptr<Property>, dolfin::Expression>
    (m, "Property")
    .def(py::init<>())
    .def_readwrite("c0", &Property::c0)
    .def_readwrite("c1", &Property::c1)
    .def_readwrite("c2", &Property::c2);
}

"""

c = CompiledExpression(compile_cpp_code(property_code).Property(),
                       c0=poro, c1=perm, c2=a_v, degree=0)

Phi = c[0]
K = c[1]
A_v = c[2]

# Create functions for boundary conditions
noslip = Constant((0, 0))
inflow = Constant((u_inlet, 0))
zero   = Constant(0)

#Define boundary
def boundary1(x):
    return x[0] < DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < DimY - DOLFIN_EPS

def boundary2(x):
    return (x[1] < DOLFIN_EPS or x[1] > DimY - DOLFIN_EPS) #or (x[0] < DOLFIN_EPS and x[1] <= 0.48) or (x[0] < DOLFIN_EPS and x[1] >= 1-0.48)

def boundary3(x):
    return x[0] > DimX - DOLFIN_EPS

# Define essential boundary
# No-slip boundary condition for velocity
bc0 = DirichletBC(W.sub(0), noslip, boundary2)
# Inflow boundary condition for velocity
bc1 = DirichletBC(W.sub(0), inflow, boundary1)
# Boundary condition for pressure at outflow
bc2 = DirichletBC(W.sub(1), zero, boundary3)
# Collect boundary conditions
bcs = [bc0, bc1, bc2]

print("Completed!")


f = Constant((0, 0))

a = mu/Phi*inner(grad(u),grad(tf1))*dx \
    + inner(grad(P),tf1)*dx \
    + mu/K*inner(u,tf1)*dx \
    + div(u)*tf2*dx
L = dot(f,tf1)*dx  - A_v*beta100*k_s*rho_f/rho_m*C_n*tf2*dx

w = Function(W)

#Assemble Eq for C
Cbc1 = DirichletBC(CFS, Constant(C_inlet), boundary1)

(u, P) = w.split()

C_mid = Ctf#(Ctf+C_n)*0.5
F = (Ctf-C_n)*Phi*tf3*dx + dt*(tf3*dot(u,grad(C_mid))*dx \
     + D*Phi*dot(grad(tf3),grad(C_mid))*dx) + dt*Phi*A_v*k_s*C_mid*tf3*dx

#Residue
res = (Ctf-C_n)*Phi+dt*(dot(u,grad(C_mid))-D*Phi*div(grad(C_mid))) + dt*Phi*A_v*k_s*C_mid

# Add SUPG stabilisation terms
vnorm = sqrt(dot(u, u))
h = 2*Circumradius(mesh)
delta = h/(2.0*vnorm)
F = F + delta*dot(u, grad(tf3))*res*dx

a1 = lhs(F)
L1 = rhs(F)

print("Completed!")



#defining output data files
vtkfileC = File('./Output/Concentration.pvd')
vtkfileP = File('./Output/Pressure.pvd')
vtkfileV = File('./Output/Velocity.pvd')
vtkfilePhi = File('./Output/Porosity.pvd')
vtkfileK = File('./Output/Permeability.pvd')



Left = AutoSubDomain(lambda x, on_bnd: near(x[0], 0) and on_bnd)
Right = AutoSubDomain(lambda x, on_bnd: near(x[0], 1) and on_bnd)
# Mark a CG1 Function with different values on the two boundaries
#V = FunctionSpace(mesh, 'CG', 1)
bcp0 = DirichletBC(CFS, 1, Left)
bcp1 = DirichletBC(CFS, 2, Right)
C_ = Function(CFS)
bcp0.apply(C_.vector())
bcp1.apply(C_.vector()) 



R_pressure = [] 
t = 0
for n in range (num_steps):
    t = t + dt
    Counter = 0
    pressure = []
    #update Properties#######################################
    for cell in cells(mesh):
      coord_celda = cell.get_vertex_coordinates()  
      center_celda_x = (coord_celda[0] + coord_celda[2] + coord_celda[4])/3
      center_celda_y = (coord_celda[1] + coord_celda[3] + coord_celda[5])/3
       
      cp = (center_celda_x, center_celda_y)
        
      CinCell = C_n(cp)
      if CinCell < 0:
        CinCell = 0.0

      poro[cell] = poro[cell] + dt*a_v[cell]*beta100*rho_f/rho_m*k_s*CinCell
      if poro[cell] > 0.999999:
        poro[cell] = 0.999999
      perm[cell] = perm_ref*(poro[cell]/poro_ref)*(pow(poro[cell]/poro_ref,2))*pow((1-poro_ref)/(1-poro[cell]),2)
      a_v[cell] = a_v0*pow((poro[cell]/poro_ref),1)*pow((perm_ref*poro[cell])/(perm[cell]*poro_ref),0.5)


    #poro.array()[:] = poro.array()[:] + dt*a_v.array()[:]*beta100*rho_f/rho_m*k_s*C_n.vector()[:]
    #poro.array()[poro.array() >= 0.99999999] = 0.99999999
    #perm.array()[:] = perm_ref*pow(pow(poro.array()[:]/poro_ref,3)*pow((1-poro_ref)/(1-poro.array()[:]),2),6)
    #a_v.array()[:] = a_v0*pow(poro_ref/poro.array()[:]*(1-poro.array()[:])/(1-poro_ref),1)

    solve(a == L, w, bcs)
    
    (u, P) = w.split()
    
    #1D decay decoupled procedure
    #tempA = project(Constant(0), CFS)
    #tempA.vector()[:] = -dt*A_v.vector()[:]*k_s*C_n.vector()[:]*Phi.vector()[:]
    #C_n.vector()[:] = C_n.vector()[:]*np.exp(tempA.vector()[:])
    #########
    
    solve(a1 == L1, C, Cbc1)
    
    #negative to zero
    C.vector()[:] = (C.vector()[:] + abs(C.vector()[:]))*0.5
    
    C_n.assign(C)
    

    if n==0 or (n+1)%2==0:  
        vtkfileP << (P, t)
        vtkfileV << (u, t)
        vtkfileC << (C, t)
        vtkfilePhi << (poro, t)
      #vtkfileK << (K, t)

      #print ("t =", t, "/", T, "sec")
        
#        L_Pressure= mean(P.vector()[P_.vector() == 1]))
#        R_Pressure= mean(P.vector()[P_.vector() == 2]))
#print ("Values on right   = ", mean(P.vector()[P_.vector() == 2]))
          
        
        for i in np.linspace(0.0035,0.39,10):
            
            point=(0.007,i)
            
            pressure.append(P(point))
            
        if mean(pressure)>= 0:
            #print(pressure)
            R_p = mean(pressure)

            R_pressure.append(R_p)
            
            print (t, R_p)

#        R_concentration = max(C.vector()[C_.vector() == 2])
#        print(t, R_concentration)

        if (R_pressure[-1] <= R_pressure[0]/100):
            break
#        j=j+1
#        
#print ("Values on left = ", mean(P.vector()[P_.vector() == 1]))
#print ("Values on right   = ", mean(P.vector()[P_.vector() == 2]))
      
#
PV_b = (u_inlet * 0.4 * t)/(Pore_volume)
#    
plot(poro)

print("Pore volume at Breakthrough: ", PV_b)
##print ("Simulation Completed!")
