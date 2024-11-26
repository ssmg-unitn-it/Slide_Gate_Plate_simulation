#%% --------------------------------------------------------------
#%% Import necessary packages

# import external libraries
import fenics as fe
import numpy as np
import math
import sys
import os
import importlib
import time
from datetime import datetime
from matplotlib import pyplot as plt

# import user defined libraries
import constitutive_model as cm
import material_parameters as mp


#%% --------------------------------------------------------------
#%% Load case selector

# define output folder descriptive name
folder_descriptor = "SGP"

# select the behaviors to activate (thermal is there by default)
elasticity_flag = 1
plasticity_flag = 1
damage_flag = 1

# specify whether or not to perform an inner loop in the time step
open_loop_flag = 0

# specify whether to use the adaptive time stepping 
adaptive_time_stepping_flag = 1

# settings for mesh refinement
adaptive_mesh_refinement_flag = 1

# specify whether to output thermal dependent material properties
check_params_flag = 0

# specify whether to output post-process quantities
postprocess_cartesian_flag = 1

#%% --------------------------------------------------------------
#%% Simulation input parameters

# define geometry file
meshfolder = "./Meshes/"
meshfile = "SGP.xdmf"

# minimum allowed mesh size
size_refinement_threshold = 0.00015

# damage value mesh refinement threshold
damage_refinement_threshold = 0.3

# crack driving force value mesh refinement threshold
H_refinement_threshold = 0.5

# minimum number of cells interested by refinement to activate it
num_cell_refinement_threshold = 1

# initial time 
sim_time = 0.0

# maximum time
tmax = 2.5*60

# time step
dt_min = 1e-7
dt_max = 30
dt_initial = 0.01
cut_factor = 2
raise_factor = 2

# specify control threshold for adaptive time stepping
damage_energy_ratio_threshold = 1.5

# specify the time interval after which to save a solution [s]
fixed_dump_interval = 30

# maximum number of iterations allowed for the staggered algo
max_number_iterations = 30

# viscous regularization parameter
eta0 = fe.Constant(5e6)

# material characteristic time 
# (minimum time to develop irreversible deformations)
tau = 0.05

# Kelvin degrees shift
Kshift = 273.15

# reference temperature
ref_Temp = Kshift + 24

# specify numerical tolerance
ntol = 1e-6

# define convection coefficients and environments temperatures
steel_hconv = fe.Constant(50e3)
steel_Temp = fe.Constant(Kshift + 1560)

outer_hconv = fe.Constant(50)
outer_Temp = fe.Constant(ref_Temp)


#%% --------------------------------------------------------------
#%% Filesystem settings

# identify the number of subplots for the convergence plot
# defaults are: solution iterations and thermal energy
convergence_plots_number = 2

# identify the load case with a descriptive identifier
load_case = ""
    
# Thermal input is always necessary
load_case += "_T"
if elasticity_flag:
    load_case += "E"
    convergence_plots_number += 1
if plasticity_flag:
    load_case += "P"    
    convergence_plots_number += 1
if damage_flag:
    load_case += "D"
    convergence_plots_number += 1
if open_loop_flag:
    load_case += "_OL"
    # no need for the iteration plot
    convergence_plots_number += -1

now = datetime.now()
date_string = now.strftime("%Y%m%dT%H%M")[2:]

outfolder =  date_string+"_"+folder_descriptor+load_case+"/"

# solution unknown fields
T_outfile = fe.File(outfolder + "temp/temp.pvd")

if elasticity_flag or plasticity_flag:
    u_outfile = fe.File(outfolder + "disp/disp.pvd")

if damage_flag:
    d_outfile = fe.File(outfolder + "dam/dam.pvd")

if plasticity_flag:
    pc_outfile = fe.File(outfolder + "pc/pc.pvd")
    Baralp_outfile = fe.File(outfolder + "Baralp/Baralp.pvd")

# post-processing fields
if postprocess_cartesian_flag:
    sig_outfile = fe.File(outfolder + "sig/sig.pvd")
    eps_outfile = fe.File(outfolder + "eps/eps.pvd")
    pinv_outfile = fe.File(outfolder + "pinv/pinv.pvd")
    qinv_outfile = fe.File(outfolder + "qinv/qinv.pvd")
    c3t_outfile = fe.File(outfolder + "c3t/c3t.pvd")

if check_params_flag:
    Emod_outfile = fe.File(outfolder + "Emod/Emod.pvd")
    alpha_outfile = fe.File(outfolder + "alpha/alpha.pvd")

# prompt for forensic
readme = input("Describe why to run this simulation: \n")
with open(outfolder + "README.md", "w") as file:
    file.writelines(readme)

#%% --------------------------------------------------------------
#%% Simulation Meta-Parameters and control functions

# Use UFLACS to speed-up assembly and limit quadrature degree
fe.parameters['form_compiler']['representation'] = 'uflacs'
fe.parameters['form_compiler']['optimize'] = True
fe.parameters['form_compiler']['quadrature_degree'] = 2


#%% --------------------------------------------------------------
#%% Mesh

# import the mesh in FEniCS
with fe.XDMFFile( meshfolder + meshfile) as xdmf:
    mesh = fe.Mesh()
    xdmf.read(mesh)
    mesh.init()

# create mesh boundaries
boundaries=fe.MeshFunction("size_t",mesh,mesh.topology().dim()-1)
boundaries.set_all(0)

# mark useful boundaries
bc_input = fe.CompiledSubDomain(
    "pow(x[0],2)+pow(x[1],2)<pow(25e-3,2)+tol && on_boundary ",
    tol = 1e-5)
bc_input.mark(boundaries, 1)

bc_outer = fe.CompiledSubDomain(
    "pow(x[0],2)+pow(x[1],2)>pow(25e-3,2)+tol && x[1]>tol && on_boundary ",
    tol=1e-5)
bc_outer.mark(boundaries, 2)

bc_sym = fe.CompiledSubDomain(
    " x[1]<tol && on_boundary ",
    tol=1e-5)
bc_sym.mark(boundaries, 3)

bc_wedge = fe.CompiledSubDomain(
    "x[0]>-148e-3 && x[0]<-119e-3 && x[1] > tol && on_boundary ",
    tol=1e-5)
bc_wedge.mark(boundaries, 4)

bc_wedge2 = fe.CompiledSubDomain(
    " x[0]>143e-3 && x[0]<194e-3 && x[1] > tol && on_boundary ",
    tol=1e-5)
bc_wedge2.mark(boundaries, 5)

# define surface integration measures
ds_inner=fe.Measure("ds",subdomain_id=1,subdomain_data=boundaries)
ds_outer=fe.Measure("ds",subdomain_id=2,subdomain_data=boundaries)


#%% --------------------------------------------------------------
#%% Define Function Spaces for solution variables

# Function space for displacement problem
SU = fe.VectorFunctionSpace(mesh, "CG", 1)
u_trl = fe.TrialFunction(SU)
u_tst = fe.TestFunction(SU)

# Function space for temperature problem
ST = fe.FunctionSpace(mesh, "CG", 1)
T_trl = fe.TrialFunction(ST)
T_tst = fe.TestFunction(ST)

# Function spaces for damage phase field problem
SD = fe.FunctionSpace(mesh, "CG", 1)
d_trl = fe.TrialFunction(SD)
d_tst = fe.TestFunction(SD)

# Function space for scalar functions projection
SF = fe.FunctionSpace(mesh, "CG", 1)

# Function spaces for tensor functions projection
SX = fe.TensorFunctionSpace(mesh, "CG", 1)

# Redefine Function space for history variable
SH = fe.FunctionSpace(mesh, "DG", 0)


#%% --------------------------------------------------------------
#%% Auxiliary functions

T_new = fe.Function(ST, name="Temperature")
T_old = fe.Function(ST, name="Temperature")

u_new = fe.Function(SU, name="Displacement")

# variable defined to store plastic deformation values
eps_p_new = fe.Function(SX, name="Plastic_deformation_tensor")
Baralp_new = fe.Function(SF, name="Cumulated_plastic_deformation_new")

pc = fe.Function(SH, name="pc")

# variables only needed for the plasticity update
if plasticity_flag:
    Nval_new = fe.Function(SX, name="Plastic_flow_direction_new")
    Nval_old = fe.Function(SX, name="Plastic_flow_direction_old")
    Dgamma_new = fe.Function(SF, name="Plastic_multiplier_increment_new")
    Dgamma_old = fe.Function(SF, name="Plastic_multiplier_increment_old")
    Baralp_old = fe.Function(SF, name="Cumulated_plastic_deformation_old")
    eps_p_old = fe.Function(SX, name="Plastic_deformation_tensor_old")
else:
    Nval_new = None
    Nval_old = None
    Dgamma_new = None
    Dgamma_old = None
    Baralp_old = None
    eps_p_old = None

d_new = fe.Function(SD, name="Damage")

if damage_flag:
    Gc = fe.Function(SH, name="Specific_fracture_energy")
    H_old = fe.Function(SD, name="Crack_driving_force_history")
else:
    Gc = None
    H_old = None

if postprocess_cartesian_flag:
    # Auxiliary functions to store output for post-processing
    sign = fe.Function(SX, name="Stress_cartesian")
    epsn = fe.Function(SX, name="Strain_cartesian")

    # Auxiliary p-q function to store output for post-processing
    invpn = fe.Function(SF, name="invp")
    invqn = fe.Function(SF, name="invq")
    inv_cos_3theta_n = fe.Function(SF, name="inv_cos_3theta")
else:
    sign = None
    epsn = None
    invpn = None
    invqn = None
    inv_cos_3theta_n = None

# thermal material properties functions
alpha = fe.Function(SH, name="alpha")
cspec = fe.Function(SH, name="cspec")
Kcond = fe.Function(SH, name="Kcond")

# elastic material properties functions
Emod = fe.Function(SH, name="Emod")
lmbda = fe.Function(SH, name="lmbda")
mu = fe.Function(SH, name="mu")

# plastic material properties functions
if plasticity_flag:
    pcT = fe.Function(SH, name="pcT")
else:
    pcT = None

if adaptive_mesh_refinement_flag:
    H_condition = fe.Function(SH)

#%% --------------------------------------------------------------
#%% Functions to define temperature varying properties

class tdp(fe.UserExpression):
    def __init__(self, mp, property, temperature, **kwargs):
        super().__init__(kwargs)
        self.mp = mp
        self.property = property
        self.temperature = temperature

    def eval_cell(self, value, x, cell):        
        # interpolate the material data to find 
        # the property at the requested temperature
        value[0] = np.interp( 
            self.temperature(x), 
            mp.exp_data[self.property]["Temp"], 
            mp.exp_data[self.property]["value"]
        )

    # specify that the output value is scalar
    def value_shape(self):
        return ()


def update_properties(temperature, space):
    alpha.assign(fe.project(tdp(mp,"alpha",temperature),space))
    cspec.assign(fe.project(tdp(mp,"cspec",temperature),space))
    Kcond.assign(fe.project(tdp(mp,"Kcond",temperature),space))
    
    if elasticity_flag:
        Emod.assign(fe.project(tdp(mp,"Emod",temperature),space))
        lmbda.assign(fe.project(Emod*nu/(1.0+nu)/(1.0-2.0*nu),space))
        mu.assign( fe.project( Emod/2.0/(1.0+nu), space ))
    
    if plasticity_flag:
        pcT.assign(fe.project(tdp(mp,"pcT",temperature),space))

    if damage_flag:
        Gc.assign(fe.project(tdp(mp,"Gc",temperature),space))


#%% --------------------------------------------------------------
#%% Initial conditions

# initialize temperature initial conditions
T_old.assign( fe.interpolate(fe.Constant(ref_Temp), ST)) 

# initialize material properties with constant values
rho = mp.rho
nu = mp.nu

# initialize material properties with initial temperature 
update_properties(T_old, SH)

if plasticity_flag:
    pc.assign( pcT )
    pc0 = mp.pc0
    Omega = mp.Omega
    alpbp = mp.alpbp
    Mbp = mp.Mbp
    mbp = mp.mbp
    betbp = mp.betbp
    gambp = mp.gambp
    Ak = mp.Ak
    delta0 = mp.delta0
else:
    pc.assign( fe.interpolate(fe.Constant(mp.pc0), SH))

if damage_flag:
    lsp = mp.lsp


#%% --------------------------------------------------------------
#%% Boundary conditions

# symmetry boundary condition
bcU_1 = fe.DirichletBC(SU.sub(1), fe.Constant(0.0), boundaries, 3)

# wedge boundary conditions
bcU_2 = fe.DirichletBC(SU, fe.Constant((0.0, 0.0)), boundaries, 4)
bcU_3 = fe.DirichletBC(SU, fe.Constant((0.0, 0.0)), boundaries, 5)

bcU = [bcU_1, bcU_2, bcU_3]

    
#%% --------------------------------------------------------------
#%% Define variational form of Thermal parabolic problem

# initialize thermal problem time step
fe_dt = fe.Constant(dt_max)

# internal thermo-elastic effect is neglected
ET = ( rho*cspec*(T_trl-T_old)/fe_dt*T_tst + \
     + Kcond*fe.dot(fe.grad(T_trl), fe.grad(T_tst)) )*fe.dx + \
     + ( steel_hconv*(T_trl - steel_Temp)*T_tst )*ds_inner + \
     + ( outer_hconv*(T_trl - outer_Temp)*T_tst )*ds_outer


#%% --------------------------------------------------------------
#%% Define variational form of Displacement

if elasticity_flag:
    EU = ( fe.inner(
            cm.sig(u_trl, T_new, d_new, eps_p_new, 
                lmbda, mu, alpha, ref_Temp, ntol),
            cm.eps(u_tst) 
        ) )*fe.dx


#%% --------------------------------------------------------------
#%% Define variational form of Phase Field

# define standard damage problem
if damage_flag:
    # Gc is embedded in the crack driving force H_new
    ED = ( -2.0*cm.H_new(u_new, T_new, eps_p_new, 
            H_old, lmbda, mu, Gc, lsp, alpha, ref_Temp)*\
         (d_tst - fe.inner(d_trl, d_tst)) +\
         fe.inner(d_trl, d_tst)/lsp +\
         lsp*fe.inner( fe.grad(d_trl), fe.grad(d_tst)) )*fe.dx


#%% --------------------------------------------------------------
#%% Mesh refinement functions

def refine_mesh(mesh_coarse, damage): 
    
    # create the refinement mask
    refinement_mask = fe.MeshFunction( 
        "bool",
        mesh_coarse, 
        mesh_coarse.topology().dim() 
    ) 
    
    H_condition.assign( fe.project( 
        cm.H_new(u_new, T_new, eps_p_new, 
            H_old, lmbda, mu, Gc, lsp, alpha, ref_Temp)*lsp
    , SH ))
    
    splitted_cells = 0
    # assign values to the empty refinement mask
    for cell in fe.cells(mesh_coarse): 
        mid = cell.midpoint() 
        refinement_mask[cell] = False
        if ( cell.h() > size_refinement_threshold and 
            ( d_new(mid) > damage_refinement_threshold or 
              H_condition(mid)>H_refinement_threshold ) 
            ):
            splitted_cells += 1
            refinement_mask[cell] = True 
    
    if splitted_cells > num_cell_refinement_threshold:
        # refine using FEniCS built in functionality
        mesh_fine = fe.refine( mesh_coarse , refinement_mask )
        return splitted_cells, mesh_fine
    else:
        return 0, mesh_coarse
   
    
def redefine_problem( mesh_fine, 
    T_coarse, T_old_coarse, u_coarse, eps_p_coarse, 
    Baralp_coarse, pc_coarse, 
    Nval_coarse, Nval_old_coarse, 
    Dgamma_coarse, Dgamma_old_coarse, 
    Baralp_old_coarse, eps_p_old_coarse, 
    d_coarse, Gc_coarse, H_old_coarse, 
    sign_coarse, epsn_coarse, 
    invpn_coarse, invqn_coarse, inv_cos_3theta_n_coarse,  
    alpha_coarse, cspec_coarse, Kcond_coarse, 
    Emod_coarse, lmbda_coarse, mu_coarse, 
    pcT_coarse):

    #%% -------------------------------------------------60
    #%% Define Function Spaces for solution variables

    # Redefine Function space for displacement problem
    SU_fine = fe.VectorFunctionSpace(mesh_fine, "CG", 1)
    u_trl_fine = fe.TrialFunction(SU_fine)
    u_tst_fine = fe.TestFunction(SU_fine)

    # Redefine Function space for temperature problem
    ST_fine = fe.FunctionSpace(mesh_fine, "CG", 1)
    T_trl_fine = fe.TrialFunction(ST_fine)
    T_tst_fine = fe.TestFunction(ST_fine)

    # Redefine Function spaces for damage phase field problem
    SD_fine = fe.FunctionSpace(mesh_fine, 'CG', 1)
    d_trl_fine = fe.TrialFunction(SD_fine)
    d_tst_fine = fe.TestFunction(SD_fine)

    # Function space for scalar functions projection
    SF_fine = fe.FunctionSpace(mesh_fine, "CG", 1)

    # Redefine Function spaces for tensor functions projection
    SX_fine = fe.TensorFunctionSpace(mesh_fine, "CG", 1)

    # Redefine Function space for history variable
    SH_fine = fe.FunctionSpace(mesh_fine, "DG", 0)

    #%% -------------------------------------------------60
    #%% Auxiliary functions

    T_new = fe.Function(ST_fine, name="Temperature")
    T_new.assign( fe.project(T_coarse, ST_fine))
    
    T_old = fe.Function(ST_fine, name="Temperature")
    T_old.assign( fe.project(T_old_coarse, ST_fine))

    u_new = fe.Function(SU_fine, name="Displacement")
    u_new.assign( fe.project(u_coarse, SU_fine))
    
    eps_p_new = fe.Function(SX_fine, name="Plastic_deformation")
    eps_p_new.assign( fe.project(eps_p_coarse, SX_fine))
        
    Baralp_new = fe.Function(SF_fine, name="Cumulated_plastic_def")
    Baralp_new.assign( fe.project(Baralp_coarse, SF_fine))
    
    pc = fe.Function(SH_fine, name="pc")
    pc.assign( fe.project(pc_coarse, SH_fine))

    if plasticity_flag:
        Nval_new = fe.Function(SX_fine, name="Plastic_flow_dir")
        Nval_new.assign( fe.project(Nval_coarse, SX_fine))
        
        Nval_old = fe.Function(SX_fine, name="Plastic_flow_dir_old")
        Nval_old.assign( fe.project(Nval_old_coarse, SX_fine))
        
        Dgamma_new = fe.Function(SF_fine, name="Plastic_mult_incr")
        Dgamma_new.assign( fe.project(Dgamma_coarse, SF_fine))
        
        Dgamma_old = fe.Function(SF_fine, name="Plastic_mult_incr_old")
        Dgamma_old.assign( fe.project(Dgamma_old_coarse, SF_fine))
        
        Baralp_old = fe.Function(SF_fine, name="Cumulated_plastic_def")
        Baralp_old.assign( fe.project(Baralp_old_coarse, SF_fine))
        
        eps_p_old = fe.Function(SX_fine, name="Plastic_def_old")
        eps_p_old.assign( fe.project(eps_p_old_coarse, SX_fine))
    else:
        Nval_new = None
        Nval_old = None
        Dgamma_new = None
        Dgamma_old = None
        Baralp_old = None
        eps_p_old = None
    
    d_new = fe.Function(SD_fine, name="Damage")
    d_new.assign( fe.project(d_coarse, SD_fine))
    
    Gc = fe.Function(SH_fine, name="Specific_fracture_energy")
    Gc.assign( fe.project(Gc_coarse, SH_fine))
    
    H_old = fe.Function(SH_fine, name="Crack_driving_force_old")
    H_old.assign( fe.project(H_old_coarse, SH_fine))

    if postprocess_cartesian_flag:
        sign = fe.Function(SX_fine, name="Stress_cartesian")
        sign.assign( fe.project( sign_coarse, SX_fine ))
        
        epsn = fe.Function(SX_fine, name="Strain_cartesian")
        epsn.assign( fe.project( epsn_coarse, SX_fine ))

        invpn = fe.Function(SF_fine, name="invp")
        invpn.assign( fe.project( invpn_coarse, SF_fine ))
        
        invqn = fe.Function(SF_fine, name="invq")
        invqn.assign( fe.project( invqn_coarse, SF_fine ))
        
        inv_cos_3theta_n = fe.Function(SF_fine, name="inv_cos_3theta")
        inv_cos_3theta_n.assign(fe.project(inv_cos_3theta_n_coarse,SF_fine))
    else:
        sign = None
        epsn = None
        invpn = None
        invqn = None
        inv_cos_3theta_n = None
    
    alpha = fe.Function(SH_fine, name="alpha")
    alpha.assign( fe.project(alpha_coarse, SH_fine))
    
    cspec = fe.Function(SH_fine, name="cspec")
    cspec.assign( fe.project(cspec_coarse, SH_fine))
    
    Kcond = fe.Function(SH_fine, name="Kcond")
    Kcond.assign( fe.project(Kcond_coarse, SH_fine))

    Emod = fe.Function(SH_fine, name="Emod")
    Emod.assign( fe.project(Emod_coarse, SH_fine)) 
    
    lmbda = fe.Function(SH_fine, name="lmbda")
    lmbda.assign( fe.project(lmbda_coarse, SH_fine))  
    
    mu = fe.Function(SH_fine, name="mu")
    mu.assign( fe.project(mu_coarse, SH_fine))  
    
    if plasticity_flag:
        pcT = fe.Function(SH_fine, name="pcT")
        pcT.assign( fe.project(pcT_coarse, SH_fine))
    else:
        pcT = None

    #%% -------------------------------------------------60
    #%% Boundary conditions    

    # create mesh boundaries
    bnds_fine = fe.MeshFunction(
        "size_t", 
        mesh_fine, 
        mesh_fine.topology().dim()-1
    )
    bnds_fine.set_all(0)

    bc_input.mark(bnds_fine, 1)
    bc_outer.mark(bnds_fine, 2)
    bc_sym.mark(bnds_fine, 3)
    bc_wedge.mark(bnds_fine, 4)
    bc_wedge2.mark(bnds_fine, 5)

    # symmetry boundary condition
    bcU_1=fe.DirichletBC(SU_fine.sub(1),fe.Constant(0.0),bnds_fine,3)

    # wedge boundary conditions
    bcU_2=fe.DirichletBC(SU_fine,fe.Constant((0.0,0.0)),bnds_fine,4)
    bcU_3=fe.DirichletBC(SU_fine,fe.Constant((0.0,0.0)),bnds_fine,5)

    bcU_fine = [bcU_1, bcU_2, bcU_3]
    
    ds_inner_fine=fe.Measure("ds",subdomain_id=1,subdomain_data=bnds_fine)
    ds_outer_fine=fe.Measure("ds",subdomain_id=2,subdomain_data=bnds_fine)

    #%% -------------------------------------------------60
    #%% Variational problems redefinition    
    
    ET_fine = ( rho*cspec*(T_trl_fine-T_old)/fe_dt*T_tst_fine + \
        +Kcond*fe.dot(fe.grad(T_trl_fine),fe.grad(T_tst_fine)))*fe.dx+\
        +(steel_hconv*(T_trl_fine-steel_Temp)*T_tst_fine)*ds_inner_fine+\
        +(outer_hconv*(T_trl_fine-outer_Temp)*T_tst_fine)*ds_outer_fine

    EU_fine = ( fe.inner( 
        cm.sig(u_trl_fine, T_new, d_new, eps_p_new, 
               lmbda, mu, alpha, ref_Temp, ntol),
        cm.eps(u_tst_fine) ) 
    )*fe.dx

    ED_fine = ( -2.0*cm.H_new(u_new, T_new, eps_p_new, H_old, 
                        lmbda, mu, Gc, lsp, alpha, ref_Temp)*\
         (d_tst_fine - fe.inner(d_trl_fine, d_tst_fine)) +\
         fe.inner(d_trl_fine, d_tst_fine)/lsp +\
         lsp*fe.inner(fe.grad(d_trl_fine),fe.grad(d_tst_fine)))*fe.dx

    return SF_fine, SX_fine, SH_fine, T_new, T_old, u_new, eps_p_new, 
        Baralp_new, pc, Nval_new, Nval_old, Dgamma_new, Dgamma_old, 
        Baralp_old, eps_p_old, d_new, Gc, H_old, sign, epsn, invpn, 
        invqn, inv_cos_3theta_n, alpha, cspec, Kcond, Emod, lmbda, 
        mu, pcT, bcU_fine, ET_fine, EU_fine, ED_fine
    

#%% --------------------------------------------------------------
#%% Functions for post-processing

def dump_solution(sim_time, T_dmp, u_dmp, d_dmp, pc_dmp, 
    Baralp_dmp, eps_p_dmp, Emod_dmp, alpha_dmp):
    
    T_outfile << T_dmp, sim_time
      
    if elasticity_flag or plasticity_flag:
        u_outfile << u_dmp, sim_time

    if plasticity_flag:
        pc_outfile << pc_dmp, sim_time
        Baralp_outfile << Baralp_dmp, sim_time

    if damage_flag:
        d_outfile << d_dmp, sim_time

    if check_params_flag:
        Emod_outfile << Emod_dmp, sim_time
        alpha_outfile << alpha_dmp, sim_time                                     

    if postprocess_cartesian_flag:
        # update post-processing variables
        sign.assign( fe.project( cm.sig(u_dmp, T_dmp, d_dmp, 
            eps_p_dmp, lmbda, mu, alpha_dmp, ref_Temp, ntol), SX ))
        epsn.assign( fe.project( cm.eps(u_dmp), SX ))   
        invpn.assign( fe.project( cm.finvp(sign), SF )) 
        invqn.assign( fe.project( cm.finvq(sign, ntol), SF )) 
        inv_cos_3theta_n.assign(fe.project(cm.finv_cos_3theta(sign,ntol),SF)) 

        sig_outfile << sign, sim_time
        eps_outfile << epsn, sim_time
        pinv_outfile << invpn, sim_time
        qinv_outfile << invqn, sim_time 
        c3t_outfile << inv_cos_3theta_n, sim_time 


def fileprint(string):
    print(string)
    with open(outfolder + "README.md", "a") as file:
        file.writelines( "\n" + string)

def checkNanEnergy(energy):
    if np.isnan(energy):
        fileprint(" Energy nan detected, interrupting simulation. ")
        return 1
    else:
        return 0


#%% --------------------------------------------------------------
#%% Variational problem resolution

# save time-steps in vector (useful for adaptive time-stepping)
time_steps = [0]

# historical quantities
thermal_energy_history = [0]
elastic_energy_history = [0]
plastic_energy_history = [0]
damage_energy_history = [0]
itr_history = [0]  

# initialize counter for solution dumping
fixed_dump_interval_counter = 0
adaptive_sol_dump_counter = 0

# output initial condition as solution
dump_solution(sim_time,T_old,u_new,d_new,pc,Baralp_new,eps_p_new,Emod,alpha)

# README file headers
info_string = f"{'[min]':^5}" + '\t' +\
                f"{'[%]':^5}"  + '\t'+\
                f"{'[sim t]':^10}"+ '\t'

if adaptive_time_stepping_flag:
    info_string +=  f"{'[dt]':^10}"+ '\t'+\
                    f"{'[Cuts]':^5}"+ '\t'+\
                    f"{'[Raise]':^5}"+ '\t'

if open_loop_flag == 0:
    info_string +=  f"{'[iters]':^5}"+ '\t'
    
info_string += f"{'[thml J]':^10}" + '\t'
if elasticity_flag:
    info_string += f"{'[elas J]':^10}" + '\t'
if plasticity_flag:
    info_string += f"{'[plst J]':^10}" + '\t' 
if damage_flag:
    info_string += f"{'[damg J]':^10}" + '\t'

if adaptive_mesh_refinement_flag:
    info_string += f"{'[N refns]':^10}" + '\t'
    info_string += f"{'[N split]':^10}" + '\t'

if adaptive_time_stepping_flag:
    info_string += f"{'[Ratio]':^10}" + '\t'

info_string += f"{'[Solution]':^10}"

fileprint( info_string )

# initializate NaN flag
NaN_flag = 0

# initialize simulation time step and dumping solution interval
dt = dt_initial
counter_dt_cuts = 0
counter_dt_raise = 0

if adaptive_mesh_refinement_flag:
    # initialize the mesh for the damage problem
    refined_mesh = mesh

# measure elapsed time for simulation
tic = time.time()

# simulation loop
while sim_time+ntol<tmax and NaN_flag == 0:
    
    # update time step in the thermal problem definition
    fe_dt.assign(dt)
    
    # solving the thermal problem and update the T_old placeholder
    fe.solve( fe.lhs(ET) == fe.rhs(ET), T_new )
    
    # update material properties with temperature 
    update_properties(T_new, SH)

    # setup elastic energy measure for loop iterations
    elastic_energy_old = 0
    elastic_energy_new = 1
    itr_index = -1
    refinements_counter = 0
    splitted_cells_history = 0

    if elasticity_flag or plasticity_flag:
        while ( itr_index < max_number_iterations and 
            abs(elastic_energy_old-elastic_energy_new)>\
                0.01*elastic_energy_new):
            itr_index += 1
            elastic_energy_old = elastic_energy_new

            if plasticity_flag:
                # update the current stress
                sig_new = cm.sig(u_new, T_new, d_new, eps_p_new, 
                                lmbda, mu, alpha, ref_Temp, ntol)
                Fval = cm.YieldFunction(sig_new, pc, pc0, Omega, 
                            Mbp, alpbp, mbp, betbp, gambp, ntol)
                
                # update the plastic flow tensor
                Nval_new.assign( fe.project( 
                    cm.dFdsig(sig_new, pc, pc0, Omega, Mbp, alpbp, 
                                mbp, betbp, gambp, ntol) ,
                SX ))

                # compute plastic multiplier factor considering time step 
                eta=eta0*(1-(1-math.exp(-dt/tau))/(dt/tau))

                # update the plastic multiplier
                Dgamma_new.assign( fe.project( 
                    cm.positive_signum(cm.finvp(sig_new),ntol)*\
                    eta*\
                    cm.Macaulay(Fval),
                SF ))
                
                # update plastic strain with forward Euler formula
                eps_p_new.assign( fe.project( 
                    eps_p_old+0.5*(
                        Dgamma_old*Nval_old + Dgamma_new*Nval_new
                    ),
                SX ))
                
                # update the scalar internal variable 
                Baralp_new.assign( fe.project( 
                    cm.Macaulay( 
                        Baralp_old + 
                        0.5*(
                            Dgamma_old*cm.tensorNorm(Nval_old) + 
                            Dgamma_new*cm.tensorNorm(Nval_new)
                        )
                    ),
                SF ))
                
                # update the hardening variable
                pc.assign( fe.project( 
                    pcT + cm.pcM(T_new, Baralp_new, Ak, delta0),
                SF ))

                # update iteration quantities
                Nval_old.assign(Nval_new)
                Dgamma_old.assign(Dgamma_new)

            # solving mechanical problem
            fe.solve( fe.lhs(EU) == fe.rhs(EU), u_new, bcU ) 
            
            # solve damage problem and update crack driving force
            if damage_flag:             

                # solving damage problem                
                fe.solve( fe.lhs(ED) == fe.rhs(ED), d_new ) 
                
                if adaptive_mesh_refinement_flag:
                    # check if mesh has to be refined and do it
                    splitted_cells,refined_mesh=refine_mesh(refined_mesh,d_new)
                    
                    if splitted_cells > 0:
                        # update counter of refinements
                        refinements_counter += 1
                        
                        # update counter of splitted cells in increment
                        splitted_cells_history += splitted_cells
                        
                        SF, SX, SH, T_new, T_old, u_new, eps_p_new,
                        Baralp_new, pc, Nval_new, Nval_old, 
                        Dgamma_new, Dgamma_old, Baralp_old, eps_p_old, 
                        d_new, Gc, H_old, sign, epsn, 
                        invpn, invqn, inv_cos_3theta_n, 
                        alpha, cspec, Kcond, 
                        Emod, lmbda, mu, 
                        pcT, bcU, ET, EU, ED = redefine_problem( 
                            refined_mesh, 
                            T_new, T_old, u_new, eps_p_new, 
                            Baralp_new, pc, 
                            Nval_new, Nval_old, 
                            Dgamma_new, Dgamma_old, 
                            Baralp_old, eps_p_old, 
                            d_new, Gc, H_old, 
                            sign, epsn, 
                            invpn, invqn, inv_cos_3theta_n, 
                            alpha, cspec, Kcond, 
                            Emod, lmbda, mu, 
                            pcT
                        )
                    
                # update damage associated energy measure by using 
                # the formula from Borden Hughes Landis Anvari Lee 2016
                damage_energy_new = fe.assemble( 
                    Gc/(4*lsp)*(
                        pow(d_new,2) + 
                        4*lsp**2*fe.inner(fe.grad(d_new),fe.grad(d_new))
                    )*fe.dx 
                )
            
            # calculate elastic energy to check convergence
            elastic_energy_new = fe.assemble( 
                fe.inner( 
                    cm.sig(u_new, T_new, d_new, eps_p_new, 
                            lmbda, mu, alpha, ref_Temp, ntol),
                    cm.eps_e(u_new, T_new, eps_p_new, alpha, ref_Temp) 
                )*fe.dx 
            )
            
            NaN_flag = checkNanEnergy(elastic_energy_new)
            
            # no need to check energy convergence in linear elastic analysis
            if ( elasticity_flag and 
                 plasticity_flag == 0 and 
                 damage_flag == 0 or 
                 open_loop_flag ):
                break
    
    # reset damage energy ratio value
    damage_energy_ratio = -1
    if len(time_steps)>1:
        # compute damage energy ratio using history values 
        # (available only after 1st time step)
        if damage_energy_history[-1] != 0:
            damage_energy_ratio=damage_energy_new/damage_energy_history[-1]
    
    # the time step check can happen only at the second calculation
    # if time step hits the minimum the solution is exported forcefully
    if( adaptive_time_stepping_flag and 
        len(time_steps)>1 and 
        ( damage_energy_ratio > damage_energy_ratio_threshold or 
        itr_index == max_number_iterations ) and 
        dt != dt_min 
        ):
        
        info_string = f"{int((time.time() - tic)/60):^5}" + '\t' +\
                        f"{sim_time/tmax*100:^5.3}" + '\t' +\
                        f"{sim_time + dt:^10.5g}"+ '\t' +\
                        f"{dt:^10.3g}"+ '\t' +\
                        f"{counter_dt_cuts:^5}"+ '\t' +\
                        f"{counter_dt_raise:^5}"+ '\t'
        
        if open_loop_flag == 0:
            info_string += f"{itr_index:^5}"+ '\t'
            
        info_string += f"{'-':^10}" + '\t'                        
        if elasticity_flag:
            info_string += f"{'-':^10}" + '\t'
        if plasticity_flag:
            info_string += f"{'-':^10}" + '\t'
        if damage_flag:
            info_string += f"{damage_energy_new:^10.3g}" + '\t'
        
        if adaptive_mesh_refinement_flag:
            info_string += f"{refinements_counter:^10.3g}" + '\t'
            info_string += f"{splitted_cells_history:^10.3g}" + '\t'
        
        info_string += f"{damage_energy_ratio:^10.3g}" + '\t'
        
        info_string += f"{'-':^10}"
        
        fileprint( info_string )
        
        # time step has to be decremented and the analysis re-run
        counter_dt_raise = 0
        counter_dt_cuts += 1
        if dt/(cut_factor*counter_dt_cuts) > dt_min:
            dt = dt/(cut_factor*counter_dt_cuts)
        else
            dt = dt_min
            
    else:
        # Solution can be exported and time step raised
        # Increment simulation time with time step used for solution
        sim_time += dt
        
        # save time steps as history output
        time_steps.append(sim_time)
        
        # save temperature value
        T_old.assign(T_new)
        
        # update thermal energy measure
        thermal_energy_new = fe.assemble(
            ( rho*cspec*(T_new - ref_Temp) )*fe.dx 
        )
        
        # export thermal energy measure as history output
        thermal_energy_history.append( thermal_energy_new )
        
        # update historic quantities after step iteration
        if plasticity_flag:
            # update plastic energy by increment before updating eps_p_old
            plastic_energy_new = plastic_energy_history[-1] + 
                fe.assemble( 
                    fe.inner( 
                        cm.sig(u_new, T_new, d_new, eps_p_new, 
                            lmbda, mu, alpha, ref_Temp, ntol), 
                        eps_p_new - eps_p_old 
                    )*fe.dx 
                )
            
            NaN_flag = checkNanEnergy(plastic_energy_new)
            
            # update plastic field variables
            eps_p_old.assign(eps_p_new)
            Baralp_old.assign(Baralp_new)

        if damage_flag:
            # update damage field variables
            H_old.assign( fe.project( 
                cm.H_new(u_new, T_new, eps_p_new, H_old, 
                        lmbda, mu, Gc, lsp, alpha, ref_Temp), 
            SH ))

        # update history variables
        if elasticity_flag:
            elastic_energy_history.append(elastic_energy_new)
        if plasticity_flag:
            plastic_energy_history.append( plastic_energy_new )
        if damage_flag:
            damage_energy_history.append( damage_energy_new )
        if open_loop_flag == 0: 
            itr_history.append(itr_index)
        
        info_string = f"{int((time.time() - tic)/60):^5}" + '\t' +\
                        f"{sim_time/tmax*100:^5.3}" + '\t' +\
                        f"{sim_time:^10.5g}"+ '\t'
        
        if adaptive_time_stepping_flag:
            info_string += f"{dt:^10.3g}"+ '\t' +\
                           f"{counter_dt_cuts:^5}"+ '\t' +\
                           f"{counter_dt_raise:^5}"+ '\t'

        if open_loop_flag == 0:
            info_string += f"{itr_index:^5}"+ '\t'
                        
        info_string += f"{thermal_energy_new:^10.3g}" + '\t'
        if elasticity_flag:
            info_string += f"{elastic_energy_new:^10.3g}" + '\t'
        if plasticity_flag:
            info_string += f"{plastic_energy_new:^10.3g}" + '\t'
        if damage_flag:
            info_string += f"{damage_energy_new:^10.3g}" + '\t'
            
        if adaptive_mesh_refinement_flag:
            info_string += f"{refinements_counter:^10.3g}" + '\t'
            info_string += f"{splitted_cells_history:^10.3g}" + '\t'

        if adaptive_time_stepping_flag:
            info_string += f"{damage_energy_ratio:^10.3g}" + '\t'


        # Compute and output values in files for post-processing in Paraview
        if( counter_dt_cuts > 0 or 
            sim_time+ntol>\
                fixed_dump_interval*(fixed_dump_interval_counter+1)
            ):
            dump_solution(sim_time,T_new,u_new,d_new,
                          pc,Baralp_new,eps_p_new,Emod,alpha)
            adaptive_sol_dump_counter += 1
            info_string += f"{adaptive_sol_dump_counter:^10}"
            fileprint( info_string )
        else:
            info_string += f"{'No dump':^10}"
            fileprint( info_string )
        
        if( sim_time+ntol>\
            fixed_dump_interval*(fixed_dump_interval_counter+1)
            ):
            # update the solution dumping counter 
            # independently from the time step
            fixed_dump_interval_counter += 1
        
        if adaptive_time_stepping_flag:
            # reset the time step cut counter
            counter_dt_cuts = 0
            counter_dt_raise += 1
            # raise time increment (with a max cap)
            if dt*(raise_factor*counter_dt_raise) <= dt_max:
                dt = dt*(raise_factor*counter_dt_raise)
            else:
                dt = dt_max

fileprint(f"Elapsed CPU time: {int((time.time() - tic)/60):5d} [min]")

with open(outfolder + "README.md", "a") as file:
    if NaN_flag == 0:
        file.writelines("\n Simulation completed successfully !!!")
    elif NaN_flag == 1:
        file.writelines("\n Simulation interrupted after NaN detection.")

if adaptive_mesh_refinement_flag:
    fileprint("Exporting refined mesh in file refined_mesh.xdmf")
    fe.XDMFFile( outfolder + "refined_mesh.xdmf" ).write(refined_mesh)

#%% --------------------------------------------------------------
#%% Post-process plots

thermal_energy_history = np.array(thermal_energy_history)
elastic_energy_history = np.array(elastic_energy_history)
plastic_energy_history = np.array(plastic_energy_history)
damage_energy_history = np.array(damage_energy_history)
itr_history = np.array(itr_history)

fig, ax = plt.subplots(convergence_plots_number, 1, 
                        sharex=True, 
                        figsize=(8, convergence_plots_number*3))
plot_index = -1

if open_loop_flag == 0:
    plot_index += 1
    ax[plot_index].plot(time_steps, itr_history,
                        '-s', 
                        color='magenta',
                        linewidth=1,
                        markersize=1,
                        label = "Iterations"
                        )
    ax[plot_index].set( ylabel="Iterations [-]" )

plot_index += 1
ax[plot_index].plot(time_steps, thermal_energy_history,
    '-o', 
    color='red',
    linewidth=1,
    markersize=1,
    label = "Thermal energy"
    )
ax[plot_index].set( ylabel="Energy [J]" )

if elasticity_flag:
    plot_index += 1
    ax[plot_index].plot(time_steps, elastic_energy_history,
        '-o', 
        color='green',
        linewidth=1,
        markersize=1,
        label = "Elastic energy"
        )
    ax[plot_index].set( ylabel="Energy [J]" )
if plasticity_flag:
    plot_index += 1
    ax[plot_index].plot(time_steps, plastic_energy_history,
        '-o', 
        color='blue',
        linewidth=1,
        markersize=1,
        label = "Dissipated Plastic energy"
        )
    ax[plot_index].set( ylabel="Energy [J]" )
if damage_flag:
    plot_index += 1
    ax[plot_index].plot(time_steps, damage_energy_history,
        '-o', 
        color='black',
        linewidth=1,
        markersize=1,
        label = "Dissipated Damage energy"
        )
    ax[plot_index].set( ylabel="Energy [J]" )

ax[plot_index].set( xlabel="Time steps")

for ax in fig.get_axes():
    ax.label_outer()
    ax.grid()

# hack for the legend
plot_index += 1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend( 
    lines[:plot_index], 
    labels[:plot_index], 
    loc='upper center', 
    ncol=plot_index, 
    bbox_to_anchor=[0.5, 0.95], 
    borderaxespad=0
)

plt.savefig(outfolder + "convergence_plot.png")
plt.close()