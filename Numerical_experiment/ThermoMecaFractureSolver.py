'''
- This file contains all the functions and classes needed to define
  and solve a thermo-mechanical problem with phase field fracture.
- The user of this module is supposed to only provide the strictly
  necessary information needed to run the case and let the solver
  of the module operate the adaptive time stepping and mesh
  refinement needed to solve the problem.
- All material properties are defined over the domain as functions
  of the position, this is needed to represent multiple materials
  on the same domain and temperature-dependent materials
- The solver works for 2D and 3D problems
- The solver is designed to be used by a topology optimization routine:
  - it allows the coexistence of different materials (with the same
    variational formulation for the material law) on the same domain
  - it allows an easy implementation of a design variable dependance
'''

import fenics as fe
import numpy as np
from datetime import datetime
import time
import os

ntol = 1e-6 # numerical tolerance
Kshift = 273.15
one = fe.Constant(1.0)
two = fe.Constant(2.0)

fe.parameters['form_compiler']['representation'] = 'uflacs'
fe.parameters['form_compiler']['optimize'] = True
fe.parameters['form_compiler']['quadrature_degree'] = 2
fe.parameters["refinement_algorithm"] = "plaza_with_parent_facets"

# set log params
# CRITICAL  = 50, // errors that may lead to data corruption and suchlike
# ERROR     = 40, // things that go boom
# WARNING   = 30, // things that may go boom later
# INFO      = 20, // information of general interest
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
# fe.set_log_active(False)
# fe.set_log_level(13)

# Utility function: Macaulay brackets
def Macaulay(var):
    return (var + abs(var))/2


# Utility function: tensor norm
def tensorNorm(tensor):
    return fe.sqrt(fe.inner(tensor, tensor))


# Utility function: positive signum function of scalar variable
# returns 0 if negative, 1 if positive
def positive_signum(var, ntol):
    return Macaulay(var)/abs(var + ntol)


# positive elastic strain (calculation for 2D meshes)
def eigenSplit_pos_2D(tensor):
    v00 = (tensor[0,0] + tensor[1,1])/2 + fe.sqrt(((tensor[0,0]-tensor[1,1])/2)**2+(tensor[1,0])**2)
    v01 = 0
    v10 = 0
    v11 = (tensor[0,0] + tensor[1,1])/2 - fe.sqrt(((tensor[0,0]-tensor[1,1])/2)**2+(tensor[1,0])**2)

    w00 = 1/fe.sqrt(1+((v00-tensor[0,0])/tensor[1,0])**2)
    w01 = 1/fe.sqrt(1+((v11-tensor[0,0])/tensor[1,0])**2)
    w10 = w00*(v00-tensor[0,0])/tensor[1,0]
    w11 = w01*(v11-tensor[0,0])/tensor[1,0]

    w = ([w00,w01],[w10,w11])
    w = fe.as_tensor(w)
    w_tr = ([w00,w10],[w01,w11])
    w_tr = fe.as_tensor(w_tr)

    v00 = fe.conditional(fe.gt(v00,0.0),v00,0.0)
    v11 = fe.conditional(fe.gt(v11,0.0),v11,0.0)

    v = ([v00,v01],[v10,v11])
    v = fe.as_tensor(v)

    return w*v*w_tr


# positive elastic strain (calculation for 3D meshes)
pi=3.14159265359
def eigenSplit_pos_3D(t):

    p1 = t[0,1]**2+t[0,2]**2+t[1,2]**2
    qq =(t[0,0]+t[1,1]+t[2,2])/3
    p2 =(t[0,0] - qq)**2 + (t[1,1] - qq)**2 + (t[2,2] - qq)**2 + 2*p1
    pp = fe.sqrt(p2/6)

    t = fe.as_tensor(t)
    B =(1/pp)*(t - qq*fe.Identity(3))
    B = fe.as_tensor(B)
    r = fe.det(B)/2

    fai = fe.acos(r)/3
    # the eigenvalues satisfy eig3 <= eig2 <= eig1
    v00 = qq + 2 * pp * fe.cos(fai)
    v22 = qq + 2 * pp * fe.cos(fai + (2*pi/3))
    v11 = 3 * qq - v00 - v22

    a1=t[:,0]-v00*fe.Constant((1., 0., 0.))
    a2=t[:,1]-v00*fe.Constant((0., 1., 0.))
    a3=t[:,0]-v11*fe.Constant((1., 0., 0.))
    a4=t[:,1]-v11*fe.Constant((0., 1., 0.))

    w00s = a1[1]*a2[2] - a1[2]*a2[1]
    w10s = a1[2]*a2[0] - a1[0]*a2[2]
    w20s = a1[0]*a2[1] - a1[1]*a2[0]

    w01s = a3[1]*a4[2] - a3[2]*a4[1]
    w11s = a3[2]*a4[0] - a3[0]*a4[2]
    w21s = a3[0]*a4[1] - a3[1]*a4[0]

    w00 = w00s/fe.sqrt(w00s**2+w10s**2+w20s**2)
    w10 = w10s/fe.sqrt(w00s**2+w10s**2+w20s**2)
    w20 = w20s/fe.sqrt(w00s**2+w10s**2+w20s**2)

    w01 = w01s/fe.sqrt(w01s**2+w11s**2+w21s**2)
    w11 = w11s/fe.sqrt(w01s**2+w11s**2+w21s**2)
    w21 = w21s/fe.sqrt(w01s**2+w11s**2+w21s**2)

    w02s = w10*w21 - w20*w11
    w12s = w20*w01 - w00*w21
    w22s = w00*w11 - w10*w01

    w02 = w02s/fe.sqrt(w02s**2+w12s**2+w22s**2)
    w12 = w12s/fe.sqrt(w02s**2+w12s**2+w22s**2)
    w22 = w22s/fe.sqrt(w02s**2+w12s**2+w22s**2)

    wp = ([w00, w01, w02],[w10, w11, w12],[w20, w21, w22])
    wp = fe.as_tensor(wp)

    wp_tr = ([w00,w10, w20],[w01,w11, w21], [w02, w12, w22])
    wp_tr = fe.as_tensor(wp_tr)

    v00 = fe.conditional(fe.gt(v00,0.0),v00,0.0)
    v11 = fe.conditional(fe.gt(v11,0.0),v11,0.0)
    v22 = fe.conditional(fe.gt(v22,0.0),v22,0.0)

    vp = ([v00,0.0,0.0],[0.0,v11,0.0],[0.0,0.0,v22])
    vp = fe.as_tensor(vp)
    return wp*vp*wp_tr


class Material:
    '''
    - object containing all material property data
    - properties are passed by the user in keyword=value format
    - if the user does not provide the correct keyword or these
      are missing, the Domain class will ask for the missing properties
    '''
    def __init__(self,
        exp_data_folder = None,
        exp_data_list = None,
        **kwargs
        ):

        self.exp_data_list = exp_data_list

        for key, value in kwargs.items():
            exec( f"self.{key} = fe.Constant({value})" )

        # thermal dependent properties data
        if exp_data_folder != None:
            self.exp_data = {}
            for property in self.exp_data_list:
                try:
                    self.exp_data[property] = {}
                    rawdata = np.loadtxt(exp_data_folder + "/" + property + ".csv", delimiter=',')
                    self.exp_data[property]["temperature"] = rawdata[:,0] + Kshift
                    self.exp_data[property]["value"] = rawdata[:,1]
                except:
                    raise RuntimeError( f"Missing {property} data." )


class material_property_field(fe.UserExpression):
    '''
    - object containing material property field variable
      expressed in FEniCS compatible format
    '''
    def __init__(
        self,
        materials_meshfunction,
        materials_ids_table,
        property,
        temperature=None,
        **kwargs
        ):

        super().__init__(kwargs)
        self.materials_meshfunction = materials_meshfunction
        self.materials_ids_table = materials_ids_table
        self.property = property
        self.temperature = temperature

    def eval_cell(self, value, x, cell):
        if self.temperature == None:
            # retrieve single property from imaterials_ids_tableorted module using getattr syntax
            value[0] = getattr(
                self.materials_ids_table[self.materials_meshfunction[cell.index]],
                self.property
            )
        else:
            # create an handle to query the material data file
            handle = self.materials_ids_table[self.materials_meshfunction[cell.index]]

            # interpolate the material data to find the property at the requested temperature
            value[0] = np.interp(
                self.temperature(x),
                handle.exp_data[self.property]["temperature"],
                handle.exp_data[self.property]["value"]
            )

    # specify that the output value is scalar
    def value_shape(self):
        return ()


class Domain:
    '''
    - object representing a finite element domain with field
      variables, the independent and the dependent ones
    - the constitutive model functions have as variables the
      problem independent variable to allow definition of trail
      and test functions
    - this object is not supposed to be used directly by the user,
      but only by the object Problem
    '''
    def __init__(self,
        mesh,
        materials_meshfunction,
        ref_temperature,
        boundaries_meshfunction,
        materials_ids_table,
        boundary_conditions_table,
        planar_Hp,
        thermal_flag = 0,
        mechanical_flag = 0,
        phase_field_damage_flag = 0,
        plasticity_flag = 0
        ):

        # Domain mesh
        self.mesh = mesh

        # Domain materials geometric locations
        self.materials_meshfunction = materials_meshfunction

        # Domain boundaries geometric locations
        self.boundaries_meshfunction = boundaries_meshfunction

        # identification dictionary (numerical id --> Material class instance)
        self.materials_ids_table = materials_ids_table

        # list of list with boundary conditions specifications
        self.boundary_conditions_table = boundary_conditions_table

        # control flags for the represented problems
        self.thermal_flag = thermal_flag
        self.mechanical_flag = mechanical_flag
        self.phase_field_damage_flag = phase_field_damage_flag
        self.plasticity_flag = plasticity_flag

        # identity matrix
        self.dimension = self.mesh.topology().dim()
        self.II = fe.Identity(self.dimension)

        if self.dimension == 2:
            if planar_Hp != "stress" and planar_Hp != "strain":
                raise RuntimeError("planar_Hp con only be stress or strain")
            self.planar_Hp = planar_Hp

        # set reference temperature attribute for domain
        self.ref_temperature = ref_temperature

        # incapsulation function to define finite element spaces
        self.define_spaces()

        # incapsulation function to define space-dependent properties
        self.define_properties()

        # incapsulation function to set fields reference values
        self.define_initial_conditions()

        # incapsulation function to define boundary conditions
        self.define_boundary_conditions()

        # incapsulation function to initialize variational forms
        self.define_variational_forms()


    def define_spaces(self):

        # Redefine Function space for history variable
        self.SH = fe.FunctionSpace(self.mesh, "DG", 0)

        # Redefine Function space and functions for temperature problem
        self.ST = fe.FunctionSpace(self.mesh, "CG", 1)
        self.T_new = fe.Function(self.ST, name="Temperature")
        self.T_old = fe.Function(self.ST, name="Temperature")

        # initial condition for T_old has to be set before properties definitions
        self.T_old.assign(fe.interpolate( self.ref_temperature, self.ST))

        if self.thermal_flag:
            self.T_trl = fe.TrialFunction(self.ST)
            self.T_tst = fe.TestFunction(self.ST)

        # Redefine Function space and functions for displacement problem
        self.SU = fe.VectorFunctionSpace(self.mesh, "CG", 1)
        self.u_new = fe.Function(self.SU, name="Displacement")

        # Redefine Function space and functions for plastic field variables
        self.SF = fe.FunctionSpace(self.mesh, "CG", 1)
        self.SX = fe.TensorFunctionSpace(self.mesh, "CG", 1)
        self.eps_p_new  = fe.Function(self.SX, name="Plastic_deformation")
        self.Baralp_new = fe.Function(self.SF, name="Accumulated_plastic_deformation")

        if self.mechanical_flag:
            self.u_trl = fe.TrialFunction(self.SU)
            self.u_tst = fe.TestFunction(self.SU)

        if self.plasticity_flag:
            self.eps_p_old  = fe.Function(self.SX, name="Plastic_deformation_tensor_old")
            self.Baralp_old = fe.Function(self.SF, name="Accumulated_plastic_deformation_old")
            self.Nval_new   = fe.Function(self.SX, name="Plastic_flow_direction_new")
            self.Nval_old   = fe.Function(self.SX, name="Plastic_flow_direction_old")
            self.Dgamma_new = fe.Function(self.SF, name="Plastic_multiplier_increment_new")
            self.Dgamma_old = fe.Function(self.SF, name="Plastic_multiplier_increment_old")
            self.pc         = fe.Function(self.SF, name="pc")

        # Redefine Function space and functions for damage problem
        self.SD = fe.FunctionSpace(self.mesh, "CG", 1)
        self.d_new = fe.Function(self.SD, name="Damage")
        self.H_old = fe.Function(self.SH, name="Crack_driving_force_historic")
        self.H_condition = fe.Function(self.SH, name="Mesh_refinement_condition")

        if self.phase_field_damage_flag:
            self.d_trl = fe.TrialFunction(self.SD)
            self.d_tst = fe.TestFunction(self.SD)


    def define_properties(self):
        # Required material properties
        self.required_properties = []

        if self.thermal_flag:
            self.required_properties += ["rho", "alpha", "cspec", "Kcond"]

        if self.mechanical_flag:
            self.required_properties += ["Emod", "nu"]

        if self.plasticity_flag:
            self.required_properties += [
                "eta0", "tau",
                "pc0", "pcT", "Omega",
                "Mbp", "alpbp", "mbp", "betbp", "gambp",
                "Ak", "delta0"
            ]

        if self.phase_field_damage_flag:
            self.required_properties += ["Gc", "lsp"]

        # identify temperature-dependent properties
        all_temperature_dependent_properties = self.materials_ids_table[0].exp_data_list

        # check that temperature-dependent properties are the same in all materials
        for material in self.materials_ids_table.values():
            if material.exp_data_list != all_temperature_dependent_properties:
                raise NotImplementedError("Required temperature-dependent properties must coincide in all materials")

        # remove not required properties from thermal dependent properties
        self.temperature_dependent_properties = []
        for property in all_temperature_dependent_properties:
            if property in self.required_properties:
                self.temperature_dependent_properties.append(property)

        # Create material properties field variables only for required properties
        for property in self.required_properties:
            exec( f"self.{property} = fe.Function(self.SH, name='{property}') ")

            try:
                if property not in self.temperature_dependent_properties:
                    # temperature-invariant property
                    command_string  = f"self.{property}.assign("
                    command_string += f"fe.project("
                    command_string += f"material_property_field("
                    command_string += f"self.materials_meshfunction,"
                    command_string += f"self.materials_ids_table,"
                    command_string += f"'{property}'"
                    command_string += f"),"
                    command_string += f"self.SH ))"
                    exec( command_string )
                else:
                    # temperature-dependent property
                    command_string  = f"self.{property}.assign("
                    command_string += f"fe.project("
                    command_string += f"material_property_field("
                    command_string += f"self.materials_meshfunction,"
                    command_string += f"self.materials_ids_table,"
                    command_string += f"'{property}',"
                    command_string += f"temperature=self.T_old,"
                    command_string += f"),"
                    command_string += f"self.SH ))"
                    exec( command_string )
            except:
                raise RuntimeError( f"Missing {property} data." )

        # Lame parameters
        if self.mechanical_flag:
            # compute dependant properties
            self.lmbda = self.Emod*self.nu/(one+self.nu)/(one-two*self.nu)
            self.mu = self.Emod/two/(one+self.nu)

            # plane stress approximation Hp
            if self.dimension == 2 and self.planar_Hp == "stress":
                self.lmbda = 2*self.lmbda*self.mu/(self.lmbda+2*self.mu)


    def define_initial_conditions(self):
        # set initial conditions
        self.T_new.assign(fe.interpolate( self.ref_temperature, self.ST))

        if self.plasticity_flag:
            # set initial condition for hardening variable
            self.pc.assign( fe.project(
                self.pcT + self.pcM()
                , self.SF )
            )


    def define_boundary_conditions(self):
        # dictionary of DirichletBC objects initialized to fenics null Constants
        # and corresponding target values as plain floats
        self.bc_values = []
        self.bc_list = []
        for bc_item in self.boundary_conditions_table:
            for ii in range(len(bc_item)-1):
                if not type( bc_item[ii+1] ) == str:
                    command_string  = "fe.DirichletBC(self.SU.sub("
                    command_string += str(ii) + "),"
                    command_string += "fe.Constant(0.0),"
                    command_string += "self.boundaries_meshfunction,"
                    command_string += str(bc_item[0]) + ")"

                    # create correspondence between identifier and final value
                    self.bc_values.append( bc_item[ii+1] )

                    # update list to pass to fenics solver
                    self.bc_list.append(eval(command_string))


    def define_variational_forms(self):
        if self.thermal_flag:
            self.fe_dt = fe.Constant(0.0)
            self.ET = (
                self.rho*self.cspec*(self.T_trl-self.T_old)/self.fe_dt*self.T_tst + \
                + self.Kcond*fe.dot(fe.grad(self.T_trl), fe.grad(self.T_tst))
                # + ( steel_hconv*(T_trl - steel_Temp)*T_tst )*ds_inner
            )*fe.dx

        if self.mechanical_flag:
            # self.EU = fe.inner( self.sig(self.u_tst), self.eps(self.u_trl) )*fe.dx
            self.EU = fe.inner( self.sig(self.u_trl), self.eps(self.u_tst) )*fe.dx

        if self.phase_field_damage_flag:
            self.ED = (
                -two*self.H_new(self.u_new)*(self.d_tst - fe.inner(self.d_trl, self.d_tst)) +\
                fe.inner(self.d_trl, self.d_tst)/self.lsp +\
                self.lsp*fe.inner( fe.grad(self.d_trl), fe.grad(self.d_tst))
            )*fe.dx


    # deformation tensor in small strain theory
    def eps(self, disp):
        return fe.sym(fe.grad(disp))


    # Thermal deformation
    def eps_T(self):
        return self.alpha*(self.T_new - self.ref_temperature)*self.II


    # elastic deformation
    def eps_e(self, disp):
        reversible_deformation = self.eps(disp)
        if self.thermal_flag:
            reversible_deformation -= self.eps_T()
        if self.plasticity_flag:
            reversible_deformation -= self.eps_p_new
        return reversible_deformation


    # normalized BP yield function
    def YieldFunction(self, sig):

        invp = - 1/3 * fe.tr(sig)
        dev = sig + invp*self.II

        # ntol in J2 avoids numerical issues
        J2 = 0.5*fe.tr(dev*dev) + ntol*self.pc0
        J3 = fe.det(dev)

        invq = fe.sqrt(3*J2)

        cos3theta = (3*(3**0.5)/2)*J3*(J2**(-3/2))

        cbp = self.Omega*self.pc0

        Phi = (invp + cbp)/(self.pc + cbp)

        finvp_val = - self.Mbp*self.pc*fe.sqrt((Phi-abs(Phi)**(self.mbp-1)*Phi)*(2*(1-self.alpbp)*Phi+self.alpbp))

        # the term under square root in finvp_val exists only if Phi in [0;1]
        finvp = fe.conditional(fe.lt(Phi, one), fe.conditional(fe.gt(Phi, 0.0), finvp_val, fe.Constant(1e5)), fe.Constant(1e5))

        gtheta_inverse = fe.cos(self.betbp * 3.14/6 - 1/3*(fe.acos(self.gambp*cos3theta)))

        F = finvp + invq*gtheta_inverse

        return F/self.pc0


    # yield function derivative
    def dFdsig(self, sig):
        vsig = fe.variable(sig)
        return fe.diff(self.YieldFunction(vsig), vsig)


    # stress calculation
    def sig(self, disp):
        return self.gdeg()*( self.lmbda*fe.tr( self.eps_e(disp) )*self.II + two*self.mu*self.eps_e(disp) )


    # pc mechanical hardening law
    def pcM(self):
        return self.Ak*self.Baralp_new/(1+self.delta0*self.Baralp_new)


    # p stress invariant = hydrostatic stress
    def finvp(self, disp):
        return - 1/3 * fe.tr(self.sig(disp))


    # crack driving force assuring irreversibility of damage
    def H_new(self, disp):

        # elastic deformation
        elastic_strain = self.eps_e(disp)

        # positive part of strain eigenvalues split
        if self.dimension == 2:
            eps_e_p = eigenSplit_pos_2D(elastic_strain)
        elif self.dimension == 3:
            eps_e_p = eigenSplit_pos_3D(elastic_strain)

        # damaged elastic strain energy (tension part)
        psiD_new = 0.5*self.lmbda*( Macaulay(fe.tr(elastic_strain)) )**2 + self.mu*fe.tr( eps_e_p*eps_e_p )

        # crack driving force normalization to allow eventual extension
        psiD_new = psiD_new/self.Gc

        # Irreversibility condition is imposed using ufl operators
        return fe.conditional( fe.lt(self.H_old, psiD_new), psiD_new, self.H_old )


    # degradation function
    def gdeg(self):
        return pow(1-self.d_new,2) + ntol


class Problem:
    '''
    - object defining the PDEs over the Domain and the solution procedure
    - main interface to the user
    '''
    def __init__(self,
        folder_descriptor,
        mesh,
        materials_meshfunction,
        boundaries_meshfunction,
        materials_ids_table,
        boundary_conditions_table,
        ref_temperature = 20,
        tmax = 1,
        dt_min = 0.1,
        dt_max = 0.1,
        dt_initial = 0.1,
        fixed_dump_interval = 0.1,
        adaptive_mesh_refinement_flag = 1,
        thermal_flag = 0,
        mechanical_flag = 0,
        phase_field_damage_flag = 0,
        plasticity_flag = 0,
        compute_reaction_flag = 0,
        planar_Hp = "undefined",
        copy_for_forensic_flag = 0
        ):

        # verify correctness of mesh info
        if mesh.topology().dim() != mesh.geometry().dim():
            print(f" mesh.geometry().dim() = {mesh.geometry().dim()} ")
            print(f" mesh.topology().dim() = {mesh.topology().dim()} ")
            print("Check your mesh, it has a non-user z=0.0 coordinate. Remove it.")
            print("Apparently there is a command line option in Meshio.")
            print("It's faster just to use Notepad++ to find and replace the `,\t0.\n` with empty.")

        # initial time
        self.sim_time = 0.0

        # maximum time
        self.tmax = tmax

        # time step
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_initial = dt_initial
        self.cut_factor = 2
        self.raise_factor = 2

        # specify the ratio between new and previous damage_energy to cut time step
        self.damage_energy_ratio_threshold = 1.5

        # specify the time interval after which to save a solution [s]
        self.fixed_dump_interval = fixed_dump_interval

        # maximum number of iterations allowed for the staggered algo
        self.max_number_iterations = 20

        # settings for mesh refinement
        if phase_field_damage_flag:
            self.adaptive_mesh_refinement_flag = adaptive_mesh_refinement_flag
        else:
            print("Adaptative mesh refinement is available only for damage problems")
            self.adaptive_mesh_refinement_flag = 0

        # setting flag for adaptive time stepping
        if dt_min != dt_max:
            if phase_field_damage_flag:
                self.adaptive_time_stepping_flag = 1
            else:
                print("Adaptative time stepping is available only for damage problems")
                self.adaptive_time_stepping_flag = 0

        # minimum allowed mesh size
        self.size_refinement_threshold = 0.000125

        # minimum value that activates refinement
        self.value_refinement_threshold = 0.3

        # from [[Freddi Mingazzi 2023]] beta coefficient adopted to anticipate
        # the refinement procedure and modulate the size of the active area
        self.beta = 0.5

        # minimum number of cells interested by refinement to activate it
        self.num_cell_refinement_threshold = 1

        # save tabular input data
        # remember that meshfunction depend on mesh refinement and must be
        # called from the refined Domain
        self.materials_ids_table = materials_ids_table
        self.boundary_conditions_table = boundary_conditions_table

        # set reference temperature attribute for domain
        self.ref_temperature = fe.Constant( Kshift + ref_temperature )

        # control flags for the represented problems
        self.thermal_flag = thermal_flag
        self.mechanical_flag = mechanical_flag
        self.phase_field_damage_flag = phase_field_damage_flag
        self.plasticity_flag = plasticity_flag
        self.compute_reaction_flag = compute_reaction_flag

        # info about planar Hp chosen
        self.planar_Hp = planar_Hp

        # problem Part definition
        self.Part = Domain(
            mesh,
            materials_meshfunction,
            self.ref_temperature,
            boundaries_meshfunction,
            materials_ids_table,
            boundary_conditions_table,
            self.planar_Hp,
            thermal_flag = self.thermal_flag,
            mechanical_flag = self.mechanical_flag,
            phase_field_damage_flag = self.phase_field_damage_flag,
            plasticity_flag = self.plasticity_flag
        )

        # provide info on properties
        print(f"Required properties: {self.Part.required_properties}")
        print(f"Temperature-dependent properties: {self.Part.temperature_dependent_properties}")

        # set control flag for iterative solver
        self.iterative_procedure_flag = 1
        if self.plasticity_flag == 0 and self.phase_field_damage_flag == 0:
            self.iterative_procedure_flag = 0
            if self.thermal_flag:
                print("Warning: no iterative procedure for thermo-meca only problem")

        # copy simulation files into the output folder for debug
        self.copy_for_forensic_flag = copy_for_forensic_flag

        # initialize file system using encapsuled function
        self.initialize_output_filesystem(folder_descriptor)


    def initialize_output_filesystem(self, folder_descriptor):
        # Outfolder specification
        load_case = "_"

        # README file headers
        info_string_header = f"{'[min]':^5}\t" +\
                             f"{'[%]':^5}\t" +\
                             f"{'[sim t]':^10}\t"

        if self.adaptive_time_stepping_flag:
            info_string_header += f"{'[dt]':^10}\t" +\
                                  f"{'[Cuts]':^5}\t" +\
                                  f"{'[Raise]':^5}\t"

        if self.iterative_procedure_flag:
            info_string_header += f"{'[iter]':^5}\t"

        if self.thermal_flag:
            load_case += "T"
            info_string_header += f"{'[thml J]':^10}\t"

        if self.mechanical_flag:
            load_case += "E"
            info_string_header += f"{'[elas J]':^10}\t"

        if self.plasticity_flag:
            load_case += "P"
            info_string_header += f"{'[plst J]':^10}\t"

        if self.phase_field_damage_flag:
            load_case += "D"
            info_string_header += f"{'[damg J]':^10}\t"

        if self.adaptive_time_stepping_flag:
            info_string_header += f"{'[Ratio]':^10}\t"

        if self.adaptive_mesh_refinement_flag:
            info_string_header += f"{'[N refns]':^10}\t" +\
                                  f"{'[N split]':^10}\t"

        if self.compute_reaction_flag:
            info_string_header += f"{'[react]':^10}\t"

        info_string_header += f"{'[Solution]':^10}"

        # setup output folder name
        now = datetime.now()
        date_string = now.strftime("%Y%m%dT%H%M")[2:] + "_"
        self.outfolder = "./" + date_string + folder_descriptor + load_case + "/"
        os.mkdir(self.outfolder)

        # printout information string
        self.fileprint(info_string_header)

        # Init output files
        self.ffile = fe.XDMFFile(self.outfolder + "field_output.xdmf")
        self.ffile.parameters["flush_output"]=True
        self.ffile.parameters["functions_share_mesh"]=True

        if self.copy_for_forensic_flag:
            # copy the simulation file to the results folder for forensics
            os.system("cp " + __file__ + " ./" + self.outfolder +
                    __file__.split("/")[-1].replace(".py", "") +
                    "_" + date_string + ".py" )


    def refine_mesh(self):

        mesh_coarse = self.Part.mesh

        # create the refinement mask
        refinement_mask = fe.MeshFunction( "bool", mesh_coarse, mesh_coarse.topology().dim() )

        self.Part.H_condition.assign( fe.project( self.Part.H_new(self.Part.u_new)*self.Part.lsp, self.Part.SH ))

        splitted_cells = 0
        # assign values to the empty refinement mask
        for cell in fe.cells(mesh_coarse):
            mid = cell.midpoint()
            refinement_mask[cell] = False
            if( cell.h() > self.size_refinement_threshold and
                    ( self.Part.d_new(mid) > self.value_refinement_threshold or
                    self.Part.H_condition(mid)>self.beta )
                ):
                splitted_cells += 1
                refinement_mask[cell] = True

        if splitted_cells > self.num_cell_refinement_threshold:
            # refine using FEniCS built in functionality
            mesh_fine = fe.refine( mesh_coarse , refinement_mask )
            return splitted_cells, mesh_fine
        else:
            return 0, mesh_coarse


    def redefine_problem_over_refined_Part(self, mesh_fine):

        refined_Part = Domain(
            mesh_fine,
            fe.adapt(self.Part.materials_meshfunction, mesh_fine),
            self.ref_temperature,
            fe.adapt(self.Part.boundaries_meshfunction, mesh_fine),
            self.materials_ids_table,
            self.boundary_conditions_table,
            self.planar_Hp,
            self.thermal_flag,
            self.mechanical_flag,
            self.phase_field_damage_flag,
            self.plasticity_flag
        )

        refined_Part.T_new.assign( fe.project(self.Part.T_new, refined_Part.ST))
        refined_Part.T_old.assign( fe.project(self.Part.T_old, refined_Part.ST))

        refined_Part.u_new.assign( fe.project(self.Part.u_new, refined_Part.SU))

        refined_Part.eps_p_new.assign( fe.project(self.Part.eps_p_new, refined_Part.SX))
        refined_Part.Baralp_new.assign( fe.project(self.Part.Baralp_new, refined_Part.SF))

        if self.plasticity_flag:
            refined_Part.eps_p_old.assign( fe.project(self.Part.eps_p_old, refined_Part.SX))
            refined_Part.Baralp_old.assign( fe.project(self.Part.Baralp_old, refined_Part.SF))
            refined_Part.Nval_new.assign( fe.project(self.Part.Nval_new, refined_Part.SX))
            refined_Part.Nval_old.assign( fe.project(self.Part.Nval_old, refined_Part.SX))
            refined_Part.Dgamma_new.assign( fe.project(self.Part.Dgamma_new, refined_Part.SF))
            refined_Part.Dgamma_old.assign( fe.project(self.Part.Dgamma_old, refined_Part.SF))

        refined_Part.d_new.assign( fe.project(self.Part.d_new, refined_Part.SD))
        refined_Part.H_old.assign( fe.project(self.Part.H_old, refined_Part.SH))

        for property in refined_Part.required_properties:
            command_string  = f"refined_Part.{property}.assign("
            command_string += f"fe.project("
            command_string += f"self.Part.{property}"
            command_string += f",refined_Part.SH ))"
            exec( command_string )

        # assign the refined part to the main one of the problem
        self.Part = refined_Part

        # bring boundary condition to current time step
        self.increment_load()


    def update_properties(self, temperature):
        # Material properties field variables
        for property in self.Part.temperature_dependent_properties:
            exec( f"self.{property} = fe.Function(self.Part.SH, name='{property}') ")
            command_string  = f"self.Part.{property}.assign("
            command_string += f"fe.project("
            command_string += f"material_property_field("
            command_string += f"self.Part.materials_meshfunction,"
            command_string += f"self.materials_ids_table,"
            command_string += f"'{property}',"
            command_string += f"temperature=temperature"
            command_string += f"),"
            command_string += f"self.Part.SH ))"
            exec( command_string )


    def run_thermo_increment(self):
        local_tic = time.time()
        fe.solve( fe.lhs(self.Part.ET) == fe.rhs(self.Part.ET), self.Part.T_new )
        print(f"Thermal problem solved in {time.time() - local_tic}")

        # update material properties with temperature
        self.update_properties(self.Part.T_new)


    def update_plasticity(self):
        # update the current stress
        sig_new = self.Part.sig(self.Part.u_new)
        Fval = self.Part.YieldFunction(sig_new)

        # update the plastic flow tensor
        self.Part.Nval_new.assign( fe.project( self.Part.dFdsig(sig_new), self.Part.SX ))

        # compute plastic multiplier factor considering time step
        self.Part.eta = self.Part.eta0*( 1 - ( 1 - fe.exp(-self.dt/self.Part.tau) )/(self.dt/self.Part.tau) )

        # update the plastic multiplier
        # self.Part.Dgamma_new.assign( fe.project( self.Part.eta*Macaulay(Fval) , self.Part.SF ))
        self.Part.Dgamma_new.assign( fe.project(
            positive_signum(self.Part.finvp(self.Part.u_new), ntol)*self.Part.eta*Macaulay(Fval),
        self.Part.SF ))

        # update plastic strain with forward Euler formula
        self.Part.eps_p_new.assign( fe.project(
            self.Part.eps_p_old + 0.5*(
                self.Part.Dgamma_old*self.Part.Nval_old +\
                self.Part.Dgamma_new*self.Part.Nval_new
            ), self.Part.SX )
        )

        # update the scalar internal variable
        self.Part.Baralp_new.assign( fe.project(
            Macaulay( self.Part.Baralp_old + 0.5*(
                self.Part.Dgamma_old*tensorNorm(self.Part.Nval_old) +\
                self.Part.Dgamma_new*tensorNorm(self.Part.Nval_new) )
            ), self.Part.SF )
        )

        # update the hardening variable
        self.Part.pc.assign( fe.project(
            self.Part.pcT + self.Part.pcM()
            , self.Part.SF )
        )

        # update iteration quantities
        self.Part.Nval_old.assign(self.Part.Nval_new)
        self.Part.Dgamma_old.assign(self.Part.Dgamma_new)


    def run_phase_field_damage_increment(self):
        # solving damage problem
        local_tic = time.time()
        fe.solve( fe.lhs(self.Part.ED) == fe.rhs(self.Part.ED), self.Part.d_new )
        print(f"Damage problem solved in {time.time() - local_tic}")

        if self.adaptive_mesh_refinement_flag:
            # check if mesh has to be refined and do it
            splitted_cells, refined_mesh = self.refine_mesh()

            if splitted_cells > 0:
                # update counter of refinements
                self.refinements_counter += 1

                # update counter of splitted cells in increment
                self.splitted_cells_history += splitted_cells

                self.redefine_problem_over_refined_Part( refined_mesh )

        # update damage associated energy measure by using the formula from Borden Hughes Landis Anvari Lee 2016
        self.damage_energy_new = fe.assemble(
            self.Part.Gc/(4*self.Part.lsp)*(
                pow(self.Part.d_new,2) +
                4*self.Part.lsp**2*fe.inner(fe.grad(self.Part.d_new), fe.grad(self.Part.d_new))
            )*fe.dx
        )


    def run_mecha_increment(self):
        # setup elastic energy measure for loop iterations
        elastic_energy_old = 0
        self.elastic_energy_new = 1
        self.itr_index = -1
        self.refinements_counter = 0
        self.splitted_cells_history = 0

        while ( self.itr_index < self.max_number_iterations and
                abs(elastic_energy_old - self.elastic_energy_new) > 0.01*self.elastic_energy_new
        ):
            self.itr_index += 1
            elastic_energy_old = self.elastic_energy_new

            if self.plasticity_flag:
                self.update_plasticity()

            # solving mechanical problem
            local_tic = time.time()
            fe.solve( fe.lhs(self.Part.EU) == fe.rhs(self.Part.EU), self.Part.u_new, self.Part.bc_list )
            print(f"Displacement problem solved in {time.time() - local_tic}")

            # solve damage problem and update crack driving force value
            if self.phase_field_damage_flag:
                self.run_phase_field_damage_increment()

            # calculate elastic energy to check convergence
            self.elastic_energy_new = fe.assemble(
                fe.inner( self.Part.sig(self.Part.u_new), self.Part.eps_e(self.Part.u_new) )*fe.dx
            )

            self.NaN_flag += self.checkNanEnergy(self.elastic_energy_new)

            # no need to check energy convergence in linear elastic analysis
            if not self.iterative_procedure_flag:
                break


    def dump_solution(self, sol_counter):
        if self.thermal_flag:
            self.ffile.write( self.Part.T_new, sol_counter )

        if self.mechanical_flag:
            self.ffile.write( self.Part.u_new, sol_counter )

        if self.plasticity_flag:
            self.ffile.write( self.Part.pc, sol_counter )
            self.ffile.write( self.Part.Baralp_new, sol_counter )

        if self.phase_field_damage_flag:
            self.ffile.write( self.Part.d_new, sol_counter )


    def fileprint(self, string):
        print(string)
        with open(self.outfolder + "history_output.md", "a") as file:
            file.writelines( "\n" + string)


    def checkNanEnergy(self, energy):
        if np.isnan(energy):
            self.fileprint(" Energy nan detected, interrupting simulation. ")
            return 1
        else:
            return 0


    def printout_refused_step(self):
        info_string = f"{int((time.time() - self.tic)/60):^5}\t" +\
                      f"{self.sim_time/self.tmax*100:^5.3}\t" +\
                      f"{self.sim_time + self.dt:^10.5g}\t"

        if self.adaptive_time_stepping_flag:
            info_string += f"{self.dt:^10.3g}\t" +\
                           f"{self.counter_dt_cuts:^5}\t" +\
                           f"{self.counter_dt_raise:^5}\t"

        if self.iterative_procedure_flag:
            info_string += f"{self.itr_index:^5}"+ '\t'

        if self.thermal_flag:
            info_string += f"{'-':^10}" + '\t'

        if self.mechanical_flag:
            info_string += f"{'-':^10}" + '\t'

        if self.plasticity_flag:
            info_string += f"{'-':^10}" + '\t'

        if self.phase_field_damage_flag:
            info_string += f"{self.damage_energy_new:^10.3g}" + '\t'

        if self.adaptive_time_stepping_flag:
            info_string += f"{self.damage_energy_ratio:^10.3g}" + '\t'

        if self.adaptive_mesh_refinement_flag:
            info_string += f"{self.refinements_counter:^10.3g}" + '\t'
            info_string += f"{self.splitted_cells_history:^10.3g}" + '\t'

        if self.compute_reaction_flag:
            info_string += f"{'-':^10}" + '\t'

        info_string += f"{'-':^10}"

        self.fileprint( info_string )


    def printout_accepted_step(self):
        info_string = f"{int((time.time() - self.tic)/60):^5}\t" +\
                  f"{self.sim_time/self.tmax*100:^5.3}\t" +\
                  f"{self.sim_time:^10.5g}\t"

        if self.adaptive_time_stepping_flag:
            info_string += f"{self.dt:^10.3g}\t" +\
                           f"{self.counter_dt_cuts:^5}\t" +\
                           f"{self.counter_dt_raise:^5}\t"

        if self.iterative_procedure_flag:
            info_string += f"{self.itr_index:^5}"+ '\t'

        if self.thermal_flag:
            info_string += f"{self.thermal_energy_new:^10.3g}\t"

        if self.mechanical_flag:
            info_string += f"{self.elastic_energy_new:^10.3g}\t"

        if self.plasticity_flag:
            info_string += f"{self.plastic_energy_new:^10.3g}\t"

        if self.phase_field_damage_flag:
            info_string += f"{self.damage_energy_new:^10.3g}\t"

        if self.adaptive_time_stepping_flag:
            info_string += f"{self.damage_energy_ratio:^10.3g}\t"

        if self.adaptive_mesh_refinement_flag:
            info_string += f"{self.refinements_counter:^10.3g}\t" +\
                           f"{self.splitted_cells_history:^10.3g}\t"

        if self.compute_reaction_flag:
            info_string += f"{self.reaction_new:^10.3g}\t"

        # Compute and output values in files for post-processing in Paraview
        if ( self.counter_dt_cuts > 0 or
             self.sim_time+ntol > self.fixed_dump_interval*(self.fixed_dump_interval_counter + 1)
        ):
            self.sol_counter += 1
            self.dump_solution(self.sol_counter)
            info_string += f"{self.sol_counter:^10}"
            self.fileprint( info_string )
        else:
            info_string += f"{'No dump':^10}"
            self.fileprint( info_string )

        if self.sim_time+ntol > self.fixed_dump_interval*(self.fixed_dump_interval_counter + 1):
            # update the solution dumping counter independently from the time step
            self.fixed_dump_interval_counter += 1


    def update_history_output(self):

        if self.thermal_flag:
            # save temperature value
            self.Part.T_old.assign(self.Part.T_new)

            # update thermal energy measure
            self.thermal_energy_new = fe.assemble( ( self.Part.rho*self.Part.cspec*(self.Part.T_new - self.ref_temperature) )*fe.dx )

            # export thermal energy measure as history output
            self.thermal_energy_history.append(self.thermal_energy_new)

            self.NaN_flag += self.checkNanEnergy(self.thermal_energy_new)

        # update historic quantities after step iteration
        if self.plasticity_flag:
            # update plastic energy by increment before updating eps_p_old
            self.plastic_energy_new = self.plastic_energy_history[-1] + \
                fe.assemble(
                    fe.inner( self.Part.sig(self.Part.u_new), self.Part.eps_p_new - self.Part.eps_p_old )*fe.dx
                )

            self.NaN_flag += self.checkNanEnergy(self.plastic_energy_new)

            # update plastic field variables
            self.Part.eps_p_old.assign(self.Part.eps_p_new)
            self.Part.Baralp_old.assign(self.Part.Baralp_new)

        if self.phase_field_damage_flag:
            # update damage field variables
            self.Part.H_old.assign( fe.project( self.Part.H_new(self.Part.u_new), self.Part.SH ))

        if self.compute_reaction_flag:
            self.reaction_new = fe.assemble( self.Part.sig(self.Part.u_new)[1,1]*fe.Measure("ds", subdomain_id=2, subdomain_data=self.Part.boundaries_meshfunction) )

        # update history variables
        if self.iterative_procedure_flag:
            self.itr_history.append(self.itr_index)
        if self.mechanical_flag:
            self.elastic_energy_history.append( self.elastic_energy_new )
        if self.plasticity_flag:
            self.plastic_energy_history.append( self.plastic_energy_new )
        if self.phase_field_damage_flag:
            self.damage_energy_history.append( self.damage_energy_new )
        if self.compute_reaction_flag:
            self.reaction_history.append( self.reaction_new )

    def advance_time_step(self):
        # Increment simulation time with time step used for solution
        self.sim_time += self.dt

        # save time steps as history output
        self.time_steps.append(self.sim_time)

        self.update_history_output()

        self.printout_accepted_step()


    def time_stepping_control(self):
        # reset damage energy ratio value
        self.damage_energy_ratio = -1
        if len(self.time_steps)>1:
            # compute damage energy ratio using history values (available only after 1st time step)
            if self.damage_energy_history[-1] != 0:
                self.damage_energy_ratio = self.damage_energy_new/self.damage_energy_history[-1]

        # the time step check can happen only at the second calculation
        # if time step hits the minimum the solution is exported forcefully
        if ( len(self.time_steps)>1 and
            ( self.damage_energy_ratio > self.damage_energy_ratio_threshold or
            self.itr_index == self.max_number_iterations ) and
            self.dt != self.dt_min
        ):

            self.printout_refused_step()

            # time step has to be decremented and the analysis re-run
            self.counter_dt_raise = 0
            self.counter_dt_cuts += 1

            if self.dt/(self.cut_factor*self.counter_dt_cuts) > self.dt_min:
                self.dt = self.dt/(self.cut_factor*self.counter_dt_cuts)
            else:
                self.dt = self.dt_min

        else:
            # Solution can be exported and time step raised
            self.advance_time_step()

            # reset the time step cut counter
            self.counter_dt_cuts = 0
            self.counter_dt_raise += 1
            # raise time increment (with a max cap)
            if self.dt*(self.raise_factor*self.counter_dt_raise) <= self.dt_max:
                self.dt = self.dt*(self.raise_factor*self.counter_dt_raise)
            else:
                self.dt = self.dt_max


    def increment_load(self):
        for ii in range(len(self.Part.bc_list)):
            if self.Part.bc_values[ii] != 0.0:

                self.Part.bc_list[ii].set_value(
                    fe.project(
                        self.Part.bc_values[ii]*(self.sim_time + self.dt)/self.tmax
                    , self.Part.SF)
                )


    def run_simulation(self):

        # save time-steps in vector (useful for adaptive time-stepping)
        self.time_steps = [0]

        # historical quantities
        self.thermal_energy_history = [0]
        self.elastic_energy_history = [0]
        self.plastic_energy_history = [0]
        self.damage_energy_history = [0]
        self.itr_history = [0]
        self.reaction_history = [0]

        # initialize counter for solution dumping
        self.fixed_dump_interval_counter = 0
        self.sol_counter = 0

        # output initial condition as solution
        self.dump_solution(self.sol_counter)

        # initializate NaN flag
        self.NaN_flag = 0

        # initialize simulation time step and dumping solution interval
        self.dt = self.dt_initial
        self.counter_dt_cuts = 0
        self.counter_dt_raise = 0

        # measure elapsed time for simulation
        self.tic = time.time()

        # simulation loop
        while self.sim_time+ntol<self.tmax and self.NaN_flag == 0:

            if self.thermal_flag:
                # update time step in the thermal problem definition
                self.Part.fe_dt.assign(self.dt)
                self.run_thermo_increment()

            if self.mechanical_flag:
                self.increment_load()
                self.run_mecha_increment()

            if self.adaptive_time_stepping_flag:
                self.time_stepping_control()
            else:
                self.advance_time_step()

        self.fileprint(f"Elapsed CPU time: {int((time.time() - self.tic)/60):5d} [min]")

        if self.NaN_flag == 0:
            self.fileprint("Simulation completed successfully !!!")
        elif self.NaN_flag == 1:
            self.fileprint("Simulation interrupted after NaN detection.")

# EOF
