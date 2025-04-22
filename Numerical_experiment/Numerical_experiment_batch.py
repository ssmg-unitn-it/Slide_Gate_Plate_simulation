import os
import sys
import fenics as fe
import ThermoMecaFractureSolver as mdl
from importMesh import readMesh

# identify input parameters
temperature  = float(sys.argv[1])
displacement = float(sys.argv[2])
try:
    time_step = float(sys.argv[3])
    mesh_refinements = int(sys.argv[4])
except:
    time_step = 0.01
    mesh_refinements = 0


# define the material
ceramic = mdl.Material(
    exp_data_folder = "data",
    exp_data_list = ["Emod", "pcT"],
    nu = 0.07,
    Gc = 10,
    lsp = 0.0005,
    eta0 = 1e7,
    tau = 0.05,
    pc0 = 351e6,
    Omega = 0.014,
    Mbp = 1.602,
    alpbp = 0.133,
    mbp = 2,
    betbp = 0.75,
    gambp = 0.7,
    Ak = 10e9,
    delta0 = 500
)


# select input geometry according to simulation type
if displacement > 0:
    mesh_file = "specimen.inp"
else:
    mesh_file = "rectangle.inp"

# import mesh, boundary markers and cell markers
mesh, boundaries, materials = readMesh("./Meshes/", mesh_file)

# rescale coordinates from meter to millimiter
mesh.coordinates()[:, :] = mesh.coordinates()[:, :]*1e-3

# use a refined mesh
print(mesh_refinements)
if mesh_refinements> 0:
    for ii in range(mesh_refinements):
        # create the refinement mask
        refinement_mask = fe.MeshFunction( "bool", mesh, mesh.topology().dim() )

        # assign values to the empty refinement mask
        for cell in fe.cells(mesh):
            refinement_mask[cell] = False
            if( cell.h() > 0.000125 ):
                refinement_mask[cell] = True
        
        # refine the mesh
        print("Refining mesh")
        mesh_fine = fe.refine( mesh , refinement_mask )
        materials = fe.adapt( materials, mesh_fine )
        boundaries = fe.adapt( boundaries, mesh_fine )
        mesh = mesh_fine

    print(f"Mesh refined {ii+1} times")

# define the boundary conditions with format: (domain_ID, ux, uy)
bcs = [
    [1, 0, 0],
    [2, "free", displacement]
]


# define the problem
pb = mdl.Problem(
    "NEXP_" + "_".join(sys.argv[1:]),
    mesh,
    materials,
    boundaries,
    { 0 : ceramic },
    bcs,
    ref_temperature = temperature,
    tmax = 1,
    dt_min = 1e-7,
    dt_max = 0.1,
    dt_initial = time_step,
    fixed_dump_interval = 0.1,
    thermal_flag = 0,
    mechanical_flag = 1,
    plasticity_flag = 1 - bool( displacement > 0),
    phase_field_damage_flag = bool( displacement > 0) + 0,
    compute_reaction_flag = 1,
    planar_Hp = "strain",
    copy_for_forensic_flag = 0
)


# launch the solver
pb.run_simulation()

