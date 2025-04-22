"""
Run a batch of simulations ranging through a given set of params
"""

#%% ---------------------------------------------------------------70
# Import necessary packages
# -----------------------------------------------------------------70
import os
import sys
import time
from datetime import datetime
import numpy as np


#%% ---------------------------------------------------------------70
# Control parameters
# -----------------------------------------------------------------70

FEM_solver = "ThermoMecaFractureSolver.py"

FEM_file = "Numerical_experiment_batch.py"

folder_descriptor = "BATCH_NEXP"

stress_strain_curves_study_flag = 0
time_step_study_flag = 0
mesh_size_study_flag = 1

displacements = [-0.0005, 0.0001]

temperatures = [20, 400, 600, 800, 1000]

time_steps = [1e-4, 1e-3, 1e-2, 1e-1]

mesh_refinement_cycles = [0, 1, 2, 3]

#%% ---------------------------------------------------------------70
# Filesystem settings
# -----------------------------------------------------------------70

now = datetime.now()
dt_string = now.strftime("%Y%m%dT%H%M")[2:]

outfolder = dt_string + "_" + folder_descriptor

# create output folder
os.system("mkdir " + outfolder)

# copy this file to the inverse analysis folder for forensics
os.system("cp " + __file__ + " ./" + outfolder + "/" +
        __file__.split("/")[-1].replace(".py", "") +
        "_" + dt_string + ".py" )

# copy the simulation setup file for forensics
os.system("cp " + FEM_file + " ./" + outfolder + "/" +
        FEM_file.split("/")[-1].replace(".py", "") +
        "_" + dt_string + ".py" )

# copy the solver file for forensics
os.system("cp " + FEM_file + " ./" + outfolder + "/" +
        FEM_solver.split("/")[-1].replace(".py", "") +
        "_" + dt_string + ".py" )

#%% ---------------------------------------------------------------70
# Batch simulations for stress-strain curves at various temperatures
# -----------------------------------------------------------------70

if stress_strain_curves_study_flag:
    for temperature in temperatures:
        for displacement in displacements:
            # launch FEM with the input parameters
            os.system("python " + FEM_file + " " + str(temperature) + " " + str(displacement) )

#%% ---------------------------------------------------------------70
# Batch simulations for studying time step effect on plasticity algo
# -----------------------------------------------------------------70

if time_step_study_flag:
    for temperature in temperatures:
        for displacement in displacements:
            if displacement < 0:
                for time_step in time_steps:
                    # launch FEM with the input parameters
                    os.system("python " + FEM_file + " " + str(temperature) + " " + str(displacement) + " " + str(time_step))

#%% ---------------------------------------------------------------70
# Batch simulation for studying mesh size effect on plasticity algo
# -----------------------------------------------------------------70

if mesh_size_study_flag:
    for displacement in displacements:
        if displacement < 0:    
            for mesh_refinements in mesh_refinement_cycles:
                # launch FEM with the input parameters
                os.system("python " + FEM_file + " " + str(temperatures[0]) + " " + str(displacement) + " " + str(time_steps[2]) + " " + str(mesh_refinements))