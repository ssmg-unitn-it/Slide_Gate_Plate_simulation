"""
- Post-processing data from history_output.md files
"""

#%% ---------------------------------------------------------------70
# Import necessary packages
# -----------------------------------------------------------------70
import numpy as np
import math
import sys
import os
import importlib
import time
from datetime import datetime
from matplotlib import pyplot as plt


#%% ---------------------------------------------------------------70
# Filesystem settings
# -----------------------------------------------------------------70

stress_strain_curves_study_flag = 1
time_step_study_flag = 0
mesh_size_study_flag = 0

folder_identifier = "NEXP"

computed_results = sorted([folder for folder in os.listdir() if folder.find(folder_identifier)>0 and folder.find("BATCH")<0 ])

now = datetime.now()
time_string = now.strftime("%Y%m%dT%H%M")[2:]

#%% ---------------------------------------------------------------70
# Control parameters
# -----------------------------------------------------------------70

# decide whether to print the figure to screen or save it
savefig_flag = 1

# scale factor from simulation time to imposed displacement
# (equal to total displacement)
time_scale_factor_tension = 0.0001
time_scale_factor_compression = -0.0005

# test temperatures set in batch file
test_temperatures = [20, 600, 1000]

# selected time step for stress-strain curves
time_step_to_plot = 0.01

# mesh refinement cycles set in batch file
mesh_refinement_cycles = [0, 1, 2, 3]

# columns identifiers for interesting results
tension_results_cols = (2, 12)
compression_results_cols = (2, 6)

#%% ---------------------------------------------------------------70
# Figure Control parameters
# -----------------------------------------------------------------70

# labels and title font-size
fs = 8

# marker size
ms = 2

# marker cycle
markers = {"20.0":'^', "600.0":'s', "1000.0":'o'}

# line width
lw = 1

# frequency of plot
fz = 5

# figure size
figsize_x_cm = 10
figsize_y_cm = 10

figsize_inches = (figsize_x_cm/2.54, figsize_y_cm/2.54)

plt.style.use('grayscale')

#%% ---------------------------------------------------------------70
# Gather data
# -----------------------------------------------------------------70

def filter_NaN(array, filter):
    return array[~np.isnan(filter)]


class computed_result:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.full_id = folder_name.split("_")[2:]

        self.temperature = float( self.full_id[0] )
        self.displacement = float( self.full_id[1] )
        try:
            self.time_step = float( self.full_id[2] )
        except:
            self.time_step = -1
        try:
            self.mesh_refinements = float( self.full_id[3] )
        except:
            self.mesh_refinements = -1
        self.descriptor = self.full_id[-1]   
    
    
        if self.displacement>0:
            columns = tension_results_cols
            scale_factor = time_scale_factor_tension
        else:
            columns = compression_results_cols
            scale_factor = time_scale_factor_compression
        
        self.data = np.genfromtxt(self.folder_name + "/history_output.md", skip_header = 2, skip_footer = 5, usecols = columns)
        self.disp_data  = np.insert( filter_NaN( self.data[:,0], self.data[:,1]), 0, 0)*scale_factor*1e3
        self.force_data = np.insert( filter_NaN( self.data[:,1], self.data[:,1]), 0, 0)/1e3

items = []
for folder in computed_results:
    items.append( computed_result(folder) )


#%% ---------------------------------------------------------------70
# Plot force displacement curves at different temperatures
# -----------------------------------------------------------------70

def analyze_monotonicity(measurement):
    drop_index = -1
    ii = 0
    while ii < len(measurement)-1:
        if measurement[ii+1] < measurement[ii]:
            drop_index = ii
            break
        ii += 1
    return drop_index

if stress_strain_curves_study_flag:
    
    # plot tensile test data
    plt.figure(figsize=figsize_inches)
    for item in items:
        if item.displacement>0:
            if item.temperature in test_temperatures:
                plot_limit = analyze_monotonicity(item.force_data)
                plt.plot(item.disp_data[:plot_limit], item.force_data[:plot_limit],
                        '-'+markers[str(item.temperature)],
                        linewidth=lw,
                        markersize=ms,
                        label = f"Temperature = {int(item.temperature):4d}°C"
                        )
            
    plt.xlabel("Displacement [mm]", fontsize=fs)
    plt.ylabel("Force [kN]", fontsize=fs)
    plt.legend(fontsize=fs)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.grid()

    if savefig_flag:
        plt.savefig( time_string + "_Tension_Reaction_force.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # plot compression test data
    plt.figure(figsize=figsize_inches)
    already_plotted_temperature = []
    for item in items:
        if item.displacement<0:
            if (item.temperature in test_temperatures and 
               item.temperature not in already_plotted_temperature):
                if item.time_step == time_step_to_plot:
                    already_plotted_temperature.append(item.temperature)
                    plt.plot(item.disp_data, item.force_data,
                            '-'+markers[str(item.temperature)],
                            linewidth=lw,
                            markersize=ms,
                            label = f"Temperature = {int(item.temperature):4d}°C"
                            )
            
    plt.xlabel("Displacement [mm]", fontsize=fs)
    plt.ylabel("Force [kN]", fontsize=fs)
    plt.legend(fontsize=fs)

    plt.ylim(bottom=-9000)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.grid()

    if savefig_flag:
        plt.savefig( time_string + "_Compression_Reaction_force.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


#%% ---------------------------------------------------------------70
# Plot Compression test data
# -----------------------------------------------------------------70

if time_step_study_flag:
    for test_temperature in test_temperatures:
        plt.figure(figsize=figsize_inches)

        for item in items:
            if item.displacement<0:
                if item.temperature == test_temperature:
                    plt.plot(item.disp_data, item.force_data,
                            '-o',
                            linewidth=lw,
                            markersize=ms,
                            label = f"time_step = {item.time_step} s"
                            )

        plt.xlabel("Displacement [mm]", fontsize=fs)
        plt.ylabel("Force [kN]", fontsize=fs)
        plt.legend(fontsize=fs)
        
        plt.ylim(bottom=-9000)
        
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)

        plt.grid()

        if savefig_flag:
            plt.savefig( time_string + "_Compression_Reaction_force_at_" + str(test_temperature) + ".png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
#%% ---------------------------------------------------------------70
# Plot mesh refinement study data
# -----------------------------------------------------------------70

if mesh_size_study_flag:
    plt.figure(figsize=figsize_inches)
    
    for mesh_refinements in mesh_refinement_cycles:
        for item in items:
            if item.displacement<0:
                if item.mesh_refinements == mesh_refinements:
                    plt.plot(item.disp_data, item.force_data,
                            '-o',
                            linewidth=lw,
                            markersize=ms,
                            label = f"Mesh refinements = {item.mesh_refinements}"
                            )

    plt.xlabel("Displacement [mm]", fontsize=fs)
    plt.ylabel("Force [kN]", fontsize=fs)
    plt.legend(fontsize=fs)
    
    plt.ylim(bottom=-9000)
    
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    plt.grid()

    if savefig_flag:
        plt.savefig( time_string + "_Mesh_refinement_study.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
