#%% --------------------------------------------------------------
#%% Constant physical parameters in SI units

# Poisson ratio
nu = 0.1

# thermal parameters
rho = 3000  	# [kg/m^3] density

# fracture parameters
lsp = 0.5*1e-3	# [m] length scale parameter

# plasticity parameters
pc0 = 350e6     #[Pa]
alpbp = 0.1
Mbp = 1.5
mbp = 2
betbp = 0.7
gambp =  0.7

# hardening parameters    
Omega = 0.01
Ak = 10e9       # [Pa]
delta0 = 500

#%% --------------------------------------------------------------
#%% Load experimental data from corresponding csv file
# This section has to compile before the Filesystem setting otherwise
# the compiler throws a weird ascii codec error (is this a bug ???)

exp_data_folder = "./data/"

# create an empty dictionary to store experimental data
exp_data = {}

measured_paramsfT = ["Emod","alpha","cspec","Kcond","pcT","Gc"]

for key in measured_paramsfT:
    exp_data[key] = {}
    rawdata = np.loadtxt(exp_data_folder + key + ".csv", delimiter=',')
    exp_data[key]["Temp"] = rawdata[:,0] + Kshift
    exp_data[key]["value"] = rawdata[:,1]