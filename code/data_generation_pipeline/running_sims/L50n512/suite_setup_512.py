import os
import csv
import subprocess

# Important paths to set before running 
output_location = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite"
run_location = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/run/L50n512_suite"
template_location = run_location + "/template"
param_arepo_file_name = "param_L50n512.txt"
slurm_script_name = "script.slurm"

gridpoints_to_init = [i for i in range(11, 50)]

######################## read grid file ###########################
cosmo_parameters = []

header = True
with open('grid_lhs_constrained_second_suite.csv', newline='') as f:
    file = csv.reader(f, delimiter=',')
    for row in file:
        if header:
            header = False
            continue
        cosmo_parameters.append([float(i) for i in row])
        

##################### loop over grid points ######################

for counter, (Omega_m, Omega_b, Omega_L, hubble_par) in enumerate(cosmo_parameters):
    print("----------------------------------")
    print("Grid Point Nr.", counter)
    print(f"Omega_m = {Omega_m:.3f}, Omega_b = {Omega_b:.3f}, Omega_L = {Omega_L:.3f}, h = {hubble_par:.3f}")
    
    if counter not in gridpoints_to_init:
        print("Not in gps to init - skipping this gp ...")
        continue

    # create output directory
    output_gridpoint = output_location+f"/gridpoint{counter}"
    os.makedirs(output_gridpoint, exist_ok=True)
    
    # create run directory
    run_gridpoint = run_location+f"/gridpoint{counter}"
    os.makedirs(run_gridpoint, exist_ok=True)
    
    # copy template to run directory
    cmd = f"cp -r {template_location}/* {run_gridpoint}"
    os.system(cmd)
    
    # edit param.txt to match gridpoint values
    path_to_param_file = run_gridpoint + f"/{param_arepo_file_name}"
    
    file_content = []
    with open(path_to_param_file, "r") as file:
        for row in file:
            if "Omega0" in row:
                row = f"Omega0	              {Omega_m:.4f}\n"
            if "OmegaLambda" in row:
                row = f"OmegaLambda           {Omega_L:.4f}\n"
            if "OmegaBaryon" in row:
                row = f"OmegaBaryon           {Omega_b:.4f}\n"
            if "HubbleParam" in row:
                row = f"HubbleParam           {hubble_par:.4f}\n"
            if "OutputDir" in row:
                row = f"OutputDir           {output_gridpoint}\n"
            file_content.append(row)

    with open(path_to_param_file, "w") as file:
        for row in file_content:
            file.write(row)
            
    # check if the edits have worked in 
    error = False
    with open(path_to_param_file, "r") as file:
        for row in file:
            if "Omega0" in row:
                if not row.split()[-1] == f"{Omega_m:.4f}":
                    print(f"Error! -> Omega0 not correctly set in {path_to_param_file}")
                    error = True
            if "OmegaLambda" in row:
                if not row.split()[-1] == f"{Omega_L:.4f}":
                    print(f"Error! -> OmegaL not correctly set in {path_to_param_file}")
                    error = True
            if "OmegaBaryon" in row:
                if not row.split()[-1] == f"{Omega_b:.4f}":
                    print(f"Error! -> OmegaB not correctly set in {path_to_param_file}")
                    error = True
            if "HubbleParam" in row:
                if not row.split()[-1] == f"{hubble_par:.4f}":
                    print(f"Error! -> HubblePar not correctly set in {path_to_param_file}")
                    error = True
            if "OutputDir" in row:
                if not row.split()[-1] == output_gridpoint:
                    print(f"Error! -> OutputDir not correctly set in {path_to_param_file}")
                    error = True
    
    if error:
        print("Skipping this gridpoint ...")
        continue
    else:
        print(f"{path_to_param_file} was edited successfully!")

    # copy ics into run location
    cmd = f"cp {run_location}/ics/ics_gp{counter}.hdf5 {run_gridpoint}/ics.hdf5"
    os.system(cmd)
    
    # check if ics.hdf5 file exists
    files_in_run_dir = os.listdir(run_gridpoint)
    if "ics.hdf5" in files_in_run_dir:
        print("ICs were sucessfully created!")
    else:
        print("Error! -> IC creation failed")
        print("Skipping this gridpoint ...")
        continue
    
    # edit name in script.slurm
    path_to_slurm_script = run_gridpoint + f"/{slurm_script_name}"
    
    file_content = []
    with open(path_to_slurm_script, "r", encoding="utf-8") as file:
        for row in file:
            if "SBATCH -J" in row:
                row = f"#SBATCH -J CLIMB-{counter}\n"
            file_content.append(row)

    with open(path_to_slurm_script, "w") as file:
        for row in file_content:
            file.write(row)
    