import subprocess
import os

run_location = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/run/L50n512_suite"
gridpoints_to_start = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for gp in gridpoints_to_start:

    run_gridpoint = run_location+f"/gridpoint{gp}"
    os.chdir(run_gridpoint)

    # submit the job script to slurm
    slurm_script = "script.slurm"
    result = subprocess.run(["sbatch", slurm_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    sbatch_output = result.stdout.strip()
    
    with open(run_location+"/slurm_job_ids.txt", "a") as myfile:
        myfile.write(f"{gp}: {sbatch_output}\n")