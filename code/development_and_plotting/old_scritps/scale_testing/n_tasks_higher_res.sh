#!/bin/bash

number_of_cpus=(144 216 288 432) # number of cpus to use

BASE_SIM_DIR="/vera/u/jerbo/arepo/run/scaling_sims"

TEMPLATE_DIR="$BASE_SIM_DIR/template_simulation_code/cosmo_box_scaling_analysis"
AREPO_DIR="/vera/u/jerbo/arepo"

for n_cpus in "${number_of_cpus[@]}"; do
    echo "=== Starting simulation for $n_cpus cpus ==="
    
    # 1. Create new simulation directory
    SIM_DIR="$BASE_SIM_DIR/sim_n${n_cpus}_cpus_higher_res"
    mkdir -p "$SIM_DIR"

    # 2. Copy template files into it
    cp -r "$TEMPLATE_DIR"/* "$SIM_DIR"/

    # 3. Set parameters for number of particles in param_music.txt
    sed -i "s/^levelmin *= *.*/levelmin           = 8/" "$SIM_DIR/param_music.txt"
    sed -i "s/^levelmin_TF *= *.*/levelmin_TF        = 8/" "$SIM_DIR/param_music.txt"
    sed -i "s/^levelmax *= *.*/levelmax           = 8/" "$SIM_DIR/param_music.txt"
    
    # 4. Set parameters for number of cpus in slurm_script.slurm
    if [ "$n_cpus" -le 72 ]; then
        sed -i "s/^#SBATCH --nodes=.*/#SBATCH --nodes=1/" "$SIM_DIR/slurm_script.slurm"
        sed -i "s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=$n_cpus/" "$SIM_DIR/slurm_script.slurm"
    else 
        n_nodes=$(( n_cpus/72 )) 
        sed -i "s/^#SBATCH --nodes=.*/#SBATCH --nodes=$n_nodes/" "$SIM_DIR/slurm_script.slurm"
        sed -i "s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=72/" "$SIM_DIR/slurm_script.slurm"
    fi
    
    # 4. Run the custom_create.py script
    python3 "$SIM_DIR/custom_create.py" "$SIM_DIR"

    # 5. Check if ICs file exists
    if [ ! -f "$SIM_DIR/ics" ]; then
        echo "ics file not found in $SIM_DIR. Skipping."
        continue
    elif [ -f "$SIM_DIR/ics" ]; then
         echo "ics file successfully created! Proceeding with simulation setup..."
    fi
    
    # 6. Build Arepo
    echo "Building Arepo..."
    cd "$AREPO_DIR" || exit # go to arepo dir
    REL_SIM_DIR="./${SIM_DIR#$AREPO_DIR/}" # set variable to relative path to simulation dir
    make CONFIG="$REL_SIM_DIR/Config.sh" BUILD_DIR="$REL_SIM_DIR/build" EXEC="$REL_SIM_DIR/Arepo" # make Arepo
    
    # 7. Go to simulation directory
    cd "$SIM_DIR" || exit
    
    # 8. Check if Arepo file exists
    if [ ! -f "$SIM_DIR/Arepo" ]; then
        echo "Arepo file not found in $SIM_DIR. Skipping."
        continue
    elif [ -f "$SIM_DIR/Arepo" ]; then
         echo "Arepo binary successfully created! Proceeding with simulation..."
    fi
    
    # 9. Submit job to Slurm
    echo "Submitting job for $n_cpus cpus"
    SBATCH_OUTPUT=$(sbatch slurm_script.slurm)
    echo "[$n_cpus cpus] $SBATCH_OUTPUT" >> "$BASE_SIM_DIR/sbatch_n_cpus_higher_res_log.txt"
    
    echo "Submitted simulation for $n_cpus cpus"
    echo "-----------------------------------------------"

done