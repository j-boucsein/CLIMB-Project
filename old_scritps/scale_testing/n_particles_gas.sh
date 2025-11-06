#!/bin/bash

number_of_particles=( 8 ) # number of particles in the simulation will be 2^(this_number*3) -> 32.789 for 5

BASE_SIM_DIR="/vera/u/jerbo/arepo/run/scaling_sims_with_gas"

TEMPLATE_DIR="$BASE_SIM_DIR/template_simulation_code/cosmo_box_star_formation"
AREPO_DIR="/vera/u/jerbo/arepo"

for n_particles in "${number_of_particles[@]}"; do
    echo "=== Starting simulation for 2^($n_particles*3) particles ==="
    
    # 1. Create new simulation directory
    SIM_DIR="$BASE_SIM_DIR/sim_n${n_particles}_particles"
    mkdir -p "$SIM_DIR"

    # 2. Copy template files into it
    cp -r "$TEMPLATE_DIR"/* "$SIM_DIR"/

    # 3. Set parameters for number of particles in param_music.txt
    sed -i "s/^levelmin *= *.*/levelmin           = $n_particles/" "$SIM_DIR/param_music.txt"
    sed -i "s/^levelmin_TF *= *.*/levelmin_TF        = $n_particles/" "$SIM_DIR/param_music.txt"
    sed -i "s/^levelmax *= *.*/levelmax           = $n_particles/" "$SIM_DIR/param_music.txt"
    
    # 4. Run the custom_create.py script
    python3 "$SIM_DIR/custom_create.py" "$SIM_DIR"

    # 5. Check if ICs file exists
    if [ ! -f "$SIM_DIR/ics" ]; then
        echo "ics file not found in $SIM_DIR. Skipping."
        continue
    elif [ -f "$SIM_DIR/ics" ]; then
         echo "ics file successfully created! Proceeding with simulation setup..."
    fi
    
    # 7. Go to simulation directory
    cd "$SIM_DIR" || exit
    
    #8. Check if Arepo file exists
    if [ ! -f "$SIM_DIR/Arepo" ]; then
        echo "Arepo file not found in $SIM_DIR. Skipping."
        continue
    elif [ -f "$SIM_DIR/Arepo" ]; then
         echo "Arepo binary successfully created! Proceeding with simulation..."
    fi
    
    # 9. Submit job to Slurm
    echo "Submitting job for 2^($n_particles*3) particles"
    SBATCH_OUTPUT=$(sbatch slurm_script.slurm)
    echo "[2^($n_particles*3) particles] $SBATCH_OUTPUT" >> "$BASE_SIM_DIR/sbatch_n_particles_log.txt"
    
    echo "Submitted simulation for 2^($n_particles*3) particles"
    echo "-----------------------------------------------"

done