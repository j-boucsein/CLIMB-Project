#!/bin/bash

BASE_SIM_DIR="/vera/u/jerbo/arepo/run/grid_search_with_gas"

TEMPLATE_DIR="$BASE_SIM_DIR/templates/cosmo_box"

AREPO_DIR="/vera/u/jerbo/arepo"

# OUTPUT_DIR_BASE="/vera/ptmp/gc/jerbo/"

grid_point=0

while read -r OmegaLambda Omegab HubbleParam Omega0; do
    
    ((counter++))
    echo "=== Starting simulation for Grid Point $counter ==="
    
    # 1. Create new simulation directory
    SIM_DIR="$BASE_SIM_DIR/grid_point_${counter}"
    mkdir -p "$SIM_DIR"

    # 2. Copy template files into it simdir
    cp -r "$TEMPLATE_DIR"/* "$SIM_DIR"/

    # 3. Set parameters in param_music.txt
    
    # calculate H0 from HubbleParam (music expects H0 while arepo expects h)
    HubbleConst=$(echo "$HubbleParam * 100" | bc -l)
    
    sed -i "s/^Omega_m *= *.*/Omega_m            = $Omega0/" "$SIM_DIR/param_music.txt"
    sed -i "s/^Omega_L *= *.*/Omega_L            = $OmegaLambda/" "$SIM_DIR/param_music.txt"
    sed -i "s/^Omega_b *= *.*/Omega_b           = $Omegab/" "$SIM_DIR/param_music.txt"
    sed -i "s/^H0 *= *.*/H0                 = $HubbleConst/" "$SIM_DIR/param_music.txt"
    
    # 4. Set parameters in param.txt  
    sed -i "s/^Omega0 *.*/Omega0                                $Omega0/" "$SIM_DIR/param.txt"
    sed -i "s/^OmegaLambda *.*/OmegaLambda                           $OmegaLambda/" "$SIM_DIR/param.txt"
    sed -i "s/^OmegaBaryon *.*/OmegaBaryon                             $Omegab/" "$SIM_DIR/param.txt"
    sed -i "s/^HubbleParam *.*/HubbleParam                           $HubbleParam/" "$SIM_DIR/param.txt"
    
    # Does not work for some reason???
    # Output_dir="$OUTPUT_DIR_BASE/grid_point_${counter}/output"
    
    # sed -i "s/^OutputDir *.*/OutputDir                               $Output_dir/" "$SIM_DIR/param.txt"
    
    # 4. Run the custom_create.py script
    echo "Creating the ics ..."
    python3 "$SIM_DIR/custom_create.py" "$SIM_DIR"

    # 5. Check if ICs file exists
    if [ ! -f "$SIM_DIR/ics" ]; then
        echo "ics file not found in $SIM_DIR. Skipping."
        continue
    elif [ -f "$SIM_DIR/ics" ]; then
         echo "ics file successfully created!"
    fi
    
    # 7. Check if Arepo file exists
    if [ ! -f "$SIM_DIR/Arepo" ]; then
        echo "Arepo file not found in $SIM_DIR. Skipping this Grid Point."
        continue
    fi
    
    # 8. Go to simulation directory
    cd "$SIM_DIR" || exit
    
    # 9. Submit job to Slurm
    echo "Submitting job for Grid Point $counter"
    SBATCH_OUTPUT=$(sbatch slurm_script.slurm)
    echo "Grid Point $counter : $SBATCH_OUTPUT" >> "$BASE_SIM_DIR/slurm_ids_log.txt"
    
    echo "Submitted simulation for Grid Point $counter"
    echo "-----------------------------------------------"
    
done < "full_random_grid.txt"