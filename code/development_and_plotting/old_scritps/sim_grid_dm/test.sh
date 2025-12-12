#!/bin/bash
counter=0
while read -r OmegaLambda Omegab HubbleParam Omega0; do
    ((counter++))
    if  [ ! -z "$HubbleParam" ];  then
        echo "=== Starting simulation for Grid Point $counter ==="
        echo $HubbleParam
    fi
    
    
done < "../sim_grid_gas/full_random_grid_rerun.txt"