import numpy as np
import subprocess
import os
import csv
import shutil
import subprocess
import re

sim_suite_location = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite"
number_of_gridpoints = 50
gps_to_reorder = []
for i in range(number_of_gridpoints):
    path = sim_suite_location + f"/gridpoint{i}"
    gps_to_reorder.append(path)

gps_to_reorder.append(sim_suite_location + f"/reference")  # add the reference run

# create the output folders
for current_gp in gps_to_reorder:
    output_dir = current_gp + "/output"
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print(f"[{output_dir}] already exists")

# check if all outputfolders were created sucessfully
error_occured = False
for current_gp in gps_to_reorder:
    content = os.listdir(current_gp)
    if 'output' not in content:
        print(f"Error: no output folder in [{current_gp}]")
        error_occured = True

assert not error_occured, "make sure everything prior worked before continuing"

# safely move all snaps and groups into output folder 

for current_gp in gps_to_reorder:
    snapdirs_to_move = []
    groups_to_move = []
    
    # src = "/vera/u/jerbo/my_ptmp/L25n128_suite/gridpoint0/snapdir_000/"
    # dst = "/vera/u/jerbo/my_ptmp/L25n128_suite/gridpoint0/output/snapdir_000/"
    print(current_gp)
    for filename in os.listdir(current_gp):
        if re.search("snapdir_*", filename):
            snapdirs_to_move.append(filename)
        if re.search("groups_*", filename):
            groups_to_move.append(filename)
    
    for snap in snapdirs_to_move:
        src = f"{current_gp}/{snap}/"
        dst = f"{current_gp}/output/{snap}/"
        # -a = archive (erhält Rechte, Zeiten, Symlinks)
        # -v = verbose
        # --remove-source-files = löscht Quelldateien nach erfolgreichem Kopieren
        cmd = ["rsync", "-av", "--remove-source-files", src, dst]
        subprocess.run(cmd, check=True)

    for group in groups_to_move:
        src = f"{current_gp}/{group}/"
        dst = f"{current_gp}/output/{group}/"
        # -a = archive (erhält Rechte, Zeiten, Symlinks)
        # -v = verbose
        # --remove-source-files = löscht Quelldateien nach erfolgreichem Kopieren
        cmd = ["rsync", "-av", "--remove-source-files", src, dst]
        subprocess.run(cmd, check=True)

# move text files to output/txt-files directory
for current_gp in gps_to_reorder:
    src = f"{current_gp}"
    dst = f"{current_gp}/output/txt-files"
    
    # Zielverzeichnis erstellen, falls es noch nicht existiert
    os.makedirs(dst, exist_ok=True)
    
    # Alle Elemente im Quellverzeichnis prüfen
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
    
        # Nur reguläre Dateien verschieben (keine Unterordner!)
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)

# move blackhole_details files to output/txt-files directory
for current_gp in gps_to_reorder:
    src = f"{current_gp}/blackhole_details"
    dst = f"{current_gp}/output/txt-files/blackhole_details"
    
    # Zielverzeichnis erstellen, falls es noch nicht existiert
    os.makedirs(dst, exist_ok=True)
    
    # Alle Elemente im Quellverzeichnis prüfen
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
    
        # Nur reguläre Dateien verschieben (keine Unterordner!)
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)

# move blackhole_mergers files to output/txt-files directory
for current_gp in gps_to_reorder:
    src = f"{current_gp}/blackhole_mergers"
    dst = f"{current_gp}/output/txt-files/blackhole_mergers"
    
    # Zielverzeichnis erstellen, falls es noch nicht existiert
    os.makedirs(dst, exist_ok=True)
    
    # Alle Elemente im Quellverzeichnis prüfen
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
    
        # Nur reguläre Dateien verschieben (keine Unterordner!)
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)

# delete empty folders
subprocess.run(["find", sim_suite_location, "-type", "d", "-empty", "-delete"], check=True)