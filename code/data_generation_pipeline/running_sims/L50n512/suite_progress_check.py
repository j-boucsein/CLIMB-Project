import os

# location of the simulation output directories
suite_location = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite"
n_gridpoints = 50
final_snapdir = "snapdir_012"  # this one means the run is done

not_started = []
running = []
finished_with_restart = []
finished_without_restart = []

for gp in range(n_gridpoints):

    gridpoint_dir = f"{suite_location}/gridpoint{gp}"

    if not os.path.isdir(gridpoint_dir):
        print(f"gridpoint{gp}: folder does not exist")
        continue

    content = os.listdir(gridpoint_dir)
    snapdirs = sorted(entry for entry in content if entry.startswith("snapdir_"))

    if len(content) == 0:
        not_started.append(gp)
    elif final_snapdir in snapdirs:
        if "restartfiles" in content:
            finished_with_restart.append(gp)
        else:
            finished_without_restart.append(gp)
    elif len(snapdirs) > 0:
        running.append(gp)
        print(f"gridpoint{gp}: running, {len(snapdirs)} snapdirs, latest {snapdirs[-1]}")
    else:
        # files are there, but no snapshot has been written yet
        running.append(gp)
        print(f"gridpoint{gp}: started, but no snapdir yet")

print()
print(f"suite: {suite_location}")
print(f"finished (with restartfiles):    {len(finished_with_restart):3d}   {finished_with_restart}")
print(f"finished (without restartfiles): {len(finished_without_restart):3d}   {finished_without_restart}")
print(f"running:                         {len(running):3d}   {running}")
print(f"not started:                     {len(not_started):3d}   {not_started}")
