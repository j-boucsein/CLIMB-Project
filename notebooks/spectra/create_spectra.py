#!/vera/u/jerbo/CLIMB-Project/env/bin/python3.12
import temet
import h5py
import os

sim_suite_location = "/vera/u/jerbo/my_ptmp/L25n256_suite"
number_of_gridpoints = 50
gps_to_do_spectra_of = []
for i in range(number_of_gridpoints):
    path = sim_suite_location + f"/gridpoint{i}"
    gps_to_do_spectra_of.append(path)

for current_gp in gps_to_do_spectra_of:
    # print(current_gp)
    sim = temet.sim(current_gp, redshift=2.0)
    x = temet.cosmo.spectrum.generate_rays_voronoi_fullbox(sim, nRaysPerDim=100)
    x = temet.cosmo.spectrum.generate_spectra_from_saved_rays(sim, ion='H I', instrument='SDSS-BOSS', nRaysPerDim=100)