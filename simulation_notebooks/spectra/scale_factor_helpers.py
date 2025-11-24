import numpy as np
from scipy.optimize import brentq


def a_matter_lambda(t, Omega_m, Omega_Lambda, h):
    H_0 = 100 * h * 1/(3.086*1e19)  # in 1/s
    a = np.power(Omega_m/Omega_Lambda * np.sinh((3*H_0*np.sqrt(Omega_Lambda)*t)/2)**2 , 1/3)
    return a


def t_matter_lambda(a, Omega_m, Omega_Lambda, h):
    H_0 = 100 * h * 1/(3.086*1e19)  # in 1/s
    t = 2/(3*H_0*np.sqrt(Omega_Lambda)) * np.arcsinh(np.sqrt(Omega_Lambda/Omega_m * a**3))
    return t


def delta_z(d_c, z_start, Omega_m, Omega_Lambda, h):
    """
    d_c (float)     : distance in cMpc/h
    a_start (float) : scale factor of box
    """
    a_start = 1/(z_start + 1)
    c = 299792.458  # in km/s
    t_tr = (d_c/h)*a_start/c *10**6 *3.086*10**13  # in s
    t_start = t_matter_lambda(a_start, Omega_m, Omega_Lambda, h)

    d_z = 1/a_matter_lambda(t_start-t_tr, Omega_m, Omega_Lambda, h) - 1/a_start 
    return d_z


def corrected_z(d_c, z_start, Omega_m, Omega_Lambda, h):
    """
    d_c (float)     : distance in cMpc/h
    a_start (float) : scale factor of box
    """
    a_start = 1/(z_start + 1)
    c = 299792.458  # in km/s
    t_tr = (d_c/h)*a_start/c *10**6 *3.086*10**13  # in s
    t_start = t_matter_lambda(a_start, Omega_m, Omega_Lambda, h)

    corr_z = 1/a_matter_lambda(t_start-t_tr, Omega_m, Omega_Lambda, h) - 1
    return corr_z


def t_box_rootfind(t_box, d_c, z_start, Omega_m, Omega_Lambda, h):
    c = 299792.458  # in km/s
    d_c = d_c * 3.086*1e19  # in km
    t_2 = t_matter_lambda(1/(z_start+1), Omega_m, Omega_Lambda, h)
    return d_c/c * a_matter_lambda(t_2 - t_box, Omega_m, Omega_Lambda, h) - t_box


def solve_for_correct_t_box(d_c, z_start, Omega_m, Omega_Lambda, h):
    c = 299792.458  # in km/s
    d_c_correct = d_c/h
    t_start = d_c_correct/(c*(z_start+1)) *10**6 *3.086*10**13  # in s

    t_box = brentq(t_box_rootfind, t_start*0.98, t_start, args=(d_c_correct, z_start, Omega_m, Omega_Lambda, h))

    return t_box


def corrected_z_exact(d_c, z_start, Omega_m, Omega_Lambda, h):

    t_2 = t_matter_lambda(1/(z_start+1), Omega_m, Omega_Lambda, h)

    t_box = solve_for_correct_t_box(d_c, z_start, Omega_m, Omega_Lambda, h)

    return 1/a_matter_lambda(t_2 - t_box, Omega_m, Omega_Lambda, h) - 1 


if __name__ == "__main__":
    #t_vals = np.logspace(0, 18, 1000)
    z_start = 2
    d_c = 25
    Omega_m = 0.3
    Omega_Lambda = 0.7
    h = 0.7
    lambda_HI = 1215.67

    exact_z = corrected_z_exact(d_c, z_start, Omega_m, Omega_Lambda, h)
    print(exact_z)

    #dz = corrected_z(d_c, z_start, Omega_m, Omega_Lambda, h)

    #dlambda = lambda_HI * dz

    #print(dlambda, dz)

    #estimated_error = 100*(d_c*1/(z_start-dz+1) - d_c*1/(z_start+1))/d_c  # in percent

    #print(f"Estimated error: {estimated_error:.2f}%")



    #plt.plot(t_vals, a_matter_lambda(t_vals, Omega_m, Omega_Lambda, h), c="r")
    #plt.savefig("plots/scale_factor_plot.pdf", format="PDF")