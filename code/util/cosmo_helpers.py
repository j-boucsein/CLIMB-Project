import numpy as np


def redshift_wavelength_forward(z, wavelength):
    """ Redshifts a wavelength forward (wavelength gets bigger with more redshift)
    
    Args:
        z (float): redshift to apply to wavelength
        wavelength (float or np.array): wavelength to be shifted

    Returns:
        (float or np.array): redshifted wavelength
    """
    return (z+1)*wavelength


def redshift_wavelength_backward(z, wavelength):
    """ Redshifts a wavelength backward (wavelength gets smaller with more redshift)
    
    Args:
        z (float): redshift to apply to wavelength
        wavelength (float or np.array): wavelength to be shifted

    Returns:
        (float or np.array): redshifted wavelength
    """
    return wavelength/(z+1)


def a_matter_lambda(t, Omega_m, Omega_Lambda, h):
    """function for scale factor in matter and lambda dominated universe

    Args:
        t (float): time at which the scale factor should be computed
        Omega_m (float): Density parameter for all matter (dark matter and baryonic matter)
        Omega_Lambda (float): Density parameter for dark energy
        h (float): hubble parameter 

    Returns:
        float: scale factor
    """
    H_0 = 100 * h * 1/(3.086*1e19)  # in 1/s
    a = np.power(Omega_m/Omega_Lambda * np.sinh((3*H_0*np.sqrt(Omega_Lambda)*t)/2)**2 , 1/3)
    return a


def t_matter_lambda(a, Omega_m, Omega_Lambda, h):
    """function for time dependend on scale factor in matter and lambda dominated universe

    Args:
        a (float): scale factor
        Omega_m (float): Density parameter for all matter (dark matter and baryonic matter)
        Omega_Lambda (float): Density parameter for dark energy
        h (float): hubble parameter 

    Returns:
        float: time in seconds
    """
    H_0 = 100 * h * 1/(3.086*1e19)  # in 1/s
    t = 2/(3*H_0*np.sqrt(Omega_Lambda)) * np.arcsinh(np.sqrt(Omega_Lambda/Omega_m * a**3))
    return t


def delta_z(d_c, z_start, Omega_m, Omega_Lambda, h):
    """compute by how much light traveling through a box with size d_c gets redshifted within that box,
       if that box sits at redshift z_start

    Args:
        d_c (float): distance in cMpc/h
        z_start (float): initial redshift of box
        Omega_m (float): Density parameter for all matter (dark matter and baryonic matter)
        Omega_Lambda (float): Density parameter for dark energy
        h (float): hubble parameter 

    Returns:
        float: redshift
    """
    a_start = 1/(z_start + 1)
    c = 299792.458  # in km/s
    t_tr = (d_c/h)*a_start/c *10**6 *3.086*10**13  # in s
    t_start = t_matter_lambda(a_start, Omega_m, Omega_Lambda, h)

    d_z = 1/a_matter_lambda(t_start-t_tr, Omega_m, Omega_Lambda, h) - 1/a_start 
    return d_z


def corrected_z(d_c, z_start, Omega_m, Omega_Lambda, h):
    """compute by how much light traveling through a box with size d_c gets redshifted in total,
       if that box sits at redshift z_start 

    Args:
        d_c (float): distance in cMpc/h
        z_start (float): initial redshift of box
        Omega_m (float): Density parameter for all matter (dark matter and baryonic matter)
        Omega_Lambda (float): Density parameter for dark energy
        h (float): hubble parameter 

    Returns:
        float: redshift
    """
    a_start = 1/(z_start + 1)
    c = 299792.458  # in km/s
    t_tr = (d_c/h)*a_start/c *10**6 *3.086*10**13  # in s
    t_start = t_matter_lambda(a_start, Omega_m, Omega_Lambda, h)

    corr_z = 1/a_matter_lambda(t_start-t_tr, Omega_m, Omega_Lambda, h) - 1
    return corr_z


