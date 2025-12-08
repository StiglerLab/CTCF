import numpy as np
import pandas as pd
import math
import time


# Set global constants
OVERSAMPLING_FACTOR = 10
KT = 1.38e-2 * 298 # kBT at room temperature in pN*nm
ETA = 0.89e-9      # Shear viscosity of water in pN s / nm^2
RHO = 0.99823e-21  # Density of water in pN s^2 nm^-4
NU = ETA / RHO     # kinematic viscosity


def load_db477x() -> pd.core.frame.DataFrame:  # Load filter data for NI447x filter
    return pd.read_csv("dB447x.txt", sep="\t", index_col=0, header=None, names=None)  # ni447x filter



def check_filters(filter_dict: dict, filter_list: list):
    """
    Check if filter_list is ok.
    :param filter_dict: FILTER_DICT linking the filter type to a function
    :param filter_list: list of dicts {'type': str, 'par1': p1, 'par2', p2, ...}, with parameters
    :return: f_sample, total_downsampling_factor
    """
    has_sample_stage = False
    f_sample = None
    subsample_factors = [1]

    for filter in filter_list:
        ftype = filter['type']
        typekey, *fparameters = filter.keys()
 
        if ftype not in filter_dict:
            raise Exception("Unknown filter type: "+ftype)     
        if ftype == 'sample':
            has_sample_stage = True
            f_sample = filter['frequency']
        if ftype == 'subsample':
            subsample_factors.append(fparameters['factor'])

    if not has_sample_stage:
        raise Exception("The filter description does not contain a sample stage")

    total_factor = math.prod(subsample_factors)
    if not float(total_factor).is_integer():
        raise Exception("The total downsampling factor is not integer")

    return f_sample, total_factor

    

def apply_filters(psd, filter_dict: dict, filter_list: list):
    f_sample, total_downsampling_factor = check_filters(filter_dict, filter_list)
    
    for filter in filter_list:
        ftype = filter['type']

        if ftype == 'sample':
            psd = filter_dict[ftype](psd, {'factor': total_downsampling_factor*OVERSAMPLING_FACTOR})
        else:
            psd = filter_dict[ftype](psd, filter)

    return np.sqrt(np.abs(np.trapz(psd, axis=0, x=psd.index)))

    

def apply_ni447x(row, db447x):
    if row.name < db447x.index[0]:
        row[0] = 1
    elif row.name > db447x.index[-1]:
        row[0] = 0
    else:
        res = 10 ** (db447x.iloc[db447x.index.get_loc(row.name, method='nearest')] / 20)
        row[0] = res.values[0]


def ni447x(psd, parameters):
    db447x = parameters['db447x']
    coefs_mag = psd.copy()
    coefs_mag.apply(apply_ni447x, axis=1, args=[db447x])
    psd_filtered = psd * coefs_mag ** 2
    return psd_filtered


#TODO: optimize
def bessel8(psd, parameters):
    f_cutoff = parameters['f_cutoff']
    f_mod = f_cutoff / 3.17962
    coefs_mag = psd.copy()
    div = [coef / f_mod for coef in coefs_mag.index]
    coefs = []
    for i in range(len(div)):
        coefs.append(2027025 / np.sqrt(81 * (-225225 * div[i] + 30030 * div[i] ** 3 - 770 *
                                             div[i] ** 5 + 4 * div[i] ** 7) ** 2 + (
                                               2027025 - 945945 * div[i] ** 2 + 51975 * div[i] ** 4 - 630 * div[
                                           i] ** 6 +
                                               div[i] ** 8) ** 2))

    coefs_mag = [coef ** 2 for coef in coefs]
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] *= coefs_mag
    return psd_filtered

#TODO: check #FIXME doesn't work
def boxcar_BUGGY(psd, parameters):
    """
    Boxcar filter: Average by combining n_avg samples into one
    """
    n_avg = parameters['n_avg']
    psd_filtered = psd.copy(deep=True)
    coefs_mag = psd.copy(deep=True)
    max_freq = psd.index[-1]  
    for x, val in coefs_mag.iterrows():
        try:
            coefs_mag.iloc[coefs_mag.index.get_loc(x, method='nearest')] = 1 / n_avg * np.abs(
                np.sin(x / max_freq * np.pi * n_avg / 2) / np.sin(x / max_freq * np.pi / 2))
        except FloatingPointError:  # To deal with case x = 0
            pass
    coefs_mag.iloc[coefs_mag.index.get_loc(coefs_mag.index[0], method='nearest')] = 1
    psd_filtered = psd * coefs_mag ** 2
    return psd_filtered


def boxcar(psd, parameters):
    """
    Boxcar filter: Average by combining n_avg samples into one
    """
    N = parameters['n_avg']      # number of points in the average
    max_freq = psd.index[-1]  
    fs = 2*max_freq              # sampling frequency (Hz)
    T = 1 / fs                   # sampling interval

    freqs = psd.index.values.astype(float).ravel()
    psd_vals = psd.values.ravel()

    omega = 2 * np.pi * freqs

    # Frequency response of an N-point moving average
    # |H(f)|^2 = (1/N^2) * (sin(NωT/2) / sin(ωT/2))^2

    num = np.sin(omega * T * N / 2)
    den = np.sin(omega * T / 2)

    # Handle the limit at f = 0 (value should be 1)
    H2 = np.zeros_like(freqs)
    small = np.isclose(den, 0)
    H2[small] = 1.0
    H2[~small] = (1 / N**2) * (num[~small] / den[~small])**2

    psd_out = psd_vals * H2
    return pd.Series(psd_out, index=psd.index)


def butterworth(psd, parameters):
    f_cutoff = parameters['f_cutoff']
    n_poles = parameters['n_poles']
    coefs_mag = [1 / np.sqrt(1 + (coef / f_cutoff) ** (2*n_poles)) for coef in psd.index]
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] *= coefs_mag
    return psd_filtered


def qpd(psd, parameters):
    """
    Filtering of a quadrant phododiode
    """
    gam = 0.44
    f_0 = 11.1e3
    freqs = psd.index.to_numpy()  # vectorized access
    coefs_mag = gam**2 + (1 - gam**2) / (1 + (freqs / f_0)**2)
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] = psd_filtered.iloc[:, 0].to_numpy() * coefs_mag
    return psd_filtered


def psd(psd, parameters):
    """
    Filtering of a position-sensitive device like the DL100-7
    """
    gam = 0.6
    f_0 = 26.695e3
    freqs = psd.index.to_numpy()  # vectorized access
    coefs_mag = gam**2 + (1 - gam**2) / (1 + (freqs / f_0)**2)
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] = psd_filtered.iloc[:, 0].to_numpy() * coefs_mag
    return psd_filtered


# Untested
def interpolate_psd(psd, n_downsample: int, n_0: int):
    """
    Helper function for psd_subsample()
    """
    n_downsample = int(n_downsample)
    # Make placeholders
    indices = [np.nan] * ((n_0 * n_downsample) - n_downsample + 1)
    values = [np.nan] * ((n_0 * n_downsample) - n_downsample + 1)
    # Fill data
    i = 0
    for x, val in psd.iterrows():
        indices[i * n_downsample] = x
        values[i * n_downsample] = val[0]
        i += 1
    # Make temp dataframe to interpolate indices
    temp_indices = pd.DataFrame(data=indices)
    # Interpolate indices
    temp_indices = temp_indices.interpolate()
    # Make dataframe for interpolating values
    temp_data = pd.DataFrame(data=values, index=temp_indices[0].to_list())
    temp_data = temp_data.interpolate('index')
    temp_data['f'] = temp_data.index
    temp_data.index = range(len(temp_data))
    return temp_data


# Untested
def psd_resample_down(psd, parameters):
    """
    Filtering as in Igor Pro resample operation
    """
    factor = parameters['factor']
    coefs_mag = psd.copy(deep=True)
    max_freq = psd.index[-1]
    freq_sample = max_freq * 2
    rs_coefs = load_resample_coefs(factor)
    midpoint_num = int((len(rs_coefs) - 1) / 2)
    x = [(-midpoint_num / freq_sample) + i * (1 / freq_sample) for i in range(len(rs_coefs))]
    rs_coefs.index = x
    length = int(len(psd) / 2) * 2
    rs_coefs_ser = rs_coefs.squeeze()  # Turn into a series so numpy's rfft works
    rs_coefs_fft = np.abs(np.fft.rfft(rs_coefs_ser, n=length))  # FFT + 0-padding
    indices_new = np.linspace(psd.index[0], psd.index[-1], int(length / 2) + 1)
    rs_coefs_fft = pd.DataFrame(data=rs_coefs_fft, index=indices_new)
    # Interpolate rs_coefs_fft to allow lookup for coefs_mag
    rs_coefs_fft_int = interpolate_psd(rs_coefs_fft, factor, len(rs_coefs_fft))
    rs_coefs_fft_int.index = rs_coefs_fft_int.iloc[:, 1]
    coefs = []
    for coef in coefs_mag.index:
        coefs.append(rs_coefs_fft_int.iloc[rs_coefs_fft_int.index.get_loc(coef, 'nearest'), 0])
    coefs = [coef ** 2 for coef in coefs]
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] *= coefs
    return psd_filtered


def load_resample_coefs(factor: int):
    try:
        rs_coef = pd.read_csv(f"rs_coefs/{factor}.txt", sep='\t', index_col=0, header=None, names=None)
    except FileNotFoundError:
        print(f"Can't resample with this factor: {factor}. Defaulted to 1.")
        rs_coef = load_resample_coefs(1)
    return rs_coef


#TODO: This can be optimized for speed
def psd_generate(k1, k2, k_d, f_sample_inf, beta_dagger1, beta_dagger2, mean_xi, diam1=1000, diam2=1000,
                 hydrodynamics='rp', bead=0) -> pd.core.frame.DataFrame:
    """
    Generate a PSD at "infinite" bandwidth, given by f_sample_inf
    """

    # Tested for RP + Bead=0 #FIXME
    gamma_1 = 6 * np.pi * ETA * diam1 / 2
    gamma_2 = 6 * np.pi * ETA * diam2 / 2
    n_pnts = 4096
    max_freq = f_sample_inf / 2
    freq = np.linspace(0, max_freq, num=n_pnts)

    if hydrodynamics == 'none':
        if bead == 0:
            theor_psd_calc = (2 * (2 * KT * (-2 * beta_dagger1 * beta_dagger2 * k_d * (
                gamma_2 * (k1 + k_d) + gamma_1 * (k2 + k_d)) + beta_dagger2 ** 2 * (
                gamma_1 * k_d ** 2 + gamma_2 * ((
                k1 + k_d) ** 2 + 4 * freq ** 2 * gamma_1 ** 2 * np.pi ** 2)) + beta_dagger1 ** 2 * (
                gamma_2 * k_d ** 2 + gamma_1 * ((
                k2 + k_d) ** 2 + 4 * freq ** 2 * gamma_2 ** 2 * np.pi ** 2)))) / (
                (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
        elif bead == 1:
            theor_psd_calc = (2 * (2 * beta_dagger1 ** 2 * KT * (gamma_2 * k_d ** 2 + gamma_1 * (
                (k2 + k_d) ** 2 + 4 * freq ** 2 * gamma_2 ** 2 * np.pi ** 2))) / (
                (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))

        elif bead == 2:
            theor_psd_calc = (2 * (2 * beta_dagger2 ** 2 * KT * (gamma_1 * k_d ** 2 + gamma_2 * (
                (k1 + k_d) ** 2 + 4 * freq ** 2 * gamma_1 ** 2 * np.pi ** 2))) / (
                (k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
        else:
            raise Exception(f'Invalid bead: {bead}')
        
    elif hydrodynamics == 'simple':  # Simple hydrodynamics
        r_12 = diam1 / 2 + diam2 / 2 + mean_xi
        g = 4 * np.pi * ETA * r_12
        if bead == 0:
            theor_psd_calc = (2 * (2 * g * KT * (-2 * beta_dagger1 * beta_dagger2 * (g ** 2 - gamma_1 * gamma_2) * (
                g * gamma_2 * k_d * (k1 + k_d) - gamma_1 * (
                gamma_2 * k1 * (k2 + k_d) - g * k_d * (k2 + k_d) + gamma_2 * k_d * (
                k2 + 2 * k_d))) - 8 * beta_dagger1 * beta_dagger2 * freq ** 2 * g ** 2 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 2 + beta_dagger2 ** 2 * (
                -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (
                k1 + k_d) + 2 * gamma_1 ** 2 * gamma_2 ** 2 * k_d * (
                k1 + k_d) - g * gamma_1 * gamma_2 * (
                gamma_1 * k_d ** 2 + gamma_2 * (k1 + k_d) ** 2) + g ** 3 * (
                gamma_1 * k_d ** 2 + gamma_2 * ((
                k1 + k_d) ** 2 + 4 * freq ** 2 * gamma_1 ** 2 * np.pi ** 2))) + beta_dagger1 ** 2 * (
                -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (
                k2 + k_d) + 2 * gamma_1 ** 2 * gamma_2 ** 2 * k_d * (
                k2 + k_d) - g * gamma_1 * gamma_2 * (
                gamma_2 * k_d ** 2 + gamma_1 * (k2 + k_d) ** 2) + g ** 3 * (
                gamma_2 * k_d ** 2 + gamma_1 * ((
                k2 + k_d) ** 2 + 4 * freq ** 2 * gamma_2 ** 2 * np.pi ** 2))))) / (
                (g ** 2 - gamma_1 * gamma_2) ** 2 * (
                k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * freq ** 2 * g ** 2 * (
                -4 * g * gamma_1 * gamma_2 * k_d * (
                gamma_2 * (k1 + k_d) + gamma_1 * (
                k2 + k_d)) + g ** 2 * (
                2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2) + 2 * gamma_1 ** 2 * gamma_2 ** 2 * (
                k1 * (k2 + k_d) + k_d * (
                k2 + 2 * k_d))) * np.pi ** 2 + 16 * freq ** 4 * g ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
            
        elif bead == 1:
            theor_psd_calc = (2 * (2 * beta_dagger1 ** 2 * g * KT * (
                -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (k2 + k_d) + 2 * (gamma_1 * gamma_2) ** 2 * k_d * (
                k2 + k_d) - g * gamma_1 * gamma_2 * (
                gamma_2 * k_d ** 2 + gamma_1 * (k2 + k_d) ** 2) + g ** 3 * (
                gamma_2 * k_d ** 2 + gamma_1 * (
                (k2 + k_d) ** 2 + 4 * freq ** 2 * gamma_2 ** 2 * np.pi ** 2)))) / (
                (gamma_1 * gamma_2) ** 2 * (k2 * k_d + k1 * (
                k2 + k_d)) ** 2 - 16 * freq ** 2 * g ** 3 * gamma_1 * gamma_2 * k_d * (
                gamma_2 * (k1 + k_d) + gamma_1 * (
                k2 + k_d)) * np.pi ** 2 + 2 * g ** 2 * gamma_1 * gamma_2 * (
                -(k2 * k_d + k1 * (
                k2 + k_d)) ** 2 + 4 * freq ** 2 * gamma_1 * gamma_2 * (
                k1 * (k2 + k_d) + k_d * (
                k2 + 2 * k_d)) * np.pi ** 2) + g ** 4 * (
                (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * (
                gamma_1 * gamma_2) ** 2 * np.pi ** 4)))

        elif bead == 2:
            theor_psd_calc = (2 * (2 * beta_dagger2 ** 2 * g * KT * (
                -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (k1 + k_d) + 2 * (gamma_1 * gamma_2) ** 2 * k_d * (
                k1 + k_d) - g * gamma_1 * gamma_2 * (
                gamma_1 * k_d ** 2 + gamma_2 * (k1 + k_d) ** 2) + g ** 3 * (
                gamma_1 * k_d ** 2 + gamma_2 * (
                (k1 + k_d) ** 2 + 4 * freq ** 2 * gamma_1 ** 2 * np.pi ** 2)))) / (
                (gamma_1 * gamma_2) ** 2 * (k1 * k_d + k2 * (
                k1 + k_d)) ** 2 - 16 * freq ** 2 * g ** 3 * gamma_1 * gamma_2 * k_d * (
                gamma_1 * (k2 + k_d) + gamma_2 * (
                k1 + k_d)) * np.pi ** 2 + 2 * g ** 2 * gamma_1 * gamma_2 * (
                -(k1 * k_d + k2 * (
                k1 + k_d)) ** 2 + 4 * freq ** 2 * gamma_1 * gamma_2 * (
                k2 * (k1 + k_d) + k_d * (
                k1 + 2 * k_d)) * np.pi ** 2) + g ** 4 * (
                (k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_1 ** 2 * (
                k1 + k_d) ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * (
                gamma_1 * gamma_2) ** 2 * np.pi ** 4)))
        else:
            raise Exception(f'Invalid bead: {bead}')
            
    elif hydrodynamics == 'rp':  # Rotne-Prager
        a1 = diam1 / 2
        a2 = diam2 / 2
        a = (a1 + a2) / 2
        r = a1 + a2 + mean_xi
        g = gamma_1 / ((6 * a) / (4 * r) - a ** 3 / r ** 3)
        v_b = beta_dagger1 * beta_dagger2
        v_g = gamma_1 * gamma_2 
        if bead == 0:
            theor_psd_calc = (2 * (2 * g * KT * (-2 * v_b * (g ** 2 - v_g) * (
                g * gamma_2 * k_d * (k1 + k_d) - gamma_1 * (
                gamma_2 * k1 * (k2 + k_d) - g * k_d * (k2 + k_d) + gamma_2 * k_d * (
                k2 + 2 * k_d))) - 8 * v_b * freq ** 2 * g ** 2 * v_g ** 2 * np.pi ** 2 + beta_dagger2 ** 2 * (
                -2 * g ** 2 * v_g * k_d * (
                k1 + k_d) + 2 * v_g ** 2 * k_d * (
                k1 + k_d) - g * v_g * (
                gamma_1 * k_d ** 2 + gamma_2 * (
                k1 + k_d) ** 2) + g ** 3 * (
                gamma_1 * k_d ** 2 + gamma_2 * ((
                k1 + k_d) ** 2 + 4 * freq ** 2 * gamma_1 ** 2 * np.pi ** 2))) + beta_dagger1 ** 2 * (
                -2 * g ** 2 * v_g * k_d * (
                k2 + k_d) + 2 * v_g ** 2 * k_d * (
                k2 + k_d) - g * v_g * (
                gamma_2 * k_d ** 2 + gamma_1 * (
                k2 + k_d) ** 2) + g ** 3 * (
                gamma_2 * k_d ** 2 + gamma_1 * ((
                k2 + k_d) ** 2 + 4 * freq ** 2 * gamma_2 ** 2 * np.pi ** 2))))) / (
                (g ** 2 - v_g) ** 2 * (
                k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * freq ** 2 * g ** 2 * (
                -4 * g * v_g * k_d * (gamma_2 * (k1 + k_d) + gamma_1 * (
                k2 + k_d)) + g ** 2 * (
                2 * v_g * k_d ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2) + 2 * v_g ** 2 * (
                k1 * (k2 + k_d) + k_d * (
                k2 + 2 * k_d))) * np.pi ** 2 + 16 * freq ** 4 * g ** 4 * v_g ** 2 * np.pi ** 4))

        elif bead == 1:
            theor_psd_calc = (2 * (2 * beta_dagger1 ** 2 * g * KT * (
                -2 * g ** 2 * v_g * k_d * (k2 + k_d) + 2 * v_g ** 2 * k_d * (k2 + k_d) - g * v_g * (
                gamma_2 * k_d ** 2 + gamma_1 * (k2 + k_d) ** 2) + g ** 3 * (
                gamma_2 * k_d ** 2 + gamma_1 * (
                (k2 + k_d) ** 2 + 4 * freq ** 2 * gamma_2 ** 2 * np.pi ** 2)))) / (v_g ** 2 * (
                k2 * k_d + k1 * (k2 + k_d)) ** 2 - 16 * freq ** 2 * g ** 3 * v_g * k_d * (gamma_2 * (
                k1 + k_d) + gamma_1 * (k2 + k_d)) * np.pi ** 2 + 2 * g ** 2 * v_g * (-(k2 * k_d + k1 * (
                k2 + k_d)) ** 2 + 4 * freq ** 2 * v_g * (k1 * (k2 + k_d) + k_d * (
                k2 + 2 * k_d)) * np.pi ** 2) + g ** 4 * ((k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * v_g * k_d ** 2 + gamma_2 ** 2 * (k1 + k_d) ** 2 + gamma_1 ** 2 * (
                k2 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * v_g ** 2 * np.pi ** 4)))

        elif bead == 2:
            theor_psd_calc = (2 * (2 * beta_dagger2 ** 2 * g * KT * (
                -2 * g ** 2 * v_g * k_d * (k1 + k_d) + 2 * v_g ** 2 * k_d * (k1 + k_d) - g * v_g * (
                gamma_1 * k_d ** 2 + gamma_2 * (k1 + k_d) ** 2) + g ** 3 * (
                gamma_1 * k_d ** 2 + gamma_2 * (
                (k1 + k_d) ** 2 + 4 * freq ** 2 * gamma_1 ** 2 * np.pi ** 2)))) / (v_g ** 2 * (
                k1 * k_d + k2 * (k1 + k_d)) ** 2 - 16 * freq ** 2 * g ** 3 * v_g * k_d * (gamma_1 * (
                k2 + k_d) + gamma_2 * (k1 + k_d)) * np.pi ** 2 + 2 * g ** 2 * v_g * (-(k1 * k_d + k2 * (
                k1 + k_d)) ** 2 + 4 * freq ** 2 * v_g * (k2 * (k1 + k_d) + k_d * (
                k1 + 2 * k_d)) * np.pi ** 2) + g ** 4 * ((k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * freq ** 2 * (
                2 * v_g * k_d ** 2 + gamma_1 ** 2 * (k2 + k_d) ** 2 + gamma_2 ** 2 * (
                k1 + k_d) ** 2) * np.pi ** 2 + 16 * freq ** 4 * v_g ** 2 * np.pi ** 4)))
        else:
            raise Exception(f'Invalid bead: {bead}')


    return pd.DataFrame(data=theor_psd_calc, index=freq)




def psd_subsample(psd, parameters):
    downsamplingfactor = parameters['factor']

    psd_arr = psd.iloc[:, 0].to_numpy()
    n_orig = len(psd_arr)
    n_per_block = n_orig // downsamplingfactor
    x = np.arange(n_per_block)

    # Even blocks (i=0,2,4...)
    even_i = np.arange(0, downsamplingfactor, 2)
    even_blocks = psd_arr[even_i[:, None]*n_per_block + x[None, :]]
    signal_even = even_blocks.sum(axis=0)

    # Odd blocks (i=1,3,5...)
    odd_i = np.arange(1, downsamplingfactor, 2)
    odd_blocks = psd_arr[(odd_i[:, None]+1)*n_per_block - 1 - x[None, :]]
    signal_odd = odd_blocks.sum(axis=0)

    # Combine signals
    signal = signal_even + signal_odd

    # Correct frequency axis
    freqs_new = np.linspace(psd.index[0], psd.index[-1] / downsamplingfactor, n_per_block)

    psd_ss = pd.DataFrame({psd.columns[0]: signal}, index=freqs_new)
   
    return psd_ss

