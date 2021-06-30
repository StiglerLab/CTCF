import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set global constants
KT = 1.38e-2 * 298
ETA = 0.89e-9
RHO = 0.99823e-21  # Water
NU = ETA / RHO  # kinematic viscosity

filter_params = {
    "ni447x": "db447x",
    "bessel8": "f_cutoff",
    "boxcar": "n_avg",
    "butterworth": "f_cutoff",
    "psd_resample_down": "factor",
    "psd_subsample": "n_downsample",
    "sample": "n_downsample",  # alias for psd_subsample
    "ss": "n_downsample"  # alias for psd_subsample
}







def load_db477x() -> pd.core.frame.DataFrame:  # Load filter data for NI447x filter
    return pd.read_csv("dB447x.txt", sep="\t", index_col=0, header=None, names=None)  # ni447x filter


def make_filter_params(db447x, n_downsample: int = 1, n_0: int = 4096, n_poles: int = 1, f_cutoff: int = 10000,
                       n_avg: int = 1, factor: int = 1):
    """
    Function to turn filter parameters into a dictionary that is later read by the parse_filter function
    :param db447x: dataframe of ni447x filter data
    :param n_downsample: downsampling factor
    :param n_0:
    :param n_poles:
    :param f_cutoff:
    :param n_avg:
    :param factor:
    :return:
    """
    return {
        "n_downsample": n_downsample,
        "n_0": n_0,
        "n_poles": n_poles,
        "f_cutoff": f_cutoff,
        "n_avg": n_avg,
        "factor": factor,
        "db447x": db447x
    }


def parse_filter(psd, parameters: dict, filter_dict: dict, filter_string: str = "", sep=";"):
    filters = filter_string.split(sep=sep)
    if len(filters) != 1 and filters[0] != '':
        for filter_name in filters:
            psd = filter_dict[filter_name](psd, parameters)
    return np.sqrt(np.abs(np.trapz(psd, axis=0, x=psd.index)))  # Gets correct area - CHECKED


def ni447x(psd, parameters):
    # Filters input PSD using NI-447x filter
    db447x = parameters["db447x"]
    coefs_mag = psd.copy(deep=True)  # !
    for x, val in psd.iterrows():  # TODO: this can probably be sped up
        if x < db447x.index[0]:
            coefs_mag.iloc[coefs_mag.index.get_loc(x, method='nearest')] = 1  # fancy lookup for float indices
        elif x > db447x.index[-1]:
            coefs_mag.iloc[coefs_mag.index.get_loc(x, method='nearest')] = 0
        else:
            res = 10 ** (db447x.iloc[db447x.index.get_loc(x, method='nearest')] / 20)
            coefs_mag.iloc[coefs_mag.index.get_loc(x, method='nearest')] = res.values[0]

    psd_filtered = psd * coefs_mag ** 2
    print(psd_filtered)
    return psd_filtered


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


def boxcar(psd, parameters):  # TODO: seems to work but compare with IGOR; might also create NaNs maybe
    n_avg = parameters['n_avg']
    psd_filtered = psd.copy(deep=True)
    coefs_mag = psd.copy(deep=True)
    max_freq = psd.index[-1]  # I think
    for x, val in coefs_mag.iterrows():
        try:
            coefs_mag.iloc[coefs_mag.index.get_loc(x, method='nearest')] = 1 / n_avg * np.abs(
                np.sin(x / max_freq * np.pi * n_avg / 2) / np.sin(x / max_freq * np.pi / 2))
        except FloatingPointError:  # To deal with case x = 0
            pass
    coefs_mag.iloc[coefs_mag.index.get_loc(coefs_mag.index[0], method='nearest')] = 1
    psd_filtered = psd * coefs_mag ** 2  # TODO: rewrite as in bessel8 or maybe not; seems to work ok
    print(psd_filtered)
    return psd_filtered


def butterworth(psd, parameters):  # TODO: Test, but should be fine now
    # n_poles = parameters['n_poles']
    f_cutoff = parameters['f_cutoff']
    coefs_mag = [1 / np.sqrt(1 + (coef / f_cutoff) ** 2) for coef in psd.index]
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] *= coefs_mag
    return psd_filtered


def qpd(psd, parameters):
    gam = 0.44
    f_0 = 11.1e3
    coefs_mag = [gam ** 2 + ((1 - gam ** 2) / (1 + (coef / f_0) ** 2)) for coef in psd.index]
    psd_filtered = psd.copy(deep=True)
    psd_filtered.iloc[:, 0] *= coefs_mag
    return psd_filtered


def interpolate_psd(psd, n_downsample: int, n_0: int):
    """
    Helper function for psd_subsample()
    :return:
    """
    # Make placeholders
    indices = [np.NaN] * ((n_0 * n_downsample) - n_downsample + 1)
    values = [np.NaN] * ((n_0 * n_downsample) - n_downsample + 1)
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


def psd_resample_down(psd, parameters):
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
    # TODO: This works but looks horrible
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


def psd_generate(k1, k2, k_d, f_sample_inf, beta_dagger1, beta_dagger2, mean_xi, diam1=1000, diam2=1000,
                 hydrodynamics='hansen rp', bead=0) -> pd.core.frame.DataFrame:
    # Tested for Hansen RP + Bead=0
    gamma_1 = 6 * np.pi * ETA * diam1 / 2
    gamma_2 = 6 * np.pi * ETA * diam2 / 2
    n_pnts = 4096
    max_freq = f_sample_inf / 2
    theor_psd = np.linspace(0, max_freq, num=n_pnts)
    theor_psd_calc = []
    if hydrodynamics == 'none':
        if bead == 0:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * KT * (-2 * beta_dagger1 * beta_dagger2 * k_d * (
                        gamma_2 * (k1 + k_d) + gamma_1 * (k2 + k_d)) + beta_dagger2 ** 2 * (
                                                             gamma_1 * k_d ** 2 + gamma_2 * ((
                                                                                                     k1 + k_d) ** 2 + 4 * x ** 2 * gamma_1 ** 2 * np.pi ** 2)) + beta_dagger1 ** 2 * (
                                                             gamma_2 * k_d ** 2 + gamma_1 * ((
                                                                                                     k2 + k_d) ** 2 + 4 * x ** 2 * gamma_2 ** 2 * np.pi ** 2)))) / (
                                              (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * (
                                              2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                                              k1 + k_d) ** 2 + gamma_1 ** 2 * (
                                                      k2 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
        elif bead == 1:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * beta_dagger1 ** 2 * KT * (gamma_2 * k_d ** 2 + gamma_1 * (
                        (k2 + k_d) ** 2 + 4 * x ** 2 * gamma_2 ** 2 * np.pi ** 2))) / (
                                              (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * (
                                              2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                                              k1 + k_d) ** 2 + gamma_1 ** 2 * (
                                                      k2 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
        elif bead == 2:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * beta_dagger2 ** 2 * KT * (gamma_1 * k_d ** 2 + gamma_2 * (
                        (k1 + k_d) ** 2 + 4 * x ** 2 * gamma_1 ** 2 * np.pi ** 2))) / (
                                              (k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * x ** 2 * (
                                              2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_1 ** 2 * (
                                              k2 + k_d) ** 2 + gamma_2 ** 2 * (
                                                      k1 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
        else:
            print(f'Invalid bead: {bead}.')
            return -1
    elif hydrodynamics == 'simple':  # Simple hydrodynamics
        r_12 = diam1 / 2 + diam2 / 2 + mean_xi
        g = 4 * np.pi * ETA * r_12
        if bead == 0:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * g * KT * (
                        -2 * beta_dagger1 * beta_dagger2 * (g ** 2 - gamma_1 * gamma_2) * (
                        g * gamma_2 * k_d * (k1 + k_d) - gamma_1 * (
                        gamma_2 * k1 * (k2 + k_d) - g * k_d * (k2 + k_d) + gamma_2 * k_d * (
                        k2 + 2 * k_d))) - 8 * beta_dagger1 * beta_dagger2 * x ** 2 * g ** 2 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 2 + beta_dagger2 ** 2 * (
                                -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (
                                k1 + k_d) + 2 * gamma_1 ** 2 * gamma_2 ** 2 * k_d * (
                                        k1 + k_d) - g * gamma_1 * gamma_2 * (
                                        gamma_1 * k_d ** 2 + gamma_2 * (k1 + k_d) ** 2) + g ** 3 * (
                                        gamma_1 * k_d ** 2 + gamma_2 * ((
                                                                                k1 + k_d) ** 2 + 4 * x ** 2 * gamma_1 ** 2 * np.pi ** 2))) + beta_dagger1 ** 2 * (
                                -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (
                                k2 + k_d) + 2 * gamma_1 ** 2 * gamma_2 ** 2 * k_d * (
                                        k2 + k_d) - g * gamma_1 * gamma_2 * (
                                        gamma_2 * k_d ** 2 + gamma_1 * (k2 + k_d) ** 2) + g ** 3 * (
                                        gamma_2 * k_d ** 2 + gamma_1 * ((
                                                                                k2 + k_d) ** 2 + 4 * x ** 2 * gamma_2 ** 2 * np.pi ** 2))))) / (
                                              (g ** 2 - gamma_1 * gamma_2) ** 2 * (
                                              k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * g ** 2 * (
                                                      -4 * g * gamma_1 * gamma_2 * k_d * (
                                                      gamma_2 * (k1 + k_d) + gamma_1 * (
                                                      k2 + k_d)) + g ** 2 * (
                                                              2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                                                              k1 + k_d) ** 2 + gamma_1 ** 2 * (
                                                                      k2 + k_d) ** 2) + 2 * gamma_1 ** 2 * gamma_2 ** 2 * (
                                                              k1 * (k2 + k_d) + k_d * (
                                                              k2 + 2 * k_d))) * np.pi ** 2 + 16 * x ** 4 * g ** 4 * gamma_1 ** 2 * gamma_2 ** 2 * np.pi ** 4))
        elif bead == 1:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * beta_dagger1 ** 2 * g * KT * (
                        -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (k2 + k_d) + 2 * (gamma_1 * gamma_2) ** 2 * k_d * (
                        k2 + k_d) - g * gamma_1 * gamma_2 * (
                                gamma_2 * k_d ** 2 + gamma_1 * (k2 + k_d) ** 2) + g ** 3 * (
                                gamma_2 * k_d ** 2 + gamma_1 * (
                                (k2 + k_d) ** 2 + 4 * x ** 2 * gamma_2 ** 2 * np.pi ** 2)))) / (
                                              (gamma_1 * gamma_2) ** 2 * (k2 * k_d + k1 * (
                                              k2 + k_d)) ** 2 - 16 * x ** 2 * g ** 3 * gamma_1 * gamma_2 * k_d * (
                                                      gamma_2 * (k1 + k_d) + gamma_1 * (
                                                      k2 + k_d)) * np.pi ** 2 + 2 * g ** 2 * gamma_1 * gamma_2 * (
                                                      -(k2 * k_d + k1 * (
                                                              k2 + k_d)) ** 2 + 4 * x ** 2 * gamma_1 * gamma_2 * (
                                                              k1 * (k2 + k_d) + k_d * (
                                                              k2 + 2 * k_d)) * np.pi ** 2) + g ** 4 * (
                                                      (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * (
                                                      2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_2 ** 2 * (
                                                      k1 + k_d) ** 2 + gamma_1 ** 2 * (
                                                              k2 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * (
                                                              gamma_1 * gamma_2) ** 2 * np.pi ** 4)))
        elif bead == 2:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * beta_dagger2 ** 2 * g * KT * (
                        -2 * g ** 2 * gamma_1 * gamma_2 * k_d * (k1 + k_d) + 2 * (gamma_1 * gamma_2) ** 2 * k_d * (
                        k1 + k_d) - g * gamma_1 * gamma_2 * (
                                gamma_1 * k_d ** 2 + gamma_2 * (k1 + k_d) ** 2) + g ** 3 * (
                                gamma_1 * k_d ** 2 + gamma_2 * (
                                (k1 + k_d) ** 2 + 4 * x ** 2 * gamma_1 ** 2 * np.pi ** 2)))) / (
                                              (gamma_1 * gamma_2) ** 2 * (k1 * k_d + k2 * (
                                              k1 + k_d)) ** 2 - 16 * x ** 2 * g ** 3 * gamma_1 * gamma_2 * k_d * (
                                                      gamma_1 * (k2 + k_d) + gamma_2 * (
                                                      k1 + k_d)) * np.pi ** 2 + 2 * g ** 2 * gamma_1 * gamma_2 * (
                                                      -(k1 * k_d + k2 * (
                                                              k1 + k_d)) ** 2 + 4 * x ** 2 * gamma_1 * gamma_2 * (
                                                              k2 * (k1 + k_d) + k_d * (
                                                              k1 + 2 * k_d)) * np.pi ** 2) + g ** 4 * (
                                                      (k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * x ** 2 * (
                                                      2 * gamma_1 * gamma_2 * k_d ** 2 + gamma_1 ** 2 * (
                                                      k1 + k_d) ** 2 + gamma_2 ** 2 * (
                                                              k1 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * (
                                                              gamma_1 * gamma_2) ** 2 * np.pi ** 4)))
        else:
            print(f'Invalid bead: {bead}.')
            return -1
    elif hydrodynamics == 'hansen rp':  # Hansen, Rotne-Prager
        a1 = diam1 / 2
        a2 = diam2 / 2
        a = (a1 + a2) / 2
        r = a1 + a2 + mean_xi
        g = gamma_1 / ((6 * a) / (4 * r) - a ** 3 / r ** 3)
        v_b = beta_dagger1 * beta_dagger2
        v_g = gamma_1 * gamma_2  # Not sure why this was a wave in IGOR
        if bead == 0:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * g * KT * (-2 * v_b * (g ** 2 - v_g) * (
                        g * gamma_2 * k_d * (k1 + k_d) - gamma_1 * (
                        gamma_2 * k1 * (k2 + k_d) - g * k_d * (k2 + k_d) + gamma_2 * k_d * (
                        k2 + 2 * k_d))) - 8 * v_b * x ** 2 * g ** 2 * v_g ** 2 * np.pi ** 2 + beta_dagger2 ** 2 * (
                                                                 -2 * g ** 2 * v_g * k_d * (
                                                                 k1 + k_d) + 2 * v_g ** 2 * k_d * (
                                                                         k1 + k_d) - g * v_g * (
                                                                         gamma_1 * k_d ** 2 + gamma_2 * (
                                                                         k1 + k_d) ** 2) + g ** 3 * (
                                                                         gamma_1 * k_d ** 2 + gamma_2 * ((
                                                                                                                 k1 + k_d) ** 2 + 4 * x ** 2 * gamma_1 ** 2 * np.pi ** 2))) + beta_dagger1 ** 2 * (
                                                                 -2 * g ** 2 * v_g * k_d * (
                                                                 k2 + k_d) + 2 * v_g ** 2 * k_d * (
                                                                         k2 + k_d) - g * v_g * (
                                                                         gamma_2 * k_d ** 2 + gamma_1 * (
                                                                         k2 + k_d) ** 2) + g ** 3 * (
                                                                         gamma_2 * k_d ** 2 + gamma_1 * ((
                                                                                                                 k2 + k_d) ** 2 + 4 * x ** 2 * gamma_2 ** 2 * np.pi ** 2))))) / (
                                              (g ** 2 - v_g) ** 2 * (
                                              k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * g ** 2 * (
                                                      -4 * g * v_g * k_d * (gamma_2 * (k1 + k_d) + gamma_1 * (
                                                      k2 + k_d)) + g ** 2 * (
                                                              2 * v_g * k_d ** 2 + gamma_2 ** 2 * (
                                                              k1 + k_d) ** 2 + gamma_1 ** 2 * (
                                                                      k2 + k_d) ** 2) + 2 * v_g ** 2 * (
                                                              k1 * (k2 + k_d) + k_d * (
                                                              k2 + 2 * k_d))) * np.pi ** 2 + 16 * x ** 4 * g ** 4 * v_g ** 2 * np.pi ** 4))
        elif bead == 1:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * beta_dagger1 ** 2 * g * KT * (
                        -2 * g ** 2 * v_g * k_d * (k2 + k_d) + 2 * v_g ** 2 * k_d * (k2 + k_d) - g * v_g * (
                        gamma_2 * k_d ** 2 + gamma_1 * (k2 + k_d) ** 2) + g ** 3 * (
                                gamma_2 * k_d ** 2 + gamma_1 * (
                                (k2 + k_d) ** 2 + 4 * x ** 2 * gamma_2 ** 2 * np.pi ** 2)))) / (v_g ** 2 * (
                        k2 * k_d + k1 * (k2 + k_d)) ** 2 - 16 * x ** 2 * g ** 3 * v_g * k_d * (gamma_2 * (
                        k1 + k_d) + gamma_1 * (k2 + k_d)) * np.pi ** 2 + 2 * g ** 2 * v_g * (-(k2 * k_d + k1 * (
                        k2 + k_d)) ** 2 + 4 * x ** 2 * v_g * (k1 * (k2 + k_d) + k_d * (
                        k2 + 2 * k_d)) * np.pi ** 2) + g ** 4 * ((k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * (
                        2 * v_g * k_d ** 2 + gamma_2 ** 2 * (k1 + k_d) ** 2 + gamma_1 ** 2 * (
                        k2 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * v_g ** 2 * np.pi ** 4)))
        elif bead == 2:
            for x in theor_psd:
                theor_psd_calc.append(2 * (2 * beta_dagger2 ** 2 * g * KT * (
                        -2 * g ** 2 * v_g * k_d * (k1 + k_d) + 2 * v_g ** 2 * k_d * (k1 + k_d) - g * v_g * (
                        gamma_1 * k_d ** 2 + gamma_2 * (k1 + k_d) ** 2) + g ** 3 * (
                                gamma_1 * k_d ** 2 + gamma_2 * (
                                (k1 + k_d) ** 2 + 4 * x ** 2 * gamma_1 ** 2 * np.pi ** 2)))) / (v_g ** 2 * (
                        k1 * k_d + k2 * (k1 + k_d)) ** 2 - 16 * x ** 2 * g ** 3 * v_g * k_d * (gamma_1 * (
                        k2 + k_d) + gamma_2 * (k1 + k_d)) * np.pi ** 2 + 2 * g ** 2 * v_g * (-(k1 * k_d + k2 * (
                        k1 + k_d)) ** 2 + 4 * x ** 2 * v_g * (k2 * (k1 + k_d) + k_d * (
                        k1 + 2 * k_d)) * np.pi ** 2) + g ** 4 * ((k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * x ** 2 * (
                        2 * v_g * k_d ** 2 + gamma_1 ** 2 * (k2 + k_d) ** 2 + gamma_2 ** 2 * (
                        k1 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * v_g ** 2 * np.pi ** 4)))
        else:
            print(f'Invalid bead: {bead}.')
            return -1
    else:  # Hansen, SM
        a1 = diam1/2
        a2 = diam2/2
        a = (a1+a2)/2
        r = diam1 / 2 + diam2 / 2 + mean_xi
        tau_a1 = a1 ** 2 / NU
        tau_a2 = a2 ** 2 / NU
        y = np.linspace(0, max_freq, n_pnts)
        alpha1 = np.sqrt(-2*np.pi*y*tau_a1*np.complex(0,1))/a1
        alpha2 = np.sqrt(-2*np.pi*y*tau_a2*np.complex(0,1))/a2
        alpha = [(alpha1[i]+alpha2[i])/2 for i in range(len(alpha1))]
        gamma1_self = [gamma_1 * (1 + alpha1[i]**2*a1**2/9) for i in range(len(alpha1))]
        gamma2_self = [gamma_2 * (1 + alpha2[i]**2*a2**2/9) for i in range(len(alpha2))]
        gamma_cross = [gamma_1 / ((3 * a) / (4 * r) * np.exp(-alpha[i] * r) * (
            1 + 5 / 9 * alpha[i] ** 2 * a ** 2 + 1 / 6 * alpha[i] ** 3 * a ** 3) * 2 - (1 / 3 - 1) * (
                                    a ** 3 / r ** 3 + 9 * a / (2 * alpha[i] ** 2 * r ** 3) - (
                                    (5 * alpha[i] ** 2 * a ** 2 + 9) * (
                                    alpha[i] ** 2 * r ** 2 + 2 * alpha[i] * r + 2) * a) / (
                                            4 * alpha[i] ** 2 * r ** 3) * np.exp(-alpha[i] * r))) for i in range(len(alpha1))]

        gamma_cross[0] = gamma_cross[1]
        v_b = beta_dagger1*beta_dagger2
        vgc = np.multiply(gamma1_self, gamma2_self)
        if bead == 1:
            for i in range(len(theor_psd)):
                x = theor_psd[i]
                theor_psd_calc.append(
            2 * (2 * beta_dagger1 ** 2 * gamma_cross[i] * KT * (
                        -2 * gamma_cross[i] ** 2 * vgc[i] * k_d * (k2 + k_d) + 2 * vgc[i] ** 2 * k_d * (k2 + k_d) - gamma_cross[i] * vgc[i] * (
                            gamma2_self[i] * k_d ** 2 + gamma1_self[i] * (k2 + k_d) ** 2) + gamma_cross[i] ** 3 * (
                                    gamma2_self[i] * k_d ** 2 + gamma1_self[i] * ((k2 + k_d) ** 2 + 4 * x ** 2 * gamma2_self[i] ** 2 * np.pi ** 2)))) / (
                        vgc[i] ** 2 * (k2 * k_d + k1 * (k2 + k_d)) ** 2 - 16 * x ** 2 * gamma_cross[i] ** 3 * vgc[i] * k_d * (
                            gamma2_self[i] * (k1 + k_d) + gamma1_self[i] * (k2 + k_d)) * np.pi ** 2 + 2 * gamma_cross[i] ** 2 * vgc[i] * (
                                    -(k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * vgc[i] * (
                                        k1 * (k2 + k_d) + k_d * (k2 + 2 * k_d)) * np.pi ** 2) + gamma_cross[i] ** 4 * (
                                    (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * (
                                        2 * vgc[i] * k_d ** 2 + gamma2_self[i] ** 2 * (k1 + k_d) ** 2 + gamma1_self[i] ** 2 * (
                                            k2 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * vgc[i] ** 2 * np.pi ** 4)))
        elif bead == 2:
            for i in range(len(theor_psd)):
                x = theor_psd[i]
                theor_psd_calc.append(
                 2 * (2 * beta_dagger2 ** 2 * gamma_cross[i] * KT * (
                            -2 * gamma_cross[i] ** 2 * vgc[i] * k_d * (k1 + k_d) + 2 * vgc[i] ** 2 * k_d * (k1 + k_d) - gamma_cross[i] * vgc[i] * (
                                gamma1_self[i] * k_d ** 2 + gamma2_self[i] * (k1 + k_d) ** 2) + gamma_cross[i] ** 3 * (
                                        gamma1_self[i] * k_d ** 2 + gamma2_self[i] * ((k1 + k_d) ** 2 + 4 * x ** 2 * gamma1_self[i] ** 2 * np.pi ** 2)))) / (
                                               vgc[i] ** 2 * (
                                                   k1 * k_d + k2 * (k1 + k_d)) ** 2 - 16 * x ** 2 * gamma_cross[i] ** 3 * vgc[i] * k_d * (
                                                           gamma1_self[i] * (k2 + k_d) + gamma2_self[i] * (
                                                               k1 + k_d)) * np.pi ** 2 + 2 * gamma_cross[i] ** 2 * vgc[i] * (
                                                           -(k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * x ** 2 * vgc[i] * (
                                                               k2 * (k1 + k_d) + k_d * (
                                                                   k1 + 2 * k_d)) * np.pi ** 2) + gamma_cross[i] ** 4 * (
                                                           (k1 * k_d + k2 * (k1 + k_d)) ** 2 + 4 * x ** 2 * (
                                                               2 * vgc[i] * k_d ** 2 + gamma1_self[i] ** 2 * (
                                                                   k2 + k_d) ** 2 + gamma2_self[i] ** 2 * (
                                                                           k1 + k_d) ** 2) * np.pi ** 2 + 16 * x ** 4 * vgc[i] ** 2 * np.pi ** 4)))
        else:
            for i in range(len(theor_psd)):
                x = theor_psd[i]
                theor_psd_calc.append(2 * (2 * gamma_cross[i] * KT * (-2 * v_b * (gamma_cross[i] **2 - vgc[i]) * (gamma_cross[i] * gamma2_self[i] * k_d * (k1 + k_d) - gamma1_self[i] * (
                        gamma2_self[i] * k1 * (k2 + k_d) - gamma_cross[i] * k_d * (k2 + k_d) + gamma2_self[i] * k_d * (
                        k2 + 2 * k_d))) - 8 * v_b * x ** 2 * gamma_cross[i] ** 2 * vgc[i][i] ** 2 * np.pi ** 2 + beta_dagger2 ** 2 * (
                                                                              -2 * gamma_cross[i] ** 2 * vgc[i][i] * k_d * (k1 + k_d) + 2 * vgc[i][i] ** 2 * k_d * (
                                                                              k1 + k_d) - gamma_cross[i] * vgc[i][i] * (
                                                                                      gamma1_self[i] * k_d ** 2 + gamma2_self[i] * (k1 + k_d) ** 2) + gamma_cross[i] ** 3 * (
                                                                                      gamma1_self[i] * k_d ** 2 + gamma2_self[i] * ((
                                                                                                                                            k1 + k_d) ** 2 + 4 * x ** 2 * gamma1_self[i] ** 2 * np.pi ** 2))) + beta_dagger1 ** 2 * (
                                                                              -2 * gamma_cross[i] ** 2 * vgc[i][i] * k_d * (k2 + k_d) + 2 * vgc[i][i] ** 2 * k_d * (
                                                                              k2 + k_d) - gamma_cross[i] * vgc[i][i] * (
                                                                                      gamma2_self[i] * k_d ** 2 + gamma1_self[i] * (k2 + k_d) ** 2) + gamma_cross[i] ** 3 * (
                                                                                      gamma2_self[i] * k_d ** 2 + gamma1_self[i] * (
                                                                                      (k2 + k_d) ** 2 + 4 * x ** 2 * gamma2_self[i] ** 2 * np.pi ** 2))))) / (
                                              (gamma_cross[i] ** 2 - vgc[i][i]) ** 2 * (k2 * k_d + k1 * (k2 + k_d)) ** 2 + 4 * x ** 2 * gamma_cross[i] ** 2 * (
                                              -4 * gamma_cross[i] * vgc[i][i] * k_d * (gamma2_self[i] * (k1 + k_d) + gamma1_self[i] * (k2 + k_d)) + gamma_cross[i] ** 2 * (
                                              2 * vgc[i][i] * k_d ** 2 + gamma2_self[i] ** 2 * (k1 + k_d) ** 2 + gamma1_self[i] ** 2 * (
                                              k2 + k_d) ** 2) + 2 * vgc[i][i] ** 2 * (k1 * (k2 + k_d) + k_d * (
                                              k2 + 2 * k_d))) * np.pi ** 2 + 16 * x ** 4 * gamma_cross[i] ** 4 * vgc[i][i] ** 2 * np.pi ** 4))
        theor_psd_calc = np.real(theor_psd_calc)
    return pd.DataFrame(data=theor_psd_calc, index=theor_psd)


def psd_subsample(psd, parameters):  # TODO: compare exact numbers to IGOR, otherwise this seems to work
    n_downsample = parameters['n_downsample']
    psd0 = psd.copy(deep=True)
    psd0['f'] = psd.index
    psd0.index = range(len(psd))
    psd_interpol = psd.copy(deep=True)
    psd_ss = psd.copy(deep=True)
    n_0 = len(psd)
    psd_interpol = interpolate_psd(psd_interpol, n_downsample, n_0)
    max_freq_ind = int(len(psd_interpol) / n_downsample)
    signal = [0] * len(psd0)
    for i in range(0, n_downsample, 2):
        f_start = i * max_freq_ind
        for x in range(len(psd_ss)):
            signal[x] += psd_interpol.iat[x + f_start, 0]
    for i in range(1, n_downsample, 2):
        f_start = (i + 1) * max_freq_ind
        for x in range(len(psd_ss)):
            signal[x] += psd_interpol.iat[f_start - x, 0]
    psd_ss[0] = signal
    psd_ss.index /= n_downsample
    return psd_ss


def read_filter(filter_string):
    filter_list = filter_string.lower().split(";")
    filters = [x.split(',')[0] for x in filter_list]
    paramaters = [x.split(',')[1] for x in filter_list]
    param_dict = {}
    for i, filter in enumerate(filters):
        (filter_params[filter])
        param_dict[filter_params[filter]] = paramaters[i]
    return ";".join(filters), param_dict





if __name__ == '__main__':
    #psd1 = psd_generate(0.9958863333266993, 0.9958863333266993, 0.1776581135500805, 100000, 0.9958863333266993,
     #                   0.9958863333266993, 467.9337734050282,bead=1)
    pass
