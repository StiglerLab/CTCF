import numpy as np


def correct_linker_soft_trap(force_combined, dist, k1_app, k2_app, beta_dagger1, beta_dagger2, k_dagger1, k_dagger2, width1, width2):
    """ Compute correction due to miscalibration and non-linear traps """
    
    kc_app = 1/(1/k1_app + 1/k2_app)
    k_dagger = (k_dagger1/k1_app + k_dagger2/k2_app)/(1/kc_app)
    beta_dagger = (beta_dagger1*k2_app*k_dagger1 + beta_dagger2*k1_app*k_dagger2)/(k2_app*k_dagger1+k1_app*k_dagger2)
    ext_corr = np.zeros(len(dist))
    dist_corr = np.zeros(len(dist))
    f_corr = np.zeros(len(dist))
    defl_corr1 = np.zeros(len(dist))
    defl_corr2 = np.zeros(len(dist))
    for i in range(len(dist)):
        kappa_1 = np.pi * force_combined[i]*k_dagger1/(2*beta_dagger*k_dagger*k1_app*width1)
        kappa_2 = np.pi * force_combined[i]*k_dagger2/(2*beta_dagger*k_dagger*k2_app*width2)
        x_soft1 = 2 * width1/np.pi * np.arcsin(kappa_1)
        x_soft2 = 2 * width2/np.pi * np.arcsin(kappa_2)

        ext_linker = dist[i] - x_soft1 - x_soft2
        f_true = force_combined[i]/(beta_dagger*k_dagger)

        defl_corr1[i] = x_soft1
        defl_corr2[i] = x_soft2
        ext_corr[i] = ext_linker
        f_corr[i] = f_true

    return ext_corr, f_corr, defl_corr1, defl_corr2
