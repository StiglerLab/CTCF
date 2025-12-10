import psd_filter
import nonlinear_trap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, minimize
import time


# Dictionary for parsing filters; append keys and filter functions to add custom filters
FILTER_DICT = {'qpd': psd_filter.filter_qpd,
               'psd': psd_filter.filter_psd,
               'bessel8': psd_filter.filter_bessel8,
               'butterworth': psd_filter.filter_butterworth,
               'sample': psd_filter.psd_subsample,
               'boxcar': psd_filter.filter_boxcar,
               'igorresample': psd_filter.filter_psd_resample_down,
               'ni447x': psd_filter.filter_ni447x,
               "subsample": psd_filter.psd_subsample}


class Trace:
    def __init__(self):
        #--- Settings
        self.hydrodynamics = 'rp'     # Hydrodynamics correction mode. Options: 'none', 'simple', 'rp'
        self.plot = True              # Toggle plotting of fit progress     
        
        #--- Measured FDC data
        self.force = None             # Uncorrected force (pN); must be set for correction
        self.dist = None              # Distance (nm) corresponding to the force data; must be set for correction
        self.stdev = None             # Standard deviation of total deflection signal (nm); must be set for correction
                                     
        self.force_mob = None         # Uncorrected force (pN) from mobile trap
        self.force_fix = None         # Uncorrected force (pN) from fixed trap
        self.stdev_mob = None         # Standard deviation of deflection signal in mobile trap (nm)
        self.stdev_fix = None         # Standard deviation of deflection signal in fixed trap (nm)

        self.mask = None              # Masking array: same size as force, dist, etc. NaN values are excluded from fit

        
        #--- Measured parameters
        self.bead_diameter_mob = 1000 # Diameter of bead in mobile trap (nm)
        self.bead_diameter_fix = 1000 # Diameter of bead in fixed trap (nm)

        self.k1_app = np.nan          # Apparent spring constant of mobile trap before correction (pN/nm)
        self.k2_app = np.nan          # Apparent spring constant of fixed trap before correction (pN/nm)

        #--- Outputs
        self.corrected = False        # Flag to indicate whether correction of the traces has been done
        self.kc_app = 1 / (1 / self.k1_app + 1 / self.k2_app)  # Apparent combined spring constant
        self.k_dagger1 = 1.0          # Miscalibration factor of k_mobile
        self.k_dagger2 = 1.0          # Miscalibration factor of k_fixed
        self.k_dagger = (self.k_dagger1 / self.k1_app + self.k_dagger2 / self.k2_app) / (1 / self.kc_app)  # Combined miscalibration factor of k_c
        self.beta_dagger1 = 1.0       # Miscalibration factor of the nm/V calibration of mobile trap
        self.beta_dagger2 = 1.0       # Miscalibration factor of the nm/V calibration of fixed trap
        self.beta_dagger = ((self.beta_dagger1 * self.k2_app * self.k_dagger1 + self.beta_dagger2 * self.k1_app * self.k_dagger2) /
                            (self.k2_app * self.k_dagger1 + self.k1_app * self.k_dagger2))  # Combined miscalibration factor of the nm/V calibration
        self.width1 = 800             # Non-harmonicity parameter mobile trap (nm)
        self.width2 = 800             # Non-harmonicity parmaeter fixed trap (nm)

        self.ext_orig = None          # Original extension data (nm)
        self.ext_corr = None          # Corrected extension data (nm)
        self.force_corr = None        # Corrected force data (pN)
        self.force_corr_mob = None    # Corrected force data (pN)
        self.force_corr_fix = None    # Corrected force data (pN)
       

        #--- Internal stack for troubleshooting      
        self.psd_orig = None    # PSD before filtering
        self.calc_sigma = None  # Calculated noise
        self.calc_sigma_mob = None
        self.calc_sigma_fix = None
        self.ratio_fitted = None       # Ratio sigma_fit / sigma_exp
        self.ratio_fitted_mob = None
        self.ratio_fitted_fix = None

        self.filters = []  # Filters applied to signal
        self.fit_counter = 0  # Keep track of fit iterations

        
    def calc_theor_sigma_var_kc(self, force, dist, k1_app, k2_app, beta_dagger1, beta_dagger2, k_dagger1, k_dagger2,
                                width1, width2, bead):
        """
        Calculates theoretical sigma as part of the fitting routine.
        Also stores corrected data.
        :param force: Force data calculated from the combination of both traps (pN)
        :param dist:  Distance data (nm)
        :param k1_app: Apparent (incorrect) k for trap 1 (pN/nm)
        :param k2_app: Apparent (incorrect) k for trap 2 (pN/nm)
        :param beta_dagger1: Correction factor beta_dagger for trap 1
        :param beta_dagger2: Correction factor beta_dagger for trap 2
        :param k_dagger1: Correction factor k_dagger for trap 1
        :param k_dagger2: Correction factor k_dagger for trap 2
        :param width1: Non-harmonicity parmeter for trap 1 (nm)
        :param width2: Non-harmonicity parmeter for trap 2 (nm)
        :param bead: Bead selection for calculation; default = 0: both beads, only used for storing data where it belongs
        :return: Theoretical noise sigma
        """
        if len(force) != len(dist):
            raise Exception("Length of force data is not identical to dist data")
        
        kc_app = 1 / (1 / k1_app + 1 / k2_app)
        k_dagger = (k_dagger1 / k1_app + k_dagger2 / k2_app) / (1 / kc_app)
        beta_dagger = (beta_dagger1 * k2_app * k_dagger1 + beta_dagger2 * k1_app * k_dagger2) / (
                    k2_app * k_dagger1 + k1_app * k_dagger2)
        ext_corr, f_corr, defl_corr1, defl_corr2 = nonlinear_trap.correct_linker_soft_trap(force, dist, k1_app, k2_app,
                                                                                           beta_dagger1, beta_dagger2,
                                                                                           k_dagger1, k_dagger2, width1,
                                                                                           width2)
        if np.isnan(defl_corr1).any() or np.isnan(defl_corr2).any():
            return 1e7*np.ones_like(force)

        def central_diff(y, x):    
            dydx = np.zeros_like(y)
            dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
            dydx[0] = (y[1] - y[0]) / (x[1] - x[0]) # fwd diff
            dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) # bwd diff
            return dydx

        df_dx = central_diff(f_corr, ext_corr)

        f_sample, total_downsampling_factor = psd_filter.check_filters(FILTER_DICT, self.filters)
        f_generate = psd_filter.OVERSAMPLING_FACTOR*f_sample

        calc_sigma = np.zeros_like(force)
        kappa1 = np.pi * force * k_dagger1/(2*beta_dagger * k_dagger * k1_app * width1)
        kappa2 = np.pi * force * k_dagger2/(2*beta_dagger * k_dagger * k2_app * width2)
        k1_eff = k1_app / k_dagger1 * np.sqrt(1 - (kappa1**2))
        k2_eff = k2_app / k_dagger2 * np.sqrt(1 - (kappa2**2))
        beta_dagger_total1 = beta_dagger1 * np.sqrt(1 - (kappa1**2))
        beta_dagger_total2 = beta_dagger2 * np.sqrt(1 - (kappa2**2))

        psd_orig = psd_filter.psd_generate(k1_eff, k2_eff, df_dx, f_generate, beta_dagger_total1, beta_dagger_total2,
                                                   ext_corr, self.bead_diameter_mob, self.bead_diameter_fix, self.hydrodynamics, bead)
        calc_sigma = psd_filter.apply_filters(psd_orig, FILTER_DICT, self.filters)
        
        self.ext_orig = dist - force / self.kc_app
 
        self.force_corr = f_corr.copy()
        self.ext_corr = dist - defl_corr1 - defl_corr2
        # abs(F_N(x)) = 2*k_0*w/pi * sin(pi*x/(2*w)) [eq. 19]:
        self.force_corr_mob = 2*(self.k1_app/self.k_dagger1)*self.width1/np.pi * np.sin(np.pi*defl_corr1/(2*self.width1))
        self.force_corr_fix = 2*(self.k2_app/self.k_dagger2)*self.width2/np.pi * np.sin(np.pi*defl_corr2/(2*self.width2))

        if bead==0:
            self.calc_sigma = calc_sigma
        if bead==1:
            self.calc_sigma_mob = calc_sigma
        if bead==2:
            self.calc_sigma_fix = calc_sigma
                   
        if np.isnan(calc_sigma).any():  # heavy penalization if NaNs are generated; prevents abort by Fit Error
            return 1e7*np.ones_like(force)
        return np.array(calc_sigma)

    
    def correct(self):
        """
        Main method that's called for correction of miscalibration after data loading.
        Calls compute_residuals in minimizer for correction.
        """
        print("\nSettings:")
        print("------------------------------------------------")
        print(f"k1_app (pN/nm): {self.k1_app:.3f}, k2_app (pN/nm): {self.k2_app:.3f}")
        print(f"diam1 (nm):     {self.bead_diameter_mob:4d},  diam2 (nm):     {self.bead_diameter_fix:4d}")
        print(f"hydrodynamics:  {self.hydrodynamics}")
        print(f"filters:        {self.filters}\n")
        
 
        if self.force is None or self.dist is None:
            raise Exception("No FDC")

        #Check if we have mobile and fix data. If so, we can do a global fit (much better!)
        if self.force_mob is not None and self.force_fix is not None and self.stdev_mob is not None and self.stdev_fix is not None:
            global_fit = True
        else:
            global_fit = False

        #global_fit = False
 
        if global_fit:
            print("Running global fit to sum, mob, fix...")
            stdev_data = np.vstack([self.stdev, self.stdev_mob, self.stdev_fix])
            force_data = np.vstack([self.force, self.force_mob, self.force_fix])
        else:
            print("Fitting sum signal only...")
            stdev_data = np.vstack([self.stdev])
            force_data = np.vstack([self.force])
           
        same_length = all(len(a) == len(stdev_data[0]) for a in stdev_data) and \
            all(len(a) == len(force_data[0]) for a in force_data) and \
            len(stdev_data[0]) == len(force_data[0])
        if not same_length:
            raise Exception("Data not same length")

        # Masking
        if self.mask is None:
            force_data_masked = force_data
            stdev_data_masked = stdev_data
            dist_masked = self.dist
        else:
            if len(self.mask) != len(force_data[0]):
                raise Exception("Mask not same length")
            force_data_masked = force_data[:,~np.isnan(self.mask)]
            stdev_data_masked = stdev_data[:,~np.isnan(self.mask)]
            dist_masked = self.dist[~np.isnan(self.mask)]
        
        # Create parameters
        fit_params = Parameters()
        fit_params.add('k1_app', value=self.k1_app, vary=0)
        fit_params.add('k2_app', value=self.k2_app, vary=0)
        fit_params.add('logbeta_dagger1', value=np.log(self.beta_dagger1), min=np.log(0.5), max=np.log(2))
        fit_params.add('logbeta_dagger2', value=np.log(self.beta_dagger2), min=np.log(0.5), max=np.log(2))
        fit_params.add('logk_dagger1', value=np.log(self.k_dagger1), min=np.log(0.5), max=np.log(2))
        fit_params.add('logk_dagger2', value=np.log(self.k_dagger2), min=np.log(0.5), max=np.log(2))
        fit_params.add('width1', value=self.width1, min=20, max=5000)
        fit_params.add('width2', value=self.width2, min=20, max=5000)
        for i, y in enumerate(force_data):
            fit_params.add(f'bead_{i}', value=i, vary=0)

        result = minimize(self.compute_residuals, fit_params, args=(self.dist, stdev_data, force_data, self.mask))#,max_nfev=10)
        print("\nDone with correction. Corrected trace.")
            
        self.beta_dagger1 = 10**result.params['logbeta_dagger1'].value
        self.beta_dagger2 = 10**result.params['logbeta_dagger2'].value
        self.k_dagger1 = 10**result.params['logk_dagger1'].value
        self.k_dagger2 = 10**result.params['logk_dagger2'].value
        self.width1 = result.params['width1'].value
        self.width2 = result.params['width2'].value

        # Update self.k_dagger1, self.k_dagger2, self.beta_dagger1 and self.beta_dagger2 with fit values
        self.kc_app = 1 / (1 / self.k1_app + 1 / self.k2_app)
        self.k_dagger = (self.k_dagger1 / self.k1_app + self.k_dagger2 / self.k2_app) / (1 / self.kc_app)
        self.beta_dagger = (self.beta_dagger1 * self.k2_app * self.k_dagger1 + self.beta_dagger2 *
                            self.k1_app * self.k_dagger2) / (self.k2_app * self.k_dagger1 + self.k1_app * self.k_dagger2)

        print("\nMiscalibration factors (1=mob, 2=fix):")
        print("------------------------------------------------")
        print(f"beta_dagger1: {self.beta_dagger1:.3f},   beta_dagger2: {self.beta_dagger2:.3f}")
        print(f"k_dagger1:    {self.k_dagger1:.3f},   k_dagger2:    {self.k_dagger2:.3f}")
        print(f"width1 (nm): {self.width1:6.1f},   width2 (nm): {self.width2:6.1f}")
        print(f"beta_dagger:  {self.beta_dagger:.3f},   k_dagger:     {self.k_dagger:.3f}")

        # Re-calculate from final solution
        self.compute_residuals(result.params, self.dist, stdev_data, force_data, quiet=True)
        self.corrected = True

        # Show result after correction
        if self.plot:
            plt.figure()
            plt.plot(self.ext_orig, self.force, 'ko', mfc='white', label='1+2 orig')
            plt.plot(self.ext_corr, self.force_corr, 'ko', mfc='k', label='1+2 corr')
            if global_fit:
                plt.plot(self.ext_orig, self.force_mob, 'go', mfc='white', label='1 orig')
                plt.plot(self.ext_corr, self.force_corr_mob, 'go', mfc='g', label='1 corr')
                plt.plot(self.ext_orig, self.force_fix, 'ro', mfc='white', label='2 orig')
                plt.plot(self.ext_corr, self.force_corr_fix, 'ro', mfc='r', label='2 corr')
                
            plt.xlabel('Extension (nm)')
            plt.ylabel('Force (pN)')
            plt.title('Correction result')
            plt.legend()
            plt.show()
            plt.ioff()




    def compute_residuals(self, params, dist, stdev_data, force_data, mask=None, quiet=False):
        """
        Calculate total residual to minimize
        Calls calc_theor_sigma_var_kc
        """
        
        ndata, nx = stdev_data.shape
        resid = np.zeros_like(stdev_data)
        # One residual per sum, mob, fix
        for i in range(ndata):
            #force_data[0,:] below is correct. We calculate the theoretical sum, mob and fix only based on the sum signal
            calc_sigma = self.calc_theor_sigma_var_kc(force_data[0,:], dist,
                                                      params['k1_app'].value, params['k2_app'].value,
                                                      10**params['logbeta_dagger1'].value, 10**params['logbeta_dagger2'].value,
                                                      10**params['logk_dagger1'].value, 10**params['logk_dagger2'].value,
                                                      params['width1'].value, params['width2'].value,
                                                      params[f'bead_{i}'].value)
            calc_sigma = np.asanyarray(calc_sigma).reshape(-1)
            ratio = stdev_data[i,:] / calc_sigma.ravel()

            resid[i,:] = stdev_data[i,:] - calc_sigma
            #resid[i,:] = np.log(stdev_data[i,:] / calc_sigma)
 
            # If masking is on: don't show ratio or calc_sigma for the masked points, and (FURTHER BELOW), remove the residuals
            if mask is not None:
                calc_sigma[np.isnan(mask)] = np.nan
                ratio[np.isnan(mask)] = np.nan
            
            if i==0: #sum
                self.ratio_fitted = ratio
                self.calc_stdev = calc_sigma
                if self.plot:
                    plt.ion()
                    plt.clf()
                    plt.subplot(211)
                    plt.plot(dist, np.log2(self.ratio_fitted), 'ko', mfc='white')
                    plt.ylabel('log2(Measured/Fitted)')
                    plt.axhline(y=0, color='k')
                    low, high = plt.ylim()
                    bound = max(abs(low), abs(high))
                    plt.ylim(-bound, bound)                 
                    plt.title("Residuals")
                    plt.subplot(212)
                    plt.plot(dist, stdev_data[i,:], 'ko', mfc='white', label='1+2 Measured')
                    plt.plot(dist, self.calc_stdev, 'k-',label='1+2 Fitted')
                    plt.xlabel('Distance (nm)')
                    plt.ylabel(r'$\sigma$ (nm)')
                    plt.title("Noise fitting")
            if i==1: #mob
                self.ratio_fitted_mob = ratio
                self.calc_stdev_mob = calc_sigma
                if self.plot:
                    plt.subplot(211)
                    plt.plot(dist, np.log2(self.ratio_fitted_mob), 'go', mfc='white')
                    plt.subplot(212)
                    plt.plot(dist, stdev_data[i,:], 'go', mfc='white', label='1 Measured')
                    plt.plot(dist, self.calc_stdev_mob, 'g-',label='1 Fitted')
            if i==2: #fix
                self.ratio_fitted_fix = ratio
                self.calc_stdev_fix = calc_sigma
                if self.plot:
                    plt.subplot(211)
                    plt.plot(dist, np.log2(self.ratio_fitted_fix), 'ro', mfc='white')
                    plt.subplot(212)
                    plt.plot(dist, stdev_data[i,:], 'ro', mfc='white', label='2 Measured')
                    plt.plot(dist, self.calc_stdev_fix, 'r-',label='2 Fitted')
                    
        if self.plot:
            plt.legend()
            plt.draw()
            plt.pause(0.001)

        if not quiet:
            print(f"ITERATION {self.fit_counter:4d}: {10**params['logbeta_dagger1'].value:.3f}, {10**params['logbeta_dagger2'].value:.3f}, {10**params['logk_dagger1'].value:.3f}, {10**params['logk_dagger2'].value:.3f}, {params['width1'].value:.3f}, {params['width2'].value:.3f}", end="\r")
            self.fit_counter += 1

        if mask is not None:
            resid = resid[:,~np.isnan(mask)]

        return resid.flatten()

    
    # Redefine __repr__ and __str__ methods
    def __repr__(self):
        return f'Trace ({len(self.force)} rows):\n{self.force} '

    def __str__(self):
        ret_string = (f"Trace with miscalibration factors:"
                      f"\n beta_dagger1: {self.beta_dagger1}\n beta_dagger2: {self.beta_dagger2}\n"
                      f" k_dagger1: {self.k_dagger1}\n k_dagger2: {self.k_dagger2}\n"
                      f" width1: {self.width1}\n width2: {self.width2}\n"
                      f" beta_dagger: {self.beta_dagger}\n k_dagger:{self.k_dagger}")
        if self.corrected:
            ret_string = ret_string + "\nCorrection is done."
        return ret_string


def correction(filename: str, k1_app: float, k2_app: float, filters: list, sheet: str = "", hydrodynamics = 'rp'):
    """
    Loads data, parses filter and runs correction
    :param filename:  File that contains data to be corrected; allowed extensions: *.csv, *.xlsx
    :param k1_app: Apparent stiffness of trap 1
    :param k2_app; Apparent stiffness of trap 2
    :param sheet: Optional sheetname argument in case multi-sheet .xlsx file is used
    :param filters: String of filters and related parameters
    :param hydrodynamics: Hydrodynamics model
    :return: Corrected trace object
    """
    #  Load data and store in Trace object
    if filename.endswith(".csv"):
        data = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        if len(sheet) != 0:
            data = pd.read_excel(filename, sheet_name=sheet, engine='openpyxl')
        else:
            data = pd.read_excel(filename, engine='openpyxl')
    else:
        print("File format not supported. Please convert data to .csv or .xlsx.")
        return -1
    trace = Trace()
    trace.k1_app = k1_app
    trace.k2_app = k2_app
    trace.force = data.iloc[:, 0]
    trace.stdev = data.iloc[:, 1]
    trace.dist = data.iloc[:, 2]

    has_mobfix_data = False
    try:  # Load data for individual traps if available
        trace.force_mob = data.iloc[:, 3]
        trace.force_fix = data.iloc[:, 4]
        trace.stdev_mob = data.iloc[:, 5]
        trace.stdev_fix = data.iloc[:, 6]
        has_mobfix_data = True
    except IndexError:
        pass

    try:
        if has_mobfix_data:
            trace.mask = data.iloc[:, 7]
        else:
            trace.mask = data.iloc[:, 4]
    except IndexError:
        pass
        
    trace.filters = filters
    trace.hydrodynamics = hydrodynamics

    # Correct
    print("start correction")
    trace.correct()

    return trace


if __name__ == "__main__":
    pass


