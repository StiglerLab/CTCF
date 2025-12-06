import psd_filter
import nonlinear_trap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import multiprocessing as mp



# Dictionary for parsing filters; append keys and filter functions to add custom filters
FILTER_DICT = {'qpd': psd_filter.qpd,
               'bessel8': psd_filter.bessel8,
               'butterworth': psd_filter.butterworth,
               'sample': psd_filter.psd_subsample,
               'boxcar': psd_filter.boxcar,
               'igorresample': psd_filter.psd_resample_down,
               'ni447x': psd_filter.ni447x,
               'sub': psd_filter.psd_subsample,
               "ss": psd_filter.psd_subsample}
# Dictionary for assigning filter parameters to the right key when using read_filter()
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


class Trace:
    def __init__(self):
        self.name = ""                # Used as an identifier in print statements

        #--- Settings
        self.hydrodynamics = 'none'   # Hydrodynamics correction mode
        self.plot = True              # Toggle plotting of fit progress     
        self.f_generate = 1e7         # Generate PSDs up to this frequency (Hz)
        
        #--- Measured FDC data
        self.force = np.empty(1)      # Uncorrected force (pN); must be set for correction
        self.dist = np.empty(1)       # Distance (nm) corresponding to the force data; must be set for correction
        self.stdev = np.empty(1)      # Standard deviation of total deflection signal (nm); must be set for correction

        self.force_mob = np.empty(1)  # Uncorrected force (pN) from mobile trap
        self.force_fix = np.empty(1)  # Uncorrected force (pN) from fixed trap
        self.stdev_mob = np.empty(1)  # Standard deviation of deflection signal in mobile trap (nm)
        self.stdev_fix = np.empty(1)  # Standard deviation of deflection signal in fixed trap (nm)

        self.mask = 1#FIXME np.empty(1)       # Masking array: same size as force, dist, etc. NaN values are excluded from fit

        
        #--- Measured parameters
        self.bead_diameter1 = 1000    # Diameter of bead in mobile trap (nm)
        self.bead_diameter2 = 1000    # Diameter of bead in fixed trap (nm)

        self.k1_app = .3              # Apparent spring constant of mobile trap before correction (pN/nm)
        self.k2_app = .3              # Apparent spring constant of fixed trap before correction (pN/nm)

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

        self.ext_orig = np.empty(1)   # Original extension data (nm)
        self.ext_corr = np.empty(1)   # Corrected extension data (nm)
        self.force_corr = np.empty(1) # Corrected force data (pN)
       

        #--- Internal stack for troubleshooting      
        self.psd_orig = np.empty(1)    # PSD before filtering
        self.calc_sigma = np.empty(1)  # Calculated noise
        self.ratio_fitted = None       # Ratio sigma_fit / sigma_exp

        self.f_bessel = 1  # Cutoff for bessel filter
        self.n_downsample = 1  # Downsampling factor of data
        self.n_boxcar = 1  # Window size of boxcar filter
        self.n_resampledown = 1
        self.filters = ''  # Filters applied to signal; separated by ';'
        self.fit_counter = 0  # Keep track of fit iterations
        self.db447x = psd_filter.load_db477x()  # Filter values for NI DB447x filter
        self.bead = 0  # Bead selection
        self.parameters = {}  # Dictionary for filter parameters

    def calc_theor_sigma_var_kc(self, force, dist, k1_app, k2_app, beta_dagger1, beta_dagger2, k_dagger1, k_dagger2,
                                width1, width2, bead):
        """
        Calculates theoretical sigma as part of the fitting routine.
        PSD generation and application of filters is multithreaded.
        :param force: Force data (pN)
        :param dist:  Distance data (nm)
        :param k1_app: Apparent (incorrect) k for trap 1 (pN/nm)
        :param k2_app: Apparent (incorrect) k for trap 2 (pN/nm)
        :param beta_dagger1: Correction factor beta_dagger for trap 1
        :param beta_dagger2: Correction factor beta_dagger for trap 2
        :param k_dagger1: Correction factor k_dagger for trap 1
        :param k_dagger2: Correction factor k_dagger for trap 2
        :param width1: Non-harmonicity parmeter for trap 1 (nm)
        :param width2: Non-harmonicity parmeter for trap 2 (nm)
        :param bead: Bead selection for calculation; default = 0: both beads
        :param filter_string: String containing the applied filters
        :return: Theoretical noise sigma
        """
        kc_app = 1 / (1 / k1_app + 1 / k2_app)
        k_dagger = (k_dagger1 / k1_app + k_dagger2 / k2_app) / (1 / kc_app)
        beta_dagger = (beta_dagger1 * k2_app * k_dagger1 + beta_dagger2 * k1_app * k_dagger2) / (
                    k2_app * k_dagger1 + k1_app * k_dagger2)
        dist = self.dist.copy()
        ext_corr, f_corr, defl_corr1, defl_corr2 = nonlinear_trap.correct_linker_soft_trap(force, dist, k1_app, k2_app,
                                                                                           beta_dagger1, beta_dagger2,
                                                                                           k_dagger1, k_dagger2, width1,
                                                                                           width2)
        if np.isnan(defl_corr1).any():
            return 1e7*np.ones_like(force)

        def central_diff(y, x):    
            dydx = np.zeros_like(y)
            dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
            dydx[0] = (y[1] - y[0]) / (x[1] - x[0]) # fwd diff
            dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) # bwd diff
            return dydx

        df_dx = central_diff(f_corr, ext_corr)

        calc_sigma = np.zeros_like(force)
        kappa1 = np.pi * force * k_dagger1/(2*beta_dagger * k_dagger * k1_app * width1)
        kappa2 = np.pi * force * k_dagger2/(2*beta_dagger * k_dagger * k2_app * width2)
        k1_eff = k1_app / k_dagger1 * np.sqrt(1 - (kappa1**2))
        k2_eff = k2_app / k_dagger2 * np.sqrt(1 - (kappa2**2))
        beta_dagger_total1 = beta_dagger1 * np.sqrt(1 - (kappa1**2))
        beta_dagger_total2 = beta_dagger2 * np.sqrt(1 - (kappa2**2))
        pool = mp.Pool(processes=4)
        psd_orig = pool.starmap(psd_filter.psd_generate, [(k1_eff[i], k2_eff[i], df_dx[i], self.f_generate, beta_dagger_total1[i], beta_dagger_total2[i], ext_corr[i], self.bead_diameter1, self.bead_diameter2, self.hydrodynamics, bead) for i in range(len(calc_sigma))])
        calc_sigma = pool.starmap(psd_filter.parse_filter, [(psd_orig[i], self.parameters, FILTER_DICT, self.filters) for i in range(len(calc_sigma))])
        pool.close()
        pool.join()
        self.ext_orig = dist - self.force / self.kc_app
        self.dist_orig = self.dist.copy()
        self.force_corr = f_corr.copy()
        self.ext_corr = ext_corr.copy()
        self.calc_sigma = calc_sigma
        dist_temp = dist.copy()        
        if np.isnan(calc_sigma).any():  # heavy penalization if NaNs are generated; prevents abort by Fit Error
            return 1e7*np.ones_like(force)
        return np.array(calc_sigma)

    def correct(self):
        """
        Main method that's called for correction of miscalibration after data loading.
        Calls fit_sigma_filter method for correction.
        """
        bead = self.bead
        if bead == 1:
            force = self.force_mob
        elif bead == 2:
            force = self.force_fix
        else:
            force = self.force
        if not self.parameters:
            self.parameters = psd_filter.make_filter_params(self.db447x, self.n_downsample, 4096, 1, self.f_bessel,
                                                                self.n_boxcar, self.n_resampledown)
            print("Setting parameters")
        if len(force) == 1 or len(self.dist) == 1:
            raise Exception("No FDC")
        sigma = self.calc_theor_sigma_var_kc(force, self.dist, self.k1_app, self.k2_app, self.beta_dagger1, self.beta_dagger2, self.k_dagger1, self.k_dagger2, self.width1, self.width2, bead)
        kc_app = 1 / (1 / self.k1_app + 1 / self.k2_app)
        self.k_dagger = (self.k_dagger1 / self.k1_app + self.k_dagger2 / self.k2_app) / (1 / kc_app)
        self.beta_dagger = np.divide(self.beta_dagger1 * self.k2_app * self.k_dagger1 + self.beta_dagger2 * self.k1_app * self.k_dagger2,
                                     self.k2_app * self.k_dagger1 + self.k1_app * self.k_dagger2)
        x = self.dist.copy(deep=True)
        if bead == 1:  # set stdev according to bead mode
            y = self.stdev_mob
        elif bead == 2:
            y = self.stdev_fix
        else:
            y = self.stdev
        if self.plot:
            plt.plot(self.dist, self.stdev)
            plt.ion()
            plt.show()
            plt.pause(0.0001)

        fmodel = Model(self.fit_sigma_filter)
        params = fmodel.make_params(logbeta_dagger1=np.log(self.beta_dagger1),
                                    logbeta_dagger2=np.log(self.beta_dagger2),
                                    logk_dagger1=np.log(self.k_dagger1),
                                    logk_dagger2=np.log(self.k_dagger2),
                                    width1=self.width1, width2=self.width2,
                                    k1_app=self.k1_app, k2_app=self.k2_app,
                                    bead=bead)
        
        params['k1_app'].vary = False
        params['k2_app'].vary = False
        params['bead'].vary = False
        params['logbeta_dagger1'].min = np.log(0.5)
        params['logbeta_dagger1'].max = np.log(2)
        params['logbeta_dagger2'].min = np.log(0.5)
        params['logbeta_dagger2'].max = np.log(2)
        params['logk_dagger1'].min = np.log(0.5)
        params['logk_dagger1'].max = np.log(2)
        params['logk_dagger2'].min = np.log(0.5)
        params['logk_dagger2'].max = np.log(2)
        params['width1'].min = 100
        params['width1'].max = 4000
        params['width2'].min = 100
        params['width2'].max = 4000

        result = fmodel.fit(y, params, x=x)
        self.beta_dagger1 = 10**result.params['logbeta_dagger1'].value
        self.beta_dagger2 = 10**result.params['logbeta_dagger2'].value
        self.k_dagger1 = 10**result.params['logk_dagger1'].value
        self.k_dagger2 = 10**result.params['logk_dagger2'].value
        self.width1 = result.params['width1'].value
        self.width2 = result.params['width2'].value

        # Update self.k_dagger1, self.k_dagger2, self.beta_dagger1 and self.beta_dagger2 with fit values
        self.kc_app = 1 / (1 / self.k1_app + self.k2_app)
        self.k_dagger = (self.k_dagger1 / self.k1_app + self.k_dagger2 / self.k2_app) / (1 / self.kc_app)
        self.beta_dagger = (self.beta_dagger1 * self.k2_app * self.k_dagger1 + self.beta_dagger2 *
                            self.k1_app * self.k_dagger2) / (self.k2_app * self.k_dagger1 + self.k1_app * self.k_dagger2)
        print(f"Corrected trace {self.name}.\n New calibration factors:"
              f"\n beta_dagger1: {self.beta_dagger1}\n beta_dagger2: {self.beta_dagger2}\n"
              f" k_dagger1: {self.k_dagger1}\n k_dagger2: {self.k_dagger2}\n"
              f" beta_dagger: {self.beta_dagger}\n k_dagger:{self.k_dagger}")
        self.calc_sigma = self.calc_theor_sigma_var_kc(self.force, self.dist, self.k1_app, self.k2_app, self.beta_dagger1,
                                             self.beta_dagger2, self.k_dagger1, self.k_dagger2, self.width1,
                                             self.width2, bead)
        self.corrected = True
        if self.plot:
            plt.ioff()
            plt.show()

    def fit_sigma_filter(self, x, logbeta_dagger1, logbeta_dagger2, logk_dagger1, logk_dagger2, width1, width2, k1_app, k2_app, bead):
        beta_dagger1 = 10**logbeta_dagger1
        beta_dagger2 = 10**logbeta_dagger2
        k_dagger1 = 10**logk_dagger1
        k_dagger2 = 10**logk_dagger2
        if bead == 1:
            ext_std = self.stdev_mob.copy()
            ext_force = self.force_mob.copy()
        elif bead == 2:
            ext_std = self.stdev_fix.copy()
            ext_force = self.force_fix.copy()
        else:
            ext_std = self.stdev.copy()
            ext_force = self.force.copy()
        sigma = self.calc_theor_sigma_var_kc(ext_force, x, k1_app, k2_app, beta_dagger1, beta_dagger2, k_dagger1,
                                             k_dagger2, width1=width1, width2=width2, bead=bead)

        self.ratio_fitted = self.stdev / sigma.ravel()

        sigma_masked = sigma.copy()
        sigma_masked = np.array(sigma_masked * self.mask)
        sigma_masked = sigma_masked[~np.isnan(sigma_masked)]  # remove NaN
 
        if len(sigma_masked) == len(sigma):
            yw = sigma_masked.copy()
        else:
            yw = sigma.copy()

        # Plot for visualization
        if self.plot:
            plt.clf()
            plt.subplot(211)
            plt.plot(x, np.log2(self.ratio_fitted), 'ko', mfc='white')
            plt.ylabel('log2(Measured/Fitted)')
            plt.axhline(y=0, color='k')
            low, high = plt.ylim()
            bound = max(abs(low), abs(high))
            plt.ylim(-bound, bound)

            plt.subplot(212)
            plt.plot(x, self.stdev, 'ko', mfc='white', label='1+2 Measured')
            plt.plot(x, yw, 'k-',label='1+2 Fitted')
            plt.xlabel('Distance (nm)')
            plt.ylabel(r'$\sigma$ (nm)')
            plt.legend()
            plt.draw()
            plt.pause(0.001)
        print(f"{self.fit_counter:4d} {beta_dagger1:.3f}, {beta_dagger2:.3f}, {k_dagger1:.3f}, {k_dagger2:.3f}, {width1:.3f}, {width2:.3f}")

        self.fit_counter += 1
        # return fitted sigma as yw
        return yw

    # FIXME Headers?
    def load_from_csv(self, path: str,):
        """
        Method for loading in data. Format: .csv file with force as $root_F, distance as $root_Dist and Stdev as $root_Stdev
        :param path: filepath of .csv file
        """
        self.name = path.split('\\')[-1]
        data = pd.read_csv(path, sep=",")
        self.dist = data.loc[:, 'Distance']
        self.force = data.loc[:, 'Force']
        self.stdev = data.loc[:, 'Stdev']
        try:  # Trap specific values are optional, this skips loading them if they aren't present
            self.force_mob = data.loc[:, 'Force_1']
            self.force_fix = data.loc[:, 'Force_2']
            self.stdev_mob = data.loc[:, 'Stdev_1']
            self.stdev_fix = data.loc[:, 'Stdev_2']
        except KeyError:
            pass

    def read_filter(self, filter_string: str):
        filter_list = filter_string.lower().split(";")
        filters = [x.split(',')[0] for x in filter_list]
        parameters = [x.split(',')[1] if len(x.split('1'))>1 else '' for x in filter_list]
        param_dict = {}
        for i, filter in enumerate(filters):
            if parameters[i]!='':
                param_dict[filter_params[filter]] = float(parameters[i])
        param_dict['db447x'] = self.db447x
        self.filters = ';'.join(filters)
        self.parameters = param_dict

    # Redefine __repr__ and __str__ methods
    def __repr__(self):
        return f'Trace {self.name} ({len(self.force)} rows):\n{self.force} '

    def __str__(self):
        ret_string = (f"Trace {self.name}\n Calibration factors:"
                      f"\n beta_dagger1: {self.beta_dagger1}\n beta_dagger2: {self.beta_dagger2}\n"
                      f" k_dagger1: {self.k_dagger1}\n k_dagger2: {self.k_dagger2}\n"
                      f" beta_dagger: {self.beta_dagger}\n k_dagger:{self.k_dagger}")
        if self.corrected:
            ret_string = f"Corrected " + ret_string
        return ret_string


def correction(filename: str, k1_app: float, k2_app: float, filters: str = "", sheet: str = ""):
    """
    Loads data, parses filter and runs correction
    :param filename:  File that contains data to be corrected; allowed extensions: *.csv, *.xlsx
    :param k1_app: Apparent stiffness of trap 1
    :param k2_app; Apparent stiffness of trap 2
    :param sheet: Optional sheetname argument in case multi-sheet .xlsx file is used
    :param filters: String of filters and related parameters
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
    try:  # Load data for individual traps if available
        trace.force_fix = data.iloc[:, 3]
        trace.force_mob = data.iloc[:, 4]
        trace.stdev_fix = data.iloc[:, 5]
        trace.stdev_mob = data.iloc[:, 6]
    except KeyError:
        pass
    # Parse filters
    trace.read_filter(filters)
    # Correct
    print("start correction")
    trace.correct()
    return trace


if __name__ == "__main__":
    pass


