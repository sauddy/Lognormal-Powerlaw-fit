'''''
Author: Sayantan

Modified: 29 May2019
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import math
import pymc

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('text', usetex=True)
plt.style.use('classic')



print("New TP_class_module is imported")


class PDF_analysis(object):
    """This read the model name the
            path to the data"""

    def __init__(self, Model, Path, binsize, upperlimit=None):
        self.Model = Model
        self.Path = Path
        self.binsize = binsize
        self.upperlimit = upperlimit
        # return(print("THE PATH IS ", Path))

    def fittingdata(self):
        '''
        General Overview: This makes the normalized-histogram of the NPDFs for a specific binsize

        Inputs: filename of the model
              : Cutoff by eys estimate to determine the power-law (optional only if needed)
              : Prefered binsize
              : Upper limit till which the date should be plotted (optional only if needed)

        Output: Xdata (bincenter)
              : ydata normalized histogram size
              : Xdata for the lognormal part (optional)
              : Ydata for the lognormal part (optional)

        '''
        # print("using fittingdata")
        filename = self.Path
        fp = open(filename, 'r')
        data = []
        for line in fp:
            t = line.strip().split()
            for value in t:
                data.append(float(value))

        fp.close()

        columndensity_array = np.asarray(data)

        bincount, bin_edge = np.histogram(np.log10(columndensity_array), self.binsize, density=True)  # i.e. divided by binwidth*N_total
        binwidth = (bin_edge[2] - bin_edge[1])
        N_total = len(columndensity_array)

        # the total normalization is binwidth*N_total*np.log(10)
        # the np.log(10) is multiplied in the fuction directly
        bincenter = (bin_edge[1:] + bin_edge[:-1]) / 2.0

        norm_const = binwidth * N_total  # this is the normalization constant
        xdata_raw = bincenter[:]
        ydata_raw = bincount
        ydata_not_normalized_raw = bincount * norm_const

        # this is added to eliminate the zero counts in the ydata

        ydata_nonzero_index = []
        for index, item in enumerate(ydata_not_normalized_raw):
            #         if (item !=0 and item !=1):
            if (item >= 1):
                ydata_nonzero_index.append(index)
    #     print(ydata_nonzero_index)

        xdata = xdata_raw[ydata_nonzero_index]  # modified Xdata
        ydata = ydata_raw[ydata_nonzero_index]  # modiffied normalized ydata
        ydata_not_normalized = ydata_not_normalized_raw[ydata_nonzero_index]  # modified non-normalized ydata

        # return xdata_new, ydata, ydata_not_normalized, norm_const

        # this is added to eliminate the data beyond a range in the value of the column density
        if self.upperlimit is not None:
            #         print("with the upper limit")
            xdata_lt_upperlimit_index = []
            for index, item in enumerate(xdata):
                if (item <= self.upperlimit):
                    xdata_lt_upperlimit_index.append(index)
        # print(ydata_nonnan)

            xdata_new = xdata[xdata_lt_upperlimit_index]
            ydata_new = ydata[xdata_lt_upperlimit_index]
            ydata_not_normalized_new = ydata_not_normalized[xdata_lt_upperlimit_index]

            xdata = xdata_new
            ydata = ydata_new
            ydata_not_normalized = ydata_not_normalized_new

        log_ydata = np.log10(ydata)  # normalized but not divided by log10
        log_ydata_not_normalized = np.log10(ydata_not_normalized)

        return xdata, log_ydata, log_ydata_not_normalized, norm_const
        # return xdata

    def plothist(self):
        '''Given the binsize and the upperlimit (optional)
        this plots the distribution of the PDF in normalised units'''

        xdata1, ydata1, ydata_not_normalized, norm_const = self.fittingdata()
        fig = plt.figure(1)
        plt.plot(xdata1, ydata_not_normalized)
        plt.xlabel("Log $\Sigma/\Sigma_{0}$", fontsize=18)
        plt.ylabel("Log n", fontsize=18)
        plt.tight_layout()
        return fig


class Fitting_the_PDF(PDF_analysis):

    def __init__(self, Model, Path, binsize, upperlimit=None, function_to_fit=None, bounds=None):
        super().__init__(Model, Path, binsize, upperlimit)

        if function_to_fit == None and bounds == None:
            # this is the default fit fucntion if the user has not defined any specific function
            self.function_to_fit = "Piecewise"
            self.bounds = ((0, 0, -7, .5), (1, 1, -2, .7))

            print("Using the default Piecewise function as no function is specified")
        else:
            self.function_to_fit = function_to_fit
            self.bounds = bounds
        # getting the raw data for fitting
        # xdata1, ydata1, ydata_not_normalized, norm_const = self.fittingdata()

    def lognormal_powerlaw_function(self, xdata, x0, sigma, alpha, TP_min, norm_const=None):
        '''
        General Overview: This is the log10 of the Piecewise lognormal and the power-law function.
                          For data less then TP the function is lognormal and for greater than TP its powerlaw
        Inputs: xdata
        Output: mean X0, Sigma
        '''
        constant_powlaw = []
        xG = np.array([xx for xx in xdata if xx <= TP_min])

        def F1(xG):
            if norm_const is not None:
                p1 = np.log10(norm_const * np.log(10) * 1 / (np.sqrt(2 * np.pi) * sigma))
            else:
                p1 = np.log10(np.log(10) * 1 / (np.sqrt(2 * np.pi) * sigma))
            p2 = np.log(10**xG)
            p3 = p1 - (p2 - x0)**2 / (2 * np.log(10) * sigma**2)
            return p3
        xL = np.array([xx for xx in xdata if xx > TP_min])
        global A2
        A2 = F1(TP_min) - alpha * TP_min
        constant_powlaw.append(A2)  # To store the value of C

        def F2(xL):
            return alpha * xL + A2
        return np.concatenate((F1(xG), F2(xL)))

    def lognormal_function(self, x, x0, sigma, norm_const=None):
        '''
        General Overview: This is the log10 of the normalized lognormal function.

        Inputs: xdata,mean,sigma, and Normalization_value optional
        if norm_const is given then it gives the non-normalised value of the function
        Output:  function
        '''
        xG = x

        def F1(xG):
            if norm_const is not None:
                p1 = np.log10(norm_const * np.log(10) * 1 / (np.sqrt(2 * np.pi) * sigma))
            else:
                p1 = np.log10(np.log(10) * 1 / (np.sqrt(2 * np.pi) * sigma))
            p2 = np.log(10**xG)
            p3 = p1 - (p2 - x0)**2 / (2 * np.log(10) * sigma**2)
            return p3
        return F1(xG)

    def model_fitting_parameters(self, function_to_fit=None, bounds=None):
        xdata1, ydata1, ydata_not_normalized, norm_const = self.fittingdata()

        if function_to_fit == None and bounds == None:
            user_defined_function = self.function_to_fit
            user_defined_bounds = self.bounds
        else:
            user_defined_function = function_to_fit
            user_defined_bounds = bounds

        if user_defined_function == "Lognormal":
            # print("Fitting lognormal function to the distribution:")
            popt, pcov = curve_fit(self.lognormal_function, xdata1, ydata1, bounds=user_defined_bounds)
            error_std = np.sqrt(np.diag(pcov))
            # print("The lognor"popt)
            # print("Fit parameters for the %s" % user_defined_function, "are using curve_fit are mu=%5.2f, sigma=%5.2f " % tuple(popt))

        elif user_defined_function == "Piecewise":
            # print("Fitting Piecewise function to the distribution:")
            popt, pcov = curve_fit(self.lognormal_powerlaw_function, xdata1, ydata1, bounds=user_defined_bounds)
            error_std = np.sqrt(np.diag(pcov))
            # print("Fit parameters for the %s" % user_defined_function, "using the curve_fit method are  mu=%5.2f, sigma=%5.2f, alpha=%5.2f, TP=%5.2f" % tuple(popt))
            # print(popt)

        return popt, error_std

    def xrange_for_chisq_fit(self, xfit, function_to_fit, norm_const, x0, sigma, alpha=None, TP_min=None):
        # print("function_to_fit", function_to_fit)
        if function_to_fit == "Lognormal":
            yfit = self.lognormal_function(xfit, x0, sigma, norm_const)
        elif function_to_fit == "Piecewise":
            # print("I am using yfit Piecewise in xrange method")
            yfit = self.lognormal_powerlaw_function(xfit, x0, sigma, alpha, TP_min, norm_const)

        ydata_nonzero = []
        for index, item in enumerate(yfit):
            if (item >= 0): ## selecting the index for yvalues greater than zero
                ydata_nonzero.append(index)
        yfit_new = yfit[ydata_nonzero]
        xfit_new = xfit[ydata_nonzero]
        xmax = np.max(xfit_new)
        xmin = np.min(xfit_new)

        return xmax, xmin  ## returns new range for the x array such that yvalues are positive only

    def get_chisquare_value(self, function_to_fit=None, bounds=None, Method=None, Best_fit_values=None):
        # print("We are inside the chisquare_method")
        xdata1, ydata1, ydata_not_normalized, norm_const = self.fittingdata()

        if Best_fit_values is None:  # if the fit values are not know then this estimates from begining

            if Method == "PYMC":
                popt = self.get_pymc_model_fit_params()
                user_defined_function = self.function_to_fit
            else:
                if function_to_fit == None and bounds == None:
                    Method = "Curve Fit"
                    popt, pcov = self.model_fitting_parameters()
                    user_defined_function = self.function_to_fit
                else:
                    Method = "Curve Fit"
                    popt, pcov = self.model_fitting_parameters(function_to_fit, bounds=bounds)
                    user_defined_function = function_to_fit
        else:  # else uses the input values to estimate the chisquare
            popt = Best_fit_values
            user_defined_function = self.function_to_fit
        xfit_test = np.linspace(np.min(xdata1), np.max(xdata1), len(ydata1))
        xmax, xmin = self.xrange_for_chisq_fit(xfit_test, user_defined_function, norm_const, *popt)
        xfit_corrected_for_chisquare = np.linspace(xmin, xmax, len(ydata1))
        if user_defined_function == "Lognormal":
            yfit_corrected = self.lognormal_function(xfit_corrected_for_chisquare, *popt, norm_const)
        elif user_defined_function == "Piecewise":
            yfit_corrected = self.lognormal_powerlaw_function(xfit_corrected_for_chisquare, *popt, norm_const)
        chisq = chisquare(ydata_not_normalized, yfit_corrected)

        # print("The Chisq value of %s" % user_defined_function, "fit using Method %s" % Method, "is %.2f" % chisq[0])
        return chisq[0]

    def best_fit_model(self, Model1, bound_model1, Model2, bound_model2):

        if Model1 is not "Lognormal":
            Temp = Model1
            Temp_Bounds = bound_model1
            Model1 = Model2
            bound_model1 = bound_model2
            Model1 = Model2
            Model2 = Temp
            bound_model2 = Temp_Bounds

        chisq_Lognormal = self.get_chisquare_value(function_to_fit=Model1, bounds=bound_model1)

        chisq_PieceWise = self.get_chisquare_value(function_to_fit=Model2, bounds=bound_model2)

        if (chisq_PieceWise <= 0.8 * chisq_Lognormal):
            print("The best model to fit is Piecewise")
        else:
            print("The best fit model is lognormal")

        return

    def model_pymc(self, guess_value_for_pymc=None):

        xdata1, ydata1, ydata_not_normalized, norm_const = self.fittingdata()
        x = xdata1
        f = ydata1

        # if the selected model in PieceWise then we take the default values
        # using the curve_fit as input, if
        # if guess values are not provided then we calculate them using curve fit

        if guess_value_for_pymc is None:
            Fit_params_Piecewise, pcov = self.model_fitting_parameters(function_to_fit="Piecewise", bounds=self.bounds)
            # print("Test Successful")
        else:
            Fit_params_Piecewise = guess_value_for_pymc
            # print("using stored values")
        p0_center = Fit_params_Piecewise[0]  # mean
        p0_width = Fit_params_Piecewise[1]  # sigma
        p0_slope = Fit_params_Piecewise[2]  # alpha
        p0_TP = Fit_params_Piecewise[3]  # Transition point

        # if the curvy fit routines are not senstive to the changes
        # then the intial conditions are imposed by hand

        if Fit_params_Piecewise[3] >= np.max(xdata1) or Fit_params_Piecewise[2] > 0:
            p0_TP = x[np.where(f == max(f))][0] + np.std(x)
            p0_slope = -3.

        p0 = [p0_center, p0_width, p0_slope, p0_TP]


#     A1 = pymc.Uniform("A1", 2.,6., value = p0[0])
        x0 = pymc.Uniform("x0", p0[0] - .4, p0[0] + .4, value=p0[0])
        sigma = pymc.Uniform("sigma", -0.01, 1., value=p0[1])
        alpha = pymc.Uniform("alpha", -7., 0., value=p0[2])
        TP_min = pymc.Uniform("TP_min", p0[3] - .4, p0[3] + .8, value=p0[3])

        @pymc.deterministic(plot=False)
        def function(x=x, x0=x0, sigma=sigma, alpha=alpha, TP_min=TP_min):
            xG = np.array([xx for xx in x if xx <= TP_min])

            def F1(xG):
                p1 = np.log10(np.log(10) * 1 / (np.sqrt(2 * np.pi) * sigma))
                p2 = np.log(10**xG)
                p3 = p1 - (p2 - x0)**2 / (2 * np.log(10) * sigma**2)
                return p3
            xL = np.array([xx for xx in x if xx > TP_min])
            A2 = F1(TP_min) - alpha * TP_min

            def F2(xL):
                return alpha * xL + A2
            return np.concatenate((F1(xG), F2(xL)))
        # self.lognormal_powerlaw_function(xdata=x, x0=x0, sigma=sigma, alpha=alpha, TP_min=TP_min)
        s = pymc.HalfNormal('s', tau=1)
        y = pymc.Normal("y", mu=function, tau=1 / s**2, value=f, observed=True)
        return locals()

    def get_pymc_model_fit_params(self, guess_value_for_pymc=None):

        MDL = pymc.MCMC(self.model_pymc(guess_value_for_pymc=guess_value_for_pymc))
        MDL.sample(20000, 5000, 1)

        x0_mean, x0_std = MDL.stats()['x0']['mean'], MDL.stats()['x0']['standard deviation']
        sigma_mean, sigma_std = MDL.stats()['sigma']['mean'], MDL.stats()['sigma']['standard deviation']
        alpha_mean, alpha_std = MDL.stats()['alpha']['mean'], MDL.stats()['alpha']['standard deviation']
        TP_min_mean, TP_min_std = MDL.stats()['TP_min']['mean'], MDL.stats()['TP_min']['standard deviation']

        x0_mean, x0_std = round(x0_mean, 2), round(x0_std, 2)
        sigma_mean, sigma_std = round(sigma_mean, 2), round(sigma_std, 2)
        alpha_mean, alpha_std = round(alpha_mean, 2), round(alpha_std, 2)
        TP_min_mean, TP_min_std = round(TP_min_mean, 2), round(TP_min_std, 2)

        # Array of best fitted parameters:
        bf_vals = [x0_mean, sigma_mean, alpha_mean, TP_min_mean]
        bf_valsUnc = [x0_std, sigma_std, alpha_std, TP_min_std]

        print("Best fit  using PYMC: mu=%5.2f, sigma=%5.2f, alpha=%5.2f, TP=%5.2f" % tuple(bf_vals))
        print("Err in best fit vals using PYMC: mu=%5.2f, sigma=%5.2f, alpha=%5.2f, TP=%5.2f" % tuple(bf_valsUnc))

        return bf_vals, bf_valsUnc

    def get_fitted_plot(self, function_to_fit=None, bounds=None, Method=None, time_in_my=None, guess_value_for_pymc=None):

        xdata1, ydata1, ydata_not_normalized, norm_const = self.fittingdata()

        fig = plt.figure(figsize=(8, 6))
        plt.step(xdata1, ydata_not_normalized, 'r', markersize=1)

        if Method == "PYMC":
            popt, pcov = self.get_pymc_model_fit_params(guess_value_for_pymc=guess_value_for_pymc)
            user_defined_function = self.function_to_fit
        else:
            if function_to_fit == None and bounds == None:  # use the user defined fucntion and bounds with the model
                Method = "Curve Fit"
                popt, pcov = self.model_fitting_parameters()
                user_defined_function = self.function_to_fit
            else:
                # use the user defined fucntion and bounds with the fit methods (usually not recommended)
                Method = "Curve Fit"
                popt, pcov = self.model_fitting_parameters(function_to_fit, bounds=bounds)
                user_defined_function = function_to_fit

        xfit_test = np.linspace(np.min(xdata1), np.max(xdata1), len(ydata1))
        xmax, xmin = self.xrange_for_chisq_fit(xfit_test, user_defined_function, norm_const, *popt)
        xfit_corrected_for_chisquare = np.linspace(xmin, xmax, len(ydata1))
        if user_defined_function == "Lognormal":  # for lognormal fucntion
            chisq_lg = self.get_chisquare_value(Best_fit_values=popt)
            yfit_corrected = self.lognormal_function(xfit_corrected_for_chisquare, *popt, norm_const)
            plt.plot(xfit_corrected_for_chisquare, yfit_corrected, "k", linewidth=2, label=r'Lognormal: $\mu$=%5.2f, $\sigma$=%5.2f,' % tuple(popt))
            plt.text(.7, 3.4, r'$\chi^2_{Curve-fit}$ =%5.2f' % chisq_lg, color='red', fontsize=12)
        elif user_defined_function == "Piecewise":  # for piecewise function with method PYMC or curve_fit
            chisq_lgpw = self.get_chisquare_value(Best_fit_values=popt)
            yfit_corrected = self.lognormal_powerlaw_function(xfit_corrected_for_chisquare, *popt, norm_const)
            plt.plot(xfit_corrected_for_chisquare, yfit_corrected, "k", linewidth=2, label=r'Piecewise: $\mu$=%5.2f, $\sigma$=%5.2f, $\alpha$=%5.2f, $\eta_{\rm {TP}}$=%5.2f' % tuple(popt))
            plt.axvspan(popt[3] + pcov[3], popt[3] + -pcov[3], alpha=0.5, color='green')
            plt.axvline(popt[3], linewidth=1, color='k', ls=':')
            # plt.text(.7, 3.4, r'$\chi^2_{Curve-fit}$ =%5.2f' % chisq_lg, color='red', fontsize=12)
            plt.text(.7, 3.4, r'$\chi^2_{MCMC}$ =%5.2f' % chisq_lgpw, color='red', fontsize=12)
            # plt.text(-.7, 1.7, r'$\alpha$ = %5.2f' % popt[2] + " $\pm$ % 5.2f" % pcov[2], color='red', fontsize=19)
            # plt.text(-.7, 1.3, r'$\eta_{\rm {TP}}$ = %5.2f ' % popt[3] + " $\pm$ % 5.2f" % pcov[3], color='red', fontsize=19)

        plt.legend(loc=3)
        plt.xlim(-0.8, 1.8)
        plt.ylim(-0.1, 4.5)
        plt.text(-.7, 4.1, r' $\beta=$ ' + '0.' + self.Model[4:6] + r',$v_{\rm a} = $' + self.Model[8] + '.' + self.Model[9] + r'$c_{\rm s}$', color='black', fontsize=20)
        plt.text(.7, 4, 'Model Selected : %s' % user_defined_function, color='blue', fontsize=12)
        plt.text(.7, 3.7, r'No of Bins = ' + str(self.binsize), color='red', fontsize=12)

        plt.minorticks_on()
        plt.text(1.0, 4.6, r'T = %.4s' % (time_in_my) + 'Myr', fontsize=20, color='black')
        plt.tick_params(axis='both', which='major', labelsize=20, length=6, width=1.5)
#     plt.tick_params(which='major',labelsize=20, length=10, width=2, direction='in')
        plt.tick_params(which='minor', length=3, width=1, direction='in')
        plt.xlabel(r"Log $\Sigma/\Sigma_{0}$", fontsize=25)
        plt.ylabel(r"Log n", fontsize=25)
        plt.tight_layout()
        # plt.savefig(self.Model + "_.eps", format='eps', dpi=300)
        # plt.close()
        return fig
