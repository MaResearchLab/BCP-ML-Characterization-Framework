import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

class PeakFitter:
    def __init__(self, sample, data_dir, output_dir, window_range, alpha=10, low_q_limit=0.01, q_and_I_in_pairs=False, plot_enabled=True):
        self.sample = sample
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.window_range = window_range
        self.alpha = alpha
        self.low_q_limit = low_q_limit
        self.q_and_I_in_pairs = q_and_I_in_pairs
        self.plot_enabled = plot_enabled

        # Setup paths
        self.data_path = os.path.join(data_dir, f"{sample}.xlsx")
        self.param_dir = os.path.join(output_dir, sample)
        self.plot_dir = os.path.join(output_dir, sample, "plots")
        os.makedirs(self.param_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.output_excel = os.path.join(self.param_dir, f"{self.sample}_fit.xlsx")

        # Load data once
        self.df = pd.read_excel(self.data_path, sheet_name=0)
        self.first_row = pd.read_excel(self.data_path, sheet_name=0, nrows=0).columns.tolist()
        self.results = []

    def _background_equation(self, x, a, b, c):
        return a + b * np.power(x, -c)

    def _peak_equation(self, x, a, b, c, d, e, f):
        return a + b*np.power(x, -c) + d*np.exp((-4*np.log(2)*(x-e)**2)/((f)**2))

    def _get_data_ranges(self):
        col_values = self.df.columns
        data_ranges = []
        start = 0
        while start < len(col_values):
            end = start
            while end + 1 < len(col_values) and not col_values[end+1].startswith('Unnamed'):
                end += 1
            if end - start >= 1:
                data_ranges.append((start, end))
            start = end + 1
        return data_ranges

    def fit_all(self):
        data_ranges = self._get_data_ranges()

        for start_col, end_col in data_ranges:
            for i in range(start_col, end_col):
                try:
                    q_index = self.first_row[2 * i] if self.q_and_I_in_pairs else self.first_row[0]
                    I_index = self.first_row[(2 * i) + 1] if self.q_and_I_in_pairs else self.first_row[i + 1]
                except IndexError:
                    continue

                print(f"\nProcessing sample: {I_index}")
                q_raw, I_raw = self.df[q_index], self.df[I_index]
                mask = q_raw > self.low_q_limit
                q_filtered, I_filtered = q_raw[mask], I_raw[mask]

                try:
                    bg_opt, _ = curve_fit(self._background_equation, q_filtered, I_filtered, p0=[1, 1, 2], maxfev=50000)
                    bg_fit = self._background_equation(q_filtered, *bg_opt)
                    corrected = I_filtered - bg_fit
                except Exception as e:
                    print(f"Background fit failed for {I_index}: {e}")
                    self._write_result(I_index, ["Background Fit Failed"] + ["N/A"]*7)
                    continue

                peak_pos, d_guess = self._estimate_peak(q_filtered.values, I_filtered.values, corrected.values)
                if peak_pos is None:
                    self._write_result(I_index, ["Disordered"] + ["N/A"]*7)
                    continue

                best_fit, q_fit_max, I_fit_max = self._fit_peak(q_filtered, I_filtered, peak_pos, d_guess)
                if best_fit:
                    popt, q_fit, I_fit, score = best_fit
                    r_squared = self._calculate_r_squared(I_fit, q_fit, popt)
                    E, F = popt[4], popt[5]  # e and f
                    self._write_result(I_index, E, F, r_squared)
                    self._plot_fit(I_index, q_fit, I_fit, q_fit_max, I_fit_max, popt)
                else:
                    self._write_result(I_index, ["Disordered"] + ["N/A"]*7)

            # Write only relevant output columns
            df_out = pd.DataFrame(self.results, columns=["Sample Name", "E (peak position)", "F (FWHM)", "R²"])
            df_out.to_excel(self.output_excel, index=False)

    def _estimate_peak(self, q, I, corrected):
        """Estimate the peak position based on max intensity in corrected data."""
        max_I = None
        max_q = None
        d_guess = None
        # first_value = corrected.index[0]
        for j in range(1, len(corrected) - 1):
            if corrected[j] > corrected[j-1] and corrected[j] > corrected[j+1]:
                if max_I is None or corrected[j] > max_I:
                    max_I = corrected[j]
                    d_guess = I[j]
                    max_q = q[j]
        return max_q, d_guess

    def _fit_peak(self, q, I, peak_pos, d_guess):
        best_score = np.inf
        best_result = None
        peak_eq = self._peak_equation
        alpha = self.alpha
        
        # Convert q to np array to accelerate calculation
        q = np.array(q)
        I = np.array(I)

        for window in self.window_range:
            mask = (q > peak_pos - window) & (q < peak_pos + window)
            q_fit, I_fit = q[mask], I[mask]
            
            try:
                popt, _ = curve_fit(peak_eq, q_fit, I_fit,
                                    p0=[0, 0, 0, d_guess, peak_pos, peak_pos / 10], maxfev=50000)
                residuals = I_fit - peak_eq(q_fit, *popt)
                ssr = np.sum(residuals ** 2)
                score = (ssr / len(q_fit)) + alpha * (1 / len(q_fit))
                if score < best_score:
                    best_score = score
                    best_result = (popt, q_fit, I_fit, score)
            except Exception as e:
                print(f"Fit failed at window ±{window:.4f}: {e}")
                continue
        # Add q and I fit to be returned to better visualize the fitting re
        return best_result, q_fit, I_fit

    def _calculate_r_squared(self, I_fit, q_fit, popt):
        y_fit = self._peak_equation(q_fit, *popt)
        ss_residuals = np.sum((I_fit - y_fit) ** 2)
        ss_total = np.sum((I_fit - np.mean(I_fit)) ** 2)
        return 1 - (ss_residuals / ss_total)

    def _plot_fit(self, I_index, q_fit, I_fit, q_fit_max, I_fit_max, popt):
        if not self.plot_enabled:
            return
        y_fit = self._peak_equation(q_fit, *popt)

        # Format the plots
        # %matplotlib inline
        rc('text', usetex=False)
        rc('mathtext', fontset='cm')
        rc('xtick', labelsize=25)   
        rc('xtick', labelsize=25)
        rc('xtick.major', size=7)  
        rc('xtick.minor', size=4)
        rc('xtick.major', width=2)
        rc('xtick.minor', width=2)
        rc('ytick', labelsize=25)
        rc('ytick.major', width=2)
        rc('ytick.minor', width=2)
        rc('ytick', labelsize=25)
        rc('ytick.major', size=7)
        rc('ytick.minor', size=4)
        rc('axes', labelsize=35) 
        rc('axes', linewidth=2) 
        rc('font',family='sans serif')
        rc('font', style='normal')
        rc('font', weight='500')
        rc('font', size='15')
        rc('axes', labelweight='500')
        rc('axes.spines', **{'right':True, 'top':True}) 
        plt.rcParams['font.family'] = 'Arial'

        #This sets the size of the entire image
        fig,ax = plt.subplots(figsize=(8,8))
        plt.scatter(q_fit_max, I_fit_max, label='Original Data', color='blue', s=100)
        plt.plot(q_fit, y_fit, label='Fitted Curve', color='red', linewidth=2.5)
        plt.xlabel('Q vector [$A^{-1}$]')
        plt.ylabel('Intensity / a.u.')
        plt.title(f"Fit for {I_index}")
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{I_index}.png"), dpi=300)
        plt.close()

    def _write_result(self, sample_name, E, F, R2):
        self.results.append([sample_name, E, F, R2])


"""
Example Useage:

if __name__ == "__main__":
    fitter = PeakFitter(
        sample="name_of_file",
        data_dir="location_to_file",
        output_dir="output_location",
        window_range=np.linspace(0.004, 0.01, 20), # Set the window fitting range
        alpha=10, # Larger values for alpha incentivizes larger fitting windows
        low_q_limit=0.0045,  # Set the value to trim the noisy low q region of the data
        q_and_I_in_pairs=False # True if there is a q for every I, otherwise False for only one q for each I
    )
    fitter.fit_all()
"""