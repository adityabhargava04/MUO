# time_calibration.py
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'serif'


from load_data import load_digitizer, apply_basic_cuts, describe

# map filename → true delay in microseconds
# if your set is different, just edit this dict
TRUE_DELAYS_US = {
    "digitizer_clock_1.txt": 1.0,
    "digitizer_clock_4.txt": 4.0,
    "digitizer_clock_10.txt": 10.0,
    "digitizer_clock_15.txt": 15.0,
    "digitizer_clock_20.txt": 20.0,
    "digitizer_clock_25.txt": 25.0,
    "digitizer_clock_30.txt": 30.0,
    "digitizer_clock_35.txt": 35.0,
    "digitizer_clock_40.txt": 40.0,
}

DATA_DIR = "./data"   # adjust if needed


def compute_mean_dt(path):
    """Return mean and std of dt_us for one clock file (with basic cuts)."""
    data = load_digitizer(path)

    # optional: amplitude cuts to throw out garbage
    mask = data["valid_dt_mask"].copy()
    mask &= data["A1"] > 0.05
    mask &= data["A2"] > 0.05

    dt_us = data["dt_us"][mask]
    mean = np.mean(dt_us)
    std = np.std(dt_us)

    return mean, std, dt_us.size


def main():
    true_vals = []
    meas_vals = []
    meas_errs = []
    labels = []

    for fname, true_dt in sorted(TRUE_DELAYS_US.items(), key=lambda kv: kv[1]):
        path = os.path.join(DATA_DIR, fname)
        mean, std, n = compute_mean_dt(path)

        true_vals.append(true_dt)
        meas_vals.append(mean)
        # error on the mean = std / sqrt(N)
        meas_errs.append(std / np.sqrt(n))
        labels.append(fname)
        print(f"{fname:25s}  true={true_dt:5.1f} us,  mean={mean:7.3f} us,  N={n}")

    true_vals = np.array(true_vals)
    meas_vals = np.array(meas_vals)
    meas_errs = np.array(meas_errs)

    # linear fit: meas = a + b * true
    coeffs = np.polyfit(true_vals, meas_vals, 1)
    b, a = coeffs[0], coeffs[1]

    print("\nTime calibration fit (measured = a + b * true):")
    print(f"a = {a:.4f} μs  (offset)")
    print(f"b = {b:.6f}     (scale factor)")

    # save calibration constants for later scripts
    np.save("./data/time_calibration_coeffs.npy", np.array([a, b]))

    # plot
    t_fit = np.linspace(min(true_vals)-1, max(true_vals)+1, 200)
    m_fit = a + b * t_fit

    plt.figure(figsize=(8, 6))
    plt.errorbar(true_vals, meas_vals, yerr=meas_errs, fmt="o", label="Data")
    plt.plot(t_fit, m_fit, label=rf"Fit: $b={a:.3f}\,\mu\text{{s}}$, $m={b:.6f}$")

    plt.xlabel(r"True delay $\Delta t_{\mathrm{true}}\,(\mu s)$")
    plt.ylabel(r"Measured delay $\Delta t_{\mathrm{meas}}\,(\mu s)$")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./figures/time_calibration.pdf",
            bbox_inches='tight',
            pad_inches=0.05)
    plt.show()


if __name__ == "__main__":
    main()