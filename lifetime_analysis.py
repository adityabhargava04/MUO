import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from load_data import load_digitizer, apply_basic_cuts


# -------------------------
#  Model: exponential + flat
# -------------------------

def exp_plus_const(t, N0, tau, B):
    return N0 * np.exp(-t / tau) + B


# -------------------------
#  Efficiency spline loading
# -------------------------

def load_efficiency_spline(path="./data/efficiency_spline.npy"):
    from scipy.interpolate import UnivariateSpline
    data = np.load(path, allow_pickle=True).item()
    knots = data["knots"]
    coeffs = data["coeffs"]
    degree = data["degree"]
    return UnivariateSpline._from_tck((knots, coeffs, degree))


# -------------------------
#  Lifetime analysis
# -------------------------

def analyze_lifetime(
    data_path="./data/MUO final filtered pulse data_overweekend.txt",
    eff_spline_path="./data/efficiency_spline.npy",
    amp1_min=0.06,
    amp2_min=0.045,
    dt_fit_range=(0.5, 30.0),   # μs
    nbins=120
):
    print("\n=== Muon Lifetime Analysis ===\n")
    print(f"Using dt as digitizer-reported (no time calibration).")

    # 1) Load raw digitizer output
    data = load_digitizer(data_path)

    # 2) Basic cuts
    base_mask = data["valid_dt_mask"]
    cut_mask = apply_basic_cuts(data, amp1_min=amp1_min, amp2_min=amp2_min)
    mask = base_mask & cut_mask
    dt_us = data["dt_us"][mask]

    # 3) Fit window
    tmin, tmax = dt_fit_range
    in_window = (dt_us > tmin) & (dt_us < tmax)
    t = dt_us[in_window]

    print(f"Entries after cuts and window {tmin}-{tmax} μs: {len(t)}")

    # 4) Efficiency correction
    if eff_spline_path is not None:
        eff_spline = load_efficiency_spline(eff_spline_path)
    else:
        eff_spline = None

    # 5) Histogram
    counts, edges = np.histogram(t, bins=nbins, range=(tmin, tmax))
    centers = 0.5 * (edges[:-1] + edges[1:])

    if eff_spline is not None:
        eff_vals = eff_spline(centers)
        eff_vals[centers < 3.5] = 1.0   # avoid overcorrecting early-time dip
        eff_vals = np.clip(eff_vals, 0.1, np.inf)  # protect against div by ~0
        y = counts / eff_vals
        yerr = np.sqrt(counts) / eff_vals
    else:
        eff_vals = None
        y = counts
        yerr = np.sqrt(counts)

    # 6) Fit exponential + flat
    fit_mask = y > 0
    x_fit = centers[fit_mask]
    y_fit = y[fit_mask]
    yerr_fit = yerr[fit_mask]

    p0 = [y_fit.max(), 2.2, y_fit.min()]  # initial: τ ≈ 2.2 μs
    popt, pcov = curve_fit(
        exp_plus_const,
        x_fit,
        y_fit,
        p0=p0,
        sigma=yerr_fit,
        absolute_sigma=True,
        maxfev=10000
    )

    N0, tau, B = popt
    dN0, dtau, dB = np.sqrt(np.diag(pcov))

    print("\nFit results (exp + const):")
    print(f"  tau = {tau:.4f} ± {dtau:.4f} μs")
    print(f"  N0  = {N0:.2e} ± {dN0:.2e}")
    print(f"  B   = {B:.2f} ± {dB:.2f}")

    # 7) Plot
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14
    })

    plt.figure(figsize=(8, 6))
    plt.errorbar(centers, y, yerr=yerr, fmt="o", alpha=0.8, label="Data")

    tt = np.linspace(tmin, tmax, 400)
    plt.plot(
        tt,
        exp_plus_const(tt, *popt),
        label=rf"Fit: $\tau={tau:.3f}\pm{dtau:.3f}\,\mu\mathrm{{s}}$"
    )

    plt.xlabel(r"Pulse separation $\Delta t$ ($\mu$s)")
    plt.ylabel(r"Counts per bin")
    plt.title("Muon Lifetime Fit (Exponential + Constant)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("./figures/muon_lifetime_fit.pdf",
                bbox_inches="tight", pad_inches=0.05)
    plt.show()

    return {
        "tau": tau, "tau_err": dtau,
        "B": B, "B_err": dB
    }


if __name__ == "__main__":
    res = analyze_lifetime()
    print("\nDone.\n")
