import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from load_data import load_digitizer, apply_basic_cuts

def exp_plus_const(t, N0, tau, B):
    return N0 * np.exp(-t / tau) + B

def pure_exp(t, N0, tau):
    return N0 * np.exp(-t / tau)

def lin(t, a, b):
    return a + b * t

#  Efficiency spline loading

def load_efficiency_spline(path="./data/efficiency_spline.npy"):
    """
    Load efficiency spline determined from calibration.
    """
    from scipy.interpolate import UnivariateSpline
    data = np.load(path, allow_pickle=True).item()
    knots = data["knots"]
    coeffs = data["coeffs"]
    degree = data["degree"]
    return UnivariateSpline._from_tck((knots, coeffs, degree))

#  Optional time calibration
def apply_time_calibration(dt_us, time_scale=1.0, time_offset_us=0.0):
    return time_scale * dt_us + time_offset_us


#  Background determination
def fit_background_exponential(centers, counts, bg_range=(10.0, 20.0)):
    """
    Fit log(D) = a + b t in a background-only region.
    """
    mask = (centers >= bg_range[0]) & (centers <= bg_range[1]) & (counts > 0)
    t_bg = centers[mask]
    D_bg = counts[mask]

    if len(t_bg) < 3:
        raise RuntimeError("Insufficient background bins.")

    y_bg = np.log(D_bg)
    sigma_y_bg = 1.0 / np.sqrt(D_bg)

    popt, pcov = curve_fit(
        lin, t_bg, y_bg,
        sigma=sigma_y_bg,
        absolute_sigma=True,
        maxfev=10000
    )

    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    return a, b, da, db

#  Main analysis function
def analyze_lifetime(
    data_path="./data/MUO final filtered pulse data_overweekend.txt",
    eff_spline_path="./data/efficiency_spline.npy",
    amp1_min=0.06,
    amp2_min=0.045,
    width1_range=None,
    width2_range=None,
    dt_fit_range_signal=(0.5, 10.0),
    dt_fit_range_total=(0.5, 30.0),
    bg_range=(10.0, 20.0),
    nbins=120,
    apply_time_calib=False,
    time_scale=1.0,
    time_offset_us=0.0,
    snr_threshold=3.0   # *** NEW: SNR cut for clean D2 semi-log plot ***
):
    print("\n=== Muon Lifetime Analysis (with SNR-clean D2 plot) ===\n")

    data = load_digitizer(data_path)
    dt_us_raw = data["dt_us"]
    amp1 = data.get("amp1", None)
    amp2 = data.get("amp2", None)
    width1 = data.get("width1", None)
    width2 = data.get("width2", None)
    base_mask = data["valid_dt_mask"]

    print(f"Total entries: {len(dt_us_raw)}")
    print(f"Valid dt entries: {np.sum(base_mask)}")

    cut_mask = apply_basic_cuts(data, amp1_min=amp1_min, amp2_min=amp2_min)

    if width1_range and width1 is not None:
        cut_mask &= (width1 >= width1_range[0]) & (width1 <= width1_range[1])
    if width2_range and width2 is not None:
        cut_mask &= (width2 >= width2_range[0]) & (width2 <= width2_range[1])

    mask = base_mask & cut_mask

    if apply_time_calib:
        dt_us = apply_time_calibration(dt_us_raw[mask], time_scale, time_offset_us)
        print("Applied time calibration.")
    else:
        dt_us = dt_us_raw[mask]
        print("Using raw digitizer dt (no calibration).")

    tmin_total, tmax_total = dt_fit_range_total
    in_window_total = (dt_us > tmin_total) & (dt_us < tmax_total)
    t = dt_us[in_window_total]

    counts, edges = np.histogram(t, bins=nbins, range=(tmin_total, tmax_total))
    centers = 0.5 * (edges[:-1] + edges[1:])

    if eff_spline_path:
        eff_spline = load_efficiency_spline(eff_spline_path)
        eff_vals = eff_spline(centers)
        eff_vals[centers < 3.5] = 1.0
        eff_vals = np.clip(eff_vals, 0.1, np.inf)
        D = counts / eff_vals
        D_err = np.sqrt(counts) / eff_vals
    else:
        D = counts.astype(float)
        D_err = np.sqrt(counts)

    a_bg, b_bg, da_bg, db_bg = fit_background_exponential(centers, D, bg_range)
    B_all = np.exp(lin(centers, a_bg, b_bg))

    D2 = D - B_all
    D2_clipped = np.clip(D2, 1e-12, None)
    D2_err = D_err

    tmin_sig, tmax_sig = dt_fit_range_signal
    mask_sig = (centers >= tmin_sig) & (centers <= tmax_sig) & (D2_clipped > 0)

    t_sig = centers[mask_sig]
    D2_sig = D2_clipped[mask_sig]
    D2_err_sig = D2_err[mask_sig]

    p0 = [D2_sig.max(), 2.2]
    popt_sig, pcov_sig = curve_fit(
        pure_exp,
        t_sig, D2_sig,
        p0=p0,
        sigma=D2_err_sig,
        absolute_sigma=True,
        maxfev=10000
    )

    N0_eff, tau_eff = popt_sig
    dN0_eff, dtau_eff = np.sqrt(np.diag(pcov_sig))

    fit_mask_lin = D > 0
    x_fit_lin = centers[fit_mask_lin]
    y_fit_lin = D[fit_mask_lin]
    yerr_fit_lin = D_err[fit_mask_lin]

    p0_lin = [y_fit_lin.max(), 2.2, y_fit_lin.min()]
    popt_lin, pcov_lin = curve_fit(
        exp_plus_const,
        x_fit_lin, y_fit_lin,
        p0=p0_lin,
        sigma=yerr_fit_lin,
        absolute_sigma=True,
        maxfev=10000
    )

    N0_lin, tau_lin, B_lin = popt_lin
    dN0_lin, dtau_lin, dB_lin = np.sqrt(np.diag(pcov_lin))

    # Plot linear version
    plt.figure(figsize=(8, 6))
    plt.errorbar(centers, D, yerr=D_err, fmt="o", alpha=0.8)
    tt_full = np.linspace(tmin_total, tmax_total, 400)
    plt.plot(tt_full, exp_plus_const(tt_full, *popt_lin),
             label=rf"Exp+Const fit: $\tau={tau_lin:.3f}\pm{dtau_lin:.3f}$ µs")
    plt.xlabel("Pulse separation dt (µs)")
    plt.ylabel("Counts per bin")
    plt.title("Muon Lifetime Fit (Exp + Const)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/muon_lifetime_linear_fit.pdf",
                bbox_inches="tight", pad_inches=0.05)
    plt.close()

    snr_mask = D2_clipped > snr_threshold * D2_err
    cent_plot = centers[snr_mask]
    D2_plot = D2_clipped[snr_mask]
    D2_err_plot = D2_err[snr_mask]

    plt.figure(figsize=(8, 6))
    plt.errorbar(cent_plot, D2_plot, yerr=D2_err_plot,
                 fmt="o", alpha=0.85,
                 label=f"D2 (SNR ≥ {snr_threshold:.0f})")

    plt.yscale("log")
    plt.xlabel("Pulse separation dt (µs)")
    plt.ylabel("Counts per bin (eff.-corr., background-subtracted)")

    # Best-fit curve
    tt = np.linspace(tmin_sig, tmax_sig, 300)
    plt.plot(tt, pure_exp(tt, N0_eff, tau_eff),
             label=rf"Best fit: $\tau_{{\rm eff}}={tau_lin:.3f}\,\mu s$")

    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xlim(-0.2, 12)
    plt.savefig("./figures/semilog_D_.pdf",
                bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("\nDone.\n")

    return {
        "tau_eff": tau_eff,
        "tau_eff_err": dtau_eff,
        "tau_lin": tau_lin,
        "background_params": (a_bg, b_bg),
    }


if __name__ == "__main__":
    res = analyze_lifetime()
    print("\nSummary:")
    for k, v in res.items():
        print(f"  {k}: {v}")


