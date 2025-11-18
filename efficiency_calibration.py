import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from load_data import load_digitizer


def compute_relative_efficiency(dt_us, bins=200, dt_range=(0, 40)):
    """
    Compute the digitizer relative efficiency curve:

    Efficiency(dt) = H(dt) / H_plateau

    where H_plateau is the max in the last 30% of the dt range.
    """

    dt_min, dt_max = dt_range
    hist, edges = np.histogram(dt_us, bins=bins, range=(dt_min, dt_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # plateau = last 30% of dt range
    plateau_start = dt_min + 0.7 * (dt_max - dt_min)
    plateau_mask = centers > plateau_start
    plateau_level = np.max(hist[plateau_mask])

    efficiency = hist / plateau_level

    return centers, efficiency, hist, plateau_level


def fit_efficiency_spline(dt_centers, efficiency, smooth_factor=1):
    mask = np.isfinite(efficiency)
    x = dt_centers[mask]
    y = efficiency[mask]

    spline = UnivariateSpline(x, y, s=smooth_factor, k=3)
    return spline


def save_spline(spline, filename="./efficiency_spline.npy"):
    knots, coeffs, degree = spline._eval_args
    data = {"knots": knots, "coeffs": coeffs, "degree": degree}
    np.save(filename, data, allow_pickle=True)
    print(f"Spline saved to {filename}")


def load_spline(filename="./efficiency_spline.npy"):
    data = np.load(filename, allow_pickle=True).item()
    knots = data["knots"]
    coeffs = data["coeffs"]
    degree = data["degree"]
    return UnivariateSpline._from_tck((knots, coeffs, degree))


def plot_efficiency_with_spline(dt_centers, eff, spline,
                                output_path="./figures/efficiency_vs_dt.pdf"):
    """
    Plot both the raw efficiency measurements and the smoothing spline.
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14
    })

    plt.figure(figsize=(8, 6))

    # Raw points
    plt.plot(dt_centers, eff, ".", alpha=0.4, label="Measured efficiency")

    # Smooth spline curve
    dt_fine = np.linspace(dt_centers[0], dt_centers[-1], 2000)
    plt.plot(dt_fine, spline(dt_fine), "-", lw=2,
             label="Spline fit", color="red")

    plt.ylim(0, 1.05)
    plt.xlabel(r"Pulse separation $\Delta t~(\mu\mathrm{s})$")
    plt.ylabel(r"Relative efficiency $\epsilon(\Delta t)$")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def analyze_efficiency_file(path,
                            bins=200,
                            dt_range=(0, 39.9995),
                            smooth_factor=1e-1):
    """
    Full pipeline:
      • Load data
      • Compute relative efficiency
      • Fit smooth spline ε(dt)
      • Plot results

    Returns:
      spline(dt)  # callable efficiency correction function
      (dt_centers, eff_raw, hist)
    """

    print(f"\nAnalyzing digitizer efficiency file:\n{path}\n")

    data = load_digitizer(path)
    dt_us = data["dt_us"]

    # Raw measurement
    dt_centers, eff, hist, plateau_level = compute_relative_efficiency(
        dt_us, bins=bins, dt_range=dt_range
    )

    print(f"Plateau count level: {plateau_level:.1f}")

    # Fit spline
    spline = fit_efficiency_spline(dt_centers, eff, smooth_factor=smooth_factor)
    save_spline(spline, "./data/efficiency_spline.npy")
    print("Spline fit complete.")

    # Plot
    plot_efficiency_with_spline(dt_centers, eff, spline)

    return spline, (dt_centers, eff, hist)


if __name__ == "__main__":
    analyze_efficiency_file("./data/Digitizer Efficiency Overnight.txt")
