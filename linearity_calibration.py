import os
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

from load_data import load_digitizer


def parse_true_amplitudes(fname):
    """
    From 'Linearity_1-25_0-50.txt' -> (1.25, 0.50)
    Assumes naming scheme: Linearity_<P1>_<P2>.txt
    where '1-25' means 1.25 V, '0-50' means 0.50 V.
    """
    base = os.path.basename(fname)
    core = base.replace("Linearity_", "").replace(".txt", "")
    part1, part2 = core.split("_")
    p1_str = part1.replace("-", ".")
    p2_str = part2.replace("-", ".")
    return float(p1_str), float(p2_str)


def collect_linearity_points(data_dir="./data"):
    """
    Scan all Linearity_*.txt files and collect:
      - true A1 amplitude (V) vs measured mean A1
      - true A2 amplitude (V) vs measured mean A2

    Returns:
      p1_true, p1_meas, p1_err
      p2_true, p2_meas, p2_err
    """

    pattern = os.path.join(data_dir, "Linearity_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(f"No files matching {pattern}")

    p1_true_list = []
    p1_meas_list = []
    p1_err_list  = []

    p2_true_list = []
    p2_meas_list = []
    p2_err_list  = []

    for path in files:
        true_p1, true_p2 = parse_true_amplitudes(path)
        data = load_digitizer(path)

        # Use all events; these are clean pulser runs
        A1 = data["A1"]
        A2 = data["A2"]

        # basic sanity cut: ignore obviously zero garbage
        A1 = A1[A1 > 0.0]
        A2 = A2[A2 > 0.0]

        if len(A1) == 0 or len(A2) == 0:
            continue

        mean_A1 = np.mean(A1)
        std_A1  = np.std(A1)
        err_A1  = std_A1 / np.sqrt(len(A1))

        mean_A2 = np.mean(A2)
        std_A2  = np.std(A2)
        err_A2  = std_A2 / np.sqrt(len(A2))

        p1_true_list.append(true_p1)
        p1_meas_list.append(mean_A1)
        p1_err_list.append(err_A1)

        p2_true_list.append(true_p2)
        p2_meas_list.append(mean_A2)
        p2_err_list.append(err_A2)

        print(f"{os.path.basename(path):30s}  "
              f"P1_true={true_p1:.2f} V,  A1_mean={mean_A1:.4f} V  "
              f"P2_true={true_p2:.2f} V,  A2_mean={mean_A2:.4f} V")

    p1_true = np.array(p1_true_list)
    p1_meas = np.array(p1_meas_list)
    p1_err  = np.array(p1_err_list)

    p2_true = np.array(p2_true_list)
    p2_meas = np.array(p2_meas_list)
    p2_err  = np.array(p2_err_list)

    return p1_true, p1_meas, p1_err, p2_true, p2_meas, p2_err


def linear_fit(true_vals, meas_vals, meas_err=None):
    """
    Fit A_meas = a + b * A_true
    If meas_err given, do weighted fit.
    Returns (a, b).
    """
    x = true_vals
    y = meas_vals

    if meas_err is not None:
        w = 1.0 / meas_err
        coeffs = np.polyfit(x, y, 1, w=w)
    else:
        coeffs = np.polyfit(x, y, 1)

    b, a = coeffs[0], coeffs[1]  # np.polyfit returns highest power first
    return a, b


def plot_linearity_two_panel(true_vals1, meas_vals1, err1, a1, b1,
                             true_vals2, meas_vals2, err2, a2, b2,
                             output_path="./figures/linearity_two_panel.pdf"):
    """
    Plot A1 and A2 linearity in a 1x2 grid using LaTeX formatting.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax1, ax2 = axes

    if err1 is not None:
        ax1.errorbar(true_vals1, meas_vals1, yerr=err1,
                     fmt="o", alpha=0.9, label=r"Data")
    else:
        ax1.plot(true_vals1, meas_vals1, "o", label=r"Data")

    xfit = np.linspace(true_vals1.min()*0.95, true_vals1.max()*1.05, 300)
    yfit = a1 + b1 * xfit

    ax1.plot(xfit, yfit, "-", lw=2,
             label=(rf"$A_{{\mathrm{{meas}}}} = a + b A_{{\mathrm{{true}}}}$" "\n" +
                    rf"$a={a1:.4f}\,\mathrm{{V}},\; b={b1:.4f}$"))

    ax1.set_title(r"\textbf{Linearity Calibration for $A_1$}")
    ax1.set_xlabel(r"True amplitude $A_{1,\mathrm{true}}\,(\mathrm{V})$")
    ax1.set_ylabel(r"Measured amplitude $A_{1,\mathrm{meas}}\,(\mathrm{V})$")
    ax1.grid(alpha=0.3)
    ax1.legend()

    if err2 is not None:
        ax2.errorbar(true_vals2, meas_vals2, yerr=err2,
                     fmt="o", alpha=0.9, label=r"Data")
    else:
        ax2.plot(true_vals2, meas_vals2, "o", label=r"Data")

    xfit2 = np.linspace(true_vals2.min()*0.95, true_vals2.max()*1.05, 300)
    yfit2 = a2 + b2 * xfit2

    ax2.plot(xfit2, yfit2, "-", lw=2,
             label=(rf"$A_{{\mathrm{{meas}}}} = a + b A_{{\mathrm{{true}}}}$" "\n" +
                    rf"$a={a2:.4f}\,\mathrm{{V}},\; b={b2:.4f}$"))

    ax2.set_title(r"\textbf{Linearity Calibration for $A_2$}")
    ax2.set_xlabel(r"True amplitude $A_{2,\mathrm{true}}\,(\mathrm{V})$")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)

    plt.show()


def main():
    data_dir = "./data"

    (p1_true, p1_meas, p1_err,
     p2_true, p2_meas, p2_err) = collect_linearity_points(data_dir=data_dir)

    # --- A1 fit
    a1, b1 = linear_fit(p1_true, p1_meas, p1_err)
    print("\nLinearity fit for A1:")
    print(f"  A1_meas = a1 + b1 * A1_true")
    print(f"  a1 = {a1:.6f} V")
    print(f"  b1 = {b1:.6f}")
    np.save("./data/linearity_A1_coeffs.npy", np.array([a1, b1]))

    # --- A2 fit
    a2, b2 = linear_fit(p2_true, p2_meas, p2_err)
    print("\nLinearity fit for A2:")
    print(f"  A2_meas = a2 + b2 * A2_true")
    print(f"  a2 = {a2:.6f} V")
    print(f"  b2 = {b2:.6f}")
    np.save("./data/linearity_A2_coeffs.npy", np.array([a2, b2]))

    # --- Two-panel plot
    plot_linearity_two_panel(
        p1_true, p1_meas, p1_err, a1, b1,
        p2_true, p2_meas, p2_err, a2, b2,
        output_path="./figures/linearity_two_panel.pdf"
    )

    print("\nSaved:")
    print("  linearity_A1_coeffs.npy  (a1, b1)")
    print("  linearity_A2_coeffs.npy  (a2, b2)")
    print("  linearity_two_panel.pdf")



if __name__ == "__main__":
    main()
