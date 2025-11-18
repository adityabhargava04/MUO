# load_data.py
import numpy as np
import re

def load_digitizer(path):
    """
    Loads a 12-column digitizer text file by skipping the standard header lines.
    Assumes fixed format:
    - line 0: metadata
    - line 1: settings
    - line 2: blank
    - line 3: column headers
    - line 4+: numeric data
    """

    raw = np.loadtxt(path, skiprows=4)  # skip metadata + header

    data = {
        # Channel 0
        "t1": raw[:,0],
        "A1": raw[:,1],
        "w1": raw[:,2],
        "t2": raw[:,3],
        "A2": raw[:,4],
        "w2": raw[:,5],

        # Channel 1
        "t1_ch1": raw[:,6],
        "A1_ch1": raw[:,7],
        "w1_ch1": raw[:,8],
        "t2_ch1": raw[:,9],
        "A2_ch1": raw[:,10],
        "w2_ch1": raw[:,11],
    }

    dt = data["t2"] - data["t1"]
    data["dt"] = dt
    data["dt_us"] = dt * 1e6

    data["valid_dt_mask"] = (dt > 0) & (dt < 1e-4)

    return data


def apply_basic_cuts(data, amp1_min=0.05, amp2_min=0.02,
                     w1_limits=None, w2_limits=None):
    """
    Returns a mask based on amplitude and optional width constraints.
    """

    mask = np.ones_like(data["t1"], dtype=bool)

    # amplitude cuts (very common)
    mask &= data["A1"] > amp1_min
    mask &= data["A2"] > amp2_min

    # width cuts (optional)
    if w1_limits is not None:
        wmin, wmax = w1_limits
        mask &= (data["w1"] > wmin) & (data["w1"] < wmax)

    if w2_limits is not None:
        wmin, wmax = w2_limits
        mask &= (data["w2"] > wmin) & (data["w2"] < wmax)

    return mask


def describe(data):
    """Print basic statistics for quick sanity checking of a dataset."""
    print("Number of events:", len(data["t1"]))

    dt = data["dt_us"]
    print(f"Δt_us: min={dt.min():.4f}, max={dt.max():.4f}, mean={dt.mean():.4f}")

    print(f"Amp1: min={data['A1'].min():.4f}, max={data['A1'].max():.4f}")
    print(f"Amp2: min={data['A2'].min():.4f}, max={data['A2'].max():.4f}")

    print(f"Width1: min={data['w1'].min():.4e}, max={data['w1'].max():.4e}")
    print(f"Width2: min={data['w2'].min():.4e}, max={data['w2'].max():.4e}")
