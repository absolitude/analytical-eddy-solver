
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(green_dir, fourier_npz, half_width_mm=80.0):
    green_dir = Path(green_dir)

    geom = pd.read_csv(green_dir / "boundary_panels.csv")
    drive = pd.read_csv(green_dir / "drive_A.csv")
    npz = np.load(fourier_npz)

    required_geom = {"x_mid_mm", "y_mid_mm"}
    required_drive = {"drive_A_real", "drive_A_imag"}

    if not required_geom.issubset(geom.columns):
        raise ValueError(f"Missing geometry columns: {required_geom - set(geom.columns)}")

    if not required_drive.issubset(drive.columns):
        raise ValueError(f"Missing drive columns: {required_drive - set(drive.columns)}")

    if len(geom) != len(drive):
        raise ValueError(
            f"Panel count mismatch: geometry={len(geom)}, drive={len(drive)}"
        )

    x_f = np.asarray(npz["x_m"], dtype=float)
    At_f = np.asarray(npz["applied_A_top_static"], dtype=complex)
    Ab_f = np.asarray(npz["applied_A_bottom_static"], dtype=complex)

    Ag = (
        drive["drive_A_real"].to_numpy(dtype=float)
        + 1j * drive["drive_A_imag"].to_numpy(dtype=float)
    )

    order_f = np.argsort(x_f)
    x_f = x_f[order_f]
    At_f = At_f[order_f]
    Ab_f = Ab_f[order_f]

    x_g = geom["x_mid_mm"].to_numpy(dtype=float) * 1e-3
    y_g = geom["y_mid_mm"].to_numpy(dtype=float) * 1e-3

    def report(label, mask, Af):
        x = x_g[mask]
        ag = Ag[mask]

        order = np.argsort(x)
        x = x[order]
        ag = ag[order]

        af = (
            np.interp(x, x_f, Af.real)
            + 1j * np.interp(x, x_f, Af.imag)
        )

        difference = ag - af
        offset = np.mean(difference)
        difference_corrected = difference - offset

        def metrics(diff):
            rms_ref = np.sqrt(np.mean(np.abs(af)**2))
            peak_ref = np.max(np.abs(af))
            return (
                100 * np.sqrt(np.mean(np.abs(diff)**2)) / rms_ref,
                100 * np.max(np.abs(diff)) / peak_ref,
            )

        rms_raw, max_raw = metrics(difference)
        rms_corr, max_corr = metrics(difference_corrected)

        print(f"\n{label}")
        print(f"  Samples: {len(x)}")
        print(f"  Green RMS |A|: {np.sqrt(np.mean(np.abs(ag)**2)):.6e} T*m")
        print(f"  Fourier RMS |A|: {np.sqrt(np.mean(np.abs(af)**2)):.6e} T*m")
        print(f"  Mean complex offset: {offset.real:.6e} + "
              f"{offset.imag:.6e}j T*m")
        print(f"  Raw relative RMS: {rms_raw:.6f}%")
        print(f"  Raw relative Max: {max_raw:.6f}%")
        print(f"  Offset-corrected RMS: {rms_corr:.6f}%")
        print(f"  Offset-corrected Max: {max_corr:.6f}%")

    central = np.abs(x_g) <= half_width_mm * 1e-3

    top = (y_g > 0) & central
    bottom = (y_g < 0) & central

    report("Upper plate", top, At_f)
    report("Lower plate", bottom, Ab_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--green-dir", required=True)
    parser.add_argument("--fourier", required=True)
    parser.add_argument("--x-half-width-mm", type=float, default=80.0)
    args = parser.parse_args()

    compare(
        args.green_dir,
        args.fourier,
        args.x_half_width_mm,
    )
