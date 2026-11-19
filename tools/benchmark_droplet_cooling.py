"""Time DropletCooling: cost per step against n_shells, and one default-size run.

    uv run python tools/benchmark_droplet_cooling.py [--full]
"""
import argparse
import time

from droplet_temperature import DropletCooling

SETUP = {"diameter_um": 18.7, "spacing_um": 85.5, "frequency_hz": 157e3, "pressure_pa": 1.6}


def timed(**kwargs):
    start = time.perf_counter()
    cool = DropletCooling(**SETUP, **kwargs)
    return cool, time.perf_counter() - start


def backends():
    """fast=False (numpy), plus fast=True when numba is installed."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return [False]
    return [False, True]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run the default 7 ms simulation")
    args = parser.parse_args()

    for fast in backends():
        timed(max_time=1e-5, fast=fast)                      # warm-up: imports, numba compile or cache load
    print("cost per step, 18.7 um droplet, default dt (shrinks above ~100 shells), 0.2 ms")
    print(f"{'backend':>8} {'n_shells':>9} {'steps':>8} {'seconds':>9} {'us/step':>9}")
    for fast in backends():
        for n_shells in (25, 50, 100, 200, 400):
            cool, seconds = timed(max_time=2e-4, n_shells=n_shells, fast=fast)
            print(f"{cool.backend:>8} {n_shells:>9} {cool.n_steps:>8} {seconds:>9.2f} "
                  f"{1e6 * seconds / cool.n_steps:>9.2f}")

    if args.full:
        for fast in backends():
            cool, seconds = timed(fast=fast)
            print(f"\ndefault run, {cool.backend}: {cool.n_steps} steps, {seconds:.1f} s, "
                  f"T(5 mm) = {float(cool.temperature(5)):.3f} K, T(65 mm) = {float(cool.temperature(65)):.3f} K")


if __name__ == "__main__":
    main()
