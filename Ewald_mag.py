# ewald_spin_ice/cli_serial.py

import os, sys, ast, argparse
import numpy as np
from pathlib import Path
import pandas as pd

from core import (
    GeometryConfig,
    EwaldConfig,
    SpinIceEwaldSolver,
    enumerate_ktriples_first_N,
    enumerate_ktriples_first_N_sc,
    enumerate_ktriples_band,
    enumerate_ktriples_nlist,
    make_tasks,
    dump_tasks_tsv,
)


def main(argv=None):
    """
    Serial driver: runs ALL tasks one-by-one on a single process.

    Usage examples:
      python -m ewald_spin_ice.cli_serial --select-by band --omega-min 1e6 --omega-max 2.154e7
      python -m ewald_spin_ice.cli_serial --select-by nlist --nlist "(1,1,1);(2,1,1)" --outdir ./bmaps
    """
    # -------------------------
    # Default simulation params
    # -------------------------
    geom = GeometryConfig(
        Lx_src=65e-6,                      # x- length of sample
        Ly_src=65e-6/(2*np.sqrt(2)),      # y-length of sample
        Lz_src=10e-6,                      # z-length of sample
        vacuum_target=20e-6,               # extent of vacuum from each edge of sample
        Nx_src_target=100,                 # number of grid points inside sample region in x-direction 
        Ny_src_target=100,                 # number of grid points inside sample region in y-direction 
        Nz_src_target=100,                 # number of grid points inside sample region in x-direction 
        z_above=0.8e-6,
    )

    ewald = EwaldConfig(
        eps=1e-9,
        L_scale=2e-6,
        use_nquad_nearfield=False,
        reltol_ewald_real=1e-3,
        reltol_direct=2e-3,
        measure_error=False,
    )

    # Defaults for mode selection
    """
    3 modes are avaiable for choosing magnetisation modes for which stray field is to be evaluated.
    
    1. SELECT_BY = "count": evaluates the stray magnetic field for the first N_K modes ordered by their energies
    2. SELECT_BY = "band" : evaluates the stray magnetic field for the modes that are within the energy(frequency) resolved band [OMEGA_MIN, OMEGA_MAX]
    3. SELECT_BY = "nlist": evaluates the stray magnetic field for the modes mentioned in NLIST
    """
    SELECT_BY = "nlist"    # "count", "band", or "nlist"
    N_K       = 1
    OMEGA_MIN = 1e6
    OMEGA_MAX = 2.154e7
    NLIST     = [(1,1,0)]

    # -------------------------
    # CLI parsing
    # -------------------------
    ap = argparse.ArgumentParser(description="Serial spin-ice Ewald solver (no multiprocessing).")
    ap.add_argument("--select-by", type=str, choices=["count","band","nlist"],
                    help="How to select k-modes (default: band).")
    ap.add_argument("--count", type=int, default=None,
                    help="If select-by=count, number of modes.")
    ap.add_argument("--omega-min", type=float, default=None,
                    help="If select-by=band, minimum omega.")
    ap.add_argument("--omega-max", type=float, default=None,
                    help="If select-by=band, maximum omega.")
    ap.add_argument("--nlist", type=str, default=None,
                    help='If select-by=nlist, semicolon-separated list like "(1,1,0);(1,1,1)".')
    ap.add_argument("--outdir", type=str, default="./bmaps_serial",
                    help="Output directory for bmap_*.dat and run_log.csv (default: ./bmaps_serial).")
    ap.add_argument("--dry-run-count", action="store_true",
                    help="Print number of tasks and exit.")
    ap.add_argument("--dump-tasks", type=str, default=None,
                    help="Write tasks TSV here and exit.")
    args = ap.parse_args(argv)

    # Allow CLI to override defaults
    if args.select_by:
        SELECT_BY = args.select_by
    if args.count is not None:
        N_K = args.count
    if args.omega_min is not None:
        OMEGA_MIN = args.omega_min
    if args.omega_max is not None:
        OMEGA_MAX = args.omega_max
    if args.nlist:
        NLIST = [ast.literal_eval(tok) for tok in args.nlist.split(";") if tok.strip()]

    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Build modes / tasks
    # -------------------------
    if SELECT_BY == "count":
        modes_full = {
            "ins": enumerate_ktriples_first_N(N_K, geom=geom),
            "sc" : enumerate_ktriples_first_N_sc(N_K, geom=geom),
        }
    elif SELECT_BY == "band":
        modes_full = {
            "ins": enumerate_ktriples_band("ins", OMEGA_MIN, OMEGA_MAX, geom=geom),
            "sc" : enumerate_ktriples_band("sc",  OMEGA_MIN, OMEGA_MAX, geom=geom),
        }
    elif SELECT_BY == "nlist":
        modes_full = enumerate_ktriples_nlist(NLIST)
    else:
        raise ValueError("SELECT_BY must be 'count', 'band', or 'nlist'.")

    # For now, keep only 'sc' like your original script
    modes = {"sc": modes_full["sc"]}
    tasks = make_tasks(modes)

    if args.dry_run_count:
        print(len(tasks))
        return

    if args.dump_tasks:
        dump_tasks_tsv(tasks, args.dump_tasks)
        print(f"Wrote {len(tasks)} tasks to {args.dump_tasks}")
        return

    # -------------------------
    # Build solver
    # -------------------------
    solver = SpinIceEwaldSolver(
        geom=geom,
        ewald=ewald,
        amp=1,
        width=None,
        center=None,
        verbose=True,
    )

    # -------------------------
    # Serial loop over tasks
    # -------------------------
    all_rows = []
    print(f"Running {len(tasks)} tasks serially into '{OUTDIR}' ...")
    for t in tasks:
        row = solver.process_one_task(t, OUTDIR)
        all_rows.append(row)

    df_log = pd.DataFrame(all_rows)
    log_path = OUTDIR / "run_log.csv"
    if log_path.exists():
        df_log.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_log.to_csv(log_path, index=False)

    g = solver.grid
    print(f"\nWrote {len(all_rows)} rows to {log_path}")
    print(f"Box: Lx={g.Lx_box*1e6:.2f} µm, Ly={g.Ly_box*1e6:.2f} µm, Lz={g.Lz_box*1e6:.2f} µm")
    print(
        f"FFT grid: Nx={g.Nx}, Ny={g.Ny}, Nz={g.Nz} | "
        f"dx={g.dx*1e6:.3f} µm, dy={g.dy*1e6:.3f} µm, dz={g.dz*1e6:.3f} µm"
    )
    print(
        f"z_obs (snapped to grid) = {g.z_obs*1e6:.3f} µm "
        f"(z_top={g.z_top*1e6:.3f} µm, z_above≈{g.z_above_eff*1e6:.3f} µm)"
    )


if __name__ == "__main__":
    main()
