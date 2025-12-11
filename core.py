

import os, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, Tuple, List

import numpy as np
import numpy.fft as fft
from scipy.special import erfc
from scipy.integrate import nquad
import pandas as pd
from math import sqrt, pi
from multiprocessing import Pool
from functools import partial

# =============================================================================
# Physical constants
# =============================================================================

μ0 = 1.0        # set to 4*np.pi*1e-7 for SI if desired
v_mode = 10.0   # m/s (spin-ice mode velocity; omega = v*|k|)

FP = np.float32
CP = np.complex64


# =============================================================================
# Config / context objects
# =============================================================================

@dataclass
class GeometryConfig:
    Lx_src: float = 65e-6
    Ly_src: Optional[float] = None  # if None, use Lx/(2√2)
    Lz_src: float = 10e-6
    vacuum_target: float = 20e-6    # per side
    Nx_src_target: int = 100
    Ny_src_target: int = 100
    Nz_src_target: int = 100
    z_above: float = 0.8e-6         # observation height above top surface

@dataclass
class EwaldConfig:
    eps: float = 1e-9               # master tolerance
    L_scale: float = 2e-6
    use_nquad_nearfield: bool = False
    reltol_ewald_real: float = 1e-3
    reltol_direct: float = 2e-3     # for bare direct integrator
    measure_error: bool = False

    @property
    def alpha(self) -> float:
        return np.pi * np.sqrt(np.log(2.0 / self.eps)) / self.L_scale

    @property
    def realcut_factor(self) -> float:
        return float(np.sqrt(np.log(1.0 / self.eps)))

@dataclass
class GridContext:
    # geometry
    Lx_src: float
    Ly_src: float
    Lz_src: float
    vacuum_target: float

    # FFT grid
    dx: float; dy: float; dz: float
    Nx: int; Ny: int; Nz: int
    Nx_src: int; Ny_src: int; Nz_src: int
    Nx_pad: int; Ny_pad: int; Nz_pad: int
    Lx_box: float; Ly_box: float; Lz_box: float
    vacuum_x: float; vacuum_y: float; vacuum_z: float

    # coordinates
    x_1d: np.ndarray
    y_1d: np.ndarray
    z_1d: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    mask_src: np.ndarray

    # observation plane
    k_obs: int
    z_obs: float
    z_top: float
    z_above_eff: float

    # convenience
    dV: float
    tol: float = 1e-15


def _is_smooth235(n: int) -> bool:
    if n < 1:
        return False
    for p in (2, 3, 5):
        while n % p == 0 and n > 1:
            n //= p
    return n == 1

def _smooth_ceil_with_parity(n_float: float, parity: int, search_pad: int = 256) -> int:
    n0 = int(math.ceil(n_float))
    m = max(1, n0)
    limit = n0 + max(32, search_pad)
    while m <= limit:
        if (m % 2) == parity and _is_smooth235(m):
            return m
        m += 1
    pow2 = 1 << int(np.ceil(np.log2(max(n0, 2))))
    if (pow2 % 2) == parity:
        return pow2
    return pow2 + 1

def _choose_1d_grid(L_src, N_src_target, V_target):
    dx = L_src / N_src_target
    N_total_target = N_src_target + 2.0 * V_target / dx
    parity = N_src_target % 2
    N_total = _smooth_ceil_with_parity(N_total_target, parity=parity)

    if N_total <= N_src_target:
        N_total = N_src_target + 2

    pad = N_total - N_src_target
    if pad % 2 != 0:
        N_total += 1
        pad = N_total - N_src_target
    N_pad = pad // 2
    V_eff = N_pad * dx
    return dx, int(N_total), int(N_src_target), int(N_pad), V_eff


def build_grid_context(geom: GeometryConfig, verbose: bool = True) -> GridContext:
    Lx_src = geom.Lx_src
    Ly_src = geom.Ly_src if geom.Ly_src is not None else Lx_src / (2.0 * np.sqrt(2.0))
    Lz_src = geom.Lz_src
    VACUUM_TARGET = geom.vacuum_target

    dx, Nx, Nx_src, Nx_pad, Vx_eff = _choose_1d_grid(Lx_src, geom.Nx_src_target, VACUUM_TARGET)
    dy, Ny, Ny_src, Ny_pad, Vy_eff = _choose_1d_grid(Ly_src, geom.Ny_src_target, VACUUM_TARGET)
    dz, Nz, Nz_src, Nz_pad, Vz_eff = _choose_1d_grid(Lz_src, geom.Nz_src_target, VACUUM_TARGET)

    Lx_box = Nx * dx
    Ly_box = Ny * dy
    Lz_box = Nz * dz

    x_1d = (np.arange(Nx, dtype=np.float64) + 0.5 - Nx_pad) * dx
    y_1d = (np.arange(Ny, dtype=np.float64) + 0.5 - Ny_pad) * dy
    z_1d = (np.arange(Nz, dtype=np.float64) + 0.5 - Nz_pad) * dz

    xs, ys, zs = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    mask_src = (
        (xs >= 0.0) & (xs < Lx_src) &
        (ys >= 0.0) & (ys < Ly_src) &
        (zs >= 0.0) & (zs < Lz_src)
    )

    z_top = Lz_src
    z_obs_target = z_top + geom.z_above
    k_obs = int(np.argmin(np.abs(z_1d - z_obs_target)))
    z_obs = float(z_1d[k_obs])
    z_above_eff = z_obs - z_top

    if verbose:
        print(f"[grid] VACUUM_TARGET = {VACUUM_TARGET*1e6:.3f} µm")
        print(
            "[grid] Effective vacuum per side: "
            f"Vx={Vx_eff*1e6:.3f} µm, Vy={Vy_eff*1e6:.3f} µm, Vz={Vz_eff*1e6:.3f} µm"
        )
        print(f"[grid] FFT sizes: Nx={Nx}, Ny={Ny}, Nz={Nz}")
        print(f"[grid] dx={dx*1e6:.4f} µm, dy={dy*1e6:.4f} µm, dz={dz*1e6:.4f} µm")
        print(
            "[grid] points in source (eff): "
            f"Nx_src={Nx_src}, Ny_src={Ny_src}, Nz_src={Nz_src}"
        )
        print(
            f"[grid] z_obs_target = {z_obs_target*1e6:.3f} µm, snapped to grid plane "
            f"k_obs={k_obs}, z_obs={z_obs*1e6:.3f} µm"
        )
        print(
            f"[grid] z_above_target = {geom.z_above*1e6:.3f} µm, "
            f"z_above_eff = {z_above_eff*1e6:.3f} µm"
        )

    return GridContext(
        Lx_src=Lx_src,
        Ly_src=Ly_src,
        Lz_src=Lz_src,
        vacuum_target=VACUUM_TARGET,
        dx=dx, dy=dy, dz=dz,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nx_src=Nx_src, Ny_src=Ny_src, Nz_src=Nz_src,
        Nx_pad=Nx_pad, Ny_pad=Ny_pad, Nz_pad=Nz_pad,
        Lx_box=Lx_box, Ly_box=Ly_box, Lz_box=Lz_box,
        vacuum_x=Vx_eff, vacuum_y=Vy_eff, vacuum_z=Vz_eff,
        x_1d=x_1d, y_1d=y_1d, z_1d=z_1d,
        xs=xs, ys=ys, zs=zs,
        mask_src=mask_src,
        k_obs=k_obs,
        z_obs=z_obs,
        z_top=z_top,
        z_above_eff=z_above_eff,
        dV=dx * dy * dz,
    )


# =============================================================================
# Solver class
# =============================================================================

class SpinIceEwaldSolver:
    """
    Encapsulates full Ewald machinery to evaluate the stray magnetic field
    arising from a finite sized magnetized material with a 
    a number of differnet magnetization mode choices,
    including spin-ice magnetization modes.
    
    """

    def __init__(
        self,
        geom: GeometryConfig,
        ewald: EwaldConfig,
        amp: float = 1.0,
        width: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
        verbose: bool = True,
    ):
        self.geom = geom
        self.ewald = ewald
        self.AMP = amp
        self.WIDTH = width
        self.CENTER = center

        self.grid = build_grid_context(geom, verbose=verbose)

        # convenience aliases
        self.alpha = ewald.alpha
        self.realcut_factor = ewald.realcut_factor
        self.alpha_formula = "alpha = pi sqrt(log(2/eps)) / L_scale"

    #=============================================================================
    # Magnetization modes
    #=============================================================================
    """
    Encapsulates full Ewald machinery to evaluate the stray magnetic field
    arising from a finite sized magnetized material with a 
    a number of differnet magnetization mode choices,
    including spin-ice magnetization modes.
    
    """
    def _build_mode(
        self,
        mode: str,
        kvec: Optional[np.ndarray] = None,
        n: Optional[Tuple[int, int, int]] = None,
        lambda_idx: Optional[int] = None,
        amp: Optional[float] = None,
        width: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ):
        g = self.grid
        Lx_src, Ly_src, Lz_src = g.Lx_src, g.Ly_src, g.Lz_src
        xs, ys, zs = g.xs, g.ys, g.zs
        mask_src = g.mask_src

        if amp is None:
            amp = self.AMP
        if width is None:
            width = self.WIDTH
        if center is None:
            center = self.CENTER

        if n is not None:
            nx, ny, nz = n
            kx = nx * np.pi / Lx_src
            ky = ny * np.pi / Ly_src
            kz = nz * np.pi / Lz_src
        else:
            if kvec is None:
                kvec = np.pi * np.array([1/Lx_src, 1/Ly_src, 2/Lz_src])
            kx, ky, kz = map(float, kvec)

        Δx = 0.0
        Δy = 0.0
        Δz = 0.0

        if center is None:
            xc = Δx + Lx_src / 2.0
            yc = Δy + Ly_src / 2.0
        else:
            xc = float(center[0])
            yc = float(center[1])

        Xc = xs - xc
        Yc = ys - yc
        σ = (0.25 * min(Lx_src, Ly_src)) if width is None else float(width)
        env = np.exp(-(Xc**2 + Yc**2) / (2 * σ * σ))

        m_x = np.zeros_like(xs, dtype=np.float64)
        m_y = np.zeros_like(xs, dtype=np.float64)
        m_z = np.zeros_like(xs, dtype=np.float64)

        all_nonzero = (kx != 0.0) and (ky != 0.0) and (kz != 0.0)

        def choose_amps():
            if not all_nonzero:
                return None, None
            if lambda_idx is None:
                return 1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)
            if lambda_idx == 0:
                return 1.0, 0.0
            if lambda_idx == 1:
                return 0.0, 1.0
            raise ValueError("lambda_idx must be None, 0, or 1")

        # NOTE: env is built but not applied in your original code;
        # if you want it, multiply m_* by env before assigning into rho_*.

        if mode == "uniform_z":
            m_z[:] = amp
        elif mode == "uniform_x":
            m_x[:] = amp

        elif mode == "spin_ice_insulating":
            kmag = np.sqrt(kx**2 + ky**2 + kz**2)
            if not all_nonzero:
                if kx == 0 and ky != 0 and kz != 0:
                    m_y[:] = ((-kz/kmag))*np.cos(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
                    m_z[:] = ((ky/kmag))*np.cos(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.sin(kz*(zs-Δz))
                elif ky == 0 and kx != 0 and kz != 0:
                    m_x[:] = ((kz/kmag))*np.sin(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
                    m_z[:] = (-(kx/kmag))*np.cos(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.sin(kz*(zs-Δz))
                elif kz == 0 and kx != 0 and ky != 0:
                    m_x[:] = (-(ky/kmag))*np.sin(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
                    m_y[:] = ((kx/kmag))*np.cos(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
            else:
                amp_1, amp_2 = choose_amps()
                m_x[:] = (amp_1*(-kx*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(ky/(np.sqrt(kmag**2 - kz**2)))) \
                         * np.sin(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
                m_y[:] = (amp_1*(-ky*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(-kx/(np.sqrt(kmag**2 - kz**2)))) \
                         * np.cos(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
                m_z[:] = (amp_1*np.sqrt(1 - (kz**2)/(kmag**2))) \
                         * np.cos(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.sin(kz*(zs-Δz))

        elif mode == "spin_ice_superconducting":
            kmag = np.sqrt(kx**2 + ky**2 + kz**2)
            if not all_nonzero:
                if kx==0 and ky!=0 and kz!=0:
                    m_x[:] = np.cos(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.sin(kz*(zs-Δz))
                elif ky==0 and kx!=0 and kz!=0:
                    m_y[:] = np.sin(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.sin(kz*(zs-Δz))
                elif kz==0 and kx!=0 and ky!=0:
                    m_z[:] = np.sin(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.cos(kz*(zs-Δz))
            else:
                amp_1, amp_2 = choose_amps()
                m_x[:] = (amp_1*(-kx*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(ky/(np.sqrt(kmag**2 - kz**2)))) \
                         * np.cos(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.sin(kz*(zs-Δz))
                m_y[:] = (amp_1*(-ky*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(-kx/(np.sqrt(kmag**2 - kz**2)))) \
                         * np.sin(kx*(xs-Δx))*np.cos(ky*(ys-Δy))*np.sin(kz*(zs-Δz))
                m_z[:] = (amp_1*np.sqrt(1 - (kz**2)/(kmag**2))) \
                         * np.sin(kx*(xs-Δx))*np.sin(ky*(ys-Δy))*np.cos(kz*(zs-Δz))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        rho_x = np.zeros_like(xs, dtype=CP)
        rho_y = np.zeros_like(xs, dtype=CP)
        rho_z = np.zeros_like(xs, dtype=CP)
        rho_x[mask_src] = m_x[mask_src].astype(FP)
        rho_y[mask_src] = m_y[mask_src].astype(FP)
        rho_z[mask_src] = m_z[mask_src].astype(FP)
        return rho_x, rho_y, rho_z

    #=============================================================================
    # Continuous magnetization  
    #=============================================================================
    """
    Build continuous magnetization modes instead of on a grid.
    Important for evaluting the relative error of the ewald summation
    """
    def _m_eval_continuous(
        self,
        xp, yp, zp,
        *,
        mode: str,
        kvec: np.ndarray,
        lambda_idx: Optional[int] = None,
        amp: Optional[float] = None,
        width: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        g = self.grid
        Lx_src, Ly_src, Lz_src = g.Lx_src, g.Ly_src, g.Lz_src
        if amp is None:
            amp = self.AMP
        if width is None:
            width = self.WIDTH
        if center is None:
            center = self.CENTER

        Δx = 0.0
        Δy = 0.0
        Δz = 0.0

        xpL, ypL, zpL = xp-Δx, yp-Δy, zp-Δz
        if center is None:
            xc = Δx + Lx_src/2.0
            yc = Δy + Ly_src/2.0
        else:
            xc = float(center[0])
            yc = float(center[1])
        Xc, Yc = xp - xc, yp - yc
        σ  = (0.25*min(Lx_src, Ly_src)) if width is None else float(width)
        env = np.exp(-(Xc**2 + Yc**2)/(2*σ*σ))

        kx, ky, kz = map(float, kvec)
        all_nonzero = (kx != 0.0) and (ky != 0.0) and (kz != 0.0)

        def choose_amps():
            if not all_nonzero:
                return None, None
            if lambda_idx is None:
                return 1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)
            if lambda_idx == 0:
                return 1.0, 0.0
            if lambda_idx == 1:
                return 0.0, 1.0
            raise ValueError("lambda_idx must be None, 0, or 1")

        if mode == "uniform_z":
            return np.array([0.0, 0.0, amp], float)
        if mode == "uniform_x":
            return np.array([amp, 0.0, 0.0], float)

        if mode == "spin_ice_insulating":
            kmag = np.sqrt(kx**2 + ky**2 + kz**2)
            if not all_nonzero:
                if kx==0 and ky!=0 and kz!=0:
                    my = ((-kz/kmag))*np.cos(kx*xpL)*np.sin(ky*ypL)*np.cos(kz*zpL)
                    mz = ((ky/kmag))*np.cos(kx*xpL)*np.cos(ky*ypL)*np.sin(kz*zpL)
                    return np.array([0.0, my, mz], float)
                if ky==0 and kx!=0 and kz!=0:
                    mx = ((kz/kmag))*np.sin(kx*xpL)*np.cos(ky*ypL)*np.cos(kz*zpL)
                    mz = (-(kx/kmag))*np.cos(kx*xpL)*np.cos(ky*ypL)*np.sin(kz*zpL)
                    return np.array([mx, 0.0, mz], float)
                if kz==0 and kx!=0 and ky!=0:
                    mx = (-(ky/kmag))*np.sin(kx*xpL)*np.cos(ky*ypL)*np.cos(kz*zpL)
                    my = ((kx/kmag))*np.cos(kx*xpL)*np.sin(ky*ypL)*np.cos(kz*zpL)
                    return np.array([mx, my, 0.0], float)
            amp_1, amp_2 = choose_amps()
            mx = (amp_1*(-kx*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(ky/(np.sqrt(kmag**2 - kz**2)))) \
                 * np.sin(kx*xpL)*np.cos(ky*ypL)*np.cos(kz*zpL)
            my = (amp_1*(-ky*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(-kx/(np.sqrt(kmag**2 - kz**2)))) \
                 * np.cos(kx*xpL)*np.sin(ky*ypL)*np.cos(kz*zpL)
            mz = (amp_1*np.sqrt(1 - (kz**2)/(kmag**2))) \
                 * np.cos(kx*xpL)*np.cos(ky*ypL)*np.sin(kz*zpL)
            return np.array([mx, my, mz], float)

        if mode == "spin_ice_superconducting":
            kmag = np.sqrt(kx**2 + ky**2 + kz**2)
            if not all_nonzero:
                if kx==0 and ky!=0 and kz!=0:
                    mx = np.cos(kx*xpL)*np.sin(ky*ypL)*np.sin(kz*zpL); return np.array([mx, 0.0, 0.0], float)
                if ky==0 and kx!=0 and kz!=0:
                    my = np.sin(kx*xpL)*np.cos(ky*ypL)*np.sin(kz*zpL); return np.array([0.0, my, 0.0], float)
                if kz==0 and kx!=0 and ky!=0:
                    mz = np.sin(kx*xpL)*np.sin(ky*ypL)*np.cos(kz*zpL); return np.array([0.0, 0.0, mz], float)
            amp_1, amp_2 = choose_amps()
            mx = (amp_1*(-kx*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(ky/(np.sqrt(kmag**2 - kz**2)))) \
                 * np.cos(kx*xpL)*np.sin(ky*ypL)*np.sin(kz*zpL)
            my = (amp_1*(-ky*kz/(kmag*np.sqrt(kmag**2 - kz**2)))+amp_2*(-kx/(np.sqrt(kmag**2 - kz**2)))) \
                 * np.sin(kx*xpL)*np.cos(ky*ypL)*np.sin(kz*zpL)
            mz = (amp_1*np.sqrt(1 - (kz**2)/(kmag**2))) \
                 * np.sin(kx*xpL)*np.sin(ky*ypL)*np.cos(kz*zpL)
            return np.array([mx, my, mz], float)

        raise ValueError(f"Unknown mode for continuous eval: {mode}")

    #=============================================================================
    # Dipole kernels
    #=============================================================================
    
    @staticmethod
    def _H_dipole_tensor(R):
        """
        Builds Dipole kernel, which propogates the magnetization in the sample
        to give stray magnetic fields.
        """
        Rx, Ry, Rz = R
        R2 = Rx*Rx + Ry*Ry + Rz*Rz
        Rm = sqrt(R2)
        if Rm == 0.0:
            return np.zeros((3,3), float)
        c1 = μ0 / (4.0*pi) * (1.0 / (Rm**3))
        c2 = μ0 / (4.0*pi) * (3.0 / (Rm**5))
        eye = np.eye(3)
        RR  = np.array([[Rx*Rx, Rx*Ry, Rx*Rz],
                        [Ry*Rx, Ry*Ry, Ry*Rz],
                        [Rz*Rx, Rz*Ry, Rz*Rz]], float)
        return c1*eye - c2*RR

    def _B_point_direct_mode(self, p, *, mode, kvec, lambda_idx=None):
        """
        Evaluates magnetic field at a given point above the sample 
        directly using a numerical integrator, used to compare against
        the magnetic field obtained from the ewald summation for benchmarking
        the error of the ewald method.
        """
        g = self.grid
        e = self.ewald
        xlim = (0.0, g.Lx_src)
        ylim = (0.0, g.Ly_src)
        zlim = (0.0, g.Lz_src)

        def integrand(mu, xp, yp, zp):
            H = self._H_dipole_tensor((p[0]-xp, p[1]-yp, p[2]-zp))
            m = self._m_eval_continuous(
                xp, yp, zp,
                mode=mode,
                kvec=kvec,
                lambda_idx=lambda_idx,
            )
            return float(H[mu, :].dot(m))

        B = np.zeros(3, float)
        for mu in (0,1,2):
            f = lambda zp, yp, xp: integrand(mu, xp, yp, zp)
            val, _err = nquad(
                f,
                [zlim, ylim, xlim],
                opts={'epsrel': e.reltol_direct, 'epsabs': 0.0},
            )
            B[mu] = val
        return B
    #=============================================================================
    # Real-space ewald integrator 
    #=============================================================================
    @staticmethod
    def _dipole_real_batch(R: np.ndarray, a: float) -> np.ndarray:
        """
        Builds the real-space dipole ewald kernel.
        """
        R2  = np.einsum('...i,...i->...', R, R)
        Rmag = np.sqrt(R2)
        x    = a * Rmag
        with np.errstate(divide='ignore', invalid='ignore'):
            erfcx = erfc(x)
            e     = np.exp(-x*x)
            g1 = erfcx / (Rmag**3) + (2*a/np.sqrt(np.pi)) * e / (Rmag**2)
            g2 = ( erfcx / (Rmag**5)
                 + (2*a/np.sqrt(np.pi)) * e / (Rmag**4)
                 + (4*a**3/(3*np.sqrt(np.pi))) * e / (Rmag**2))
        g1 = np.nan_to_num(g1, nan=0.0, posinf=0.0, neginf=0.0)
        g2 = np.nan_to_num(g2, nan=0.0, posinf=0.0, neginf=0.0)
        eye = np.eye(3)
        return (μ0 / (4*np.pi)) * (g1[:, None, None] * eye
                                   - 3.0 * g2[:, None, None] * R[:, :, None] * R[:, None, :])

    @staticmethod
    def _H_ewald_real_single(R, a):
        """
        Builds the real-space dipole ewald kernel.
        """
        Rx, Ry, Rz = R
        R2 = Rx*Rx + Ry*Ry + Rz*Rz
        Rm = sqrt(R2)
        if Rm == 0.0:
            return np.zeros((3,3), float)
        x = a * Rm
        erfcx = erfc(x)
        e = np.exp(-x*x)
        g1 = erfcx / (Rm**3) + (2*a/np.sqrt(pi)) * e / (Rm**2)
        g2 = ( erfcx / (Rm**5)
             + (2*a/np.sqrt(pi)) * e / (Rm**4)
             + (4*a**3/np.sqrt(pi)) * e / (Rm**2))
        eye = np.eye(3)
        RR  = np.array([[Rx*Rx, Rx*Ry, Rx*Rz],
                        [Ry*Rx, Ry*Ry, Ry*Rz],
                        [Rz*Rx, Rz*Ry, Rz*Rz]], float)
        return (μ0 / (4.0*pi)) * (g1*eye - 3.0*g2*RR)

    def _B_point_ewald_real(self, p, *, mode, kvec, lambda_idx=None):
        """
        Evaluates the real-space (short-range) contribution to the magnetic field
        in the ewald summation
        """
        g = self.grid
        e = self.ewald
        alpha_val = self.alpha
        Rc = self.realcut_factor / alpha_val
        px, py, pz = p

        x_min = max(0.0,          px - Rc)
        x_max = min(g.Lx_src,     px + Rc)
        y_min = max(0.0,          py - Rc)
        y_max = min(g.Ly_src,     py + Rc)
        z_min = max(0.0,          pz - Rc)
        z_max = min(g.Lz_src,     pz + Rc)

        if (x_max <= x_min) or (y_max <= y_min) or (z_max <= z_min):
            return np.zeros(3, float)

        xlim = (x_min, x_max)
        ylim = (y_min, y_max)
        zlim = (z_min, z_max)

        def integrand(mu, xp, yp, zp):
            Rx = px - xp
            Ry = py - yp
            Rz = pz - zp
            R2 = Rx*Rx + Ry*Ry + Rz*Rz
            if R2 > Rc*Rc:
                return 0.0
            H = self._H_ewald_real_single((Rx, Ry, Rz), alpha_val)
            m = self._m_eval_continuous(
                xp, yp, zp,
                mode=mode,
                kvec=kvec,
                lambda_idx=lambda_idx,
            )
            return float(H[mu, :].dot(m))

        B = np.zeros(3, float)
        for mu in (0, 1, 2):
            f = lambda zp, yp, xp: integrand(mu, xp, yp, zp)
            val, _err = nquad(
                f,
                [zlim, ylim, xlim],
                opts={'epsrel': e.reltol_ewald_real, 'epsabs': 0.0}
            )
            B[mu] = val
        return B

    # -----------------------------
    # Reciprocal-space ewald integrator
    # -----------------------------

    def _build_reciprocal_fields_vector(self, rho_x, rho_y, rho_z):
        """
        Evaluates reciprocal-space contribtion to the magnetic field 
        evaluated by the ewald summation method using FFTs
        """
        g = self.grid
        alpha = self.alpha

        rho_k_x = fft.fftn(rho_x, axes=(0,1,2))
        rho_k_y = fft.fftn(rho_y, axes=(0,1,2))
        rho_k_z = fft.fftn(rho_z, axes=(0,1,2))

        qx_1d = (2*np.pi) * fft.fftfreq(g.Nx, d=g.dx)
        qy_1d = (2*np.pi) * fft.fftfreq(g.Ny, d=g.dy)
        qz_1d = (2*np.pi) * fft.fftfreq(g.Nz, d=g.dz)
        qx, qy, qz = np.meshgrid(qx_1d, qy_1d, qz_1d, indexing='ij')
        qx = qx.astype(FP); qy = qy.astype(FP); qz = qz.astype(FP)

        q2 = (qx*qx + qy*qy + qz*qz).astype(FP)
        mask = (q2 > 0)

        gauss = np.zeros_like(q2, dtype=FP)
        gauss[mask] = np.exp(- q2[mask] / (FP(4.0)*FP(alpha)*FP(alpha)))
        s = np.zeros_like(q2, dtype=FP)
        s[mask] = (FP(μ0) * gauss[mask]) / q2[mask]

        dot = (qx.astype(CP)*rho_k_x + qy.astype(CP)*rho_k_y + qz.astype(CP)*rho_k_z)

        Fx_k = (s.astype(CP) * qx.astype(CP)) * dot
        Fy_k = (s.astype(CP) * qy.astype(CP)) * dot
        Fz_k = (s.astype(CP) * qz.astype(CP)) * dot

        field_r_x = fft.ifftn(Fx_k, axes=(0,1,2))
        field_r_y = fft.ifftn(Fy_k, axes=(0,1,2))
        field_r_z = fft.ifftn(Fz_k, axes=(0,1,2))

        return field_r_x, field_r_y, field_r_z

    # =============================================================================
    # Ewald grid function
    #=============================================================================

    def _make_ewald_grid_fn(self, field_r_x, field_r_y, field_r_z,
                            rho_x, rho_y, rho_z,
                            mode_name=None, kvec=None, lambda_idx=None):
        """
        Builds full grid that contains the sample and vacuum.
        The magnetic field is evaluated in the vacuum surrounding the sample 
        using the Ewald summation method.
        """
        g = self.grid
        e = self.ewald
        alpha = self.alpha

        if not e.use_nquad_nearfield:
            src_x  = g.xs[g.mask_src].ravel().astype(np.float64)
            src_y  = g.ys[g.mask_src].ravel().astype(np.float64)
            src_z  = g.zs[g.mask_src].ravel().astype(np.float64)
            src_xyz = np.column_stack((src_x, src_y, src_z))

            mX = rho_x[g.mask_src].real.astype(np.float64).ravel()
            mY = rho_y[g.mask_src].real.astype(np.float64).ravel()
            mZ = rho_z[g.mask_src].real.astype(np.float64).ravel()
            m_vecs = np.column_stack((mX, mY, mZ))

        def B_ijk(i: int, j: int, k: int, real_cut: float) -> np.ndarray:
            p = np.array([g.x_1d[i], g.y_1d[j], g.z_1d[k]], dtype=float)

            Brec = np.array([
                field_r_x[i, j, k],
                field_r_y[i, j, k],
                field_r_z[i, j, k]
            ], dtype=np.complex128)

            if e.use_nquad_nearfield:
                Breal = self._B_point_ewald_real(
                    p,
                    mode=mode_name,
                    kvec=kvec,
                    lambda_idx=lambda_idx,
                )
                return Brec.real.astype(np.float64) + Breal
            else:
                Rc = real_cut
                R = p[None, :] - src_xyz
                R2 = np.einsum('ij,ij->i', R, R)
                near_mask = (R2 < (Rc*Rc)) & (R2 > g.tol)

                Bnear = np.zeros(3, dtype=np.float64)
                if np.any(near_mask):
                    Rnear = R[near_mask]
                    Dnear = self._dipole_real_batch(Rnear, alpha)
                    mnear = m_vecs[near_mask]
                    Bnear = np.einsum('nij,nj->i', Dnear, mnear) * g.dV

                return Brec.real.astype(np.float64) + Bnear

        return B_ijk

    # -----------------------------
    # B-map on (x,y) at fixed z
    # -----------------------------
    
    def _compute_B_map_dataframe_on_fft_grid(self, B_ijk_fn, k_z: int, real_cut_factor: float):
        """
        Evaluates the magnetic field on a constant-z plane above the sample
        """
        g = self.grid
        alpha = self.alpha
        real_cut = real_cut_factor / alpha
        rows = []
        z_val = g.z_1d[k_z]
        for i in range(g.Nx):
            xi = g.x_1d[i]
            for j in range(g.Ny):
                yi = g.y_1d[j]
                Bp = B_ijk_fn(i, j, k_z, real_cut=real_cut)
                rows.append((xi, yi, z_val, Bp[0], Bp[1], Bp[2]))
        return pd.DataFrame(rows, columns=["xpos","ypos","zpos","bx","by","bz"])

    # =============================================================================
    # Parallelization helpers (Optimized for  BU's Shared Computing Cluster )
    # =============================================================================

    def process_one_task(self, task: Dict[str, Any], outdir: str | Path) -> Dict[str, Any]:
        outdir = str(outdir)
        Path(outdir).mkdir(parents=True, exist_ok=True)

        mode_name  = task["mode_name"]
        tag        = task["tag"]
        nx_i, ny_i, nz_i = task["n"]
        lam       = task["lambda_idx"]

        g = self.grid
        e = self.ewald

        kvec = np.array([
            np.pi*nx_i / g.Lx_src,
            np.pi*ny_i / g.Ly_src,
            np.pi*nz_i / g.Lz_src
        ], dtype=float)
        kmag  = float(np.linalg.norm(kvec))
        omega = v_mode * kmag
        k_scale_label = "pi/L"

        all_nonzero = (kvec != 0.0).all()
        if all_nonzero and lam is None:
            raise RuntimeError("Internal: missing polarization for all-nonzero k")
        if (not all_nonzero) and (lam is not None):
            raise RuntimeError("Internal: unexpected polarization when some k component is zero")

        rho_x, rho_y, rho_z = self._build_mode(
            mode_name, kvec=kvec, lambda_idx=lam
        )

        field_r_x, field_r_y, field_r_z = self._build_reciprocal_fields_vector(rho_x, rho_y, rho_z)

        B_ijk = self._make_ewald_grid_fn(
            field_r_x, field_r_y, field_r_z,
            rho_x, rho_y, rho_z,
            mode_name=mode_name,
            kvec=kvec,
            lambda_idx=lam,
        )

        if lam is None:
            fname_bmap = f"bmap_{tag}_{nx_i}_{ny_i}_{nz_i}.dat"
        else:
            fname_bmap = f"bmap_{tag}_{nx_i}_{ny_i}_{nz_i}_{lam}.dat"
        fpath_bmap = os.path.join(outdir, fname_bmap)

        df_bmap = self._compute_B_map_dataframe_on_fft_grid(
            B_ijk, g.k_obs, real_cut_factor=self.realcut_factor
        )
        df_bmap.to_csv(fpath_bmap, index=False)

        # Edge point diagnostics
        if e.measure_error:
            i_edge = g.Nx_pad + g.Nx_src - 1
            y_target = g.Ly_src/2.0 + g.Ly_src/5.0
            j_candidates = np.arange(g.Ny_pad, g.Ny_pad + g.Ny_src)
            y_candidates = g.y_1d[j_candidates]
            j_edge_local = int(np.argmin(np.abs(y_candidates - y_target)))
            j_edge = j_candidates[j_edge_local]
            k_edge = g.k_obs

            x_edge = float(g.x_1d[i_edge])
            y_edge = float(g.y_1d[j_edge])
            z_edge = float(g.z_1d[k_edge])
            p_edge = np.array([x_edge, y_edge, z_edge], float)

            B_ew_edge = B_ijk(i_edge, j_edge, k_edge, real_cut=self.realcut_factor/self.alpha)
            B_di_edge = self._B_point_direct_mode(
                p_edge,
                mode=mode_name,
                kvec=kvec,
                lambda_idx=lam,
            )
            rel_err_edge = float(
                np.linalg.norm(B_di_edge - B_ew_edge)
            ) / max(float(np.linalg.norm(B_di_edge)), 1e-20)
        else:
            x_edge = y_edge = np.nan
            B_ew_edge = np.array([np.nan, np.nan, np.nan], float)
            B_di_edge = np.array([np.nan, np.nan, np.nan], float)
            rel_err_edge = np.nan

        row = {
            "mode": mode_name,
            "tag": tag,
            "k_scale_used": k_scale_label,
            "nx": nx_i, "ny": ny_i, "nz": nz_i,
            "kx": kvec[0], "ky": kvec[1], "kz": kvec[2],
            "k_mag": kmag, "omega": omega,
            "lambda": (np.nan if lam is None else int(lam)),

            "Lx_src": g.Lx_src, "Ly_src": g.Ly_src, "Lz_src": g.Lz_src,
            "Lx_box": g.Lx_box, "Ly_box": g.Ly_box, "Lz_box": g.Lz_box,
            "z_obs": g.z_obs,
            "vacuum_target": g.vacuum_target,
            "vacuum_x": g.vacuum_x,
            "vacuum_y": g.vacuum_y,
            "vacuum_z": g.vacuum_z,
            "alpha_formula": self.alpha_formula,
            "alpha_value": self.alpha,
            "Rc": self.realcut_factor/self.alpha,
            "dx": g.dx, "dy": g.dy, "dz": g.dz,
            "Nx_fft": g.Nx, "Ny_fft": g.Ny, "Nz_fft": g.Nz,
            "Nx_map": g.Nx, "Ny_map": g.Ny,

            "bmap_file": os.path.basename(fpath_bmap),

            "probe_edge_x": x_edge,
            "probe_edge_y": y_edge,

            "B_direct_edge_x": B_di_edge[0],
            "B_direct_edge_y": B_di_edge[1],
            "B_direct_edge_z": B_di_edge[2],
            "B_ewald_edge_x":  B_ew_edge[0],
            "B_ewald_edge_y":  B_ew_edge[1],
            "B_ewald_edge_z":  B_ew_edge[2],
            "rel_error_edge":  rel_err_edge,
            "use_nquad_nearfield": int(e.use_nquad_nearfield),
        }
        print(
            f"Saved {fname_bmap} | tag={tag}, n=({nx_i},{ny_i},{nz_i}), λ={lam}, "
            f"|k|={kmag:.3e}, rel_err_edge={rel_err_edge:.2e}"
        )
        return row

    def run_one_task_and_write_row(self, task, outdir: str | Path):
        row = self.process_one_task(task, outdir)
        tag = row["tag"]; nx=row["nx"]; ny=row["ny"]; nz=row["nz"]; lam=row["lambda"]
        lam_s = "None" if (isinstance(lam, float) and math.isnan(lam)) or lam is None else str(int(lam))
        one_row_log = Path(outdir) / f"runlog_{tag}_{nx}_{ny}_{nz}_{lam_s}.csv"
        with one_row_log.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Wrote per-task log: {one_row_log}")

    def run_in_parallel(self, tasks, outdir: str | Path, max_procs: int):
        outdir = str(outdir)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        worker = partial(self.process_one_task, outdir=outdir)
        with Pool(processes=max_procs) as pool:
            for row in pool.imap_unordered(worker, tasks):
                yield row


# =============================================================================
# Mode enumeration helpers 
# =============================================================================
"""
Section for enumarating magnetization modes of spin ice. Aids code in choosing modes
for which stray magnetic field is to be evaluted for a given user choice in spin_ice_field.py
"""
def enumerate_ktriples_first_N(N=30, geom: GeometryConfig | None = None):
    if geom is None:
        geom = GeometryConfig()
    Lx_src = geom.Lx_src
    Ly_src = geom.Ly_src if geom.Ly_src is not None else Lx_src/(2*np.sqrt(2))
    Lz_src = geom.Lz_src
    triples = []
    r = 0
    while len(triples) < N:
        r += 1
        cand = []
        for nx in range(0, r+1):
            for ny in range(0, r+1):
                for nz in range(0, r+1):
                    if nx==0 and ny==0 and nz==0:
                        continue
                    if (nx>0) + (ny>0) + (nz>0) < 2:
                        continue
                    kx = np.pi*nx / Lx_src
                    ky = np.pi*ny / Ly_src
                    kz = np.pi*nz / Lz_src
                    kmag = np.sqrt(kx*kx + ky*ky + kz*kz)
                    cand.append(((nx,ny,nz), kmag))
        allc = {c[0]:c[1] for c in cand}
        triples = sorted(allc.items(), key=lambda kv: kv[1])
    return [t[0] for t in triples[:N]]

def enumerate_ktriples_first_N_sc(N=30, geom: GeometryConfig | None = None):
    if geom is None:
        geom = GeometryConfig()
    Lx_src = geom.Lx_src
    Ly_src = geom.Ly_src if geom.Ly_src is not None else Lx_src/(2*np.sqrt(2))
    Lz_src = geom.Lz_src
    triples = []
    r = 0
    while len(triples) < N:
        r += 1
        cand = []
        for nx in range(0, r+1):
            for ny in range(0, r+1):
                for nz in range(0, r+1):
                    if not (nx>0 and ny>0):
                        continue
                    kx = np.pi*nx / Lx_src
                    ky = np.pi*ny / Ly_src
                    kz = np.pi*nz / Lz_src
                    kmag = np.sqrt(kx*kx + ky*ky + kz*kz)
                    cand.append(((nx,ny,nz), kmag))
        allc = {c[0]: c[1] for c in cand}
        triples = sorted(allc.items(), key=lambda kv: kv[1])
    return [t[0] for t in triples[:N]]

def enumerate_ktriples_band(tag: str, omega_min: float, omega_max: float,
                            geom: GeometryConfig | None = None):
    if geom is None:
        geom = GeometryConfig()
    Lx_src = geom.Lx_src
    Ly_src = geom.Ly_src if geom.Ly_src is not None else Lx_src/(2*np.sqrt(2))
    Lz_src = geom.Lz_src

    k_scale = np.pi
    k_min = max(0.0, omega_min / v_mode)
    k_max = max(0.0, omega_max / v_mode)
    if k_max < k_min:
        k_min, k_max = k_max, k_min

    nx_max = int(np.floor(k_max * Lx_src / k_scale)) + 1
    ny_max = int(np.floor(k_max * Ly_src / k_scale)) + 1
    nz_max = int(np.floor(k_max * Lz_src / k_scale)) + 1

    triples = []
    for nx in range(0, nx_max+1):
        for ny in range(0, ny_max+1):
            for nz in range(0, nz_max+1):
                if nx==0 and ny==0 and nz==0:
                    continue
                if (nx>0) + (ny>0) + (nz>0) < 2:
                    continue
                kx = k_scale*nx / Lx_src
                ky = k_scale*ny / Ly_src
                kz = k_scale*nz / Lz_src
                kmag = float(np.sqrt(kx*kx + ky*ky + kz*kz))
                omega = v_mode * kmag
                if (omega >= omega_min) and (omega <= omega_max):
                    triples.append((nx,ny,nz))

    def kmag_of(n):
        nx,ny,nz = n
        kx = k_scale*nx / Lx_src
        ky = k_scale*ny / Ly_src
        kz = k_scale*nz / Lz_src
        return np.sqrt(kx*kx + ky*ky + kz*kz)
    triples = sorted(triples, key=kmag_of)
    return triples

def enumerate_ktriples_nlist(nlist):
    cleaned = []
    for t in nlist:
        nx, ny, nz = int(t[0]), int(t[1]), int(t[2])
        cleaned.append((nx, ny, nz))
    return {"ins": cleaned, "sc": cleaned}


#=====================================================================================================
#Task builder for paralllel evalation of magnetic field (Optimized for BU's Shared computing Cluster)
#=====================================================================================================
def make_tasks(modes_dict: Dict[str, List[Tuple[int,int,int]]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    tag_to_mode = {
        "ins": "spin_ice_insulating",
        "sc":  "spin_ice_superconducting",
    }
    for tag, triples in modes_dict.items():
        mode_name = tag_to_mode.get(tag)
        if mode_name is None:
            raise ValueError(f"Unknown tag '{tag}' (expected 'ins' or 'sc').")
        for nx, ny, nz in triples:
            all_nonzero = (nx != 0) and (ny != 0) and (nz != 0)
            lambda_list = ([0, 1] if all_nonzero else [None])
            for lam in lambda_list:
                tasks.append({
                    "mode_name": mode_name,
                    "tag": tag,
                    "n": (int(nx), int(ny), int(nz)),
                    "lambda_idx": lam if lam is None else int(lam),
                })
    return tasks

def dump_tasks_tsv(tasks, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["mode_name","tag","nx","ny","nz","lambda_idx"])
        for t in tasks:
            nx, ny, nz = t["n"]
            w.writerow([
                t["mode_name"],
                t["tag"],
                nx, ny, nz,
                "" if t["lambda_idx"] is None else int(t["lambda_idx"])
            ])

def load_tasks_tsv(path: str | Path):
    path = Path(path)
    tasks = []
    with path.open("r") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            lam = None if row["lambda_idx"]=="" else int(row["lambda_idx"])
            tasks.append({
                "mode_name": row["mode_name"],
                "tag": row["tag"],
                "n": (int(row["nx"]), int(row["ny"]), int(row["nz"])),
                "lambda_idx": lam,
            })
    return tasks
