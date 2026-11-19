# Classes to time evolve initial condition according to the Schrödinger equation
import math

import torch
import torch.fft as tfft

hbar = 6.582119569e-1  # meV * ps

@torch.jit.script
def sq_norm(x):
    return x.real * x.real + x.imag * x.imag

class SsfmGPGPU:
    """
    A class for solving the Gross-Pitaevskii equation in 2D using the split step
    Fourier method. This class relies on PyTorch since it has a relatively
    comfortable interface for GPU computing and a big development team with major
    backers behind it. For this specific application, though, something like
    CuPy might be more optimal and have fewer unused features.
    """

    __slots__ = (
        "dev",
        "psi",
        "psik",
        "kxv",
        "kyv",
        "nR",
        "dt",
        "t",
        "R",
        "pump",
        "Gamma",
        "Gammainv",
        "m",
        "kTimeEvo",
        "alpha",
        "G",
        "eta",
        "constV",
        "constpart",
    )

    def __init__(
        self,
        dev,
        psi0,
        dx,
        dy,
        m,
        nR0,
        alpha,
        Gamma,
        gammalp,
        R,
        pump,
        G,
        eta,
        constV,
        dt,
    ):
        self.psi = psi0
        # self.n = psi.shape[0]
        self.psik = tfft.fft2(self.psi)
        # self.gridX = gridX.type(dtype=torch.cfloat).to(device=dev)
        # self.gridY = gridY.type(dtype=torch.cfloat).to(device=dev)
        # dx = gridX[0, 1] - gridX[0, 0]
        # dy = gridY[1, 0] - gridY[0, 0]
        kxmax = torch.pi / dx
        kymax = torch.pi / dy
        dkx = 2 * kxmax / psi0.size(dim=0)
        dky = 2 * kymax / psi0.size(dim=1)
        kx = torch.arange(-kxmax, kxmax, dkx, device=dev)
        ky = torch.arange(-kymax, kymax, dky, device=dev)
        kx = tfft.fftshift(kx)
        ky = tfft.fftshift(ky)
        kxv, kyv = torch.meshgrid(kx, ky, indexing="ij")
        self.t = 0
        self.dt = dt
        self.nR = nR0
        self.R = R
        self.pump = pump
        self.Gamma = Gamma
        self.Gammainv = 1.0 / Gamma
        self.m = m
        squareK = kxv * kxv + kyv * kyv
        self.kTimeEvo = torch.exp(-0.5j * hbar * squareK * dt / m)
        self.alpha = alpha
        self.eta = eta
        self.G = G
        self.constV = constV
        self.constpart = self.constV + G * eta * self.Gammainv * self.pump

    def updatePump(self, pump, dev):
        self.pump = pump
        self.constpart = (self.constV + self.G * self.eta * self.Gammainv * pump).to(
            device=dev
        )

    def V(self):
        return (
            self.constpart
            + self.alpha * self.psi.conj() * self.psi
            + (self.G + 0.5j * self.R) * self.nR
        )

    def halfRStepPsi(self):
        halfRTimeEvo = torch.exp(-0.5j * self.dt * self.V())
        self.psi = self.psi * halfRTimeEvo

    def rStepNR(self):
        rTimeEvo = torch.exp(
            -self.dt * (self.Gamma + self.R * sq_norm(self.psi))
        )
        self.nR = rTimeEvo * self.nR + self.pump * self.dt

    def psikNormSqr(self):
        return tfft.fftshift(self.psik.conj()) * tfft.fftshift(self.psik)

    def step(self):
        self.rStepNR()
        self.halfRStepPsi()
        tfft.fft2(self.psi, out=self.psik)
        self.psik = self.psik * self.kTimeEvo
        tfft.ifft2(self.psik, out=self.psi)
        self.halfRStepPsi()
        self.t += self.dt

