import argparse
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import torch

from solver import SsfmGPGPU

def gaussian(X, Y, mu_x, mu_y, sigma_x, sigma_y, A):
    return A * torch.exp((-(X - mu_x)**2 / (2 * sigma_x**2) - (Y - mu_y)**2 / (2*sigma_y**2)))

parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit("Need to specify config")

with open(f"{args.config}", "rb") as f:
    pars = tomllib.load(f)


dev = torch.device("cuda")  # Change this to cpu if you don't have an nvidia gpu
startX = pars["startX"]
endX = pars["endX"]
samplesX = pars["samplesX"]
startY = pars["startY"]
endY = pars["endY"]
samplesY = pars["samplesY"]
dx = (endX - startX) / samplesX
dy = (endY - startY) / samplesY
x = torch.arange(startX, endX, dx, device=dev)
y = torch.arange(startY, endY, dy, device=dev)
gridX, gridY = torch.meshgrid(x, y, indexing="xy")
kxmax = np.pi / dx
kymax = np.pi / dy
dkx = 2 * kxmax / samplesX
psi = 1e-6 * (torch.rand((samplesY, samplesX), dtype=torch.cfloat, device=dev) + 1j * torch.rand((samplesY, samplesX), dtype=torch.cfloat, device=dev))

ones = torch.ones_like(gridX)
barrier = 20 * ((torch.abs(gridX) > 45) | (torch.abs(gridY) > 45))
constV = -0.5j * pars["gammalp"] * torch.ones((samplesY, samplesX), device=dev) - 1j * barrier#((gridX / 20)**2 + (gridY / 20)**2)**5

pump =  gaussian(gridX, gridY, 0, 0, pars["sigma"], pars["sigma"], pars["pumpStrength"])
nR = torch.zeros((samplesY, samplesX), device=dev)
gpsim = SsfmGPGPU(
    dev=dev,
    psi0=psi,
    dx=dx,
    dy=dy,
    m=pars["m"],
    nR0=nR,
    alpha=pars["alpha"],
    Gamma=pars["Gamma"],
    gammalp=pars["gammalp"],
    R=pars["R"],
    pump=pump,
    G=pars["G"],
    eta=pars["eta"],
    constV=constV,
    dt=pars["dt"],
)  # the constructor takes care of sending all arrays to the appropriate device

plt.ion()
fig, ax = plt.subplots(2)
im = ax[0].imshow((gpsim.psi.conj() * gpsim.psi).cpu().numpy().real, aspect="auto", extent=[startX, endX, startY, endY], origin="lower")
im2 = ax[1].imshow(gpsim.nR.cpu().numpy(), aspect="auto", extent=[startX, endX, startY, endY], origin="lower")
cbar1 = fig.colorbar(im, ax=ax[0])
cbar2 = fig.colorbar(im2, ax=ax[1])
plt.show()
while True:
    gpsim.step()
    gpsim.step()
    gpsim.step()
    gpsim.step()
    gpsim.step()
    data = (gpsim.psi.conj() * gpsim.psi).cpu().numpy().real
    nrdata = gpsim.nR.cpu().numpy()
    im.set_data(data)
    im.set_clim(vmin=0, vmax=np.max(data))
    im2.set_data(nrdata)
    im2.set_clim(vmin=0, vmax=np.max(nrdata))
    # cbar.set_clim(vmin=0, vmax=np.max(data))
    # cbar.draw_all()
    fig.canvas.draw()
    fig.canvas.flush_events()
