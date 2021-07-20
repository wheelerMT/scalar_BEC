import cupy as cp
import grid
import wavefunction

# Grid variables
Nx, Ny = 512, 512
dx, dy = 1, 1
dkx, dky = 2 * cp.pi / (Nx * dx), 2 * cp.pi / (Ny * dy)

# Generate grid objects
Rgrid = grid.Grid(Nx, Ny, dx, dy)  # Real-space grid
Kgrid = grid.Grid(Nx, Ny, dkx, dky)  # K-space grid
Kgrid.fftshift()  # Shifts k-space grid into correct orientation

# Wavefunction parameters:
c0 = 3e-5  # Interaction strength
N = 3.2e9   # Atom number
wfn = wavefunction.Wavefunction(Rgrid, c0, N, system_type='periodic')   # Generate wavefunction obj
