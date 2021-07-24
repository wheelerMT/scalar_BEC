import cupy as cp
from grid import Grid
from wavefunction import Wavefunction
from simulation import Simulation
from phase import Phase


# Grid variables
Nx, Ny = 1024, 1024
dx, dy = 1, 1
dkx, dky = 2 * cp.pi / (Nx * dx), 2 * cp.pi / (Ny * dy)

# Generate grid objects
Rgrid = Grid(Nx, Ny, dx, dy)  # Real-space grid
Kgrid = Grid(Nx, Ny, dkx, dky)  # K-space grid
Kgrid.fftshift()  # Shifts k-space grid into correct orientation

# Set up simulation object
dt = 1e-2
nframe = 20000  # Number of timesteps to save data
sim = Simulation(dt, nframe)

# Wavefunction parameters:
c0 = 3e-5  # Interaction strength
N = 3.2e9   # Atom number
wfn = Wavefunction(Rgrid, c0, N, system_type='periodic')   # Generate wavefunction obj

# Generate phase profile:
nvort = 100  # Number of vortices
thresh = 0.1  # Threshold for vortex separation
phase = Phase(nvort, thresh, Rgrid, 'random')

# Generate the initial state
wfn.generate_initial_state(Rgrid, phase)

# Do imaginary time evolution
sim.imaginary_time(wfn, Kgrid, 500)

# Do real time evolution
sim.real_time(wfn, Kgrid, 5000)
