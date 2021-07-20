import numpy as np
from scalar import grid


class Wavefunction:
    def __init__(self, grid: grid.Grid, g: float, N: int, system_type: str) -> None:
        self.psi = np.empty((grid.Nx, grid.Ny), dtype='complex64')
        self.psi_k = np.empty((grid.Nx, grid.Ny), dtype='complex64')
        self.g = g
        self.atom_number = N
        self.system_type = system_type  # Trapped or periodic system

        # If a periodic domain, define periodic domain specific vars
        if system_type == 'periodic':
            self.n_0 = N / (grid.len_x * grid.len_y)
            self.V = 0  # Periodic box potential

        self.phase = None

    def generate_initial_state(self, grid: grid.Grid, phase=None):
        """
        :param grid:
        :param phase: Array of phase profile.
        """

        if self.system_type == 'periodic':
            self.psi = np.sqrt(self.n_0) * np.exp(1j * phase)
            self.phase = phase

        elif self.system_type == 'trapped':
            "generate trap initial state"

