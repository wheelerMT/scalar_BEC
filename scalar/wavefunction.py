import cupy as cp
from grid import Grid
from phase import Phase


class Wavefunction:
    def __init__(self, grid: Grid, g: float, N: float, system_type: str) -> None:
        self.psi = cp.empty((grid.Nx, grid.Ny), dtype='complex64')
        self.psi_k = cp.empty((grid.Nx, grid.Ny), dtype='complex64')
        self.g = g
        self.atom_number = N
        self.system_type = system_type  # Trapped or periodic system
        self.dx, self.dy = grid.dx, grid.dy

        # If a periodic domain, define periodic domain specific vars
        if system_type == 'periodic':
            self.n_0 = N / (grid.len_x * grid.len_y)
            self.V = 0  # Periodic box potential

        self.phase = cp.empty((grid.Nx, grid.Ny))

    def generate_initial_state(self, grid: Grid, phase: Phase):
        """
        :param grid:
        """

        if self.system_type == 'periodic':
            self._generate_phase(phase)
            self.psi = cp.sqrt(self.n_0) * cp.exp(1j * self.phase)
            self.psi_k = cp.fft.fft2(self.psi)

        elif self.system_type == 'trapped':
            "generate trap initial state"

    def _generate_phase(self, phase: Phase):
        self.phase = phase.phase

    def calc_atom_num(self, k_space=False):
        if k_space:
            return self.dx * self.dy * cp.sum(cp.abs(cp.fft.ifft2(self.psi_k)) ** 2)
        else:
            return self.dx * self.dy * cp.sum(cp.abs(self.psi) ** 2)
