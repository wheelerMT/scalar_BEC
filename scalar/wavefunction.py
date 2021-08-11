import cupy as cp
from grid import Grid
from phase import Phase


class Wavefunction:
    """Generates the Wavefunction object. Contains information
    about the BEC such as interaction strength and atom number.

    Attributes
    ----------
    psi : ndarray(complex)
        2D :obj:`ndarray` containing wavefunction values in position space.
    psi_k : ndarray(complex)
        2D :obj:`ndarray` containing wavefunction values in reciprocal space.
    g : float
        Interaction strength of condensate.
    atom_number : int
        Atom number of condensate.
    system_type : str
        Type of system condensate is contained in, e.g. 'periodic' or 'trapped'.
    dx : float
        Taken from Grid object - grid spacing in the x-direction.
    dx : float
        Taken from Grid object - grid spacing in the y-direction.
    """

    def __init__(self, grid: Grid, g: float, N: float, system_type: str):
        """Instantiate a wavefunction object.
        Sets BEC parameters using the parameters provided. Constructs empty positional
        and reciprocal wavefunction data arrays.

        Parameters
        ----------
        grid : Grid
            Real-space Grid object for the simulation.
        g : float
            Interaction strength of condensate.
        N : int
            Atom number of condensate.
        system_type : str
            The type of system used within the simulations - e.g., 'periodic' or 'trapped'.
        """

        self.psi = cp.empty((grid.nx, grid.ny), dtype='complex64')
        self.psi_k = cp.empty((grid.nx, grid.ny), dtype='complex64')
        self.g = g
        self.atom_number = N
        self.system_type = system_type  # Trapped or periodic system
        self.dx, self.dy = grid.dx, grid.dy

        # If a periodic domain, define periodic domain specific vars
        if system_type == 'periodic':
            self.n_0 = N / (grid.len_x * grid.len_y)
            self.V = 0  # Periodic box potential

        self.phase = cp.empty((grid.nx, grid.ny))

    def generate_initial_state(self, grid: Grid, phase: Phase):
        """Generates initial state for the condensate.
        For a periodic system, it generates a uniform background density coupled
        with a phase provided from a :class:`Phase` object.
        For a trapped system, it generates the Thomas-Fermi profile.

        Parameters
        ----------
        grid : Grid
            Real-space Grid object for the simulation.
        phase : Phase
            Phase object associated with this wavefunction.

        """

        if self.system_type == 'periodic':
            self._generate_phase(phase)
            self.psi = cp.sqrt(self.n_0) * cp.exp(1j * self.phase)
            self.psi_k = cp.fft.fft2(self.psi)

        elif self.system_type == 'trapped':
            raise NotImplementedError

    def _generate_phase(self, phase: Phase) -> None:
        """Updates wavefunction phase using :class:`Phase` object.

        Parameters
        ----------
        phase : Phase
            Phase object associated with this wavefunction.

        """

        self.phase = phase.phase

    def calc_atom_num(self, k_space=False):
        """Calculates the atom number of the system.

        Parameters
        ----------
        k_space : bool
            Determines whether to use k-space (True) wavefunction or
            real-space (False) wavefunction to calculate the atom number,
            defaults to False.

        Returns
        -------
        int
            Atom number of the system.
        """

        if k_space:
            return int(self.dx * self.dy * cp.sum(cp.abs(cp.fft.ifft2(self.psi_k)) ** 2))
        else:
            return int(self.dx * self.dy * cp.sum(cp.abs(self.psi) ** 2))
