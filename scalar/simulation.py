from grid import Grid
from wavefunction import Wavefunction
import cupy as cp


def _renormalise_atom_num(wfn: Wavefunction):
    wfn.psi_k = cp.fft.fft2(cp.sqrt(wfn.atom_number) * cp.fft.ifft2(wfn.psi_k)
                             / cp.sqrt(wfn.calc_atom_num(k_space=True)))


def _fix_phase(wfn: Wavefunction):
    wfn.psi = cp.fft.ifft2(wfn.psi_k)
    wfn.psi *= cp.exp(1j * wfn.phase) / cp.exp(1j * cp.angle(wfn.psi))
    wfn.psi_k = cp.fft.fft2(wfn.psi)


class Simulation:
    """Simulation object for the numerics.
    This contains the split-step evolution steps for
    propagating the wavefunction forward in time.

    Attributes
    ----------
    dt : float
        Time step used in the simulation.
    nframe : int
        Number of time steps before saving next dataset.
    """

    def __init__(self, dt: float, nframe: int):
        """Instantiate Simulation object.

        Parameters
        ----------
        dt : float
            Time step used in the simulation.
        nframe : int
            Number of time steps before saving next dataset.
        """

        self.dt = dt
        self.nframe = nframe

    def _kinetic_step(self, wfn: Wavefunction, Kgrid: Grid) -> None:
        """Performs half a kinetic step evolution.

        Parameters
        ----------
        wfn : Wavefunction
            The :class:`Wavefunction` object.
        Kgrid : Grid
            K-space :class:`Grid`.
        """

        wfn.psi_k *= cp.exp(-0.25 * 1j * self.dt * Kgrid.squared)

    def _potential_step(self, wfn: Wavefunction) -> None:
        """Performs a full potential evolution step.

        Parameters
        ----------
        wfn : Wavefunction
            The :class:`Wavefunction` object.
        """

        wfn.psi *= cp.exp(-1j * self.dt * (wfn.V + wfn.g * cp.abs(wfn.psi) ** 2))

    def imaginary_time(self, wfn: Wavefunction, Kgrid: Grid, nt: int) -> None:
        """Performs imaginary time evolution. Automatically updates :obj:`dt`
        to -1j * :obj:`dt` and reverts back at the end of the evolution.

        Parameters
        ----------
        wfn : Wavefunction
            The :class:`Wavefunction` object.
        Kgrid : Grid
            K-space :class:`Grid`.
        nt : int
            Number of time steps.
        """

        self.dt *= -1j  # Switch to imaginary time

        # Do imaginary time evolution
        for i in range(nt):
            self._kinetic_step(wfn, Kgrid)  # First Kinetic step

            wfn.psi = cp.fft.ifft2(wfn.psi_k)  # Inverse FFT

            self._potential_step(wfn)   # Potential step

            wfn.psi_k = cp.fft.fft2(wfn.psi)    # Forward FFT

            self._kinetic_step(wfn, Kgrid)  # Last kinetic step

            _renormalise_atom_num(wfn)

            _fix_phase(wfn)

            # Print current time:
            if i % 50 == 0:
                print(f"t = {i * self.dt}")

        self.dt *= 1j   # Switch back to real time

    def real_time(self, wfn: Wavefunction, Kgrid: Grid, nt: int):
        """Performs real time evolution.

        Parameters
        ----------
        wfn : Wavefunction
            The :class:`Wavefunction` object.
        Kgrid : Grid
            K-space :class:`Grid`.
        nt : int
            Number of time steps.
        """

        # Do real time evolution
        for i in range(nt):
            self._kinetic_step(wfn, Kgrid)  # First Kinetic step

            wfn.psi = cp.fft.ifft2(wfn.psi_k)  # Inverse FFT

            self._potential_step(wfn)  # Potential step

            wfn.psi_k = cp.fft.fft2(wfn.psi)  # Forward FFT

            self._kinetic_step(wfn, Kgrid)  # Last kinetic step

            # Print current time:
            if i % 50 == 0:
                print(f"t = {i * self.dt}")
